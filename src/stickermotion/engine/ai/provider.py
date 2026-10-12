"""Replicate client with spend control, and the two models the AI chain calls.

- Budget: checked before every paid call (``CostLedger.check``).
- 402 (insufficient credit): no retry, ``PaymentRequiredError``.
- 429 (rate limit): Replicate may still have created the prediction, so before
  the delayed retry the client lists recent predictions and cancels the ones this run
  created that are still queued or running. Then it retries, up to ``max_retries`` times.
  Predictions of other clients in the same process are never cancelled: their ids
  are in a shared registry, and while another create of the same model is in flight an
  unknown prediction cannot be attributed, so it is left alone.
- Every outcome is journaled. The token is read from ``REPLICATE_API_TOKEN`` (environment,
  then ``.env``) and is redacted from every journal line and error message.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from stickermotion.engine.ai.costs import CostLedger, Price, summarize_input

API = "https://api.replicate.com/v1"
TOKEN_ENV = "REPLICATE_API_TOKEN"
USER_AGENT = "stickermotion/0.1"
# Images up to this size go inline as data URIs, larger ones through the Files API.
DATA_URI_LIMIT = 256 * 1024
ACTIVE = ("starting", "processing")


class ProviderError(RuntimeError):
    """Any failed call to the generation provider."""


class MissingTokenError(ProviderError):
    """REPLICATE_API_TOKEN is not set (environment or .env)."""


class PaymentRequiredError(ProviderError):
    """HTTP 402: not enough credit. Never retried."""


class RateLimitedError(ProviderError):
    """HTTP 429 persisted after the allowed retries."""


class PredictionFailedError(ProviderError):
    """The prediction failed, was cancelled or timed out."""


class ApiError(ProviderError):
    """HTTP error from the API, with status and response text."""

    def __init__(self, status: int, text: str, retry_after: float | None = None) -> None:
        super().__init__(f"HTTP {status}: {text.strip()}")
        self.status = status
        self.text = text
        self.retry_after = retry_after


Transport = Callable[[str, str, dict[str, str], bytes | None], bytes]


def urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        header = exc.headers.get("Retry-After") if exc.headers else None
        retry_after = float(header) if header and header.replace(".", "", 1).isdigit() else None
        raise ApiError(exc.code, exc.read().decode("utf-8", "replace"), retry_after) from None


def read_token(dotenv: Path | None = Path(".env")) -> str | None:
    """Token from the environment, else from a ``KEY=value`` .env file."""
    token = os.environ.get(TOKEN_ENV)
    if token:
        return token.strip()
    if dotenv is None or not dotenv.is_file():
        return None
    for line in dotenv.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.strip() == TOKEN_ENV:
            return value.strip().strip('"').strip("'") or None
    return None


def png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _retry_after(exc: ApiError, default: float) -> float:
    if exc.retry_after is not None:
        return exc.retry_after
    try:
        value = json.loads(exc.text).get("retry_after")
    except (ValueError, AttributeError):
        value = None
    return float(value) if isinstance(value, int | float) else default


class PredictionRegistry:
    """Process-wide, thread-safe record of predictions created by any ``ReplicateClient``.

    ``begin``/``end`` bracket every create call, so an active prediction that is neither
    known nor covered by another in-flight create of the same model can only come from a
    create that answered 429 (an orphan).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: set[str] = set()
        self._pending: dict[str, int] = {}

    def begin(self, model: str) -> None:
        with self._lock:
            self._pending[model] = self._pending.get(model, 0) + 1

    def end(self, model: str, prediction_id: str | None = None) -> None:
        with self._lock:
            if prediction_id:
                self._ids.add(prediction_id)
            left = self._pending.get(model, 0) - 1
            if left > 0:
                self._pending[model] = left
            else:
                self._pending.pop(model, None)

    def known(self, prediction_id: str) -> bool:
        with self._lock:
            return prediction_id in self._ids

    def pending(self, model: str) -> int:
        with self._lock:
            return self._pending.get(model, 0)


REGISTRY = PredictionRegistry()
"""Shared by every client of the process (web worker threads, CLI)."""


@dataclass
class ReplicateClient:
    token: str | None
    ledger: CostLedger
    transport: Transport = urllib_transport
    sleep: Callable[[float], None] = time.sleep
    max_retries: int = 3
    retry_delay_s: float = 12.0
    poll_s: float = 2.0
    timeout_s: float = 900.0
    registry: PredictionRegistry = field(default_factory=lambda: REGISTRY)

    def __post_init__(self) -> None:
        token = self.token
        previous = self.ledger.redact

        def redact(text: str) -> str:
            text = previous(text)
            return text.replace(token, "***") if token else text

        self.ledger.redact = redact

    def redact(self, text: str) -> str:
        return self.ledger.redact(text)

    # HTTP

    def _headers(self, content_type: str | None = "application/json") -> dict[str, str]:
        if not self.token:
            raise MissingTokenError(f"{TOKEN_ENV} is not set (environment or .env)")
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": USER_AGENT}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _call(self, method: str, url: str, body: bytes | None, headers: dict[str, str]) -> bytes:
        try:
            return self.transport(method, url, headers, body)
        except ApiError as exc:
            raise ApiError(exc.status, self.redact(exc.text), exc.retry_after) from None
        except OSError as exc:
            raise ProviderError(self.redact(f"{method} {url}: {exc}")) from None

    def _json(self, method: str, url: str, body: dict[str, Any] | None = None) -> Any:
        payload = json.dumps(body).encode() if body is not None else None
        headers = self._headers("application/json" if payload else None)
        return json.loads(self._call(method, url, payload, headers))

    def image_ref(self, payload: bytes, filename: str) -> str:
        """Data URI for small images, Files API URL for large ones."""
        if len(payload) <= DATA_URI_LIMIT:
            return "data:image/png;base64," + base64.b64encode(payload).decode()
        boundary = uuid.uuid4().hex
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="content"; '
                f'filename="{filename}"\r\nContent-Type: image/png\r\n\r\n'
            ).encode()
            + payload
            + f"\r\n--{boundary}--\r\n".encode()
        )
        headers = self._headers(f"multipart/form-data; boundary={boundary}")
        data = json.loads(self._call("POST", f"{API}/files", body, headers))
        return str(data["urls"]["get"])

    def download(self, url: str) -> bytes:
        return self._call("GET", url, None, {"User-Agent": USER_AGENT})

    # Orphans after 429

    def cancel_orphans(
        self, model: str, since: datetime, model_input: dict[str, Any] | None = None
    ) -> list[str]:
        """Cancel predictions of ``model`` this call created that are still queued/running.

        Replicate can create a prediction even when the create call answered 429. An active
        prediction of the model created after the call started is such an orphan only if
        no client of this process knows its id, its input (when the listing carries one)
        equals ``model_input``, and no other create of the same model is in flight
        (otherwise it may be a neighbour's prediction that is about to be registered).
        """
        try:
            listing = self._json("GET", f"{API}/predictions")
        except ProviderError:
            return []
        cancelled = []
        for item in listing.get("results", []):
            created = _parse_time(item.get("created_at"))
            pid = str(item.get("id"))
            if (
                item.get("model") != model
                or item.get("status") not in ACTIVE
                or created is None
                or created < since
                or self.registry.known(pid)
            ):
                continue
            if "input" in item and model_input is not None:
                if item["input"] != model_input:
                    continue  # another caller's prediction
            elif self.registry.pending(model):
                continue  # cannot be told apart from a neighbour's in-flight create
            url = (item.get("urls") or {}).get("cancel") or f"{API}/predictions/{pid}/cancel"
            try:
                self._json("POST", url, {})
                cancelled.append(pid)
            except ProviderError:
                continue
        return cancelled

    # Run

    def _create(self, model: str, model_input: dict[str, Any]) -> dict[str, Any]:
        """POST the prediction; the registry covers the call until its id is known."""
        self.registry.begin(model)
        try:
            prediction: dict[str, Any] = self._json(
                "POST", f"{API}/models/{model}/predictions", {"input": model_input}
            )
        except BaseException:
            self.registry.end(model)
            raise
        self.registry.end(model, str(prediction.get("id")))
        return prediction

    def predict(
        self, model: str, model_input: dict[str, Any], price: Price, step: str
    ) -> dict[str, Any]:
        """Create a prediction, wait for it and journal the cost. Returns the prediction."""
        params = summarize_input(model_input)
        self.ledger.check(price.estimate_usd, f"{step} ({model})")
        self._headers()  # fail on a missing token before anything else
        since = datetime.now(UTC) - timedelta(seconds=5)
        attempt = 0
        while True:
            try:
                prediction = self._create(model, model_input)
                break
            except ApiError as exc:
                if exc.status == 402:
                    self.ledger.record(
                        step=step, model=model, params=params, status="402", error=str(exc)
                    )
                    raise PaymentRequiredError(
                        f"{step}: Replicate answered 402 (insufficient credit); not retried. "
                        "Top up the balance and run again."
                    ) from None
                if exc.status != 429:
                    self.ledger.record(
                        step=step,
                        model=model,
                        params=params,
                        status=str(exc.status),
                        error=str(exc),
                    )
                    raise
                cancelled = self.cancel_orphans(model, since, model_input)
                attempt += 1
                delay = _retry_after(exc, self.retry_delay_s)
                self.ledger.record(
                    step=step,
                    model=model,
                    params=params,
                    status="429",
                    error=str(exc),
                    cancelled=cancelled,
                    retry=attempt <= self.max_retries,
                    retry_in_s=delay,
                )
                if attempt > self.max_retries:
                    raise RateLimitedError(
                        f"{step}: still rate limited after {self.max_retries} retries"
                    ) from None
                self.sleep(delay)
        prediction = self._wait(prediction)
        succeeded = prediction.get("status") == "succeeded"
        metrics = prediction.get("metrics") or {}
        self.ledger.record(
            step=step,
            model=model,
            params=params,
            prediction_id=prediction.get("id"),
            status=prediction.get("status"),
            cost_usd=price.actual(metrics, succeeded),
            predict_time_s=metrics.get("predict_time"),
            metrics=metrics,
            error=None if succeeded else self.redact(str(prediction.get("error"))),
        )
        if not succeeded:
            raise PredictionFailedError(
                f"{step}: prediction {prediction.get('id')} {prediction.get('status')}: "
                f"{self.redact(str(prediction.get('error')))}"
            )
        return prediction

    def _wait(self, prediction: dict[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_s
        while prediction.get("status") not in ("succeeded", "failed", "canceled"):
            if time.monotonic() > deadline:
                pid = prediction.get("id")
                with contextlib.suppress(ProviderError):
                    self._json("POST", f"{API}/predictions/{pid}/cancel", {})
                prediction = {**prediction, "status": "canceled", "error": "timeout"}
                break
            self.sleep(self.poll_s)
            prediction = self._json("GET", prediction["urls"]["get"])
        return prediction


def output_text(output: Any) -> str:
    """Language model output: a string or a list of streamed chunks."""
    if isinstance(output, list):
        return "".join(str(part) for part in output)
    return str(output or "")


def output_url(output: Any) -> str:
    url = output[0] if isinstance(output, list) else output
    if not isinstance(url, str) or not url:
        raise PredictionFailedError(f"no output URL in {output!r}")
    return url
