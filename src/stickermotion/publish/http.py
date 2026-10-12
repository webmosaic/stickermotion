"""Outgoing HTTP for the publishers: a replaceable transport, multipart bodies, errors.

Only outgoing calls (ADR-017): the service never reads bot updates and never connects to
the Discord gateway. Tests replace the transport with fakes, so no request leaves the
machine. Bot tokens are part of URLs (Telegram) or headers (Discord); every message that
can reach the user, a log line or the database goes through ``redact`` first.
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

TIMEOUT_S = 60
# Longest raw platform error kept for the publication log.
RAW_LIMIT = 2000


class PublishError(RuntimeError):
    """A publication failed; ``str(exc)`` is a user-facing Ukrainian message.

    ``raw`` is the platform's own error response (secrets removed) for the publication log,
    empty when there is none.
    """

    raw: str = ""


class NotConfiguredError(PublishError):
    """The platform lacks environment variables; ``missing`` lists them."""

    def __init__(self, platform: str, missing: Iterable[str]) -> None:
        self.missing = tuple(missing)
        super().__init__(f"{platform} не налаштовано: бракує {', '.join(self.missing)}.")


class NetworkError(PublishError):
    """The platform could not be reached at all."""


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8") or "null")
        except (UnicodeDecodeError, ValueError):
            return None


Transport = Callable[[str, str, dict[str, str], bytes | None], Response]
"""``(method, url, headers, body) -> Response``; HTTP errors are responses, not raises."""


def urllib_transport(
    method: str, url: str, headers: dict[str, str], body: bytes | None
) -> Response:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return Response(response.status, response.read(), dict(response.headers.items()))
    except urllib.error.HTTPError as exc:
        headers_out = dict(exc.headers.items()) if exc.headers else {}
        return Response(exc.code, exc.read(), headers_out)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # The URL may carry a token (Telegram), so it never goes into the message.
        reason = getattr(exc, "reason", exc)
        raise NetworkError(f"Немає зв'язку з платформою ({type(reason).__name__}).") from None


def body_snippet(body: bytes, limit: int = 200) -> str:
    """Short single-line excerpt of a non-JSON response body for error messages."""
    text = " ".join(body.decode("utf-8", errors="replace").split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def raw_text(body: bytes, limit: int = RAW_LIMIT) -> str:
    """Whole response body as one line (capped at ``limit`` characters) for the log."""
    text = " ".join(body.decode("utf-8", errors="replace").split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def redact(text: str, *secrets_: str | None) -> str:
    for secret in secrets_:
        if secret:
            text = text.replace(secret, "***")
    return text


@dataclass(frozen=True)
class FilePart:
    field: str
    filename: str
    data: bytes
    content_type: str = ""


def multipart(fields: dict[str, str], files: Iterable[FilePart] = ()) -> tuple[bytes, str]:
    """``multipart/form-data`` body and its content type."""
    boundary = "sm" + secrets.token_hex(16)
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            + value.encode("utf-8")
            + b"\r\n"
        )
    for item in files:
        kind = (
            item.content_type
            or mimetypes.guess_type(item.filename)[0]
            or "application/octet-stream"
        )
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{item.field}"; '
                f'filename="{item.filename}"\r\nContent-Type: {kind}\r\n\r\n'
            ).encode()
            + item.data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
