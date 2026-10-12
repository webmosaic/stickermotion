"""A 429 in one client never cancels a neighbour's prediction (fake transport)."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stickermotion.engine.ai.costs import CostLedger, Price
from stickermotion.engine.ai.provider import (
    API,
    ApiError,
    PredictionRegistry,
    ReplicateClient,
)

MODEL = "owner/model"
PRICE = Price(estimate_usd=0.01, flat_usd=0.01)
TOKEN = "fake-test-token"


class RaceFake:
    """Replicate with two callers: A's create is slow (the prediction already exists on
    Replicate while the response is on its way), B's first create answers 429 after
    Replicate created an orphan anyway."""

    def __init__(self, list_inputs: bool = False) -> None:
        self.list_inputs = list_inputs
        self.lock = threading.Lock()
        self.predictions: dict[str, dict[str, Any]] = {}
        self.cancelled: list[str] = []
        self.a_created = threading.Event()
        self.release_a = threading.Event()
        self.b_creates = 0

    def _new(self, pid: str, model_input: dict[str, Any]) -> dict[str, Any]:
        item = {
            "id": pid,
            "model": MODEL,
            "status": "starting",
            "input": model_input,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "urls": {
                "get": f"{API}/predictions/{pid}",
                "cancel": f"{API}/predictions/{pid}/cancel",
            },
        }
        with self.lock:
            self.predictions[pid] = item
        return item

    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> bytes:
        if method == "POST" and url.endswith(f"/models/{MODEL}/predictions"):
            model_input = json.loads(body or b"{}")["input"]
            if model_input["prompt"] == "A":
                item = self._new("pA", model_input)
                self.a_created.set()
                assert self.release_a.wait(5)
                return json.dumps(item).encode()
            self.b_creates += 1
            if self.b_creates == 1:
                self._new("pB-orphan", model_input)
                raise ApiError(429, '{"retry_after": 1}')
            return json.dumps(self._new("pB", model_input)).encode()
        if method == "GET" and url == f"{API}/predictions":
            with self.lock:
                items = [dict(item) for item in self.predictions.values()]
            if not self.list_inputs:
                for item in items:
                    item.pop("input")
            return json.dumps({"results": items}).encode()
        if method == "POST" and url.endswith("/cancel"):
            pid = url.split("/predictions/")[1].split("/")[0]
            with self.lock:
                self.cancelled.append(pid)
                self.predictions[pid]["status"] = "canceled"
            return b"{}"
        if method == "GET" and "/predictions/" in url:
            pid = url.rsplit("/", 1)[1]
            with self.lock:
                item = self.predictions[pid]
                if item["status"] != "canceled":
                    item["status"] = "succeeded"
                    item["output"] = "ok"
                    item["metrics"] = {"predict_time": 1.0}
                return json.dumps(item).encode()
        raise AssertionError(f"unexpected call {method} {url}")


def _client(tmp_path: Path, name: str, fake: RaceFake, registry: PredictionRegistry) -> Any:
    ledger = CostLedger(tmp_path / f"{name}.jsonl", 1.0)
    return ReplicateClient(TOKEN, ledger, transport=fake, sleep=lambda _s: None, registry=registry)


def _race(tmp_path: Path, fake: RaceFake) -> dict[str, Any]:
    """A starts first and is mid-create when B gets its 429."""
    registry = PredictionRegistry()
    a = _client(tmp_path, "a", fake, registry)
    b = _client(tmp_path, "b", fake, registry)
    results: dict[str, Any] = {}

    def run_a() -> None:
        try:
            results["a"] = a.predict(MODEL, {"prompt": "A"}, PRICE, "a")
        except Exception as exc:  # surfaced by the assertions below
            results["a"] = exc

    thread = threading.Thread(target=run_a)
    thread.start()
    assert fake.a_created.wait(5)
    try:
        results["b"] = b.predict(MODEL, {"prompt": "B"}, PRICE, "b")
    finally:
        fake.release_a.set()
        thread.join(5)
    results["b_journal"] = [
        json.loads(line) for line in (tmp_path / "b.jsonl").read_text().splitlines()
    ]
    return results


def test_429_does_not_cancel_neighbour_in_flight(tmp_path: Path) -> None:
    """The listing has no input: the unknown predictions are left alone while A's create
    is in flight, so A's prediction survives."""
    fake = RaceFake(list_inputs=False)
    results = _race(tmp_path, fake)
    assert "pA" not in fake.cancelled
    assert results["a"]["status"] == "succeeded"
    assert results["b"]["status"] == "succeeded"
    assert results["b_journal"][0]["status"] == "429"
    assert results["b_journal"][0]["cancelled"] == []


def test_429_cancels_only_own_orphan_when_input_is_listed(tmp_path: Path) -> None:
    """With the input in the listing, B cancels its own orphan and never A's prediction."""
    fake = RaceFake(list_inputs=True)
    results = _race(tmp_path, fake)
    assert fake.cancelled == ["pB-orphan"]
    assert results["a"]["status"] == "succeeded"
    assert results["b"]["status"] == "succeeded"
    assert results["b_journal"][0]["cancelled"] == ["pB-orphan"]


def test_429_does_not_cancel_a_registered_neighbour(tmp_path: Path) -> None:
    """A prediction another client already registered is never an orphan."""
    fake = RaceFake()
    registry = PredictionRegistry()
    fake._new("pA", {"prompt": "A"})
    registry.begin(MODEL)
    registry.end(MODEL, "pA")
    b = _client(tmp_path, "b", fake, registry)
    b.predict(MODEL, {"prompt": "B"}, PRICE, "b")
    assert fake.cancelled == ["pB-orphan"]


def test_registry_counts_in_flight_creates() -> None:
    registry = PredictionRegistry()
    registry.begin(MODEL)
    registry.begin(MODEL)
    assert registry.pending(MODEL) == 2
    registry.end(MODEL)
    registry.end(MODEL, "p1")
    assert registry.pending(MODEL) == 0
    assert registry.known("p1") and not registry.known("p2")
