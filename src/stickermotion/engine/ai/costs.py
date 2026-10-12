"""Spend control: a per-run budget checked before every paid call, and the cost journal.

Each paid call (and each cache hit, with cost 0) appends one JSON line to
``<out>/costs.jsonl``: time, run id, step, model, input parameters (image data replaced
by a short marker), prediction id, status, cost in USD. The API key never reaches the
journal: every entry passes through ``redact`` before it is written.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BUDGET_USD = 0.10


class BudgetExceededError(RuntimeError):
    """A paid call would push this run's spend over its budget; nothing was called."""


@dataclass(frozen=True)
class Price:
    """Published price of a model on Replicate (checked 2026-10-08).

    ``flat_usd``: per output (video). ``input_per_m`` / ``output_per_m``: USD per million
    tokens (language models). ``estimate_usd`` is the upper bound checked against the
    budget before the call.
    """

    estimate_usd: float
    flat_usd: float | None = None
    input_per_m: float = 0.0
    output_per_m: float = 0.0

    def actual(self, metrics: dict[str, Any], succeeded: bool) -> float:
        """Cost of a finished prediction from its Replicate metrics."""
        if self.flat_usd is not None:
            return self.flat_usd if succeeded else 0.0
        # Replicate reports token_input_count / token_output_count (seen 2026-10-08 on
        # openai/gpt-5-nano); the other spelling is accepted too.
        tokens_in = metrics.get("token_input_count", metrics.get("input_token_count"))
        tokens_out = metrics.get("token_output_count", metrics.get("output_token_count"))
        if tokens_in is None and tokens_out is None:
            # No billing metrics reported: book the upper bound rather than nothing.
            return self.estimate_usd if succeeded else 0.0
        tokens_in = float(tokens_in or 0)
        tokens_out = float(tokens_out or 0)
        return round(tokens_in * self.input_per_m / 1e6 + tokens_out * self.output_per_m / 1e6, 6)


def summarize_input(model_input: dict[str, Any]) -> dict[str, Any]:
    """Model input for the journal: data URIs and file URLs replaced by markers."""

    def short(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("data:"):
            return f"<{value.split(';', 1)[0][5:]} {len(value)} chars>"
        if isinstance(value, str) and value.startswith("https://api.replicate.com/v1/files"):
            return "<uploaded file>"
        if isinstance(value, list):
            return [short(v) for v in value]
        return value

    return {k: short(v) for k, v in model_input.items()}


@dataclass
class CostLedger:
    """Budget of one run plus its journal file."""

    journal: Path
    budget_usd: float = DEFAULT_BUDGET_USD
    redact: Callable[[str], str] = lambda text: text
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spent_usd: float = 0.0
    entries: list[dict[str, Any]] = field(default_factory=list)

    def check(self, estimate_usd: float, what: str) -> None:
        """Raise before a paid call whose upper-bound cost does not fit the budget."""
        if self.spent_usd + estimate_usd > self.budget_usd + 1e-9:
            raise BudgetExceededError(
                f"{what}: up to {estimate_usd:.4f}$ would exceed the run budget "
                f"{self.budget_usd:.4f}$ (spent {self.spent_usd:.4f}$); nothing was called"
            )

    def record(self, **entry: Any) -> dict[str, Any]:
        """Append an entry (cost_usd is added to the run's spend)."""
        cost = float(entry.get("cost_usd") or 0.0)
        self.spent_usd = round(self.spent_usd + cost, 6)
        full = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": self.run_id,
            **entry,
            "cost_usd": cost,
        }
        text = self.redact(json.dumps(full, ensure_ascii=False))
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        clean = json.loads(text)
        self.entries.append(clean)
        return clean
