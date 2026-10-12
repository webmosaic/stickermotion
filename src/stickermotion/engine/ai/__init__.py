"""AI animation engine (ADR-010, ADR-011).

PNG -> action description (vision LLM or manual prompt) -> framing on green -> Wan 2.2
i2v fast (first frame = last frame, cached by inputs) -> chroma key -> loop ->
MasterAnimation. Paid calls go through ``provider.ReplicateClient`` with a per-run budget
and a cost journal.
"""

from stickermotion.engine.ai.chain import ChainResult, CutMethod, process
from stickermotion.engine.ai.costs import DEFAULT_BUDGET_USD, BudgetExceededError, CostLedger
from stickermotion.engine.ai.describe import Description, describe, manual
from stickermotion.engine.ai.framing import Framing, Margins, frame_for_model
from stickermotion.engine.ai.generate import Generation, generate
from stickermotion.engine.ai.provider import (
    MissingTokenError,
    PaymentRequiredError,
    ProviderError,
    RateLimitedError,
    ReplicateClient,
    read_token,
)

__all__ = [
    "DEFAULT_BUDGET_USD",
    "BudgetExceededError",
    "ChainResult",
    "CostLedger",
    "CutMethod",
    "Description",
    "Framing",
    "Generation",
    "Margins",
    "MissingTokenError",
    "PaymentRequiredError",
    "ProviderError",
    "RateLimitedError",
    "ReplicateClient",
    "describe",
    "frame_for_model",
    "generate",
    "manual",
    "process",
    "read_token",
]
