"""Action description: a vision LLM looks at the character and writes the video prompt.

The prompt follows a fixed template: what the character does (a natural action that
follows from its pose and emotion), what the decorative elements do, then fixed
requirements appended in code (static camera, flat background, return to the starting
pose, no new objects, same style), so they are always present.

Model: ``openai/gpt-5-nano`` on Replicate (same key), 0.05$ per million input tokens and
0.40$ per million output tokens (checked 2026-10-08) - the cheapest vision-capable LLM
in Replicate's vision collections with a published price. One description is about
0.0002$, two orders of magnitude below a Wan generation (0.05$).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

from stickermotion.engine.ai.costs import BudgetExceededError, Price
from stickermotion.engine.ai.provider import (
    MissingTokenError,
    PaymentRequiredError,
    PredictionFailedError,
    ReplicateClient,
    output_text,
    png_bytes,
)

log = logging.getLogger("stickermotion.ai")

DESCRIBE_MODEL = "openai/gpt-5-nano"
# Upper bound for the budget check: ~1.5k input tokens (image + instructions) and at most
# MAX_COMPLETION_TOKENS output tokens at the published price, rounded up.
DESCRIBE_PRICE = Price(estimate_usd=0.002, input_per_m=0.05, output_per_m=0.40)
MAX_COMPLETION_TOKENS = 2000
DESCRIBE_IMAGE_SIDE = 512
# Background for the retry after a moderation flag (ADR-016).
NEUTRAL = (128, 128, 128, 255)

# Positive wording only: Wan 2.2 fast has no negative prompt and reads words that
# name mouth actions (even inside "never ...") as a cue to perform them.
FIXED_REQUIREMENTS = (
    "At the end the character returns exactly to the starting pose. "
    "Hands and skin keep exactly their original colors in every frame, also while moving. "
    "Natural face, jaw and head proportions in every frame. "
    "The mouth keeps the size and shape of the original drawing and changes only as much "
    "as the described action itself requires. "
    "Hands and fingers keep their original size, shape and delicate character in every "
    "frame, also when moving towards the camera. "
    "Static camera, flat plain green background, no camera movement, no zoom, "
    "no new objects, same art style."
)

SYSTEM_PROMPT = (
    "You write prompts for an image-to-video model that animates a single drawn "
    "character for a looping chat sticker. Reply with JSON only."
)

INSTRUCTION = (
    "Look at the character on the green background. Describe one short natural action "
    "that follows from its pose and emotion (2-3 seconds, the kind of motion a sticker "
    "loops), then what the decorative elements around it do, if there are any.\n"
    "Rules: present tense, concrete body parts and directions, at most 60 words for the "
    "action and 25 for the decorations; keep every limb and accessory inside the "
    "picture; no camera moves, no new objects, no background, no style words; describe "
    "any mouth movement as small, gentle or slight, mention the mouth only when the "
    'action itself needs it; never describe the mouth with words like "wide", "huge" '
    'or "enormous".\n'
    'Answer as JSON: {"action": "<the character ...>", "decorations": "<... or empty>"}'
)


@dataclass(frozen=True)
class Description:
    prompt: str
    """Full prompt sent to the video model."""
    source: str
    """"manual" or the vision model name."""
    action: str
    decorations: str = ""


def build_prompt(action: str, decorations: str = "") -> str:
    """Template: action, decorations, fixed requirements."""

    def sentence(text: str) -> str:
        text = text.strip().rstrip(".")
        return text[:1].upper() + text[1:] + "."

    parts = [sentence(action)]
    if decorations.strip():
        parts.append(sentence(decorations))
    parts.append(FIXED_REQUIREMENTS)
    return " ".join(parts)


def manual(prompt: str) -> Description:
    """A prompt given by the user wins over the automatic one; fixed requirements are added."""
    text = prompt.strip()
    if not text:
        raise ValueError("manual prompt is empty")
    return Description(build_prompt(text), "manual", text)


def parse_reply(text: str) -> tuple[str, str]:
    """(action, decorations) from the model reply; plain text is taken as the action."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            data: Any = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and str(data.get("action", "")).strip():
            return str(data["action"]).strip(), str(data.get("decorations") or "").strip()
    cleaned = " ".join(text.split())
    if not cleaned:
        raise ValueError("the vision model returned an empty description")
    return cleaned, ""


def describe_input(image_ref: str) -> dict[str, Any]:
    """Model input for gpt-5-nano (the frame inlined or uploaded as ``image_ref``)."""
    return {
        "system_prompt": SYSTEM_PROMPT,
        "prompt": INSTRUCTION,
        "image_input": [image_ref],
        "reasoning_effort": "minimal",
        "verbosity": "low",
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }


def small_frame(frame: Image.Image, side: int = DESCRIBE_IMAGE_SIDE) -> Image.Image:
    """Downscaled copy for the vision model (fewer image tokens)."""
    copy = frame.convert("RGB")
    copy.thumbnail((side, side), Image.Resampling.LANCZOS)
    return copy


def is_moderation_error(exc: BaseException) -> bool:
    """Replicate's E005: the input was flagged as sensitive (failed, not billed)."""
    return isinstance(exc, PredictionFailedError) and "flagged as sensitive" in str(exc)


def on_neutral(image: Image.Image) -> Image.Image:
    """The image laid over the neutral grey (ADR-016)."""
    background = Image.new("RGBA", image.size, NEUTRAL)
    background.alpha_composite(image.convert("RGBA"))
    return background


def ask_vision(
    client: ReplicateClient,
    model_input: Callable[[str], dict[str, Any]],
    picture: Image.Image,
    price: Price,
    step: str,
    retry_picture: Image.Image | None = None,
) -> dict[str, Any]:
    """One vision model call; on Replicate's moderation flag (E005) once more on grey.

    ``retry_picture`` is what the second attempt sends (the subject on the neutral grey);
    None repeats ``picture``, for pictures that are on the grey already. Any other error
    and a second flag are raised.
    """
    filename = f"{step}.png"
    ref = client.image_ref(png_bytes(picture), filename)
    try:
        return client.predict(DESCRIBE_MODEL, model_input(ref), price, step)
    except PredictionFailedError as exc:
        if not is_moderation_error(exc):
            raise
    if retry_picture is not None:
        ref = client.image_ref(png_bytes(retry_picture), filename)
    return client.predict(DESCRIBE_MODEL, model_input(ref), price, step)


# Errors that stop a whole run: a helper step does not fall back on them, since the video
# generation right after it would fail the same way.
FATAL_ERRORS: tuple[type[BaseException], ...] = (
    BudgetExceededError,
    MissingTokenError,
    PaymentRequiredError,
)


def record_fallback(client: ReplicateClient, step: str, exc: BaseException, used: str) -> str:
    """Journal (cost 0) and log that a helper step fell back; returns the warning text."""
    error = client.redact(f"{type(exc).__name__}: {exc}")
    client.ledger.record(
        step=step, model=DESCRIBE_MODEL, status="fallback", error=error, fallback=used
    )
    message = f"{step}: vision model failed ({error}); fallback: {used}"
    log.warning(message)
    return message


def describe(
    client: ReplicateClient, frame: Image.Image, neutral: Image.Image | None = None
) -> Description:
    """Ask the vision model for the action and build the video prompt.

    ``neutral``: the character for the retry on grey after a moderation flag (ADR-016);
    without it a flag is raised at once.
    """
    image = small_frame(frame)
    if neutral is None:
        ref = client.image_ref(png_bytes(image), "describe.png")
        prediction = client.predict(DESCRIBE_MODEL, describe_input(ref), DESCRIBE_PRICE, "describe")
    else:
        prediction = ask_vision(
            client,
            describe_input,
            image,
            DESCRIBE_PRICE,
            "describe",
            small_frame(on_neutral(neutral)),
        )
    action, decorations = parse_reply(output_text(prediction.get("output")))
    return Description(build_prompt(action, decorations), DESCRIBE_MODEL, action, decorations)
