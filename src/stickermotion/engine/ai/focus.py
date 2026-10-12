"""Key zone for emotes: crop the face / head with its gesture so it reads at 28 px.

Emote targets (Twitch GIF, Discord emoji, Telegram emoji) are shown at about 28 px, where
the whole composition turns into a blur. ``--emote-focus`` picks a zone of the input PNG:

- ``full``: the whole composition (no crop, as before);
- ``X,Y,W,H``: a manual rectangle in input PNG pixels;
- ``auto``: the cheap vision model (``describe.DESCRIBE_MODEL``) returns the rectangle
  as JSON; an invalid reply or an implausible rectangle falls back to ``full`` with a
  warning. So does a model error: a moderation flag (E005) is retried once,
  then the step falls back and the run goes on.

The zone is mapped onto the generated video through the framing, then grown to hold
the motion within it (alpha of every loop frame, not one frame) and made square.
Sticker targets and OBS always get the full composition.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from stickermotion.engine.ai.costs import Price
from stickermotion.engine.ai.describe import (
    DESCRIBE_MODEL,
    DESCRIBE_PRICE,
    FATAL_ERRORS,
    ask_vision,
    record_fallback,
    small_frame,
)
from stickermotion.engine.ai.framing import Framing
from stickermotion.engine.ai.provider import ReplicateClient, output_text

AUTO = "auto"
FULL = "full"
FOCUS_PRICE: Price = DESCRIBE_PRICE
FOCUS_BACKGROUND = (128, 128, 128)

# A plausible key zone covers this share of the character's box (area).
MIN_AREA = 0.02
MAX_AREA = 0.85
# ...and is at least this opaque (share of visible pixels inside it).
MIN_OPAQUE = 0.15
# Alpha that counts as visible when measuring the zone and the motion.
ALPHA_VISIBLE = 128
# Motion slightly past the zone is kept up to this far around it (share of its side).
MOTION_REACH = 0.12
# Free space around the focused square (share of its side).
FOCUS_MARGIN = 0.06

SYSTEM_PROMPT = (
    "You locate regions in images of drawn characters for chat emotes. Reply with JSON only."
)
INSTRUCTION = (
    "This character will be shown as a tiny 28 px chat emote. Find the smallest, roughly "
    "square rectangle that still carries its emotion: the face and head, plus a hand only "
    "where it touches the face. Leave out raised arms, the body and decorations; the face "
    "should fill most of the rectangle.\n"
    "Coordinates are fractions of the image width and height, origin at the top-left "
    "corner. Answer as JSON: "
    '{"x": <left 0..1>, "y": <top 0..1>, "w": <width 0..1>, "h": <height 0..1>, '
    '"what": "<few words>"}'
)

Box = tuple[int, int, int, int]
"""(left, top, right, bottom) in pixels, right / bottom exclusive."""


class FocusError(ValueError):
    """A focus rectangle could not be parsed or is implausible."""


@dataclass(frozen=True)
class FocusSpec:
    mode: str
    """``auto``, ``full`` or ``manual``."""
    box: Box | None = None
    """Manual rectangle in input PNG pixels."""


def parse_spec(text: str) -> FocusSpec:
    """``auto`` | ``full`` | ``X,Y,W,H`` (input PNG pixels)."""
    value = text.strip().lower()
    if value in (AUTO, FULL):
        return FocusSpec(value)
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise FocusError(f"emote focus must be auto, full or X,Y,W,H; got {text!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise FocusError(f"X,Y,W,H must be integers; got {text!r}") from exc
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise FocusError(f"X,Y must be >= 0 and W,H > 0; got {text!r}")
    return FocusSpec("manual", (x, y, x + w, y + h))


def source_bbox(image: Image.Image, threshold: int = 8) -> Box:
    """Box of the visible character in the input PNG (same threshold as the preprocess)."""
    alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    visible = alpha >= threshold
    if not visible.any():
        raise FocusError("the input has no visible pixels")
    rows = np.flatnonzero(visible.any(axis=1))
    cols = np.flatnonzero(visible.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def check_box(box: Box, image: Image.Image) -> Box:
    """Clip a rectangle in input pixels to the character's box and check it is plausible."""
    sx0, sy0, sx1, sy1 = source_bbox(image)
    x0, y0 = max(box[0], sx0), max(box[1], sy0)
    x1, y1 = min(box[2], sx1), min(box[3], sy1)
    if x1 <= x0 or y1 <= y0:
        raise FocusError(f"rectangle {box} misses the character {(sx0, sy0, sx1, sy1)}")
    area = (x1 - x0) * (y1 - y0) / ((sx1 - sx0) * (sy1 - sy0))
    if not MIN_AREA <= area <= MAX_AREA:
        raise FocusError(
            f"rectangle {box} covers {area:.0%} of the character "
            f"(expected {MIN_AREA:.0%}..{MAX_AREA:.0%})"
        )
    alpha = np.asarray(image.convert("RGBA").getchannel("A"))[y0:y1, x0:x1]
    opaque = float((alpha >= ALPHA_VISIBLE).mean())
    if opaque < MIN_OPAQUE:
        raise FocusError(f"rectangle {box} is mostly transparent ({opaque:.0%} visible)")
    return x0, y0, x1, y1


def parse_reply(text: str) -> tuple[float, float, float, float]:
    """(x, y, w, h) fractions from the vision model reply; raises FocusError."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise FocusError(f"no JSON in the reply: {text[:200]!r}")
    try:
        data: Any = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise FocusError(f"invalid JSON in the reply: {text[:200]!r}") from exc
    if not isinstance(data, dict):
        raise FocusError("the reply is not a JSON object")
    try:
        values = tuple(float(data[k]) for k in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError) as exc:
        raise FocusError(f"the reply lacks numeric x, y, w, h: {data!r}") from exc
    x, y, w, h = values
    if not all(math.isfinite(v) for v in values):
        raise FocusError(f"non-finite coordinates: {data!r}")
    if any(v > 1.5 for v in values):
        raise FocusError(f"coordinates are not fractions of the image: {data!r}")
    if w <= 0 or h <= 0 or x < -0.05 or y < -0.05 or x + w > 1.05 or y + h > 1.05:
        raise FocusError(f"rectangle outside the image: {data!r}")
    return x, y, w, h


def fractions_to_box(fractions: tuple[float, float, float, float], crop: Box) -> Box:
    """Fractions of the cropped character image -> input PNG pixels."""
    x, y, w, h = fractions
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    return (
        crop[0] + round(max(0.0, x) * cw),
        crop[1] + round(max(0.0, y) * ch),
        crop[0] + round(min(1.0, x + w) * cw),
        crop[1] + round(min(1.0, y + h) * ch),
    )


def vision_image(image: Image.Image) -> tuple[Image.Image, Box]:
    """The character cropped to its box on a neutral grey, downscaled; and that box."""
    crop = source_bbox(image)
    sprite = image.convert("RGBA").crop(crop)
    canvas = Image.new("RGBA", sprite.size, (*FOCUS_BACKGROUND, 255))
    canvas.alpha_composite(sprite)
    return small_frame(canvas.convert("RGB")), crop


def focus_input(image_ref: str) -> dict[str, Any]:
    return {
        "system_prompt": SYSTEM_PROMPT,
        "prompt": INSTRUCTION,
        "image_input": [image_ref],
        "reasoning_effort": "minimal",
        "verbosity": "low",
        "max_completion_tokens": 1000,
    }


@dataclass
class Focus:
    """Resolved key zone for emotes."""

    requested: str
    """What was asked: ``auto``, ``full`` or ``manual``."""
    mode: str
    """What is used: ``auto``, ``manual`` or ``full`` (after a fallback)."""
    source_box: Box | None = None
    """Zone in input PNG pixels (None for full)."""
    reply: str | None = None
    fractions: tuple[float, float, float, float] | None = None
    cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.source_box is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "mode": self.mode,
            "source_box": list(self.source_box) if self.source_box else None,
            "model": DESCRIBE_MODEL if self.requested == AUTO else None,
            "reply": self.reply,
            "fractions": list(self.fractions) if self.fractions else None,
            "cost_usd": self.cost_usd,
            "warnings": self.warnings,
        }


def resolve(spec: FocusSpec, image: Image.Image, client: ReplicateClient | None) -> Focus:
    """Zone in input pixels for a spec; ``auto`` calls the vision model once."""
    if spec.mode == FULL:
        return Focus(FULL, FULL)
    if spec.mode == "manual":
        assert spec.box is not None
        try:
            return Focus("manual", "manual", check_box(spec.box, image))
        except FocusError as exc:
            return Focus("manual", FULL, warnings=[f"manual focus rejected: {exc}; using full"])
    if client is None:
        raise ValueError("auto focus needs a client")
    picture, crop = vision_image(image)
    spent = client.ledger.spent_usd
    try:
        # The picture is on the grey already: a moderation flag (E005) repeats it once.
        prediction = ask_vision(client, focus_input, picture, FOCUS_PRICE, "focus")
    except FATAL_ERRORS:
        raise
    except Exception as exc:  # ADR-013: any model error falls back to the full composition
        warning = record_fallback(client, "focus", exc, FULL)
        cost = round(client.ledger.spent_usd - spent, 6)
        return Focus(AUTO, FULL, cost_usd=cost, warnings=[warning])
    cost = round(client.ledger.spent_usd - spent, 6)
    reply = output_text(prediction.get("output"))
    try:
        fractions = parse_reply(reply)
        box = check_box(fractions_to_box(fractions, crop), image)
    except FocusError as exc:
        return Focus(
            AUTO,
            FULL,
            reply=reply,
            cost_usd=cost,
            warnings=[f"auto focus rejected: {exc}; using full"],
        )
    return Focus(AUTO, AUTO, box, reply, fractions, cost)


# Mapping onto the video and growing for motion


def to_video_box(
    box: Box, image: Image.Image, framing: Framing, video_size: tuple[int, int]
) -> Box:
    """Input PNG pixels -> pixels of the generated video (through the framing placement).

    The framing scaled the character's alpha box onto ``framing.placement``; the video has
    the aspect of the model input frame, so placement fractions map onto it directly.
    """
    sx0, sy0, sx1, sy1 = source_bbox(image)
    px0, py0, px1, py1 = framing.pixel_box(framing.placement, video_size)
    kx = (px1 - px0) / (sx1 - sx0)
    ky = (py1 - py0) / (sy1 - sy0)
    return (
        round(px0 + (box[0] - sx0) * kx),
        round(py0 + (box[1] - sy0) * ky),
        round(px0 + (box[2] - sx0) * kx),
        round(py0 + (box[3] - sy0) * ky),
    )


def _bbox(mask: np.ndarray) -> Box | None:
    if not mask.any():
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _clip(box: Box, size: tuple[int, int]) -> Box:
    return max(0, box[0]), max(0, box[1]), min(size[0], box[2]), min(size[1], box[3])


def motion_square(zone: Box, frames: Sequence[Image.Image]) -> Box:
    """Square around what the zone holds over the whole loop.

    The visible alpha of every frame is united inside the zone grown by ``MOTION_REACH``
    of its side (a head tilting or a fist moving slightly past the zone stays whole,
    while a limb that leaves the zone for good does not drag the square along). Its box
    plus ``FOCUS_MARGIN`` on each side, made square around its centre, is the crop.
    """
    if not frames:
        raise ValueError("no frames")
    size = frames[0].size
    zone = _clip(zone, size)
    if zone[2] <= zone[0] or zone[3] <= zone[1]:
        raise FocusError("the focus zone is outside the video frame")
    side0 = max(zone[2] - zone[0], zone[3] - zone[1])
    reach = round(side0 * MOTION_REACH)
    near = _clip((zone[0] - reach, zone[1] - reach, zone[2] + reach, zone[3] + reach), size)
    union = np.zeros((near[3] - near[1], near[2] - near[0]), dtype=bool)
    for frame in frames:
        union |= np.asarray(frame.getchannel("A").crop(near)) >= ALPHA_VISIBLE
    local = _bbox(union)
    box = (
        (local[0] + near[0], local[1] + near[1], local[2] + near[0], local[3] + near[1])
        if local
        else zone
    )
    side = round(max(box[2] - box[0], box[3] - box[1]) * (1 + 2 * FOCUS_MARGIN))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0 = round(cx - side / 2), round(cy - side / 2)
    return x0, y0, x0 + side, y0 + side


def crop_frames(frames: Sequence[Image.Image], box: Box) -> list[Image.Image]:
    """Every frame cropped to ``box``; areas outside the video frame are transparent."""
    side_w, side_h = box[2] - box[0], box[3] - box[1]
    out = []
    for frame in frames:
        canvas = Image.new("RGBA", (side_w, side_h), (0, 0, 0, 0))
        canvas.paste(frame.crop(box), (0, 0))
        out.append(canvas)
    return out
