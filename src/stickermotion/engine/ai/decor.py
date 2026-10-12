"""Decor as separate layers: found in the input PNG, animated procedurally, laid over the clip.

Wan treats a moon and stars next to the character as part of one scene and drags them along
with the head. Here they are taken out of the model input and animated on their own:

1. ``find_candidates``: connected components of the input alpha. The largest one is the
   character; smaller ones that stand apart from it (a gap of ``min_gap``) and are neither
   specks nor large detached body parts become decor candidates.
2. ``classify``: the cheap vision model (``describe.DESCRIBE_MODEL``) confirms each candidate
   is decor and not part of the character (an object in a hand), and picks a motion:
   twinkle, sway, float or pulse. On any model error the shape heuristic decides (a
   moderation flag, E005, is retried once first).
3. ``compose``: every decor element is resampled once from the full-resolution input pixels
   to the master scale and moved by its own periodic pose (preset engine transforms, own
   phase, a whole number of cycles per loop, so the loop stays seamless), then laid under
   the cut character frames at its place in the original composition.

``--decor auto`` (default) does all of this when decor is found; ``off`` removes the decor
from the result; ``together`` keeps it in the model input as before.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from stickermotion.core.alpha import resize_premultiplied
from stickermotion.core.models import MasterAnimation
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
from stickermotion.engine.transform import FrameTransform, SpriteTransformer

DECOR_PRICE: Price = DESCRIBE_PRICE
VISION_BACKGROUND = (128, 128, 128)
BOX_COLOR = (255, 0, 0)
# Free space around the composed master (share of its side), as in chain.MASTER_MARGIN.
MASTER_MARGIN = 0.04
# The composed master is upscaled to at least this side so decor is rendered straight from
# the input pixels at the Telegram sticker size instead of being upscaled twice.
MIN_MASTER_SIDE = 512

Box = tuple[int, int, int, int]
"""(left, top, right, bottom) in pixels, right / bottom exclusive."""


class DecorMode(StrEnum):
    AUTO = "auto"
    """Separate layers when decor is found (default)."""
    OFF = "off"
    """Decor removed from the model input and not laid back: the character alone."""
    TOGETHER = "together"
    """Decor stays in the model input and moves with the scene (the old behaviour)."""


class Motion(StrEnum):
    TWINKLE = "twinkle"
    SWAY = "sway"
    FLOAT = "float"
    PULSE = "pulse"


@dataclass(frozen=True)
class DecorParams:
    """Thresholds of the component search (shares are of the character's area / image side)."""

    alpha_min: int = 8
    """Alpha that counts as a visible pixel (same as the preprocess)."""
    min_area: float = 0.0005
    """Smaller components (share of the character's area) are specks; they stay put."""
    max_area: float = 0.25
    """Larger detached components are parts of the character, not decor."""
    min_gap: float = 0.01
    """Closer components (share of the image's longer side) belong to the character."""

    def gap_px(self, size: tuple[int, int]) -> int:
        return max(2, round(self.min_gap * max(size)))


# Connected components


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected components of a boolean mask; returns (int32 labels, count).

    Run-length union-find: each row is split into runs, runs touching a run of the row
    above (diagonals included) are merged. Labels are 1..count, 0 is background.
    """
    height, width = mask.shape
    parent: list[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    provisional = np.zeros((height, width), dtype=np.int32)
    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        row = mask[y]
        if not row.any():
            previous = []
            continue
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
        current: list[tuple[int, int, int]] = []
        j = 0
        for start, end in zip(edges[0::2].tolist(), edges[1::2].tolist(), strict=True):
            while j < len(previous) and previous[j][1] < start:
                j += 1
            label = 0
            k = j
            while k < len(previous) and previous[k][0] <= end:
                other = find(previous[k][2])
                if label == 0:
                    label = other
                elif other != label:
                    low, high = sorted((label, other))
                    parent[high] = low
                    label = low
                k += 1
            if label == 0:
                label = len(parent)
                parent.append(label)
            provisional[y, start:end] = label
            current.append((start, end, label))
        previous = current
    roots = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    unique = np.unique(roots[1:]) if len(parent) > 1 else np.array([], dtype=np.int32)
    lut = np.zeros(len(parent), dtype=np.int32)
    remap = {int(root): index + 1 for index, root in enumerate(unique)}
    for index in range(1, len(parent)):
        lut[index] = remap[int(roots[index])]
    return lut[provisional], len(unique)


def _bbox(mask: np.ndarray) -> Box:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    image = Image.fromarray((mask * 255).astype(np.uint8), "L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0


@dataclass(frozen=True)
class Candidate:
    index: int
    box: Box
    area: int
    fill: float
    """Share of its bounding box the component covers (stars and crescents are low)."""
    mask: np.ndarray = field(repr=False, compare=False)
    """Full-image boolean mask of the component grown by its faint fringe."""


def find_candidates(
    image: Image.Image, params: DecorParams | None = None
) -> tuple[list[Candidate], Box | None]:
    """Detached decor candidates and the box of the character (largest component)."""
    params = params or DecorParams()
    alpha = np.asarray(image.convert("RGBA").getchannel("A"))
    visible = alpha >= params.alpha_min
    labels, count = label_components(visible)
    if count == 0:
        return [], None
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    areas[0] = 0
    main = int(np.argmax(areas))
    main_mask = labels == main
    main_box = _bbox(main_mask)
    gap = params.gap_px(image.size)
    faint = alpha > 0
    height, width = alpha.shape
    found: list[Candidate] = []
    for label in range(1, count + 1):
        if label == main:
            continue
        area = int(areas[label])
        share = area / areas[main]
        if not params.min_area <= share <= params.max_area:
            continue
        own = labels == label
        x0, y0, x1, y1 = _bbox(own)
        wx0, wy0 = max(0, x0 - gap), max(0, y0 - gap)
        wx1, wy1 = min(width, x1 + gap), min(height, y1 + gap)
        near = _dilate(own[wy0:wy1, wx0:wx1], gap)
        if (near & main_mask[wy0:wy1, wx0:wx1]).any():
            continue
        # Its faint antialiased fringe (alpha below alpha_min) goes along with it.
        grown = np.zeros_like(own)
        fringe = _dilate(own[wy0:wy1, wx0:wx1], 2) & faint[wy0:wy1, wx0:wx1]
        grown[wy0:wy1, wx0:wx1] = fringe & ~main_mask[wy0:wy1, wx0:wx1]
        grown |= own
        fill = area / ((x1 - x0) * (y1 - y0))
        found.append(Candidate(len(found) + 1, (x0, y0, x1, y1), area, round(fill, 3), grown))
    found.sort(key=lambda c: (c.box[1], c.box[0]))
    found = [
        Candidate(i + 1, c.box, c.area, c.fill, c.mask) for i, c in enumerate(found)
    ]  # numbered top to bottom for the vision model
    return found, main_box


# Classification


# Motions that resize or fade an element; a dominant element (a moon among stars) must not
# get them even when the vision model asks for it (seen live: the moon called a star).
FLICKER = (Motion.TWINKLE, Motion.PULSE)
# An element at least this many times larger than every other decor element is dominant.
DOMINANT_RATIO = 4.0


def dominant(candidate: Candidate, candidates: Sequence[Candidate]) -> bool:
    others = [c.area for c in candidates if c.index != candidate.index]
    return bool(others) and candidate.area >= DOMINANT_RATIO * max(others)


def heuristic_motion(candidate: Candidate, largest_area: int) -> Motion:
    """Large decor sways (a moon) or floats, small spiky decor twinkles, small round pulses."""
    if candidate.area >= 0.5 * largest_area:
        return Motion.SWAY if candidate.fill < 0.5 else Motion.FLOAT
    return Motion.TWINKLE if candidate.fill < 0.6 else Motion.PULSE


@dataclass(frozen=True)
class DecorItem:
    index: int
    box: Box
    """Box in input PNG pixels."""
    area: int
    motion: Motion
    phase: float
    what: str = ""
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "box": list(self.box),
            "area": self.area,
            "what": self.what,
            "motion": self.motion.value,
            "phase": round(self.phase, 3),
            "source": self.source,
        }


DECOR_SYSTEM = (
    "You look at drawn characters prepared as chat stickers and tell decorations apart from "
    "the character. Reply with JSON only."
)
DECOR_INSTRUCTION = (
    "Red numbered boxes mark elements that are not attached to the character's body. For "
    "each number decide whether it is a decoration floating around the character (stars, "
    "moon, hearts, Zzz, sparkles, music notes, clouds) or part of the character or an "
    "object it holds or wears. For decorations pick one gentle motion: twinkle (stars, "
    "sparkles), sway (moon, hanging things), float (clouds, hearts, Zzz), pulse (hearts, "
    "glows).\n"
    'Answer as JSON: {"items": [{"id": 1, "decor": true, "what": "<one word>", '
    '"motion": "twinkle|sway|float|pulse"}]}'
)


def vision_image(image: Image.Image, candidates: Sequence[Candidate]) -> Image.Image:
    """The composition on grey with numbered boxes around the candidates, downscaled."""
    canvas = Image.new("RGBA", image.size, (*VISION_BACKGROUND, 255))
    canvas.alpha_composite(image.convert("RGBA"))
    draw = ImageDraw.Draw(canvas)
    side = max(image.size)
    width = max(2, side // 250)
    try:
        font: Any = ImageFont.load_default(size=max(12, side // 30))
    except TypeError:  # Pillow without scalable default font
        font = ImageFont.load_default()
    for c in candidates:
        x0, y0, x1, y1 = c.box
        pad = width * 2
        draw.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), outline=BOX_COLOR, width=width)
        draw.text(
            (x0 - pad, max(0, y0 - pad - side // 28)), str(c.index), fill=BOX_COLOR, font=font
        )
    return small_frame(canvas.convert("RGB"))


def decor_input(image_ref: str) -> dict[str, Any]:
    return {
        "system_prompt": DECOR_SYSTEM,
        "prompt": DECOR_INSTRUCTION,
        "image_input": [image_ref],
        "reasoning_effort": "minimal",
        "verbosity": "low",
        "max_completion_tokens": 1000,
    }


def parse_reply(text: str) -> dict[int, tuple[bool, str, Motion | None]]:
    """{id: (is decor, what, motion or None)}; raises ValueError on an unusable reply."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON in the reply: {text[:200]!r}")
    data: Any = json.loads(match.group(0))
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("the reply has no items list")
    result: dict[int, tuple[bool, str, Motion | None]] = {}
    for item in items:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            index = int(item["id"])
        except (TypeError, ValueError):
            continue
        motion_text = str(item.get("motion") or "").strip().lower()
        motion = Motion(motion_text) if motion_text in Motion.__members__.values() else None
        decor = item.get("decor")
        is_decor = decor is True or str(decor).strip().lower() in ("true", "yes", "1")
        result[index] = (is_decor, str(item.get("what") or "").strip().lower()[:30], motion)
    return result


def phase_for(index: int) -> float:
    """Spread phases with the golden ratio so neighbours never move in step."""
    return (index * 0.6180339887) % 1.0


# The split


@dataclass
class DecorPlan:
    """What the pipeline does with the decor of one input."""

    requested: DecorMode
    mode: DecorMode
    """What is used: ``auto`` (separate layers), ``off`` or ``together`` (none found too)."""
    character: Image.Image = field(repr=False)
    """The input with the decor removed (the input itself for ``together``)."""
    items: list[DecorItem] = field(default_factory=list)
    sprites: list[Image.Image] = field(default_factory=list, repr=False)
    """Each item's own pixels cropped to its box, full input resolution."""
    rejected: list[dict[str, Any]] = field(default_factory=list)
    """Candidates the vision model kept with the character."""
    reply: str | None = None
    cost_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def separate(self) -> bool:
        """Decor is out of the model input (``auto`` with items, or ``off``)."""
        return self.mode in (DecorMode.AUTO, DecorMode.OFF) and bool(self.items)

    @property
    def animated(self) -> bool:
        return self.mode is DecorMode.AUTO and bool(self.items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested.value,
            "mode": self.mode.value,
            "model": DESCRIBE_MODEL if self.reply is not None else None,
            "items": [i.to_dict() for i in self.items],
            "rejected": self.rejected,
            "reply": self.reply,
            "cost_usd": self.cost_usd,
            "warnings": self.warnings,
        }


def plan(
    image: Image.Image,
    mode: DecorMode | str = DecorMode.AUTO,
    client: ReplicateClient | None = None,
    params: DecorParams | None = None,
) -> DecorPlan:
    """Find decor, confirm it with the vision model (if ``client``) and split it off."""
    requested = DecorMode(mode)
    image = image.convert("RGBA")
    if requested is DecorMode.TOGETHER:
        return DecorPlan(requested, DecorMode.TOGETHER, image)
    candidates, _ = find_candidates(image, params)
    if not candidates:
        return DecorPlan(requested, requested, image)
    largest = max(c.area for c in candidates)
    verdicts: dict[int, tuple[bool, str, Motion | None]] = {}
    reply: str | None = None
    cost = 0.0
    warnings: list[str] = []
    source = "heuristic"
    if client is not None:
        spent = client.ledger.spent_usd
        prediction: dict[str, Any] | None
        try:
            # The picture is on the grey already: a moderation flag (E005) repeats it once.
            prediction = ask_vision(
                client, decor_input, vision_image(image, candidates), DECOR_PRICE, "decor"
            )
        except FATAL_ERRORS:
            raise
        except Exception as exc:  # ADR-018: any model error leaves it to the shape heuristic
            warnings.append(record_fallback(client, "decor", exc, "shape heuristic"))
            prediction = None
        cost = round(client.ledger.spent_usd - spent, 6)
        if prediction is not None:
            reply = output_text(prediction.get("output"))
            try:
                verdicts = parse_reply(reply)
                source = DESCRIBE_MODEL
            except (ValueError, json.JSONDecodeError) as exc:
                warnings.append(f"decor reply unusable ({exc}); shape heuristic used")
    items: list[DecorItem] = []
    sprites: list[Image.Image] = []
    rejected: list[dict[str, Any]] = []
    character = np.asarray(image).copy()
    source_data = np.asarray(image)
    for c in candidates:
        verdict = verdicts.get(c.index)
        if verdict is not None and not verdict[0]:
            rejected.append({"index": c.index, "box": list(c.box), "what": verdict[1]})
            continue
        if verdict is None and source != "heuristic":
            warnings.append(f"decor item {c.index} missing in the reply; heuristic used")
        motion = (verdict[2] if verdict else None) or heuristic_motion(c, largest)
        if motion in FLICKER and dominant(c, candidates):
            fixed = heuristic_motion(c, largest)
            warnings.append(
                f"decor item {c.index}: {motion.value} -> {fixed.value} (dominant size)"
            )
            motion = fixed
        what = verdict[1] if verdict else ""
        item_source = source if verdict else "heuristic"
        items.append(
            DecorItem(c.index, c.box, c.area, motion, phase_for(len(items)), what, item_source)
        )
        own = np.zeros_like(source_data)
        own[c.mask] = source_data[c.mask]
        sprites.append(Image.fromarray(own, "RGBA").crop(_grow(c.box, 2, image.size)))
        character[c.mask] = 0
    return DecorPlan(
        requested,
        requested,
        Image.fromarray(character, "RGBA") if items else image,
        items,
        sprites,
        rejected,
        reply,
        cost,
        warnings,
    )


def _grow(box: Box, pad: int, size: tuple[int, int]) -> Box:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(size[0], box[2] + pad),
        min(size[1], box[3] + pad),
    )


# Motion


@dataclass(frozen=True)
class MotionStyle:
    cycles: int
    """Whole cycles per loop: the pose at phase 1 equals phase 0."""
    scale: float
    """Peak relative size change."""
    angle_deg: float
    lift: float
    """Vertical travel, share of the element's height."""
    fade: float
    """Opacity dip at the low point (0 keeps it opaque)."""


STYLES: dict[Motion, MotionStyle] = {
    Motion.TWINKLE: MotionStyle(cycles=2, scale=0.14, angle_deg=6.0, lift=0.0, fade=0.35),
    Motion.SWAY: MotionStyle(cycles=1, scale=0.0, angle_deg=5.0, lift=0.0, fade=0.0),
    Motion.FLOAT: MotionStyle(cycles=1, scale=0.0, angle_deg=0.0, lift=0.06, fade=0.0),
    Motion.PULSE: MotionStyle(cycles=2, scale=0.08, angle_deg=0.0, lift=0.0, fade=0.0),
}


def pose(motion: Motion, t: float, phase: float, height: float) -> tuple[FrameTransform, float]:
    """Transform around the element's centre and its opacity at loop phase ``t``."""
    style = STYLES[motion]
    angle = 2.0 * math.pi * (style.cycles * t + phase)
    wave = math.sin(angle)
    scale = 1.0 + style.scale * wave
    opacity = 1.0 - style.fade * (0.5 - 0.5 * wave)
    transform = FrameTransform(
        scale_x=scale,
        scale_y=scale,
        angle_deg=style.angle_deg * wave,
        dy=-style.lift * height * wave,
    )
    return transform, opacity


def motion_pad(motion: Motion, width: float, height: float) -> float:
    """How far the element can reach past its box while moving (pixels)."""
    style = STYLES[motion]
    radius = math.hypot(width, height) / 2
    grown = radius * (1 + style.scale) - min(width, height) / 2
    return max(grown, style.lift * height) + 2


# Composition


@dataclass(frozen=True)
class PlacedDecor:
    item: DecorItem
    sprite: Image.Image
    """Full-resolution pixels; ``item.box`` grown by 2 px is where they sit in the input."""
    box: tuple[float, float, float, float]
    """Where the sprite sits on the generated video (pixels, may exceed the frame)."""


def place(decor: DecorPlan, framing: Framing, video_size: tuple[int, int]) -> list[PlacedDecor]:
    """Map every decor sprite from input pixels onto the video through the framing.

    The framing scaled the character's box (in the input) onto ``framing.placement``; the
    decor keeps its offset from the character in the same scale.
    """
    alpha = np.asarray(decor.character.getchannel("A")) >= 8
    sx0, sy0, sx1, sy1 = _bbox(alpha)
    px0, py0, px1, py1 = framing.pixel_box(framing.placement, video_size)
    kx = (px1 - px0) / (sx1 - sx0)
    ky = (py1 - py0) / (sy1 - sy0)
    placed = []
    size = decor.character.size
    for item, sprite in zip(decor.items, decor.sprites, strict=True):
        x0, y0, x1, y1 = _grow(item.box, 2, size)
        box = (px0 + (x0 - sx0) * kx, py0 + (y0 - sy0) * ky, px0 + (x1 - sx0) * kx,
               py0 + (y1 - sy0) * ky)  # fmt: skip
        placed.append(PlacedDecor(item, sprite, box))
    return placed


def _alpha_box(frames: Sequence[Image.Image]) -> Box:
    boxes = [b for b in (f.getchannel("A").getbbox() for f in frames) if b is not None]
    if not boxes:
        raise ValueError("every frame is fully transparent after the cut")
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


@dataclass
class Composition:
    master: MasterAnimation
    box: Box
    """Square on the video (pixels) the master covers."""
    scale: float
    """Master pixels per video pixel."""

    def to_dict(self) -> dict[str, Any]:
        return {"video_square": list(self.box), "scale": round(self.scale, 4)}


def compose(
    frames: Sequence[Image.Image],
    fps: float,
    placed: Sequence[PlacedDecor],
    min_side: int = MIN_MASTER_SIDE,
    margin: float = MASTER_MARGIN,
) -> Composition:
    """Master square holding the cut character frames and the moving decor under them."""
    count = len(frames)
    left, top, right, bottom = (float(v) for v in _alpha_box(frames))
    pads = []
    for p in placed:
        w, h = p.box[2] - p.box[0], p.box[3] - p.box[1]
        pad = motion_pad(p.item.motion, w, h)
        pads.append(pad)
        left, top = min(left, p.box[0] - pad), min(top, p.box[1] - pad)
        right, bottom = max(right, p.box[2] + pad), max(bottom, p.box[3] + pad)
    side_v = round(max(right - left, bottom - top) * (1 + 2 * margin))
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x0, y0 = round(cx - side_v / 2), round(cy - side_v / 2)
    box = (x0, y0, x0 + side_v, y0 + side_v)
    scale = max(1.0, min_side / side_v)
    side = max(1, round(side_v * scale))

    layers = []
    for p, pad_v in zip(placed, pads, strict=True):
        w = max(1, round((p.box[2] - p.box[0]) * scale))
        h = max(1, round((p.box[3] - p.box[1]) * scale))
        sprite = resize_premultiplied(p.sprite, (w, h))
        pad = math.ceil(pad_v * scale) + 4
        canvas = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
        canvas.paste(sprite, (pad, pad))
        transformer = SpriteTransformer(canvas, (pad + w / 2, pad + h / 2))
        at = (round((p.box[0] - x0) * scale) - pad, round((p.box[1] - y0) * scale) - pad)
        layers.append((p.item, transformer, at, h))

    out = []
    for index, frame in enumerate(frames):
        character = Image.new("RGBA", (side_v, side_v), (0, 0, 0, 0))
        character.paste(frame.crop(box), (0, 0))
        if side != side_v:
            character = resize_premultiplied(character, (side, side))
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        t = index / count
        for item, transformer, at, h in layers:
            transform, opacity = pose(item.motion, t, item.phase, h)
            sprite = transformer.render(transform)
            if opacity < 1.0:
                a = np.asarray(sprite).copy()
                a[..., 3] = np.rint(a[..., 3] * opacity).astype(np.uint8)
                sprite = Image.fromarray(a, "RGBA")
            layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
            layer.paste(sprite, at)
            canvas.alpha_composite(layer)
        canvas.alpha_composite(character)
        out.append(canvas)
    return Composition(MasterAnimation(frames=out, fps=fps, loop=True), box, scale)


NAMES_UK = {
    "moon": "місяць",
    "crescent": "місяць",
    "star": "зірка",
    "stars": "зірка",
    "heart": "сердечко",
    "zzz": "Zzz",
    "z": "Zzz",
    "sparkle": "блиск",
    "sparkles": "блиск",
    "cloud": "хмаринка",
    "note": "нота",
    "bubble": "бульбашка",
}


def summary_uk(items: Sequence[dict[str, Any]]) -> str:
    """User-facing list such as "місяць, зірка ×3" for the web UI."""
    counts: dict[str, int] = {}
    for item in items:
        name = NAMES_UK.get(str(item.get("what") or "").lower(), "елемент")
        counts[name] = counts.get(name, 0) + 1
    return ", ".join(n if c == 1 else f"{n} ×{c}" for n, c in counts.items())
