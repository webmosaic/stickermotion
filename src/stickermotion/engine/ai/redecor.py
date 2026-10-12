"""Old decor out of an existing clip, so a liked motion can get the separate decor.

Clips generated before decor separation (or with ``--decor together``) carry the moon and
stars in the video, moving with the head. To reuse such a clip without paying for a new one:

1. ``infer_mapping``: where the input PNG sits on the model input frame. The visible box
   of the model input is matched with the box of the whole composition (decor in the
   frame) or of the character alone (decor already out), by aspect ratio.
2. ``decor_in_clip``: whether the model input really shows the decor (its mapped pixels
   are not the key colour). Only then is anything removed.
3. ``track``: the decor moves with the head, so its place is followed frame by frame:
   the decor pixels of the model input are a template matched (masked SSD) on the raw
   video frames around the previous position, first as a group, then each item on its
   own within a few pixels.
4. ``remove``: in every cut frame the connected components of alpha that are not the
   character (the largest component) and lie mostly inside the tracked decor zone are
   cleared with their faint fringe. The character must not lose more than ``tolerance``
   of its area outside the zone; otherwise the frame is left as it was and a warning is
   recorded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from stickermotion.engine.ai.decor import Box, Candidate, _bbox, _dilate, label_components
from stickermotion.engine.ai.framing import GREEN, Color, Framing, Margins
from stickermotion.engine.ai.matting import chroma_key, frame_key

ALPHA_MIN = 8
# A mapped decor mask counts as present in the model input when at least this share of
# its pixels differs from the key colour.
PRESENT_SHARE = 0.5
KEY_DISTANCE = 60.0
GROUP_RADIUS = 10
ITEM_RADIUS = 5
TEMPLATE_PAD = 6
ZONE_PX = 4
MIN_INSIDE = 0.5
TOLERANCE = 0.02


@dataclass(frozen=True)
class Mapping:
    """Input PNG pixels -> video pixels: ``video = offset + input * scale``."""

    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    reference: str
    """``composition`` (decor was in the frame) or ``character``."""

    @classmethod
    def from_framing(
        cls,
        framing: Framing,
        subject: Image.Image,
        size: tuple[int, int],
        reference: str = "character",
    ) -> Mapping:
        """Mapping of a known framing: the subject's box lands on ``framing.placement``.

        ``subject`` is what the framing placed: the character alone (``reference``
        ``character``) or the whole composition with the decor (``composition``).
        """
        sx0, sy0, sx1, sy1 = visible_box(subject)
        width, height = size
        x0, y0, x1, y1 = framing.placement
        kx = (x1 - x0) * width / (sx1 - sx0)
        ky = (y1 - y0) * height / (sy1 - sy0)
        return cls(kx, ky, x0 * width - sx0 * kx, y0 * height - sy0 * ky, reference)

    def box(self, box: Box) -> tuple[float, float, float, float]:
        return (
            self.offset_x + box[0] * self.scale_x,
            self.offset_y + box[1] * self.scale_y,
            self.offset_x + box[2] * self.scale_x,
            self.offset_y + box[3] * self.scale_y,
        )

    def mask(self, mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Boolean input-sized mask drawn onto a video frame of ``size``."""
        height, width = mask.shape
        scaled = Image.fromarray((mask * 255).astype(np.uint8), "L").resize(
            (max(1, round(width * self.scale_x)), max(1, round(height * self.scale_y))),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("L", size, 0)
        canvas.paste(scaled, (round(self.offset_x), round(self.offset_y)))
        return np.asarray(canvas) > 64

    def framing(self, character: Image.Image, size: tuple[int, int]) -> Framing:
        """Framing whose placement is the character's box on the video (for ``decor.place``)."""
        x0, y0, x1, y1 = self.box(visible_box(character))
        width, height = size
        placement = (x0 / width, y0 / height, x1 / width, y1 / height)
        return Framing(size, GREEN, placement, Margins())

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": [round(self.scale_x, 5), round(self.scale_y, 5)],
            "offset": [round(self.offset_x, 2), round(self.offset_y, 2)],
            "reference": self.reference,
        }


def framing_from_dict(data: dict[str, Any]) -> Framing:
    """``Framing.to_dict`` back into a ``Framing`` (raw clip metadata)."""
    margins = data.get("margins") or {}
    return Framing(
        (int(data["canvas_size"][0]), int(data["canvas_size"][1])),
        (int(data["background"][0]), int(data["background"][1]), int(data["background"][2])),
        (
            float(data["placement"][0]),
            float(data["placement"][1]),
            float(data["placement"][2]),
            float(data["placement"][3]),
        ),
        Margins(**margins),
    )


def visible_box(image: Image.Image) -> Box:
    alpha = np.asarray(image.convert("RGBA").getchannel("A")) >= ALPHA_MIN
    if not alpha.any():
        raise ValueError("the image has no visible pixels")
    return _bbox(alpha)


def model_input_box(model_input: Image.Image, key: Color = GREEN) -> Box:
    """Visible box of the model input frame after the chroma key."""
    rgb = model_input.convert("RGB")
    return visible_box(chroma_key(rgb, frame_key(rgb, key)))


def _ratio(box: Box) -> float:
    return math.log((box[2] - box[0]) / (box[3] - box[1]))


def infer_mapping(
    original: Image.Image, character: Image.Image, model_input: Image.Image
) -> Mapping:
    """Mapping of input pixels onto the model input frame (and so onto the clip)."""
    target = model_input_box(model_input)
    full = visible_box(original)
    alone = visible_box(character)
    reference, source = min(
        (("composition", full), ("character", alone)),
        key=lambda pair: abs(_ratio(pair[1]) - _ratio(target)),
    )
    sx = (target[2] - target[0]) / (source[2] - source[0])
    sy = (target[3] - target[1]) / (source[3] - source[1])
    return Mapping(sx, sy, target[0] - source[0] * sx, target[1] - source[1] * sy, reference)


def decor_in_clip(
    model_input: Image.Image, masks: Sequence[np.ndarray], key: Color = GREEN
) -> list[bool]:
    """For each mapped decor mask: whether the model input shows something there."""
    rgb = np.asarray(model_input.convert("RGB"), dtype=np.float32)
    observed = np.asarray(frame_key(model_input.convert("RGB"), key), dtype=np.float32)
    off_key = np.sqrt(((rgb - observed) ** 2).sum(-1)) > KEY_DISTANCE
    present = []
    for mask in masks:
        inner = mask & ~_dilate(~mask, 1)  # skip the antialiased rim
        area = int(inner.sum()) or int(mask.sum())
        hits = int((off_key & (inner if inner.any() else mask)).sum())
        present.append(area > 0 and hits / area >= PRESENT_SHARE)
    return present


# Tracking


Shift = tuple[int, int]


@dataclass
class _Template:
    box: Box
    pixels: np.ndarray
    weight: np.ndarray

    @classmethod
    def build(cls, model_input: np.ndarray, mask: np.ndarray, pad: int) -> _Template:
        x0, y0, x1, y1 = _bbox(mask)
        height, width = mask.shape
        box = (max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad))
        pixels = model_input[box[1] : box[3], box[0] : box[2]]
        weight = _dilate(mask, pad)[box[1] : box[3], box[0] : box[2]].astype(np.float32)
        return cls(box, pixels, weight / max(1.0, float(weight.sum())))

    def score(self, frame: np.ndarray, shift: Shift) -> float:
        x0, y0, x1, y1 = self.box
        dx, dy = shift
        height, width = frame.shape[:2]
        if x0 + dx < 0 or y0 + dy < 0 or x1 + dx > width or y1 + dy > height:
            return math.inf
        diff = frame[y0 + dy : y1 + dy, x0 + dx : x1 + dx] - self.pixels
        return float(((diff * diff).sum(-1) * self.weight).sum())

    def search(self, frame: np.ndarray, around: Shift, radius: int) -> tuple[Shift, float]:
        best: tuple[float, Shift] = (math.inf, around)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                shift = (around[0] + dx, around[1] + dy)
                value = self.score(frame, shift)
                if value < best[0]:
                    best = (value, shift)
        return best[1], best[0]


def track(
    frames: Sequence[Image.Image],
    model_input: Image.Image,
    masks: Sequence[np.ndarray],
    group_radius: int = GROUP_RADIUS,
    item_radius: int = ITEM_RADIUS,
    pad: int = TEMPLATE_PAD,
) -> list[list[Shift]]:
    """Per frame, the shift of every decor mask (video pixels) from its model input place."""
    reference = np.asarray(model_input.convert("RGB"), dtype=np.float32)
    group_mask = np.zeros(reference.shape[:2], dtype=bool)
    for mask in masks:
        group_mask |= mask
    group = _Template.build(reference, group_mask, pad)
    items = [_Template.build(reference, mask, max(2, pad - 2)) for mask in masks]
    at: Shift = (0, 0)
    out: list[list[Shift]] = []
    for frame in frames:
        data = np.asarray(frame.convert("RGB"), dtype=np.float32)
        at, _ = group.search(data, at, group_radius)
        out.append([item.search(data, at, item_radius)[0] for item in items])
    return out


def shifted(mask: np.ndarray, shift: Shift) -> np.ndarray:
    """``mask`` moved by ``shift`` (pixels pushed out of the frame are dropped)."""
    dx, dy = shift
    height, width = mask.shape
    out = np.zeros_like(mask)
    xs, xd = (0, dx) if dx >= 0 else (-dx, 0)
    ys, yd = (0, dy) if dy >= 0 else (-dy, 0)
    w, h = width - abs(dx), height - abs(dy)
    if w > 0 and h > 0:
        out[yd : yd + h, xd : xd + w] = mask[ys : ys + h, xs : xs + w]
    return out


# Removal


@dataclass
class Removal:
    frames: list[Image.Image]
    removed_px: list[int] = field(default_factory=list)
    """Pixels cleared in each frame."""
    residual_px: list[int] = field(default_factory=list)
    """Visible pixels left in the tracked decor zone that are not part of the character."""
    touching_px: list[int] = field(default_factory=list)
    """Character pixels inside the tracked decor zone (decor merged with the character)."""
    character_loss: list[float] = field(default_factory=list)
    """Share of the character's area (outside the zone) cleared in each frame."""
    reverted: list[int] = field(default_factory=list)
    """Frames left unchanged because the character would lose more than the tolerance."""
    shifts: list[list[Shift]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def span(values: Sequence[float]) -> list[float]:
            return [round(min(values), 4), round(max(values), 4)] if values else [0, 0]

        travel = [
            [max(abs(s[i][0]) for s in self.shifts), max(abs(s[i][1]) for s in self.shifts)]
            for i in range(len(self.shifts[0]) if self.shifts else 0)
        ]
        return {
            "frames": len(self.frames),
            "removed_px": span(self.removed_px),
            "residual_px_max": max(self.residual_px, default=0),
            "touching_px_max": max(self.touching_px, default=0),
            "character_loss_max": round(max(self.character_loss, default=0.0), 5),
            "reverted_frames": self.reverted,
            "max_shift_px": travel,
            "warnings": self.warnings,
        }


def remove(
    frames: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    shifts: Sequence[Sequence[Shift]],
    zone_px: int = ZONE_PX,
    min_inside: float = MIN_INSIDE,
    tolerance: float = TOLERANCE,
) -> Removal:
    """Clear the old decor from cut RGBA frames along the tracked positions."""
    result = Removal([], shifts=[list(s) for s in shifts])
    for index, (frame, frame_shifts) in enumerate(zip(frames, shifts, strict=True)):
        data = np.asarray(frame.convert("RGBA")).copy()
        alpha = data[..., 3]
        visible = alpha >= ALPHA_MIN
        zone = np.zeros(visible.shape, dtype=bool)
        for mask, shift in zip(masks, frame_shifts, strict=True):
            zone |= shifted(mask, shift)
        zone = _dilate(zone, zone_px)
        labels, count = label_components(visible)
        if count == 0:
            result.frames.append(frame)
            for values in (result.removed_px, result.residual_px, result.touching_px):
                values.append(0)
            result.character_loss.append(0.0)
            continue
        areas = np.bincount(labels.ravel(), minlength=count + 1)
        areas[0] = 0
        main = int(np.argmax(areas))
        main_mask = labels == main
        clear = np.zeros(visible.shape, dtype=bool)
        for label in np.unique(labels[zone & visible]):
            if label == 0 or label == main:
                continue
            own = labels == label
            if (own & zone).sum() / areas[label] < min_inside:
                continue
            clear |= _dilate(own, 2) & (alpha > 0) & ~main_mask
        outside = visible & ~zone
        loss = float((clear & outside).sum()) / max(1, int(outside.sum()))
        if loss > tolerance:
            result.reverted.append(index)
            result.warnings.append(
                f"frame {index}: removal would cut {loss:.1%} of the character; kept as is"
            )
            clear[:] = False
        data[clear] = 0
        left = (data[..., 3] >= ALPHA_MIN) & zone & ~main_mask
        result.frames.append(Image.fromarray(data, "RGBA"))
        result.removed_px.append(int(clear.sum()))
        result.residual_px.append(int(left.sum()))
        result.touching_px.append(int((zone & main_mask).sum()))
        result.character_loss.append(0.0 if loss > tolerance else loss)
    if max(result.touching_px, default=0) > 0:
        result.warnings.append(
            "the decor zone overlaps the character in some frames "
            f"(up to {max(result.touching_px)} px); decor merged with the character stays"
        )
    return result


def candidate_masks(
    candidates: Sequence[Candidate], mapping: Mapping, size: tuple[int, int]
) -> list[np.ndarray]:
    """Decor candidate masks drawn on the video frame; empty ones are dropped."""
    masks = [mapping.mask(c.mask, size) for c in candidates]
    return [m for m in masks if m.any()]
