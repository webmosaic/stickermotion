"""Model input frame: the character on a flat keyable background in the aspect Wan accepts.

The character is cropped to its alpha, scaled into a box that leaves room for motion and
placed on a 16:9 canvas (Wan 2.2 at 480p renders 832x480, at 720p 1280x720). Free space is
larger above and at the sides than below: in early tests a raised arm reached the top edge of the
frame.

``margins_for`` picks tighter margins from the action text (more headroom only when
the action raises arms, stretches or jumps) and lets the character use the full canvas
width, so it fills more of the model frame and the clip carries more detail.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from stickermotion.core.alpha import resize_premultiplied
from stickermotion.preprocess import normalize

Color = tuple[int, int, int]

GREEN: Color = (0, 255, 0)
WAN_CANVAS = (832, 480)
WAN_CANVASES: dict[str, tuple[int, int]] = {"480p": WAN_CANVAS, "720p": (1280, 720)}


def canvas_for(resolution: str) -> tuple[int, int]:
    """Model input canvas for a Wan resolution (the size Wan renders at that setting)."""
    try:
        return WAN_CANVASES[resolution]
    except KeyError:
        raise ValueError(
            f"resolution must be one of {', '.join(WAN_CANVASES)}, got {resolution!r}"
        ) from None


@dataclass(frozen=True)
class Margins:
    """Free space around the character, as fractions of the canvas height.

    ``top`` is headroom for raised arms and stretches, ``side`` the minimum space left
    and right inside the central square of the canvas, ``bottom`` the gap under the
    character. ``full_width``: the character's box spans the whole canvas width minus
    ``side`` on each side instead of the central square.
    """

    top: float = 0.30
    side: float = 0.08
    bottom: float = 0.04
    full_width: bool = False

    def __post_init__(self) -> None:
        for name in ("top", "side", "bottom"):
            value = getattr(self, name)
            if not 0 <= value < 0.5:
                raise ValueError(f"margin {name} must be in [0, 0.5), got {value}")
        if self.top + self.bottom >= 0.9:
            raise ValueError("top + bottom margins leave no room for the character")


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Framing:
    """Where the character sits on the model input frame.

    Boxes are (left, top, right, bottom) as fractions of the canvas width / height, so
    they map onto a generated video of any resolution with the same aspect.
    """

    canvas_size: tuple[int, int]
    background: Color
    placement: Box
    margins: Margins

    def pixel_box(self, box: Box, size: tuple[int, int]) -> tuple[int, int, int, int]:
        width, height = size
        return (
            round(box[0] * width),
            round(box[1] * height),
            round(box[2] * width),
            round(box[3] * height),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def tight_crop(image: Image.Image) -> Image.Image:
    """The character cropped to its visible pixels (alpha checked by the preprocess)."""
    prepared = normalize(image, canvas_size=max(image.size), headroom=0.0)
    return prepared.image.crop(prepared.bbox)


def flatten(image: Image.Image, background: Color) -> Image.Image:
    """Composite an RGBA image over a solid colour."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    bg = np.asarray(background, dtype=np.float32) / 255.0
    rgb = rgba[..., :3] * alpha + bg * (1.0 - alpha)
    return Image.fromarray(np.rint(rgb * 255.0).astype(np.uint8), "RGB")


def frame_for_model(
    image: Image.Image,
    canvas_size: tuple[int, int] = WAN_CANVAS,
    margins: Margins | None = None,
    background: Color = GREEN,
) -> tuple[Image.Image, Framing]:
    """Place the character on a flat canvas with motion headroom; returns (RGB frame, Framing).

    The character's box is ``height * (1 - 2 * side)`` wide (inside the central square;
    ``width - 2 * side * height`` with ``full_width``)
    and ``height * (1 - top - bottom)`` tall; the character is scaled to fit it, centred
    horizontally and stood on the bottom margin.
    """
    margins = margins or Margins()
    width, height = canvas_size
    if width < height:
        raise ValueError("portrait canvases are not supported")
    sprite = tight_crop(image)
    if margins.full_width:
        box_w = width - 2 * margins.side * height
    else:
        box_w = height * (1 - 2 * margins.side)
    box_h = height * (1 - margins.top - margins.bottom)
    scale = min(box_w / sprite.width, box_h / sprite.height)
    size = (max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale)))
    sprite = resize_premultiplied(sprite, size)
    left = (width - size[0]) // 2
    top = round(height * (1 - margins.bottom)) - size[1]
    canvas = Image.new("RGBA", canvas_size, (*background, 255))
    canvas.alpha_composite(sprite, (left, top))
    placement = (left / width, top / height, (left + size[0]) / width, (top + size[1]) / height)
    framing = Framing(canvas_size, background, placement, margins)
    return canvas.convert("RGB"), framing


# Tighter framing for one action

TIGHT = Margins(top=0.12, side=0.04, bottom=0.03, full_width=True)
"""Margins for an action that stays within the character's silhouette (yawn, blink, nod)."""
RAISED = Margins(top=0.24, side=0.04, bottom=0.03, full_width=True)
"""Margins for an action that reaches upwards (raised arms, stretching, jumping)."""

_UPWARD = re.compile(
    r"\b(rais\w*|lift\w*|stretch\w*|jump\w*|hop\w*|leap\w*|bounc\w*|reach\w*|"
    r"wav(e|es|ed|ing)|throw\w*|toss\w*|overhead|upward\w*|"
    r"(arms?|hands?|paws?|fists?) up|up (over|above))\b",
    re.IGNORECASE,
)


def margins_for(action: str) -> Margins:
    """Headroom from the motion description: raised arms and jumps need more above."""
    return RAISED if _UPWARD.search(action) else TIGHT
