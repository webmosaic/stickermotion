"""Generated clip -> transparent loop -> MasterAnimation.

video on green --cut (chroma key per frame | rembg + key)--> RGBA frames --loop (trim |
ping-pong)--> tight square around the union of alpha --> MasterAnimation. The exporters
take it from there unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from PIL import Image, ImageFilter

from stickermotion.core.models import MasterAnimation
from stickermotion.engine.ai.framing import GREEN, Color
from stickermotion.engine.ai.looping import LoopChoice, make_loop
from stickermotion.engine.ai.matting import RembgMatter, chroma_key, frame_key
from stickermotion.engine.ai.skin import SkinFix

MASTER_MARGIN = 0.04
# The outermost pixels of a decoded clip are unreliable (encoder edge, off-colour key);
# the framing keeps the character away from them, so they are always cleared.
EDGE_CLEAR_PX = 2
# Opaque specks smaller than this (side, px) are codec noise, not part of the character.
SPECK_PX = 5


class CutMethod(StrEnum):
    CHROMA = "chroma"
    REMBG = "rembg"


def clean_alpha(frame: Image.Image) -> Image.Image:
    """Clear the frame edge and drop isolated specks (an opening of the visible mask)."""
    data = np.asarray(frame).copy()
    e = EDGE_CLEAR_PX
    if e:
        data[:e] = 0
        data[-e:] = 0
        data[:, :e] = 0
        data[:, -e:] = 0
    visible = Image.fromarray(((data[..., 3] > 0) * 255).astype(np.uint8), "L")
    size = 2 * (SPECK_PX // 2) + 1
    opened = visible.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
    # Keep every pixel near a surviving body (soft edges, thin details next to it).
    keep = np.asarray(opened.filter(ImageFilter.MaxFilter(size + 4))) > 0
    data[~keep] = 0
    return Image.fromarray(data, "RGBA")


def cut(
    frames: Sequence[Image.Image],
    method: CutMethod = CutMethod.CHROMA,
    background: Color = GREEN,
    matter: RembgMatter | None = None,
    skin: SkinFix | None = None,
) -> list[Image.Image]:
    """Cut every full frame into RGBA with the key measured on its own border.

    ``skin`` repairs skin that drifted towards magenta, right after the key.
    """
    if method is CutMethod.CHROMA:
        keyed = [chroma_key(f, frame_key(f, background)) for f in frames]
    else:
        matter = matter or RembgMatter()
        keyed = [matter(f, frame_key(f, background)) for f in frames]
    if skin is not None:
        keyed = [skin(f) for f in keyed]
    return [clean_alpha(f) for f in keyed]


def union_square(
    frames: Sequence[Image.Image], margin: float = MASTER_MARGIN
) -> tuple[int, int, int, int]:
    """Square box around the union of non-transparent pixels of all frames."""
    boxes = [b for b in (f.getchannel("A").getbbox() for f in frames) if b is not None]
    if not boxes:
        raise ValueError("every frame is fully transparent after the cut")
    left = min(b[0] for b in boxes)
    top = min(b[1] for b in boxes)
    right = max(b[2] for b in boxes)
    bottom = max(b[3] for b in boxes)
    side = round(max(right - left, bottom - top) * (1 + 2 * margin))
    cx, cy = (left + right) / 2, (top + bottom) / 2
    x0, y0 = round(cx - side / 2), round(cy - side / 2)
    return x0, y0, x0 + side, y0 + side


def to_master(frames: Sequence[Image.Image], fps: float) -> MasterAnimation:
    """Tight square master; areas outside the video frame are transparent."""
    box = union_square(frames)
    side = box[2] - box[0]
    out = []
    for frame in frames:
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(frame.crop(box), (0, 0))
        out.append(canvas)
    return MasterAnimation(frames=out, fps=fps, loop=True)


@dataclass(frozen=True)
class ChainResult:
    master: MasterAnimation
    loop: LoopChoice
    cut: CutMethod
    edge_touch: dict[str, bool]
    """Which video frame edges the cut character touches in any frame (cropped limbs)."""


def edge_touch(frames: Sequence[Image.Image], px: int = 3) -> dict[str, bool]:
    """Whether opaque pixels reach each edge of the video frame in any frame.

    Checked on the ``px`` rows / columns just inside the cleared edge band.
    """
    touch = {"top": False, "left": False, "right": False, "bottom": False}
    e = EDGE_CLEAR_PX
    for frame in frames:
        alpha = frame.getchannel("A")
        w, h = alpha.size
        regions = {
            "top": (0, e, w, e + px),
            "bottom": (0, h - e - px, w, h - e),
            "left": (e, 0, e + px, h),
            "right": (w - e - px, 0, w - e, h),
        }
        for side, box in regions.items():
            if not touch[side] and alpha.crop(box).getextrema()[1] >= 128:
                touch[side] = True
    return touch


def process(
    frames: Sequence[Image.Image],
    fps: float,
    method: CutMethod = CutMethod.CHROMA,
    background: Color = GREEN,
    matter: RembgMatter | None = None,
    skin: SkinFix | None = None,
    clean: Callable[[list[Image.Image]], list[Image.Image]] | None = None,
) -> ChainResult:
    """``clean`` runs on the cut frames before the loop (old decor removal)."""
    rgba = cut(frames, method, background, matter, skin)
    if clean is not None:
        rgba = clean(rgba)
    loop = make_loop(rgba, fps)
    return ChainResult(to_master(loop.frames, fps), loop, method, edge_touch(rgba))
