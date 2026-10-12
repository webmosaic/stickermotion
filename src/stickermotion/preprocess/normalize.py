"""Normalize a character PNG: check alpha, crop by alpha, fit onto a square canvas with headroom."""

from __future__ import annotations

import numpy as np
from PIL import Image

from stickermotion.core.alpha import resize_premultiplied
from stickermotion.core.models import PreparedImage


class NoAlphaError(ValueError):
    """Input has no usable transparency (no alpha channel or fully opaque)."""


class EmptyImageError(ValueError):
    """Input has an alpha channel but nothing above the alpha threshold."""


_ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa", "La"}


def _has_alpha_channel(image: Image.Image) -> bool:
    if image.mode in _ALPHA_MODES:
        return True
    return image.mode in {"P", "L", "RGB"} and "transparency" in image.info


def _to_rgba(image: Image.Image) -> Image.Image:
    if not _has_alpha_channel(image):
        raise NoAlphaError(
            f"image mode {image.mode} has no alpha channel; export the character as PNG "
            "with a transparent background (background removal is not supported yet)"
        )
    return image.convert("RGBA")


def normalize(
    image: Image.Image,
    canvas_size: int = 1024,
    headroom: float = 0.18,
    alpha_threshold: int = 8,
) -> PreparedImage:
    """Prepare a character image for animation.

    - pixels with alpha below `alpha_threshold` become fully transparent (RGB = 0);
    - the image is cropped to the remaining non-transparent pixels;
    - the crop is scaled (premultiplied, no edge fringes) so its longer side fills
      `canvas_size * (1 - 2 * headroom)` and centred on a square transparent canvas,
      leaving at least `headroom * canvas_size` free on every side for preset motion.

    Raises NoAlphaError for inputs without transparency, EmptyImageError when nothing
    is left after thresholding.
    """
    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")
    if not 0 <= headroom < 0.5:
        raise ValueError("headroom must be in [0, 0.5)")
    if not 0 <= alpha_threshold <= 255:
        raise ValueError("alpha_threshold must be in [0, 255]")

    rgba = _to_rgba(image)
    data = np.asarray(rgba).copy()
    alpha = data[..., 3]
    if (alpha == 255).all():
        raise NoAlphaError(
            "image is fully opaque; export the character as PNG with a transparent "
            "background (background removal is not supported yet)"
        )

    data[alpha < alpha_threshold] = 0
    visible = data[..., 3] > 0
    if not visible.any():
        raise EmptyImageError(f"no pixels with alpha >= {alpha_threshold}")

    rows = np.flatnonzero(visible.any(axis=1))
    cols = np.flatnonzero(visible.any(axis=0))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    crop = Image.fromarray(data[top:bottom, left:right], mode="RGBA")

    available = canvas_size * (1 - 2 * headroom)
    scale = available / max(crop.width, crop.height)
    new_w = max(1, round(crop.width * scale))
    new_h = max(1, round(crop.height * scale))

    scaled = resize_premultiplied(crop, (new_w, new_h))
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    off_x = (canvas_size - new_w) // 2
    off_y = (canvas_size - new_h) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = np.asarray(scaled)
    result = Image.fromarray(canvas, mode="RGBA")

    result_alpha = np.asarray(result)[..., 3] > 0
    rows = np.flatnonzero(result_alpha.any(axis=1))
    cols = np.flatnonzero(result_alpha.any(axis=0))
    bbox = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    anchor = ((bbox[0] + bbox[2]) / 2, float(bbox[3]))

    return PreparedImage(
        image=result,
        bbox=bbox,
        anchor=anchor,
        source_size=image.size,
        scale=scale,
    )
