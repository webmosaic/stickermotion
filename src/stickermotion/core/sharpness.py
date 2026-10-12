"""Sharpness metric for comparing exports: variance of the Laplacian.

The Laplacian of the luminance responds to fine detail; a blurrier frame has a lower
variance. Only opaque pixels away from the silhouette edge are measured, so the contrast
between the character and transparency (or the chat background) does not count.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

OPAQUE = 128
EDGE_PX = 3


def laplacian_variance(frame: Image.Image, opaque: int = OPAQUE, edge_px: int = EDGE_PX) -> float:
    """Variance of the 4-neighbour Laplacian of luminance (0..255) inside the opaque area."""
    rgba = frame.convert("RGBA")
    luma = np.asarray(rgba.convert("L"), dtype=np.float32)
    solid = rgba.getchannel("A").point(lambda a: 255 if a >= opaque else 0)
    if edge_px > 0:
        solid = solid.filter(ImageFilter.MinFilter(2 * edge_px + 1))
    mask = np.asarray(solid)[1:-1, 1:-1] > 0
    lap = luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:] - 4 * luma[1:-1, 1:-1]
    values = lap[mask]
    if values.size < 2:
        return 0.0
    return float(values.var())
