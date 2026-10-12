"""Premultiplied-alpha helpers shared by preprocess, presets and exporters.

Resampling straight (non-premultiplied) RGBA blends the colour of transparent pixels into
semi-transparent edges and produces dark or light fringes. Doing the math on premultiplied
colour avoids that: transparent pixels contribute nothing regardless of their RGB.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

TRANSPARENT_BELOW = 128
"""A pixel counts as transparent when its alpha is below this value (ADR-008).

Lossy VP9 alpha turns a fully opaque plane into values around 252-254, so "alpha < 255"
would accept opaque videos; real transparent backgrounds decode to alpha near 0.
"""

GIF_ALPHA_THRESHOLD = TRANSPARENT_BELOW
"""Binary-alpha cut for GIF: pixels with alpha >= this stay opaque, the rest become
transparent. Equal to TRANSPARENT_BELOW, so the exporter and the validator agree."""


def premultiply(image: Image.Image) -> np.ndarray:
    """Convert an image to a float32 HxWx4 array in [0, 1] with RGB multiplied by alpha."""
    data = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    data[..., :3] *= data[..., 3:4]
    return data


def unpremultiply(data: np.ndarray) -> Image.Image:
    """Convert a premultiplied float array back to an RGBA image.

    Values are clipped to the valid range (colour never exceeds alpha) and fully
    transparent pixels get RGB = 0.
    """
    if data.ndim != 3 or data.shape[2] != 4:
        raise ValueError(f"expected HxWx4 array, got shape {data.shape}")
    alpha = np.clip(data[..., 3], 0.0, 1.0)
    color = np.clip(data[..., :3], 0.0, alpha[..., None])
    safe_alpha = np.where(alpha > 0, alpha, 1.0)[..., None]
    straight = color / safe_alpha
    out = np.empty(data.shape, dtype=np.uint8)
    out[..., :3] = np.rint(straight * 255.0).astype(np.uint8)
    out[..., 3] = np.rint(alpha * 255.0).astype(np.uint8)
    out[out[..., 3] == 0, :3] = 0
    return Image.fromarray(out, mode="RGBA")


def resize_premultiplied(
    image: Image.Image,
    size: tuple[int, int],
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resize an RGBA image without edge fringes by resampling premultiplied channels."""
    data = premultiply(image)
    channels = [
        np.asarray(Image.fromarray(data[..., i], mode="F").resize(size, resample)) for i in range(4)
    ]
    return unpremultiply(np.stack(channels, axis=-1))
