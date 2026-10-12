"""Cutting the character out of generated frames back into transparency.

- ``chroma_key`` (main path): we painted the background green, so alpha comes from how
  green a pixel is (G - max(R, B)), which stays linear across soft edges. The key is
  measured on each frame's border because models drift the background colour slightly.
  Colours are fixed only in the edge band (``edge_colors``): semi-transparent pixels take
  the colour of the nearby opaque interior, and a narrow band of opaque pixels inside the
  contour has green capped at max(R, B). The rest of the character is left untouched.
- ``RembgMatter`` (optional extra ``rembg``): a background-removal model decides what is
  the character, the colour key gives the exact soft edge. For shadows and smudges a key
  cannot separate.

Both return straight RGBA frames with RGB = 0 under alpha = 0.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter

from stickermotion.engine.ai.framing import GREEN, Color

# Alpha below this (0..1) is compression noise and is dropped.
ALPHA_FLOOR = 0.04
BORDER_PX = 8
# Opaque pixels this close to the contour (px) get green spill capped (codec chroma bleed).
SPILL_BAND_PX = 3
# Alpha at or above this counts as opaque interior.
SOLID_ALPHA = 0.999
# Search radii (px) for the interior colour a semi-transparent pixel takes.
BLEED_RADII = (2, 4, 8, 16)
DEFAULT_REMBG_MODEL = "isnet-anime"


def is_green_key(key: Color) -> bool:
    r, g, b = key
    return g - max(r, b) > 128


def key_alpha(
    rgb: np.ndarray, key: Color, bg_tolerance: float = 0.12, fg_level: float = 0.0
) -> np.ndarray:
    """Alpha in [0, 1] from an RGB float array in [0, 1] for a green key.

    Greenness relative to the key's greenness is ~1 on the background and <= 0 on a
    character without green; alpha ramps from 0 at ``1 - bg_tolerance`` to 1 at
    ``fg_level``.
    """
    if not is_green_key(key):
        raise ValueError(f"not a green key: {key}")
    key_arr = np.asarray(key, dtype=np.float32) / 255.0
    key_g = float(key_arr[1] - max(key_arr[0], key_arr[2]))
    g = (rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2])) / key_g
    top = 1.0 - bg_tolerance
    return np.clip((top - g) / (top - fg_level), 0.0, 1.0).astype(np.float32)


def unmix(rgb: np.ndarray, alpha: np.ndarray, key: Color) -> np.ndarray:
    """Foreground colour from ``observed = a * fg + (1 - a) * key``."""
    key_arr = np.asarray(key, dtype=np.float32) / 255.0
    a = alpha[..., None]
    safe = np.where(a > 1e-3, a, 1.0)
    return np.clip((rgb - (1.0 - a) * key_arr) / safe, 0.0, 1.0)


def despill_green(rgb: np.ndarray) -> np.ndarray:
    """Cap green at max(r, b): no green tint left on edges."""
    out = rgb.copy()
    out[..., 1] = np.minimum(out[..., 1], np.maximum(out[..., 0], out[..., 2]))
    return out


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a (2r+1)^2 window per pixel (zero outside the image), via an integral image."""
    pad = np.pad(
        values, [(radius + 1, radius)] + [(radius + 1, radius)] + [(0, 0)] * (values.ndim - 2)
    )
    integral = pad.cumsum(axis=0).cumsum(axis=1)
    k = 2 * radius + 1
    return integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Pixels of ``mask`` whose whole (2r+1)^2 neighbourhood is in the mask."""
    if radius <= 0:
        return mask.copy()
    k = 2 * radius + 1
    counts = _box_sum(mask.astype(np.float64), radius)
    return counts >= k * k - 0.5


def bleed(
    rgb: np.ndarray, source: np.ndarray, radii: tuple[int, ...] = BLEED_RADII
) -> tuple[np.ndarray, np.ndarray]:
    """Mean colour of ``source`` pixels around every pixel (smallest radius that has any).

    Returns (colours, found); ``found`` is False where no source pixel is within the
    largest radius.
    """
    out = np.zeros_like(rgb)
    found = np.zeros(source.shape, dtype=bool)
    weights = source.astype(np.float64)
    weighted = rgb.astype(np.float64) * weights[..., None]
    for radius in radii:
        count = _box_sum(weights, radius)
        total = _box_sum(weighted, radius)
        new = (count > 0.5) & ~found
        out[new] = (total[new] / count[new][:, None]).astype(rgb.dtype)
        found |= new
        if found.all():
            break
    return out, found


def edge_colors(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Colours with green spill removed in the edge band only.

    - Opaque pixels within ``SPILL_BAND_PX`` of the contour: green capped at max(R, B).
    - Semi-transparent pixels: the mean colour of the (cleaned) opaque pixels nearby.
      Un-mixing them from the key divides codec noise by a small alpha and pushes green
      below zero, which turns the edge magenta; the interior colour has no such shift.
      With no opaque pixel nearby (isolated thin details) the observed colour with green
      capped is used: a neutral, never magenta, colour.
    - Everything else keeps its colour (green parts of the character survive).
    """
    solid = alpha >= SOLID_ALPHA
    out = rgb.copy()
    band = solid & ~erode(solid, SPILL_BAND_PX)
    out[band] = despill_green(rgb[band][None])[0]
    semi = (alpha > 0) & ~solid
    if semi.any():
        colours, found = bleed(out, solid)
        fallback = despill_green(rgb)
        out[semi] = np.where(found[semi][:, None], colours[semi], fallback[semi])
    return out


def to_rgba(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    alpha = np.where(alpha < ALPHA_FLOOR, 0.0, alpha)
    out = np.empty((*alpha.shape, 4), dtype=np.uint8)
    out[..., :3] = np.rint(rgb * 255.0).astype(np.uint8)
    out[..., 3] = np.rint(alpha * 255.0).astype(np.uint8)
    out[out[..., 3] == 0, :3] = 0
    return Image.fromarray(out, "RGBA")


def chroma_key(frame: Image.Image, key: Color = GREEN) -> Image.Image:
    """Cut one RGB frame by its flat green background (with despill)."""
    rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    alpha = key_alpha(rgb, key)
    return to_rgba(edge_colors(rgb, alpha), alpha)


def border_color(frame: Image.Image, px: int = BORDER_PX) -> Color:
    """Median colour of the top rows and the side columns of a frame (bottom skipped)."""
    rgb = np.asarray(frame.convert("RGB"))
    border = np.concatenate(
        [rgb[:px].reshape(-1, 3), rgb[:, :px].reshape(-1, 3), rgb[:, -px:].reshape(-1, 3)]
    )
    r, g, b = (int(v) for v in np.median(border, axis=0))
    return (r, g, b)


def frame_key(frame: Image.Image, sent: Color = GREEN) -> Color:
    """Observed border colour when it is still a green key, else the colour we sent."""
    observed = border_color(frame)
    return observed if is_green_key(observed) else sent


def combine_masks(
    model_alpha: np.ndarray, keyed: np.ndarray, grow_px: int = 3, solid: float = 0.95
) -> np.ndarray:
    """The model mask decides what is foreground, the colour key decides the exact edge.

    Outside the (binarised, grown) model mask only pixels the key sees as solid survive,
    so detached props stay while shadows (a darkened key colour) disappear.
    """
    hard = (model_alpha >= 0.5) | (keyed >= solid)
    grown = Image.fromarray((hard * 255).astype(np.uint8), "L")
    if grow_px > 0:
        grown = grown.filter(ImageFilter.MaxFilter(2 * grow_px + 1))
    return keyed * (np.asarray(grown, dtype=np.float32) / 255.0)


MaskFn = Callable[[Image.Image], Image.Image]


class RembgUnavailableError(RuntimeError):
    """rembg is not installed (pip install -e '.[rembg]')."""


class RembgMatter:
    """rembg mask + green key edge. ``mask_fn`` replaces rembg in tests."""

    def __init__(self, model: str = DEFAULT_REMBG_MODEL, mask_fn: MaskFn | None = None) -> None:
        self.model = model
        self._mask_fn = mask_fn

    def _mask(self, frame: Image.Image) -> Image.Image:
        if self._mask_fn is None:
            try:
                import onnxruntime as ort
                from rembg import new_session, remove
            except ImportError as exc:
                raise RembgUnavailableError(
                    "rembg is not installed; pip install -e '.[rembg]' or use the chroma cut"
                ) from exc
            opts = ort.SessionOptions()
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
            session = new_session(self.model, sess_opts=opts)

            def run(image: Image.Image) -> Image.Image:
                return remove(image, session=session, only_mask=True, post_process_mask=False)

            self._mask_fn = run
        return self._mask_fn(frame)

    def __call__(self, frame: Image.Image, key: Color = GREEN) -> Image.Image:
        rgb_img = frame.convert("RGB")
        mask = self._mask(rgb_img).convert("L")
        if mask.size != rgb_img.size:
            mask = mask.resize(rgb_img.size, Image.Resampling.BILINEAR)
        rgb = np.asarray(rgb_img, dtype=np.float32) / 255.0
        model_alpha = np.asarray(mask, dtype=np.float32) / 255.0
        alpha = combine_masks(model_alpha, key_alpha(rgb, key))
        return to_rgba(edge_colors(rgb, alpha), alpha)
