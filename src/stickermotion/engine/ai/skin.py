"""Skin hue repair after the cut: skin that drifted towards magenta / lilac gets its hue back.

The video model sometimes turns a moving hand lilac in part of the frames, even
with a skin colour requirement in the prompt. The fix runs on cut RGBA frames:

- The skin reference is measured on the input PNG: opaque pixels in the warm skin hue
  range give the hue centre and the saturation / brightness spread (``SkinTone``).
- A frame pixel is a candidate when its saturation and brightness are close to the skin
  reference and its hue lies below the skin hue range, shifted towards magenta by
  ``MIN_SHIFT_DEG`` .. ``MAX_SHIFT_DEG``.
- Colours the character really has are protected twice: hues of that band the input PNG
  itself uses (pink clothes, a sleep mask, lips) are excluded from the saturation those
  colours have on the input upwards (hue limit; paler lilac skin of the same hue is still
  repaired), and pixels
  near places where the model input frame already has a candidate colour are left alone
  (input frame mask).
- A candidate keeps its saturation and brightness; its hue is rotated to the skin centre,
  with a soft ramp at the band edge so corrected and untouched skin meet without a seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

# Warm hue range (degrees) that counts as skin on the input PNG.
SKIN_HUE_RANGE = (0.0, 50.0)
SKIN_MIN_SAT = 0.12
SKIN_MAX_SAT = 0.8
SKIN_MIN_VAL = 0.35
# The input must have at least this share of skin pixels among opaque ones.
MIN_SKIN_SHARE = 0.02
# Hue shift below the skin range (degrees) that is repaired; the ramp starts at half of MIN.
MIN_SHIFT_DEG = 12.0
MAX_SHIFT_DEG = 80.0
# Tolerance around the reference saturation / brightness spread.
SAT_MARGIN = 0.08
VAL_MARGIN = 0.05
# A hue bin of the band used by at least this share of opaque input pixels is protected.
NATIVE_HUE_SHARE = 0.005
HUE_BIN_DEG = 5
# Native colours are protected only from this percentile of their saturation up: lilac
# skin on the live clip shares hues with the pink sleep mask (335-345 deg) but is paler.
NATIVE_SAT_PERCENTILE = 2.0
# Radius (px of the clip) around candidate-coloured pixels of the model input frame.
PROTECT_RADIUS_PX = 6
SOLID_ALPHA = 128
# Pixels this close (px) to the cut contour are left to ``edge_colors``: the outlines of
# stripes and the mask are pale pink there and must not turn skin-coloured.
CONTOUR_PX = 1
# Pixels this close (px) to a pink pixel of the current frame (native hue and saturation)
# are left alone: the anti-aliased rim of clothes is paler than the clothes themselves.
NEAR_NATIVE_PX = 1


@dataclass(frozen=True)
class SkinTone:
    """Skin reference from the input PNG (hue in degrees, saturation / value in 0..1)."""

    hue: float
    hue_low: float
    sat_low: float
    sat_high: float
    val_low: float
    native_bins: frozenset[int] = frozenset()
    """Hue bins (``HUE_BIN_DEG`` wide) inside the band that the input itself uses."""
    native_sat_low: float = 0.0
    """Saturation from which a pixel of a native hue counts as that colour (protected)."""


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> float HxWx3 (hue in degrees 0..360, saturation and value 0..1)."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(axis=-1)
    mn = x.min(axis=-1)
    delta = mx - mn
    safe = np.where(delta > 0, delta, 1.0)
    hue = np.where(
        mx == r,
        ((g - b) / safe) % 6.0,
        np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0),
    )
    hue = np.where(delta > 0, hue * 60.0, 0.0)
    sat = np.where(mx > 0, delta / np.where(mx > 0, mx, 1.0), 0.0)
    return np.stack([hue, sat, mx], axis=-1).astype(np.float32)


def to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Inverse of ``to_hsv``; returns uint8 RGB."""
    h = (hsv[..., 0] % 360.0) / 60.0
    s, v = hsv[..., 1], hsv[..., 2]
    c = v * s
    x = c * (1.0 - np.abs(h % 2.0 - 1.0))
    m = v - c
    zeros = np.zeros_like(h)
    sector = np.floor(h).astype(np.int32) % 6
    choices = [
        np.stack([c, x, zeros], -1),
        np.stack([x, c, zeros], -1),
        np.stack([zeros, c, x], -1),
        np.stack([zeros, x, c], -1),
        np.stack([x, zeros, c], -1),
        np.stack([c, zeros, x], -1),
    ]
    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    for index, choice in enumerate(choices):
        rgb = np.where((sector == index)[..., None], choice, rgb)
    rgb = rgb + m[..., None]
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _shift(hue: np.ndarray, tone: SkinTone) -> np.ndarray:
    """How far (degrees) a hue lies below the skin range, going down through red to magenta."""
    return (tone.hue_low - hue) % 360.0


def skin_reference(image: Image.Image) -> SkinTone | None:
    """Skin tone of the input PNG, or None when the character shows too little skin."""
    rgba = np.asarray(image.convert("RGBA"))
    opaque = rgba[..., 3] >= 250
    if not opaque.any():
        return None
    hsv = to_hsv(rgba[..., :3])[opaque]
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    skin = (
        (hue >= SKIN_HUE_RANGE[0])
        & (hue <= SKIN_HUE_RANGE[1])
        & (sat >= SKIN_MIN_SAT)
        & (sat <= SKIN_MAX_SAT)
        & (val >= SKIN_MIN_VAL)
    )
    if skin.mean() < MIN_SKIN_SHARE:
        return None
    s_hue, s_sat, s_val = hue[skin], sat[skin], val[skin]
    tone = SkinTone(
        hue=float(np.median(s_hue)),
        hue_low=float(np.percentile(s_hue, 5)),
        sat_low=float(np.percentile(s_sat, 5)),
        sat_high=float(np.percentile(s_sat, 95)),
        val_low=float(np.percentile(s_val, 5)),
    )
    # Hue limit: colours of the band the character really has (pink clothes, lips).
    shift = _shift(hue, tone)
    in_band = (shift >= MIN_SHIFT_DEG / 2) & (shift <= MAX_SHIFT_DEG) & (sat >= tone.sat_low)
    bins = np.floor(hue[in_band] / HUE_BIN_DEG).astype(np.int64)
    counts = np.bincount(bins, minlength=360 // HUE_BIN_DEG)
    used = np.flatnonzero(counts >= NATIVE_HUE_SHARE * opaque.sum())
    n = 360 // HUE_BIN_DEG
    native = {int((b + d) % n) for b in used for d in (-1, 0, 1)}
    native_sat = sat[in_band][np.isin(bins % n, used)]
    sat_floor = float(np.percentile(native_sat, NATIVE_SAT_PERCENTILE)) if native_sat.size else 0.0
    return SkinTone(
        **{**tone.__dict__, "native_bins": frozenset(native), "native_sat_low": sat_floor}
    )


def candidates(hsv: np.ndarray, tone: SkinTone) -> np.ndarray:
    """Correction weight 0..1 per pixel (no alpha or protection applied)."""
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    shift = _shift(hue, tone)
    ramp = np.clip((shift - MIN_SHIFT_DEG / 2) / (MIN_SHIFT_DEG / 2), 0.0, 1.0)
    weight = np.where(shift <= MAX_SHIFT_DEG, ramp, 0.0)
    close = (
        (sat >= tone.sat_low - SAT_MARGIN)
        & (sat <= tone.sat_high + SAT_MARGIN)
        & (val >= max(tone.val_low, SKIN_MIN_VAL) - VAL_MARGIN)
    )
    close &= ~native_mask(hsv, tone)
    return np.where(close, weight, 0.0).astype(np.float32)


def native_mask(hsv: np.ndarray, tone: SkinTone) -> np.ndarray:
    """Pixels of a colour the input itself has in the band (native hue and saturation)."""
    if not tone.native_bins:
        return np.zeros(hsv.shape[:2], dtype=bool)
    bins = np.floor(hsv[..., 0] / HUE_BIN_DEG).astype(np.int64) % (360 // HUE_BIN_DEG)
    native = np.isin(bins, np.fromiter(tone.native_bins, dtype=np.int64))
    return native & (hsv[..., 1] >= tone.native_sat_low)


def protect_mask(
    model_input: Image.Image, tone: SkinTone, size: tuple[int, int] | None = None
) -> np.ndarray:
    """Pixels near band-coloured places of the model input frame (input frame mask).

    ``size`` is the clip frame size; the mask is scaled to it when the clip resolution
    differs from the model input (same aspect).
    """
    hsv = to_hsv(np.asarray(model_input.convert("RGB")))
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    shift = _shift(hue, tone)
    banded = (
        (shift >= MIN_SHIFT_DEG / 2)
        & (shift <= MAX_SHIFT_DEG)
        & (sat >= tone.sat_low - SAT_MARGIN)
        & (val >= SKIN_MIN_VAL - VAL_MARGIN)
    )
    mask = Image.fromarray((banded * 255).astype(np.uint8), "L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    if PROTECT_RADIUS_PX > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * PROTECT_RADIUS_PX + 1))
    return np.asarray(mask) > 0


@dataclass(frozen=True)
class SkinFix:
    """Skin reference plus the protected area of the model input frame."""

    tone: SkinTone
    protect: np.ndarray

    @classmethod
    def build(
        cls, image: Image.Image, model_input: Image.Image, size: tuple[int, int]
    ) -> SkinFix | None:
        """None when the input PNG has too little skin to take a reference from."""
        tone = skin_reference(image)
        if tone is None:
            return None
        return cls(tone, protect_mask(model_input, tone, size))

    def to_dict(self) -> dict[str, object]:
        """Reference values for the manifest."""
        tone = self.tone
        return {
            "hue_deg": round(tone.hue, 1),
            "hue_low_deg": round(tone.hue_low, 1),
            "sat": [round(tone.sat_low, 3), round(tone.sat_high, 3)],
            "val_low": round(tone.val_low, 3),
            "protected_hues_deg": sorted(b * HUE_BIN_DEG for b in tone.native_bins),
            "protected_sat_from": round(tone.native_sat_low, 3),
            "protected_px": int(self.protect.sum()),
        }

    def _weights(self, rgba: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        weight = candidates(hsv, self.tone)
        weight[rgba[..., 3] < SOLID_ALPHA] = 0.0
        if self.protect.shape == weight.shape:
            weight[self.protect] = 0.0
        if not weight.any():
            return weight
        solid = Image.fromarray(((rgba[..., 3] >= SOLID_ALPHA) * 255).astype(np.uint8), "L")
        if CONTOUR_PX > 0:
            inner = np.asarray(solid.filter(ImageFilter.MinFilter(2 * CONTOUR_PX + 1))) > 0
            weight[~inner] = 0.0
        pink = native_mask(hsv, self.tone)
        if NEAR_NATIVE_PX > 0 and pink.any():
            grown = Image.fromarray((pink * 255).astype(np.uint8), "L")
            near = grown.filter(ImageFilter.MaxFilter(2 * NEAR_NATIVE_PX + 1))
            weight[np.asarray(near) > 0] = 0.0
        return weight

    def weights(self, frame: Image.Image) -> np.ndarray:
        """Correction weight per pixel of an RGBA frame (0 = untouched)."""
        rgba = np.asarray(frame.convert("RGBA"))
        return self._weights(rgba, to_hsv(rgba[..., :3]))

    def __call__(self, frame: Image.Image) -> Image.Image:
        """The frame with magenta-shifted skin turned back to the skin hue."""
        rgba = np.asarray(frame.convert("RGBA")).copy()
        hsv = to_hsv(rgba[..., :3])
        weight = self._weights(rgba, hsv)
        touched = weight > 0
        if not touched.any():
            return frame
        # Only the touched pixels are converted back (the frame is mostly untouched).
        sub = hsv[touched]
        # Rotate the hue towards the skin centre along the short way round the circle.
        delta = (self.tone.hue - sub[:, 0] + 180.0) % 360.0 - 180.0
        sub[:, 0] = (sub[:, 0] + delta * weight[touched]) % 360.0
        rgba[touched, :3] = to_rgb(sub[None])[0]
        return Image.fromarray(rgba, "RGBA")
