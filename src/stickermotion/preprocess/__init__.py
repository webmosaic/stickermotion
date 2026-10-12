"""Preprocessing: alpha check, crop by alpha, padding, squaring, upscale.

Background removal is a fallback only for inputs without alpha (ADR-005).
"""

from stickermotion.preprocess.normalize import EmptyImageError, NoAlphaError, normalize

__all__ = ["EmptyImageError", "NoAlphaError", "normalize"]
