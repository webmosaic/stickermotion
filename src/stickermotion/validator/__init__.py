"""Validator: checks exported files against platform limits from core.specs."""

from stickermotion.validator.validate import FfmpegNotFoundError, UnreadableFileError, validate

__all__ = ["FfmpegNotFoundError", "UnreadableFileError", "validate"]
