"""Exporters: one per ExportTarget, sharing a common interface.

Each exporter converts a MasterAnimation into a platform file, tunes compression until
the file fits the limits from core.specs and returns an ExportResult (ADR-001).
"""

from stickermotion.exporters.base import ExportFitError, FrameCache
from stickermotion.exporters.formats import FfmpegMissingError
from stickermotion.exporters.targets import DEFAULT_MATTE, export, export_target

__all__ = [
    "DEFAULT_MATTE",
    "ExportFitError",
    "FfmpegMissingError",
    "FrameCache",
    "export",
    "export_target",
]
