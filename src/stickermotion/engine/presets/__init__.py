"""Procedural animation presets.

Each preset takes an RGBA image plus parameters (duration, intensity, AnimationClass)
and returns a MasterAnimation (ADR-002, ADR-003).
"""

from stickermotion.engine.presets.base import Preset, PresetInfo, PresetParams
from stickermotion.engine.presets.library import Bounce, SquashStretch, Wiggle
from stickermotion.engine.presets.registry import get_preset, list_presets

__all__ = [
    "Bounce",
    "Preset",
    "PresetInfo",
    "PresetParams",
    "SquashStretch",
    "Wiggle",
    "get_preset",
    "list_presets",
]
