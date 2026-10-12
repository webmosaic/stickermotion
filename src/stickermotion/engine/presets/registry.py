"""Preset registry: look up preset classes by name."""

from __future__ import annotations

from stickermotion.engine.presets.base import Preset, PresetInfo
from stickermotion.engine.presets.library import Bounce, SquashStretch, Wiggle

_PRESETS: dict[str, type[Preset]] = {
    preset.name: preset for preset in (Bounce, Wiggle, SquashStretch)
}


def get_preset(name: str) -> type[Preset]:
    """Preset class by name; instantiate it with the class and parameters to render."""
    try:
        return _PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(_PRESETS))
        raise KeyError(f"unknown preset {name!r}; known: {known}") from None


def list_presets() -> list[PresetInfo]:
    """Descriptions of all registered presets, sorted by name."""
    return [_PRESETS[name].info() for name in sorted(_PRESETS)]
