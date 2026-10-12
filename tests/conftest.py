from __future__ import annotations

import pytest
from PIL import Image

from presets_helpers import FIXTURE_NAMES, load_fixture
from stickermotion.core.models import AnimationClass, MasterAnimation
from stickermotion.engine.presets import get_preset, list_presets

PRESET_NAMES = [info.name for info in list_presets()]
CLASSES = [AnimationClass.EMOTE, AnimationClass.STICKER]


@pytest.fixture(scope="session")
def canvases() -> dict[str, Image.Image]:
    return {name: load_fixture(name) for name in FIXTURE_NAMES}


@pytest.fixture(scope="session")
def renders(
    canvases: dict[str, Image.Image],
) -> dict[tuple[str, str, AnimationClass], MasterAnimation]:
    """Every preset x class x fixture at defaults (intensity 1), rendered once per session."""
    return {
        (fixture, preset, anim_class): get_preset(preset)(anim_class).render(canvas)
        for fixture, canvas in canvases.items()
        for preset in PRESET_NAMES
        for anim_class in CLASSES
    }
