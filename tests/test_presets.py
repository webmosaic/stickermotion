from __future__ import annotations

from dataclasses import astuple

import numpy as np
import pytest
from PIL import Image

from conftest import CLASSES, PRESET_NAMES
from presets_helpers import FIXTURE_NAMES, FIXTURES, on_canvas
from stickermotion.core.models import AnimationClass, MasterAnimation
from stickermotion.engine.presets import (
    Bounce,
    Preset,
    PresetParams,
    SquashStretch,
    Wiggle,
    get_preset,
    list_presets,
)
from stickermotion.engine.presets.base import CLASS_DEFAULTS, smooth_bump
from stickermotion.engine.transform import FrameTransform, SpriteTransformer

Renders = dict[tuple[str, str, AnimationClass], MasterAnimation]
COMBOS = [(f, p, c) for f in FIXTURE_NAMES for p in PRESET_NAMES for c in CLASSES]


def _alpha(frame: Image.Image) -> np.ndarray:
    return np.asarray(frame.getchannel("A")).astype(np.float32) / 255.0


def _motion(animation: MasterAnimation) -> float:
    """Largest mean alpha difference between any frame and the first one."""
    first = _alpha(animation.frames[0])
    return max(float(np.abs(_alpha(f) - first).mean()) for f in animation.frames)


# Registry and parameters


def test_registry_lists_three_presets() -> None:
    assert PRESET_NAMES == ["bounce", "squash_stretch", "wiggle"]
    for info in list_presets():
        assert info.description
        assert set(info.supported_classes) == {AnimationClass.EMOTE, AnimationClass.STICKER}


def test_get_preset_returns_classes() -> None:
    assert get_preset("bounce") is Bounce
    assert get_preset("wiggle") is Wiggle
    assert get_preset("squash_stretch") is SquashStretch


def test_get_preset_unknown() -> None:
    with pytest.raises(KeyError, match="unknown preset"):
        get_preset("spin")


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_class_defaults(name: str) -> None:
    preset_cls = get_preset(name)
    emote = preset_cls(AnimationClass.EMOTE).params
    sticker = preset_cls("sticker").params
    assert emote.animation_class is AnimationClass.EMOTE
    assert emote.fps == CLASS_DEFAULTS[AnimationClass.EMOTE][1]
    assert sticker.fps == CLASS_DEFAULTS[AnimationClass.STICKER][1]
    assert emote.intensity == sticker.intensity == 1.0
    # Emote: shorter cycle (ADR-002).
    assert emote.duration_s < sticker.duration_s


def test_explicit_params_override_defaults() -> None:
    params = Wiggle("emote", duration_s=1.5, fps=12, intensity=0.4).params
    assert (params.duration_s, params.fps, params.intensity) == (1.5, 12, 0.4)
    assert params.frame_count == 18


def test_frame_count_minimum() -> None:
    params = PresetParams(animation_class="emote", duration_s=0.01, fps=10, intensity=1)
    assert params.frame_count == 2


@pytest.mark.parametrize(
    "kwargs", [{"intensity": 1.1}, {"intensity": -0.1}, {"fps": 0}, {"duration_s": 0}]
)
def test_invalid_params(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        Bounce("sticker", **kwargs)


def test_unsupported_class() -> None:
    with pytest.raises(ValueError, match="does not support"):
        Bounce("overlay")


def test_phases_never_reach_one() -> None:
    phases = Bounce("emote").phases()
    assert phases[0] == 0.0
    assert max(phases) < 1.0
    assert len(phases) == Bounce("emote").params.frame_count


def test_smooth_bump() -> None:
    assert smooth_bump(0) == pytest.approx(0)
    assert smooth_bump(1) == pytest.approx(0)
    assert smooth_bump(0.5) == pytest.approx(1)


def test_render_rejects_non_rgba(canvases: dict[str, Image.Image]) -> None:
    with pytest.raises(ValueError, match="RGBA"):
        Bounce().render(canvases["char_square.png"].convert("RGB"))


def test_render_rejects_character_touching_edge() -> None:
    image = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    with pytest.raises(ValueError, match="edge"):
        Wiggle().render(image)


# Rendering


@pytest.mark.parametrize(("fixture", "preset", "anim_class"), COMBOS)
def test_renders_master_animation(
    renders: Renders, fixture: str, preset: str, anim_class: AnimationClass
) -> None:
    animation = renders[(fixture, preset, anim_class)]
    params = get_preset(preset)(anim_class).params
    assert animation.loop is True
    assert animation.size == (1024, 1024)
    assert animation.fps == params.fps
    assert animation.frame_count == params.frame_count
    assert _motion(animation) > 0.005, "preset must visibly move"


@pytest.mark.parametrize("name", PRESET_NAMES)
@pytest.mark.parametrize("anim_class", CLASSES)
@pytest.mark.parametrize("t", [0.0, 0.13, 0.5, 0.77])
def test_pose_is_periodic(name: str, anim_class: AnimationClass, t: float) -> None:
    preset = get_preset(name)(anim_class)
    now = astuple(preset.pose(t, 500, 1.0))
    next_loop = astuple(preset.pose(t + 1.0, 500, 1.0))
    assert now == pytest.approx(next_loop, abs=1e-9)


@pytest.mark.parametrize(("fixture", "preset", "anim_class"), COMBOS)
def test_seamless_loop(
    renders: Renders,
    canvases: dict[str, Image.Image],
    fixture: str,
    preset: str,
    anim_class: AnimationClass,
) -> None:
    animation = renders[(fixture, preset, anim_class)]
    instance = get_preset(preset)(anim_class)
    canvas = canvases[fixture]
    transformer = SpriteTransformer(canvas)
    amplitude = instance.fit_amplitude(transformer)
    height = transformer.bbox[3] - transformer.bbox[1]

    # The frame after the last one, computed by formula at phase 1.0, equals the first.
    following = transformer.render(instance.pose(1.0, height, amplitude))
    diff = np.abs(_alpha(following) - _alpha(animation.frames[0])).max()
    assert diff <= 1 / 255

    # The last frame is not a duplicate of the first.
    assert not np.array_equal(np.asarray(animation.frames[-1]), np.asarray(animation.frames[0]))

    # No jump at the wrap: last -> first step is no bigger than the largest regular step.
    alphas = [_alpha(f) for f in animation.frames]
    steps = [float(np.abs(b - a).mean()) for a, b in zip(alphas, alphas[1:], strict=False)]
    wrap = float(np.abs(alphas[0] - alphas[-1]).mean())
    assert wrap <= max(steps) * 1.05 + 1e-6


@pytest.mark.parametrize(("fixture", "preset", "anim_class"), COMBOS)
def test_character_stays_inside_canvas(
    renders: Renders, fixture: str, preset: str, anim_class: AnimationClass
) -> None:
    for frame in renders[(fixture, preset, anim_class)].frames:
        alpha = np.asarray(frame.getchannel("A"))
        border = np.concatenate([alpha[0], alpha[-1], alpha[:, 0], alpha[:, -1]])
        assert border.max() == 0


def test_tight_canvas_reduces_amplitude() -> None:
    """With little margin the preset shrinks its motion instead of leaving the canvas."""
    with Image.open(FIXTURES / "char_tall.png") as source:
        canvas = on_canvas(source, canvas=256, margin=0.04)
    preset = Bounce("emote")
    amplitude = preset.fit_amplitude(SpriteTransformer(canvas))
    assert 0 <= amplitude < 1
    animation = preset.render(canvas)
    for frame in animation.frames:
        alpha = np.asarray(frame.getchannel("A"))
        assert alpha[0].max() == 0 and alpha[:, 0].max() == 0 and alpha[:, -1].max() == 0


def _solid_colour_canvas(canvas: Image.Image, colour: tuple[int, int, int]) -> Image.Image:
    """Same alpha as the fixture, single colour, black RGB where transparent."""
    out = Image.new("RGBA", canvas.size, (*colour, 255))
    out.putalpha(canvas.getchannel("A"))
    pixels = np.asarray(out).copy()
    pixels[pixels[..., 3] == 0] = 0
    return Image.fromarray(pixels, "RGBA")


@pytest.mark.parametrize("preset", PRESET_NAMES)
@pytest.mark.parametrize("anim_class", CLASSES)
def test_no_fringe_on_transformed_frames(
    canvases: dict[str, Image.Image], preset: str, anim_class: AnimationClass
) -> None:
    colour = (240, 200, 40)
    canvas = _solid_colour_canvas(canvases["char_wide_soft.png"], colour)
    animation = get_preset(preset)(anim_class, duration_s=0.5, fps=8).render(canvas)
    for frame in animation.frames:
        pixels = np.asarray(frame).astype(int)
        alpha = pixels[..., 3]
        edge = (alpha >= 8) & (alpha < 255)
        assert edge.sum() > 100
        assert np.abs(pixels[edge][:, :3] - np.array(colour)).max() <= 6
        assert pixels[alpha == 0][:, :3].max() == 0


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_emote_moves_more_than_sticker(renders: Renders, fixture: str, preset: str) -> None:
    emote = renders[(fixture, preset, AnimationClass.EMOTE)]
    sticker = renders[(fixture, preset, AnimationClass.STICKER)]
    assert _motion(emote) > _motion(sticker) * 1.3
    assert emote.duration_s < sticker.duration_s
    assert emote.frame_count < sticker.frame_count


@pytest.mark.parametrize("preset_cls", [Bounce, Wiggle, SquashStretch])
def test_emote_and_sticker_styles_differ(preset_cls: type[Preset]) -> None:
    styles = preset_cls.styles  # type: ignore[attr-defined]
    assert styles[AnimationClass.EMOTE] != styles[AnimationClass.STICKER]
    assert (
        preset_cls.default_duration_s[AnimationClass.EMOTE]
        < preset_cls.default_duration_s[AnimationClass.STICKER]
    )


def test_intensity_scales_motion(canvases: dict[str, Image.Image]) -> None:
    canvas = canvases["char_square.png"]
    low = Wiggle("sticker", duration_s=1, fps=8, intensity=0.3).render(canvas)
    high = Wiggle("sticker", duration_s=1, fps=8, intensity=1.0).render(canvas)
    assert _motion(low) < _motion(high)


def test_zero_intensity_is_static(canvases: dict[str, Image.Image]) -> None:
    canvas = canvases["char_square.png"]
    animation = SquashStretch("emote", intensity=0.0).render(canvas)
    assert _motion(animation) < 1 / 255


def test_bounce_lands_with_squash() -> None:
    preset = Bounce("emote")
    style = preset.styles[AnimationClass.EMOTE]
    contact_mid = preset.pose(style.contact / 2, 500, 1.0)
    assert contact_mid.scale_y < 1 < contact_mid.scale_x
    assert contact_mid.dy == 0
    apex = preset.pose(style.contact + (1 - style.contact) / 2, 500, 1.0)
    assert apex.dy == pytest.approx(-style.jump * 500)


def test_wiggle_rotates_both_ways() -> None:
    preset = Wiggle("sticker")
    angles = [preset.pose(t, 500, 1.0).angle_deg for t in preset.phases()]
    assert max(angles) > 0 > min(angles)
    assert max(angles) == pytest.approx(-min(angles), rel=0.05)


@pytest.mark.parametrize("anim_class", CLASSES)
def test_squash_stretch_preserves_area(anim_class: AnimationClass) -> None:
    preset = SquashStretch(anim_class)
    poses: list[FrameTransform] = [preset.pose(t, 500, 1.0) for t in preset.phases()]
    for pose in poses:
        assert pose.scale_x * pose.scale_y == pytest.approx(1.0)
    assert min(p.scale_y for p in poses) < 1 < max(p.scale_y for p in poses)


def test_render_accepts_prepared_image() -> None:
    """Output of preprocess.normalize renders directly; its anchor is the pivot (ADR-008)."""
    from stickermotion.preprocess import normalize

    with Image.open(FIXTURES / "char_tall.png") as source:
        prepared = normalize(source, canvas_size=256)
    from_prepared = Wiggle("emote", duration_s=0.4, fps=10).render(prepared)
    from_image = Wiggle("emote", duration_s=0.4, fps=10).render(prepared.image)
    assert from_prepared.frame_count == from_image.frame_count == 4
    for a, b in zip(from_prepared.frames, from_image.frames, strict=True):
        diff = np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).max()
        assert diff <= 2  # anchor equals the default pivot (bottom centre of bbox)
