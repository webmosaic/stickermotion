import pytest

from stickermotion.core.models import AnimationClass, ExportTarget
from stickermotion.core.specs import KB, MB, SPECS, get_spec


def test_all_targets_present() -> None:
    assert set(SPECS) == set(ExportTarget)
    for target, spec in SPECS.items():
        assert spec.target is target
        assert get_spec(target) is spec


def test_twitch_gif() -> None:
    spec = get_spec(ExportTarget.TWITCH_GIF)
    assert spec.formats == ("gif",)
    assert spec.square
    auto, manual = spec.size_variants
    assert (auto.min_side, auto.max_side, auto.max_bytes) == (112, 4096, 1 * MB)
    assert manual.fixed_sizes == (28, 56, 112)
    assert manual.max_bytes == 512 * KB
    assert spec.max_frames == 60
    assert spec.max_flashes_per_s == 3
    assert spec.binary_alpha
    assert spec.max_bytes == 1 * MB
    assert spec.animation_class is AnimationClass.EMOTE


def test_discord_sticker() -> None:
    spec = get_spec(ExportTarget.DISCORD_STICKER)
    assert spec.formats[0] == "apng"
    assert set(spec.formats) == {"apng", "gif", "png", "lottie"}
    assert spec.size_variants[0].fixed_sizes == (320,)
    assert spec.max_bytes == 512 * KB
    assert spec.max_duration_s == 5
    assert spec.animation_class is AnimationClass.STICKER


def test_discord_emoji() -> None:
    spec = get_spec(ExportTarget.DISCORD_EMOJI)
    assert set(spec.formats) == {"gif", "webp", "apng"}
    assert spec.square
    assert spec.size_variants[0].fixed_sizes == (128,)
    assert spec.max_bytes == 256 * KB
    assert spec.animation_class is AnimationClass.EMOTE


def test_telegram_sticker() -> None:
    spec = get_spec(ExportTarget.TELEGRAM_STICKER)
    assert spec.formats == ("webm",)
    assert spec.codec == "vp9"
    assert not spec.square
    assert spec.size_variants[0].fixed_sizes == (512,)
    assert spec.max_duration_s == 3
    assert spec.max_fps == 30
    assert spec.max_bytes == 256 * KB
    assert spec.requires_alpha
    assert not spec.audio_allowed
    assert spec.animation_class is AnimationClass.STICKER


def test_telegram_emoji() -> None:
    spec = get_spec(ExportTarget.TELEGRAM_EMOJI)
    assert spec.formats == ("webm",)
    assert spec.codec == "vp9"
    assert spec.square
    assert spec.size_variants[0].fixed_sizes == (100,)
    # "the rest of the limits as for video stickers"
    sticker = get_spec(ExportTarget.TELEGRAM_STICKER)
    assert spec.max_duration_s == sticker.max_duration_s
    assert spec.max_fps == sticker.max_fps
    assert spec.audio_allowed == sticker.audio_allowed
    # Video emoji fail far below the documented 256 KB: own 64 KB target.
    assert spec.max_bytes == 64 * KB
    assert spec.animation_class is AnimationClass.EMOTE


def test_obs_overlay() -> None:
    spec = get_spec(ExportTarget.OBS_OVERLAY)
    assert spec.formats[0] == "webm"
    assert spec.codec == "vp9"
    assert spec.requires_alpha
    assert spec.max_bytes is None
    assert spec.max_frames is None
    assert spec.animation_class is AnimationClass.OVERLAY


@pytest.mark.parametrize("target", list(ExportTarget))
def test_every_target_requires_alpha(target: ExportTarget) -> None:
    # ADR-005: full transparency is mandatory in every export.
    assert get_spec(target).requires_alpha


def test_recommended_sizes() -> None:
    # ADR-007: only the Discord emoji size is a recommendation (warning, not violation).
    recommended = {
        target for target, spec in SPECS.items() if any(v.recommended for v in spec.size_variants)
    }
    assert recommended == {ExportTarget.DISCORD_EMOJI}
