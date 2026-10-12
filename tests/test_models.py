from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from stickermotion.core.models import (
    AnimationClass,
    ExportResult,
    ExportTarget,
    MasterAnimation,
)


def _frames(n: int, size: tuple[int, int] = (64, 32), mode: str = "RGBA") -> list[Image.Image]:
    return [Image.new(mode, size) for _ in range(n)]


def test_animation_class_values() -> None:
    assert {c.value for c in AnimationClass} == {"emote", "sticker", "overlay"}


def test_export_target_values() -> None:
    assert {t.value for t in ExportTarget} == {
        "twitch_gif",
        "discord_sticker",
        "discord_emoji",
        "telegram_sticker",
        "telegram_emoji",
        "obs_overlay",
    }


def test_master_animation_properties() -> None:
    anim = MasterAnimation(frames=_frames(30), fps=15)
    assert anim.frame_count == 30
    assert anim.size == (64, 32)
    assert anim.duration_s == pytest.approx(2.0)
    assert anim.loop is True


def test_master_animation_loop_flag() -> None:
    assert MasterAnimation(frames=_frames(1), fps=1, loop=False).loop is False


def test_master_animation_requires_frames() -> None:
    with pytest.raises(ValidationError):
        MasterAnimation(frames=[], fps=10)


@pytest.mark.parametrize("fps", [0, -1])
def test_master_animation_requires_positive_fps(fps: float) -> None:
    with pytest.raises(ValidationError):
        MasterAnimation(frames=_frames(2), fps=fps)


@pytest.mark.parametrize("mode", ["RGB", "L", "P", "LA"])
def test_master_animation_rejects_non_rgba(mode: str) -> None:
    with pytest.raises(ValidationError, match="RGBA"):
        MasterAnimation(frames=_frames(2, mode=mode), fps=10)


def test_master_animation_rejects_mixed_sizes() -> None:
    frames = [Image.new("RGBA", (10, 10)), Image.new("RGBA", (10, 11))]
    with pytest.raises(ValidationError, match="size"):
        MasterAnimation(frames=frames, fps=10)


def test_master_animation_rejects_non_image() -> None:
    with pytest.raises(ValidationError):
        MasterAnimation(frames=["not an image"], fps=10)  # type: ignore[list-item]


def _result(**overrides: object) -> ExportResult:
    data: dict[str, object] = {
        "path": Path("out/a.gif"),
        "target": ExportTarget.TWITCH_GIF,
        "width": 112,
        "height": 112,
        "size_bytes": 1000,
        "frame_count": 10,
        "duration_s": 1.0,
        "valid": True,
    }
    data.update(overrides)
    return ExportResult(**data)  # type: ignore[arg-type]


def test_export_result_valid() -> None:
    result = _result()
    assert result.violations == []
    assert result.target is ExportTarget.TWITCH_GIF


def test_export_result_invalid_with_violations() -> None:
    result = _result(valid=False, violations=["too heavy"])
    assert result.violations == ["too heavy"]


def test_export_result_target_from_string() -> None:
    assert _result(target="telegram_sticker").target is ExportTarget.TELEGRAM_STICKER


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 0},
        {"height": -1},
        {"size_bytes": -1},
        {"frame_count": 0},
        {"duration_s": -0.1},
        {"target": "tiktok"},
        {"valid": True, "violations": ["x"]},
    ],
)
def test_export_result_rejects_bad_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _result(**overrides)


def test_export_result_warnings_do_not_affect_validity() -> None:
    result = _result(warnings=["size 96 px, recommended 128"])
    assert result.valid
    assert result.warnings == ["size 96 px, recommended 128"]
    assert _result().warnings == []
