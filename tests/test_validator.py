"""Validator tests. All media files are generated here with Pillow and ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from stickermotion.core.models import ExportTarget
from stickermotion.core.specs import KB, MB
from stickermotion.validator import UnreadableFileError, validate

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not installed")


# ---------- frame generators ----------


def _frames(
    n: int,
    size: tuple[int, int],
    transparent: bool = True,
    noise: bool = False,
    seed: int = 0,
) -> list[Image.Image]:
    """Moving disc on a transparent (or opaque white) background; optional RGB noise."""
    rng = np.random.default_rng(seed)
    w, h = size
    frames = []
    for i in range(n):
        if noise:
            arr = rng.integers(0, 256, size=(h, w, 4), dtype=np.uint8)
            arr[..., 3] = 255
            if transparent:
                arr[: h // 4, :, :] = 0
            frames.append(Image.fromarray(arr, "RGBA"))
            continue
        bg = (0, 0, 0, 0) if transparent else (255, 255, 255, 255)
        image = Image.new("RGBA", size, bg)
        r = min(w, h) // 4
        cx = w // 2 + int(r * 0.5 * np.sin(i / max(n, 1) * 2 * np.pi))
        cy = h // 2
        # Colour depends on the index so no two frames are identical (encoders merge duplicates).
        color = (240, 80, (i * 4) % 256, 255)
        ImageDraw.Draw(image).ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
        frames.append(image)
    return frames


def _save_gif(path: Path, frames: list[Image.Image], duration_ms: int = 50) -> Path:
    first, *rest = frames
    if first.getchannel("A").getextrema()[0] == 255:
        first, *rest = [f.convert("RGB") for f in frames]
    first.save(path, save_all=True, append_images=rest, duration=duration_ms, loop=0, disposal=2)
    return path


def _save_apng(path: Path, frames: list[Image.Image], duration_ms: int = 50) -> Path:
    first, *rest = frames
    if first.getchannel("A").getextrema()[0] == 255:
        first, *rest = [f.convert("RGB") for f in frames]
    first.save(path, format="PNG", save_all=True, append_images=rest, duration=duration_ms, loop=0)
    return path


def _save_webp(path: Path, frames: list[Image.Image], duration_ms: int = 50, **kw: object) -> Path:
    first, *rest = frames
    if first.getchannel("A").getextrema()[0] == 255:
        first, *rest = [f.convert("RGB") for f in frames]
    first.save(
        path, format="WEBP", save_all=True, append_images=rest, duration=duration_ms, loop=0, **kw
    )
    return path


def _save_webm(
    path: Path,
    frames: list[Image.Image],
    fps: int = 30,
    pix_fmt: str = "yuva420p",
    codec: str = "libvpx-vp9",
    audio: bool = False,
    extra: tuple[str, ...] = ("-b:v", "200k", "-deadline", "realtime", "-cpu-used", "8"),
) -> Path:
    src = path.with_suffix("")
    src.mkdir()
    for i, frame in enumerate(frames):
        frame.save(src / f"f{i:04d}.png")
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(src / "f%04d.png")]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
        cmd += ["-shortest", "-c:a", "libopus"]
    cmd += ["-c:v", codec, "-pix_fmt", pix_fmt, *extra]
    if codec == "libvpx":
        cmd += ["-auto-alt-ref", "0"]  # VP8 cannot encode alpha with alt-ref frames
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _assert_violation(result, *needles: str) -> None:  # type: ignore[no-untyped-def]
    assert not result.valid
    text = " | ".join(result.violations)
    for needle in needles:
        assert needle in text, f"{needle!r} not in {result.violations}"


# ---------- GIF: Twitch ----------


def test_gif_valid_auto(tmp_path: Path) -> None:
    path = _save_gif(tmp_path / "ok.gif", _frames(20, (112, 112)))
    result = validate(path, ExportTarget.TWITCH_GIF)
    assert result.valid, result.violations
    assert result.violations == [] and result.warnings == []
    assert (result.width, result.height, result.frame_count) == (112, 112, 20)
    assert result.duration_s == pytest.approx(1.0)
    assert result.target is ExportTarget.TWITCH_GIF


def test_gif_valid_manual(tmp_path: Path) -> None:
    for side in (28, 56, 112):
        path = _save_gif(tmp_path / f"m{side}.gif", _frames(10, (side, side)))
        assert validate(path, "twitch_gif", size_variant="manual").valid


def test_gif_manual_rejects_other_size(tmp_path: Path) -> None:
    path = _save_gif(tmp_path / "m64.gif", _frames(10, (64, 64)))
    _assert_violation(validate(path, "twitch_gif", size_variant="manual"), "expected 28/56/112")
    # 64 is also below the auto minimum of 112.
    _assert_violation(validate(path, "twitch_gif", size_variant="auto"), "below minimum 112")


def test_gif_too_heavy(tmp_path: Path) -> None:
    path = _save_gif(tmp_path / "heavy.gif", _frames(4, (720, 720), noise=True))
    assert path.stat().st_size > 1 * MB
    _assert_violation(validate(path, "twitch_gif"), "weight", "exceeds 1024.0 KB")


def test_gif_manual_weight_limit_is_512kb(tmp_path: Path) -> None:
    path = _save_gif(tmp_path / "heavy112.gif", _frames(60, (112, 112), noise=True))
    assert 512 * KB < path.stat().st_size <= 1 * MB
    assert validate(path, "twitch_gif", size_variant="auto").valid
    _assert_violation(validate(path, "twitch_gif", size_variant="manual"), "exceeds 512.0 KB")


def test_gif_not_square(tmp_path: Path) -> None:
    path = _save_gif(tmp_path / "rect.gif", _frames(10, (160, 120)))
    _assert_violation(validate(path, "twitch_gif"), "not square")


def test_gif_too_large_side(tmp_path: Path) -> None:
    path = _save_gif(tmp_path / "big.gif", _frames(1, (4100, 4100)))
    _assert_violation(validate(path, "twitch_gif"), "above maximum 4096")


def test_gif_too_many_frames(tmp_path: Path) -> None:
    path = _save_gif(tmp_path / "many.gif", _frames(61, (112, 112)))
    _assert_violation(validate(path, "twitch_gif"), "61 frames exceeds 60")


def test_gif_without_alpha(tmp_path: Path) -> None:
    path = _save_gif(tmp_path / "opaque.gif", _frames(10, (112, 112), transparent=False))
    _assert_violation(validate(path, "twitch_gif"), "no transparent pixels")


def test_wrong_format_for_target(tmp_path: Path) -> None:
    path = _save_apng(tmp_path / "x.png", _frames(10, (112, 112)))
    _assert_violation(validate(path, "twitch_gif"), "format apng not allowed")


def test_unknown_size_variant(tmp_path: Path) -> None:
    path = _save_gif(tmp_path / "ok.gif", _frames(2, (112, 112)))
    with pytest.raises(ValueError, match="no size variant"):
        validate(path, "twitch_gif", size_variant="huge")


def test_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "junk.gif"
    path.write_bytes(b"not an image at all")
    with pytest.raises(UnreadableFileError):
        validate(path, "twitch_gif")


# ---------- APNG: Discord sticker ----------


def test_apng_valid(tmp_path: Path) -> None:
    path = _save_apng(tmp_path / "ok.png", _frames(40, (320, 320)))
    result = validate(path, ExportTarget.DISCORD_STICKER)
    assert result.valid, result.violations
    assert result.frame_count == 40
    assert result.duration_s == pytest.approx(2.0)


def test_apng_too_heavy(tmp_path: Path) -> None:
    path = _save_apng(tmp_path / "heavy.png", _frames(3, (320, 320), noise=True))
    assert path.stat().st_size > 512 * KB
    _assert_violation(validate(path, "discord_sticker"), "exceeds 512.0 KB")


def test_apng_wrong_size(tmp_path: Path) -> None:
    path = _save_apng(tmp_path / "small.png", _frames(10, (256, 256)))
    _assert_violation(validate(path, "discord_sticker"), "side 256 px, expected 320")


def test_apng_too_long(tmp_path: Path) -> None:
    path = _save_apng(tmp_path / "long.png", _frames(30, (320, 320)), duration_ms=200)
    _assert_violation(validate(path, "discord_sticker"), "duration 6.00 s exceeds 5 s")


def test_apng_without_alpha(tmp_path: Path) -> None:
    path = _save_apng(tmp_path / "opaque.png", _frames(10, (320, 320), transparent=False))
    _assert_violation(validate(path, "discord_sticker"), "no transparent pixels")


def test_apng_rgba_but_fully_opaque(tmp_path: Path) -> None:
    # Alpha channel present in metadata but no transparent pixel: still rejected.
    frames = _frames(5, (320, 320), transparent=False)
    first, *rest = frames
    first.save(tmp_path / "o.png", format="PNG", save_all=True, append_images=rest, duration=50)
    with Image.open(tmp_path / "o.png") as image:
        assert image.mode == "RGBA"
    _assert_violation(validate(tmp_path / "o.png", "discord_sticker"), "no transparent pixels")


# ---------- WEBP: Discord emoji ----------


def test_webp_valid(tmp_path: Path) -> None:
    path = _save_webp(tmp_path / "ok.webp", _frames(20, (128, 128)))
    result = validate(path, ExportTarget.DISCORD_EMOJI)
    assert result.valid, result.violations
    assert result.warnings == []
    assert result.frame_count == 20


def test_webp_non_recommended_size_is_warning(tmp_path: Path) -> None:
    path = _save_webp(tmp_path / "96.webp", _frames(10, (96, 96)))
    result = validate(path, "discord_emoji")
    assert result.valid
    assert result.violations == []
    assert any("recommended" in w and "96" in w for w in result.warnings)


def test_webp_not_square(tmp_path: Path) -> None:
    path = _save_webp(tmp_path / "rect.webp", _frames(10, (128, 96)))
    _assert_violation(validate(path, "discord_emoji"), "not square")


def test_webp_too_heavy(tmp_path: Path) -> None:
    path = _save_webp(tmp_path / "heavy.webp", _frames(8, (128, 128), noise=True), lossless=True)
    assert path.stat().st_size > 256 * KB
    _assert_violation(validate(path, "discord_emoji"), "exceeds 256.0 KB")


def test_webp_without_alpha(tmp_path: Path) -> None:
    path = _save_webp(tmp_path / "opaque.webp", _frames(10, (128, 128), transparent=False))
    _assert_violation(validate(path, "discord_emoji"), "no transparent pixels")


def test_gif_valid_for_discord_emoji(tmp_path: Path) -> None:
    path = _save_gif(tmp_path / "e.gif", _frames(10, (128, 128)))
    assert validate(path, "discord_emoji").valid


# ---------- WEBM: Telegram ----------


@needs_ffmpeg
def test_webm_valid_sticker(tmp_path: Path) -> None:
    path = _save_webm(tmp_path / "ok.webm", _frames(60, (512, 512)))
    result = validate(path, ExportTarget.TELEGRAM_STICKER)
    assert result.valid, result.violations
    assert (result.width, result.height) == (512, 512)
    assert result.frame_count == 60
    assert result.duration_s == pytest.approx(2.0, abs=0.05)


@needs_ffmpeg
def test_webm_sticker_longer_side_512_not_square(tmp_path: Path) -> None:
    path = _save_webm(tmp_path / "wide.webm", _frames(30, (512, 256)))
    assert validate(path, "telegram_sticker").valid


@needs_ffmpeg
def test_webm_valid_emoji(tmp_path: Path) -> None:
    path = _save_webm(tmp_path / "emoji.webm", _frames(30, (100, 100)))
    assert validate(path, ExportTarget.TELEGRAM_EMOJI).valid


@needs_ffmpeg
def test_webm_too_heavy(tmp_path: Path) -> None:
    frames = _frames(8, (512, 512), noise=True)
    extra = ("-crf", "4", "-b:v", "0", "-deadline", "realtime", "-cpu-used", "8")
    path = _save_webm(tmp_path / "heavy.webm", frames, extra=extra)
    assert path.stat().st_size > 256 * KB
    _assert_violation(validate(path, "telegram_sticker"), "exceeds 256.0 KB")


@needs_ffmpeg
def test_webm_wrong_size(tmp_path: Path) -> None:
    path = _save_webm(tmp_path / "400.webm", _frames(30, (400, 400)))
    _assert_violation(validate(path, "telegram_sticker"), "longer side 400 px, expected 512")


@needs_ffmpeg
def test_webm_too_long(tmp_path: Path) -> None:
    path = _save_webm(tmp_path / "long.webm", _frames(120, (512, 512)))
    _assert_violation(validate(path, "telegram_sticker"), "duration 4.00 s exceeds 3 s")


@needs_ffmpeg
def test_webm_too_fast(tmp_path: Path) -> None:
    path = _save_webm(tmp_path / "fast.webm", _frames(60, (512, 512)), fps=60)
    _assert_violation(validate(path, "telegram_sticker"), "60.00 fps exceeds 30 fps")


@needs_ffmpeg
def test_webm_without_alpha_pix_fmt(tmp_path: Path) -> None:
    path = _save_webm(tmp_path / "yuv.webm", _frames(30, (512, 512)), pix_fmt="yuv420p")
    _assert_violation(
        validate(path, "telegram_sticker"), "no alpha channel in pixel format", "no transparent"
    )


@needs_ffmpeg
def test_webm_alpha_checked_by_decoding(tmp_path: Path) -> None:
    # yuva420p with alpha_mode flag, but every pixel is opaque: metadata alone would pass.
    frames = _frames(30, (512, 512), transparent=False)
    path = _save_webm(tmp_path / "opaque_yuva.webm", frames)
    result = validate(path, "telegram_sticker")
    assert "no alpha channel in pixel format" not in " ".join(result.violations)
    _assert_violation(result, "no transparent pixels")


@needs_ffmpeg
def test_webm_with_audio(tmp_path: Path) -> None:
    path = _save_webm(tmp_path / "audio.webm", _frames(30, (512, 512)), audio=True)
    _assert_violation(validate(path, "telegram_sticker"), "audio not allowed")
    # OBS overlay allows audio.
    assert validate(path, "obs_overlay").valid


@needs_ffmpeg
def test_webm_wrong_codec(tmp_path: Path) -> None:
    path = _save_webm(tmp_path / "vp8.webm", _frames(30, (512, 512)), codec="libvpx")
    _assert_violation(validate(path, "telegram_sticker"), "codec vp8, expected vp9")


@needs_ffmpeg
def test_webm_not_allowed_for_gif_target(tmp_path: Path) -> None:
    path = _save_webm(tmp_path / "x.webm", _frames(10, (112, 112)))
    _assert_violation(validate(path, "twitch_gif"), "format webm not allowed")
