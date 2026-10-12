from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from presets_helpers import FIXTURE_NAMES, FIXTURES
from stickermotion.core.alpha import (
    GIF_ALPHA_THRESHOLD,
    TRANSPARENT_BELOW,
    resize_premultiplied,
)
from stickermotion.core.models import AnimationClass, ExportTarget, MasterAnimation
from stickermotion.core.specs import KB, get_spec
from stickermotion.engine.presets import get_preset
from stickermotion.exporters import DEFAULT_MATTE, ExportFitError, export, export_target
from stickermotion.exporters.base import (
    Timing,
    bleed_color,
    distribute_delays,
    fit_to_limit,
    plan_timings,
    sample_indices,
)
from stickermotion.exporters.formats import matte_and_binarize, shared_palette_frames
from stickermotion.pipeline import master_class
from stickermotion.preprocess import normalize

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

# Small master canvas keeps tests fast; exporters only downscale from it.
TEST_CANVAS = 512
TWITCH_DARK = np.array([24, 24, 27])


@pytest.fixture(scope="module")
def masters() -> dict[tuple[str, AnimationClass], MasterAnimation]:
    """One master per fixture and class (bounce at defaults) on a 512 px canvas."""
    out = {}
    for name in FIXTURE_NAMES:
        with Image.open(FIXTURES / name) as image:
            prepared = normalize(image, canvas_size=TEST_CANVAS)
        for anim_class in (AnimationClass.EMOTE, AnimationClass.STICKER):
            out[(name, anim_class)] = get_preset("bounce")(anim_class).render(prepared)
    return out


def _master(frames: int, fps: float, size: int = 64) -> MasterAnimation:
    """Synthetic loop: an opaque square moving one step per frame."""
    images = []
    for i in range(frames):
        frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        frame.paste((200, 60, 60, 255), (8 + i % 8, 16, 40 + i % 8, 48))
        images.append(frame)
    return MasterAnimation(frames=images, fps=fps)


# Timing


def test_sample_indices_no_duplicate_first_frame() -> None:
    assert sample_indices(60, 30) == tuple(range(0, 60, 2))
    indices = sample_indices(42, 20)
    assert indices[0] == 0 and indices[-1] < 42
    assert len(set(indices)) == 20
    with pytest.raises(ValueError):
        sample_indices(10, 11)


def test_plan_timings_respects_target_limits() -> None:
    spec = get_spec(ExportTarget.TELEGRAM_STICKER)
    timings = plan_timings(_master(240, 60), spec)
    best = timings[0]
    assert best.duration_s == 3.0  # 4 s loop played faster, not cut
    assert best.fps <= 30
    assert best.frame_count == 90
    assert [t.frame_count for t in timings] == sorted({t.frame_count for t in timings})[::-1]
    assert min(t.fps for t in timings) >= 8 - 1e-9


def test_plan_timings_twitch_frame_cap() -> None:
    spec = get_spec(ExportTarget.TWITCH_GIF)
    best = plan_timings(_master(90, 30), spec, max_fps=50)[0]
    assert best.frame_count == 60
    assert best.duration_s == pytest.approx(3.0)


def test_plan_timings_discord_sticker_duration() -> None:
    best = plan_timings(_master(180, 30), get_spec(ExportTarget.DISCORD_STICKER))[0]
    assert best.duration_s == 5.0
    assert best.frame_count == 150


@pytest.mark.parametrize(("source", "count"), [(60, 30), (42, 20), (90, 60), (16, 16)])
def test_thinned_loop_is_seamless(source: int, count: int) -> None:
    """No duplicate of the first frame at the end; the wrap step equals a regular step."""
    indices = sample_indices(source, count)
    assert 0 not in indices[1:]
    steps = np.diff([*indices, source])
    assert steps.max() - steps.min() <= 1
    assert indices[-1] != source  # the frame after the last is the first one, not a copy


def test_distribute_delays_multiples_and_total() -> None:
    delays = distribute_delays(30, 1.0, 10, min_units=2)
    assert all(d % 10 == 0 and d >= 20 for d in delays)
    assert sum(delays) == 1000
    assert set(delays) == {30, 40}
    assert sum(distribute_delays(16, 0.8, 1)) == 800


# Fit loop


def test_fit_to_limit_lowers_quality_then_fps(tmp_path: Path) -> None:
    sizes = {(10, "hi"): 500, (10, "lo"): 300, (5, "hi"): 200, (5, "lo"): 100}
    timings = [Timing(tuple(range(10)), 1.0), Timing(tuple(range(0, 10, 2)), 1.0)]

    def encode(path: Path, timing: Timing, quality: str) -> None:
        path.write_bytes(b"x" * sizes[(timing.frame_count, quality)])

    path = tmp_path / "f.bin"
    timing, quality, attempts = fit_to_limit(
        path, timings, ["hi", "lo"], encode, 250, lambda t, q: f"{t.frame_count} {q}"
    )
    assert (timing.frame_count, quality) == (5, "hi")
    assert len(attempts) == 3

    with pytest.raises(ExportFitError) as info:
        fit_to_limit(path, timings, ["hi", "lo"], encode, 50, lambda t, q: f"{t.frame_count} {q}")
    assert "could not fit within" in str(info.value)
    assert len(info.value.attempts) == 4
    assert "5 lo: 0.1 KB" in str(info.value)


def _noise_master(size: int, frames: int, fps: float, seed: int = 1) -> MasterAnimation:
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(frames):
        data = rng.integers(0, 256, (size, size, 4), dtype=np.uint8)
        data[..., 3] = np.where(data[..., 3] > 40, 255, 0)
        images.append(Image.fromarray(data, "RGBA"))
    return MasterAnimation(frames=images, fps=fps)


def test_heavy_noise_fits_or_explains(tmp_path: Path) -> None:
    """Pure noise either fits the limit after reductions or fails with every tried step."""
    master = _noise_master(320, 60, 30)
    try:
        result = export(master, "discord_sticker", tmp_path)
    except ExportFitError as exc:
        message = str(exc)
        assert "discord_sticker.png: could not fit within 512.0 KB" in message
        assert "lossless" in message and "colors 64" in message and "fps" in message
        assert exc.result is not None and not exc.result.valid
        assert any("exceeds 512.0 KB" in v for v in exc.result.violations)
    else:
        assert result.valid and len(result.attempts) > 1


def test_noise_gif_reduces_until_it_fits(tmp_path: Path) -> None:
    master = _noise_master(112, 20, 20)
    result = export(master, "twitch_gif", tmp_path, size=112)
    assert result.valid, result.violations
    assert result.size_bytes <= 512 * KB


# Targets


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
@pytest.mark.parametrize(
    "target",
    [
        ExportTarget.TWITCH_GIF,
        ExportTarget.DISCORD_STICKER,
        ExportTarget.DISCORD_EMOJI,
        pytest.param(ExportTarget.TELEGRAM_STICKER, marks=needs_ffmpeg),
        pytest.param(ExportTarget.TELEGRAM_EMOJI, marks=needs_ffmpeg),
        pytest.param(ExportTarget.OBS_OVERLAY, marks=needs_ffmpeg),
    ],
)
def test_target_export_passes_validator(
    masters: dict[tuple[str, AnimationClass], MasterAnimation],
    tmp_path: Path,
    fixture: str,
    target: ExportTarget,
) -> None:
    master = masters[(fixture, master_class(target))]
    results = export_target(master, target, tmp_path)
    for result in results:
        assert result.valid, (result.path.name, result.violations)
        assert result.path.exists()
        assert result.target is target
    sides = sorted(max(r.width, r.height) for r in results)
    expected = {
        ExportTarget.TWITCH_GIF: [28, 56, 112, 224],
        ExportTarget.DISCORD_STICKER: [320],
        ExportTarget.DISCORD_EMOJI: [128],
        ExportTarget.TELEGRAM_STICKER: [512],
        ExportTarget.TELEGRAM_EMOJI: [100],
        ExportTarget.OBS_OVERLAY: [TEST_CANVAS],
    }[target]
    assert sides == expected


def test_twitch_variants_and_loop(
    masters: dict[tuple[str, AnimationClass], MasterAnimation], tmp_path: Path
) -> None:
    master = masters[("char_square.png", AnimationClass.EMOTE)]
    results = export_target(master, "twitch_gif", tmp_path)
    assert [r.size_variant for r in results] == ["manual", "manual", "manual", "auto"]
    with Image.open(results[2].path) as gif:
        assert gif.info["loop"] == 0
        delays = []
        for index in range(gif.n_frames):
            gif.seek(index)
            delays.append(gif.info["duration"])
    assert all(d % 10 == 0 for d in delays)
    assert sum(delays) == pytest.approx(master.duration_s * 1000, abs=10)


def test_telegram_sticker_longer_side_512(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    frames = [Image.new("RGBA", (200, 100), (0, 0, 0, 0)) for _ in range(10)]
    for i, frame in enumerate(frames):
        frame.paste((10, 200, 10, 255), (20 + i, 20, 80 + i, 80))
    result = export(MasterAnimation(frames=frames, fps=10), "telegram_sticker", tmp_path)
    assert (result.width, result.height) == (512, 256)
    assert result.valid, result.violations


# GIF edges


def _soft_edge_frame(color: tuple[int, int, int]) -> Image.Image:
    """Light character with a wide soft edge; black RGB under transparent pixels."""
    size = 112
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.hypot(xx - size / 2, yy - size / 2)
    alpha = np.clip((40 - dist) / 10, 0, 1)
    data = np.zeros((size, size, 4), dtype=np.uint8)
    data[..., :3] = color
    data[..., 3] = np.rint(alpha * 255)
    data[data[..., 3] == 0] = 0
    return Image.fromarray(data, "RGBA")


def _composite(gif_frame: Image.Image, background: tuple[int, int, int]) -> np.ndarray:
    canvas = Image.new("RGBA", gif_frame.size, (*background, 255))
    canvas.alpha_composite(gif_frame.convert("RGBA"))
    return np.asarray(canvas.convert("RGB")).astype(int)


def _contour(alpha: np.ndarray) -> np.ndarray:
    """Opaque pixels with at least one transparent 4-neighbour."""
    opaque = alpha > 0
    padded = np.pad(opaque, 1)
    neighbours_clear = (
        ~padded[:-2, 1:-1] | ~padded[2:, 1:-1] | ~padded[1:-1, :-2] | ~padded[1:-1, 2:]
    )
    return opaque & neighbours_clear


def test_matte_and_binarize() -> None:
    frame = Image.new("RGBA", (2, 1))
    frame.putpixel((0, 0), (255, 255, 255, 200))
    frame.putpixel((1, 0), (255, 255, 255, 100))
    rgb, opaque = matte_and_binarize(frame, (0, 0, 0))
    assert opaque.tolist() == [[True, False]]
    assert rgb[0, 0].tolist() == [200, 200, 200]
    assert GIF_ALPHA_THRESHOLD == TRANSPARENT_BELOW


def _contour_excess(gif_path: Path, reference: Image.Image) -> float:
    """95th percentile of how much lighter GIF contour pixels are on #18181B than the
    correctly alpha-composited reference frame."""
    with Image.open(gif_path) as gif:
        gif.seek(0)
        frame = gif.convert("RGBA")
    contour = _contour(np.asarray(frame)[..., 3])
    assert contour.sum() > 20
    dark = (24, 24, 27)
    got = _composite(frame, dark)[contour].mean(axis=1)
    want = _composite(reference, dark)[contour].mean(axis=1)
    return float(np.percentile(got - want, 95))


@pytest.mark.parametrize("fixture", FIXTURE_NAMES)
def test_gif_no_light_fringe_on_dark(
    masters: dict[tuple[str, AnimationClass], MasterAnimation], tmp_path: Path, fixture: str
) -> None:
    """Contour pixels on #18181B are no lighter than the true (full alpha) composite."""
    master = masters[(fixture, AnimationClass.EMOTE)]
    reference = resize_premultiplied(master.frames[0], (56, 56))
    result = export(master, "twitch_gif", tmp_path / "default", size=56)
    assert _contour_excess(result.path, reference) <= 12, "light fringe on the contour"
    # Control: matting onto white produces exactly the fringe the test must catch.
    white = export(master, "twitch_gif", tmp_path / "white", size=56, matte=(255, 255, 255))
    assert _contour_excess(white.path, reference) > 30


def test_gif_matte_controls_edge_colour(tmp_path: Path) -> None:
    """A white soft-edged character: no light halo on dark with the default matte."""
    white = _soft_edge_frame((255, 255, 255))
    master = MasterAnimation(frames=[white, white.rotate(5)], fps=10)
    dark_result = export(master, "twitch_gif", tmp_path / "dark", size=112)
    light_result = export(master, "twitch_gif", tmp_path / "light", size=112, matte=(255, 255, 255))
    with Image.open(dark_result.path) as gif:
        dark_frame = gif.convert("RGBA")
    with Image.open(light_result.path) as gif:
        light_frame = gif.convert("RGBA")
    contour = _contour(np.asarray(dark_frame)[..., 3])
    on_dark = _composite(dark_frame, tuple(DEFAULT_MATTE))[contour].mean()
    on_dark_light_matte = _composite(light_frame, tuple(DEFAULT_MATTE))[
        _contour(np.asarray(light_frame)[..., 3])
    ].mean()
    # With the dark matte the edge blends towards the chat background.
    assert on_dark < on_dark_light_matte - 40
    # The edge colour is a blend between matte and white, within ~threshold alpha.
    assert on_dark <= 255 * 0.75 + TWITCH_DARK.mean() * 0.25 + 10


# Colour handling


def test_bleed_color_keeps_alpha_and_fills_transparent() -> None:
    frame = _soft_edge_frame((250, 120, 10))
    bled = np.asarray(bleed_color(frame))
    original = np.asarray(frame)
    assert np.array_equal(bled[..., 3], original[..., 3])
    near = (original[..., 3] == 0) & (bled[..., :3].sum(axis=-1) > 0)
    assert near.any()
    assert np.abs(bled[near][:, :3].astype(int) - [250, 120, 10]).max() <= 3


def test_shared_palette_frames_one_palette() -> None:
    frames = [_soft_edge_frame((250, 120, 10)), _soft_edge_frame((10, 120, 250))]
    out = shared_palette_frames(frames, 64)
    assert all(f.mode == "P" for f in out)
    assert out[0].getpalette() == out[1].getpalette()


def test_downscale_has_no_dark_rim(tmp_path: Path) -> None:
    """Premultiplied downscale: a yellow soft edge stays yellow at 128 px.

    The WEBP exporter starts lossless, but libwebp still rounds colour of low-alpha
    pixels, hence the tolerance."""
    big = _soft_edge_frame((240, 200, 40)).resize((448, 448), Image.Resampling.BICUBIC)
    data = np.asarray(big).copy()
    data[data[..., 3] == 0] = 0
    big = Image.fromarray(data, "RGBA")
    result = export(MasterAnimation(frames=[big, big], fps=10), "discord_emoji", tmp_path)
    with Image.open(result.path) as webp:
        pixels = np.asarray(webp.convert("RGBA")).astype(int)
    edge = (pixels[..., 3] >= 16) & (pixels[..., 3] < 255)
    assert edge.sum() > 20
    assert np.abs(pixels[edge][:, :3] - [240, 200, 40]).max() <= 16
    # A naive straight-alpha downscale would pull edges towards the black RGB (> 100 off).


def test_unknown_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export(_master(4, 10), "myspace_gif", tmp_path)


def test_telegram_emoji_fit_keeps_duration_and_fps_floor() -> None:
    """64 KB emoji target; quality first, then fps down to 15, never shorter."""
    from stickermotion.exporters.targets import TELEGRAM_EMOJI_MIN_FPS, _job

    frames = [Image.new("RGBA", (64, 64), (200, 100, 50, 255)) for _ in range(81)]
    master = MasterAnimation(frames=frames, fps=30.0, loop=True)
    emoji = _job(master, ExportTarget.TELEGRAM_EMOJI, {})
    assert emoji.max_bytes == 64 * KB
    assert all(t.fps >= TELEGRAM_EMOJI_MIN_FPS - 1e-6 for t in emoji.timings)
    assert all(math.isclose(t.duration_s, 2.7) for t in emoji.timings)
    assert len(emoji.timings) > 1  # fps can still drop once quality is exhausted
    sticker = _job(master, ExportTarget.TELEGRAM_STICKER, {})
    assert min(t.fps for t in sticker.timings) < TELEGRAM_EMOJI_MIN_FPS
