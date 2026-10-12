"""Edge tint after the chroma key, Wan clip length parameters, emote focus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from stickermotion.ai_pipeline import animate
from stickermotion.core.models import ExportTarget
from stickermotion.engine.ai import frame_for_model
from stickermotion.engine.ai.describe import DESCRIBE_MODEL
from stickermotion.engine.ai.focus import (
    FULL,
    FocusError,
    FocusSpec,
    check_box,
    crop_frames,
    fractions_to_box,
    motion_square,
    parse_reply,
    parse_spec,
    resolve,
    source_bbox,
    to_video_box,
)
from stickermotion.engine.ai.framing import GREEN
from stickermotion.engine.ai.generate import (
    WAN_MODEL,
    WAN_PARAMS,
    generate,
    wan_params,
    wan_price,
)
from stickermotion.engine.ai.matting import (
    bleed,
    chroma_key,
    edge_colors,
    erode,
    key_alpha,
)
from stickermotion.engine.ai.video import write_mp4
from stickermotion.validator import validate
from test_ai import TOKEN, FakeReplicate, _character, _client, _clip_frames, _journal

SKIN = (232, 190, 160)


@pytest.fixture(scope="module")
def green_clip(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Short H.264 clip on a green background, as Wan delivers it (same as in test_ai)."""
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    write_mp4(path, _clip_frames(), 16.0)
    return path.read_bytes()


@pytest.fixture
def character_png(tmp_path: Path) -> Path:
    path = tmp_path / "hero.png"
    _character().save(path)
    return path


# Edge tint


def _magenta(rgb: np.ndarray) -> np.ndarray:
    """(R + B) / 2 - G: positive means a shift towards magenta."""
    return (rgb[..., 0] + rgb[..., 2]) / 2 - rgb[..., 1]


def edge_shift(rgba: Image.Image) -> float:
    """Mean magenta index of semi-transparent edge pixels minus that of the nearby interior."""
    data = np.asarray(rgba, dtype=np.float32) / 255.0
    alpha = data[..., 3]
    solid = alpha >= 0.999
    interior, found = bleed(data[..., :3], solid, (4,))
    band = (alpha > 0.1) & (alpha < 0.95) & found
    assert band.sum() > 50, "no edge band to measure"
    return float((_magenta(data[..., :3]) - _magenta(interior))[band].mean())


def _soft_skin_on_green(noise: float = 0.0, seed: int = 0) -> Image.Image:
    """Skin disc with a soft (blurred) edge on the green key, plus codec-like noise and a
    green spill ring just inside the contour."""
    size = 160
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((30, 30, 130, 130), fill=255)
    soft = np.asarray(mask.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32) / 255.0
    fg = np.empty((size, size, 3), dtype=np.float32)
    fg[:] = np.asarray(SKIN, dtype=np.float32) / 255.0
    # Spill: the outer 3 px of the opaque disc pick up some green.
    inner = np.asarray(mask.filter(ImageFilter.MinFilter(7)), dtype=np.float32) / 255.0
    ring = (soft > 0.99) & (inner < 0.5)
    fg[ring, 1] = np.minimum(1.0, fg[ring, 1] + 0.15)
    key = np.asarray(GREEN, dtype=np.float32) / 255.0
    rgb = fg * soft[..., None] + key * (1 - soft[..., None])
    if noise:
        rng = np.random.default_rng(seed)
        rgb = rgb + rng.normal(0, noise, rgb.shape).astype(np.float32)
    return Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")


def test_edge_of_skin_on_green_does_not_turn_magenta() -> None:
    for noise in (0.0, 0.03):
        out = chroma_key(_soft_skin_on_green(noise))
        assert edge_shift(out) <= 0.02, noise
        data = np.asarray(out)
        edge = (data[..., 3] > 25) & (data[..., 3] < 240)
        # Every edge pixel keeps the warm skin hue: R > G >= B-ish, no G below B.
        px = data[edge].astype(int)
        assert (px[:, 0] >= px[:, 1]).all()
        assert (px[:, 1] >= px[:, 2] - 8).mean() > 0.97


def test_spill_is_capped_only_in_the_edge_band() -> None:
    alpha = np.zeros((40, 40), dtype=np.float32)
    alpha[5:35, 5:35] = 1.0
    alpha[4, 5:35] = 0.5
    rgb = np.zeros((40, 40, 3), dtype=np.float32)
    rgb[...] = (0.2, 0.8, 0.3)  # a green shirt: green is part of the character
    out = edge_colors(rgb, alpha)
    # Deep inside the contour the green stays.
    assert out[20, 20, 1] == pytest.approx(0.8)
    # In the opaque band next to the contour green is capped at max(R, B).
    assert out[5, 20, 1] == pytest.approx(0.3)
    assert out[6, 20, 1] == pytest.approx(0.3)
    # Semi-transparent pixels take the cleaned interior colour next to them.
    assert out[4, 20, 1] == pytest.approx(0.3)


def test_erode_and_bleed_helpers() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 1:8] = True
    assert erode(mask, 1).sum() == 25 and erode(mask, 0).sum() == mask.sum()
    rgb = np.zeros((9, 9, 3), dtype=np.float32)
    rgb[mask] = (1.0, 0.5, 0.0)
    colours, found = bleed(rgb, mask, (1,))
    assert found[0, 0] and colours[0, 0] == pytest.approx((1.0, 0.5, 0.0))
    far = np.zeros((40, 40), dtype=bool)
    far[0, 0] = True
    _, found_far = bleed(np.zeros((40, 40, 3), dtype=np.float32), far, (2,))
    assert not found_far[30, 30]


def test_isolated_semi_transparent_pixel_is_neutral() -> None:
    # A thin detail with no opaque pixel nearby: the observed colour with green capped.
    rgb = np.zeros((40, 40, 3), dtype=np.float32)
    rgb[...] = (0.0, 1.0, 0.0)
    rgb[20, 20] = (0.5, 0.75, 0.45)
    alpha = key_alpha(rgb, GREEN)
    out = edge_colors(rgb, alpha)
    assert alpha[20, 20] > 0
    assert out[20, 20, 1] <= max(out[20, 20, 0], out[20, 20, 2]) + 1e-6
    assert out[20, 20, 1] >= min(out[20, 20, 0], out[20, 20, 2]) - 1e-6


# Generation parameters


def test_wan_params_override_and_validate() -> None:
    assert wan_params() == WAN_PARAMS
    params = wan_params(frames=97, fps=20)
    assert params["num_frames"] == 97 and params["frames_per_second"] == 20
    assert WAN_PARAMS["frames_per_second"] == 30  # ADR-015 default, untouched
    assert WAN_PARAMS["num_frames"] == 81
    with pytest.raises(ValueError, match="81..121"):
        wan_params(frames=49)
    with pytest.raises(ValueError, match="5..30"):
        wan_params(fps=60)


def test_wan_price_follows_frames_not_fps() -> None:
    assert wan_price(WAN_PARAMS).flat_usd == 0.05
    assert wan_price(wan_params(fps=30)).flat_usd == 0.05
    assert wan_price(wan_params(frames=121)).flat_usd == pytest.approx(0.0747)


def test_generate_sends_params_and_keys_cache_by_them(tmp_path: Path, green_clip: bytes) -> None:
    fake = FakeReplicate(green_clip)
    frame, _ = frame_for_model(_character())
    cache = tmp_path / "cache"
    default = generate(_client(tmp_path, fake), frame, "p", cache)
    slow = generate(_client(tmp_path, fake), frame, "p", cache, params=wan_params(fps=16))
    assert not slow.cached and slow.cache_key != default.cache_key
    sent = fake.creates(WAN_MODEL)
    assert sent[0]["input"]["frames_per_second"] == 30 and sent[0]["input"]["num_frames"] == 81
    assert sent[1]["input"]["frames_per_second"] == 16
    again = generate(_client(tmp_path, fake), frame, "p", cache, params=wan_params(fps=16))
    assert again.cached
    longer = generate(_client(tmp_path / "b", fake), frame, "p", cache, params=wan_params(97))
    assert longer.cost_usd == pytest.approx(0.0599)


def test_animate_records_clip_params(
    tmp_path: Path, green_clip: bytes, character_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    fake = FakeReplicate(green_clip)
    result = animate(
        character_png,
        prompt="hop",
        targets=[ExportTarget.TELEGRAM_EMOJI],
        out_dir=tmp_path,
        transport=fake,
        dotenv=None,
        fps=20,
    )
    assert result.manifest["ai"]["params"]["frames_per_second"] == 20
    assert fake.creates(WAN_MODEL)[0]["input"]["frames_per_second"] == 20


# Emote focus: parsing and validation


def _portrait() -> Image.Image:
    """300x400 PNG: head (circle) on top of a body (rectangle), transparent elsewhere."""
    image = Image.new("RGBA", (300, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((100, 40, 200, 140), fill=(230, 190, 160, 255))
    draw.rectangle((60, 140, 240, 390), fill=(200, 40, 60, 255))
    return image


def test_parse_spec() -> None:
    assert parse_spec("auto") == FocusSpec("auto")
    assert parse_spec(" FULL ") == FocusSpec("full")
    assert parse_spec("10,20,30,40") == FocusSpec("manual", (10, 20, 40, 60))
    for bad in ("1,2,3", "a,b,c,d", "0,0,0,5", "-1,0,5,5"):
        with pytest.raises(FocusError):
            parse_spec(bad)


def test_parse_reply_valid_and_invalid() -> None:
    reply = '```json\n{"x": 0.2, "y": 0.05, "w": 0.5, "h": 0.4, "what": "face"}\n```'
    assert parse_reply(reply) == (0.2, 0.05, 0.5, 0.4)
    for bad in (
        "a face near the top",
        '{"x": 0.1, "y": 0.1}',
        '{"x": "left", "y": 0, "w": 1, "h": 1}',
        '{"x": 120, "y": 40, "w": 300, "h": 200}',  # pixels, not fractions
        '{"x": 0.8, "y": 0.1, "w": 0.6, "h": 0.2}',  # outside the image
        '{"x": 0.1, "y": 0.1, "w": 0, "h": 0.2}',
        "[1, 2, 3]",
    ):
        with pytest.raises(FocusError):
            parse_reply(bad)


def test_check_box_rejects_implausible_rectangles() -> None:
    image = _portrait()
    assert source_bbox(image) == (60, 40, 241, 391)
    assert check_box((90, 30, 210, 150), image) == (90, 40, 210, 150)
    with pytest.raises(FocusError, match="misses"):
        check_box((0, 0, 50, 30), image)
    with pytest.raises(FocusError, match="covers"):
        check_box((60, 40, 241, 391), image)  # the whole character
    with pytest.raises(FocusError, match="covers"):
        check_box((140, 80, 145, 85), image)  # a speck
    sparse = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    ImageDraw.Draw(sparse).rectangle((0, 0, 299, 3), fill=(255, 0, 0, 255))
    ImageDraw.Draw(sparse).rectangle((0, 296, 299, 299), fill=(255, 0, 0, 255))
    with pytest.raises(FocusError, match="transparent"):
        check_box((50, 50, 250, 250), sparse)


def test_fractions_map_onto_the_cropped_character() -> None:
    assert fractions_to_box((0.25, 0.0, 0.5, 0.25), (100, 200, 300, 600)) == (150, 200, 250, 300)
    assert fractions_to_box((-0.02, 0.9, 0.5, 0.2), (0, 0, 100, 100)) == (0, 90, 48, 100)


def test_manual_focus_and_fallback_to_full() -> None:
    image = _portrait()
    ok = resolve(parse_spec("90,30,120,120"), image, None)
    assert ok.mode == "manual" and ok.source_box == (90, 40, 210, 150) and not ok.warnings
    bad = resolve(parse_spec("0,0,20,20"), image, None)
    assert bad.mode == FULL and not bad.active and "rejected" in bad.warnings[0]
    assert resolve(parse_spec("full"), image, None).mode == FULL


def test_auto_focus_calls_vision_model(tmp_path: Path, green_clip: bytes) -> None:
    reply = '{"x": 0.2, "y": 0.0, "w": 0.6, "h": 0.32, "what": "head"}'
    fake = FakeReplicate(green_clip, reply)
    client = _client(tmp_path, fake)
    focus = resolve(parse_spec("auto"), _portrait(), client)
    assert focus.mode == "auto" and focus.active
    # Character box (60, 40)-(241, 391): 181 x 351 px.
    assert focus.source_box == (96, 40, 205, 152)
    assert focus.fractions == (0.2, 0.0, 0.6, 0.32)
    sent = fake.creates(DESCRIBE_MODEL)[0]["input"]
    assert "28 px" in sent["prompt"] and sent["image_input"][0].startswith("data:image/png")
    entry = _journal(tmp_path / "costs.jsonl")[0]
    assert entry["step"] == "focus" and entry["cost_usd"] == focus.cost_usd > 0


@pytest.mark.parametrize(
    "reply",
    [
        "I think the face is at the top.",
        '{"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}',
        '{"x": 0.9, "y": 0.9, "w": 0.01, "h": 0.01}',
    ],
)
def test_auto_focus_falls_back_to_full(tmp_path: Path, green_clip: bytes, reply: str) -> None:
    fake = FakeReplicate(green_clip, reply)
    focus = resolve(parse_spec("auto"), _portrait(), _client(tmp_path, fake))
    assert focus.requested == "auto" and focus.mode == FULL and not focus.active
    assert focus.reply == reply and "auto focus rejected" in focus.warnings[0]


# Emote focus: coordinates and motion


def test_input_box_maps_to_video_through_framing() -> None:
    image = _portrait()
    frame, framing = frame_for_model(image)
    head = (100, 40, 200, 140)
    zone = to_video_box(head, image, framing, frame.size)
    rgba = chroma_key(frame)
    head_px = np.asarray(rgba)[zone[1] : zone[3], zone[0] : zone[2]]
    # The mapped zone lands on the skin-coloured head in the model input frame.
    centre = head_px[head_px.shape[0] // 2, head_px.shape[1] // 2]
    assert abs(int(centre[0]) - 230) < 6 and abs(int(centre[1]) - 190) < 6
    # Same placement on a video of another resolution with the same aspect.
    double = to_video_box(head, image, framing, (frame.width * 2, frame.height * 2))
    assert all(abs(d - 2 * z) <= 2 for d, z in zip(double, zone, strict=True))


def _frames_with_moving_hand(count: int = 12) -> list[Image.Image]:
    """Head at (100..160, 100..160); a fist at its right edge rises slightly past the zone
    top and returns; a raised arm far to the right waves (outside the key zone)."""
    frames = []
    for i in range(count):
        img = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((100, 100, 160, 160), fill=(230, 190, 160, 255))
        draw.rectangle((40, 175, 220, 299), fill=(200, 40, 60, 255))  # body
        lift = round(45 * np.sin(np.pi * i / (count - 1)))
        draw.rectangle((150, 140 - lift, 164, 156 - lift), fill=(230, 190, 160, 255))  # fist
        draw.rectangle((300, 20 + lift, 315, 160), fill=(230, 190, 160, 255))  # far arm
        frames.append(img)
    return frames


def test_motion_square_covers_motion_over_all_frames() -> None:
    frames = _frames_with_moving_hand()
    zone = (100, 100, 160, 160)
    box = motion_square(zone, frames)
    assert box[2] - box[0] == box[3] - box[1]
    # The fist at its highest point (middle frame, top at y=95) is inside the square.
    assert box[1] <= 95 and box[2] >= 164
    # A square from one frame only (the first, fist low) would miss that.
    first_only = motion_square(zone, frames[:1])
    assert first_only[1] > box[1]
    # The far arm and the body below are not pulled in.
    assert box[2] < 300 and box[3] < 190
    assert box[2] - box[0] <= 90


def test_motion_square_keeps_static_zone_tight() -> None:
    frames = [_frames_with_moving_hand()[0]] * 3
    box = motion_square((100, 100, 160, 160), frames)
    assert 64 <= box[2] - box[0] <= 75


def test_crop_frames_pads_outside_the_video() -> None:
    frames = _frames_with_moving_hand(2)
    out = crop_frames(frames, (-10, -10, 50, 50))
    assert out[0].size == (60, 60) and out[0].getpixel((0, 0))[3] == 0


# Pipeline


def test_animate_focus_only_for_emotes_and_page_compares(
    tmp_path: Path, green_clip: bytes, character_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    fake = FakeReplicate(green_clip)
    targets = [ExportTarget.TELEGRAM_EMOJI, ExportTarget.TELEGRAM_STICKER]
    full = animate(
        character_png,
        prompt="hop",
        targets=targets,
        out_dir=tmp_path / "a",
        transport=fake,
        dotenv=None,
        emote_focus="full",
    )
    focused = animate(
        character_png,
        prompt="hop",
        targets=targets,
        out_dir=tmp_path / "b",
        transport=fake,
        dotenv=None,
        emote_focus="40,20,80,60",
    )
    ai = focused.manifest["ai"]["emote_focus"]
    assert ai["mode"] == "manual" and ai["video_square"]

    # Sticker: the same full composition in both runs; emote: a different (cropped) source.
    def size(result: object, name: str) -> tuple[int, int]:
        item = next(r for r in [*result.results, *result.compare] if r.path.name == name)
        return item.width, item.height

    assert size(full, "telegram_sticker.webm") == size(focused, "telegram_sticker.webm")
    full_master = full.manifest["ai"]["master_size"]
    focus_side = ai["video_square"][2] - ai["video_square"][0]
    assert focus_side < full_master[0]
    assert (focused.root / "full" / "telegram_emoji.webm").is_file()
    assert [r.path.name for r in focused.compare] == ["telegram_emoji.webm"]
    for item in [*focused.results, *focused.compare]:
        assert validate(item.path, item.target, item.size_variant).valid
    page = (focused.root / "index.html").read_text()
    assert "full composition vs focus" in page
    assert 'src="full/telegram_emoji.webm"' in page and 'src="telegram_emoji.webm"' in page
    manifest = json.loads((focused.root / "manifest.json").read_text())
    assert manifest["ai"]["emote_focus"]["pairs"][0]["full"] == "full/telegram_emoji.webm"
    # Full composition: no focus block, no comparison files.
    assert (
        full.compare == []
        and "full composition vs focus" not in (full.root / "index.html").read_text()
    )


def test_animate_auto_focus_fallback_keeps_full(
    tmp_path: Path, green_clip: bytes, character_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    fake = FakeReplicate(green_clip, "no idea")
    result = animate(
        character_png,
        prompt="hop",
        targets=[ExportTarget.TELEGRAM_EMOJI],
        out_dir=tmp_path,
        transport=fake,
        dotenv=None,
        emote_focus="auto",
    )
    focus = result.manifest["ai"]["emote_focus"]
    assert focus["requested"] == "auto" and focus["mode"] == FULL
    assert result.compare == [] and "rejected" in focus["warnings"][0]
    assert result.manifest["ai"]["costs_usd"]["focus"] > 0


def test_auto_focus_not_called_without_emote_targets(
    tmp_path: Path, green_clip: bytes, character_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    fake = FakeReplicate(green_clip)
    animate(
        character_png,
        prompt="hop",
        targets=[ExportTarget.TELEGRAM_STICKER],
        out_dir=tmp_path,
        transport=fake,
        dotenv=None,
        emote_focus="auto",
    )
    steps = [e["step"] for e in _journal(tmp_path / "costs.jsonl")]
    assert "focus" not in steps


# ADR-015 defaults


def test_adr015_defaults_auto_focus_30fps_and_skin_color(
    tmp_path: Path, green_clip: bytes, character_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    fake = FakeReplicate(green_clip, '{"x": 0.2, "y": 0.1, "w": 0.6, "h": 0.5, "what": "face"}')
    result = animate(
        character_png,
        prompt="hop",
        targets=[ExportTarget.TELEGRAM_EMOJI],
        out_dir=tmp_path,
        transport=fake,
        dotenv=None,
    )
    assert result.manifest["ai"]["emote_focus"]["requested"] == "auto"
    sent = fake.creates(WAN_MODEL)[0]["input"]
    assert sent["num_frames"] == 81 and sent["frames_per_second"] == 30
    assert "Hands and skin keep exactly their original colors" in sent["prompt"]
