"""Decor as separate layers, tighter framing, 720p, Telegram sticker sharpness."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from stickermotion.ai_pipeline import animate
from stickermotion.core.models import ExportTarget, MasterAnimation
from stickermotion.core.sharpness import laplacian_variance
from stickermotion.engine.ai import decor
from stickermotion.engine.ai.decor import (
    DECOR_SYSTEM,
    DecorMode,
    Motion,
    compose,
    find_candidates,
    label_components,
    place,
    plan,
    pose,
    summary_uk,
)
from stickermotion.engine.ai.describe import DESCRIBE_MODEL
from stickermotion.engine.ai.framing import (
    RAISED,
    TIGHT,
    Margins,
    canvas_for,
    frame_for_model,
    margins_for,
)
from stickermotion.engine.ai.generate import WAN_MODEL, wan_params, wan_price
from stickermotion.engine.ai.video import write_mp4
from stickermotion.exporters import export
from stickermotion.exporters.base import Timing, Unsharp, fit_candidates, sharpen
from stickermotion.exporters.formats import Vp9Rate, encode_webm
from stickermotion.exporters.targets import telegram_sticker_order
from stickermotion.validator import validate
from test_ai import TOKEN, FakeReplicate, _client, _journal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_quality  # noqa: E402


def _scene() -> Image.Image:
    """Character (disc + arm), a moon, two stars, a speck and a blob touching the body."""
    image = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((60, 80, 240, 290), fill=(200, 40, 60, 255))
    draw.rectangle((230, 150, 300, 170), fill=(230, 180, 150, 255))  # arm, attached
    draw.ellipse((300, 140, 316, 180), fill=(90, 60, 30, 255))  # held object, 0 px gap
    draw.ellipse((300, 20, 380, 100), fill=(250, 210, 60, 255))  # moon: a thin crescent
    draw.ellipse((318, 10, 398, 90), fill=(0, 0, 0, 0))
    draw.polygon([(290, 120), (296, 132), (284, 132)], fill=(250, 220, 80, 255))  # star
    draw.polygon([(350, 200), (358, 216), (342, 216)], fill=(250, 220, 80, 255))  # star
    draw.point((20, 20), fill=(255, 255, 255, 255))  # speck
    return image


# Components and candidates


def test_label_components_8_connected_and_merging() -> None:
    mask = np.zeros((6, 8), dtype=bool)
    mask[0, 0] = mask[1, 1] = True  # diagonal neighbours: one component
    mask[0, 4:7] = True  # a U shape whose arms join below
    mask[1, 4] = mask[1, 6] = mask[2, 4:7] = True
    mask[5, 0] = True
    labels, count = label_components(mask)
    assert count == 3
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 4] == labels[0, 6] == labels[2, 5]
    assert len({labels[0, 0], labels[0, 4], labels[5, 0]}) == 3
    assert labels[3, 3] == 0


def test_find_candidates_keeps_detached_decor_only() -> None:
    candidates, main = find_candidates(_scene())
    assert main is not None and main[0] == 60
    boxes = [c.box for c in candidates]
    assert len(candidates) == 3, boxes  # moon + 2 stars; speck and held object stay
    assert all(c.box[0] >= 284 for c in candidates)
    assert [c.index for c in candidates] == [1, 2, 3]
    assert candidates[0].box[1] < candidates[1].box[1] < candidates[2].box[1]


def test_plan_without_client_uses_shape_heuristic_and_removes_decor() -> None:
    image = _scene()
    result = plan(image, DecorMode.AUTO)
    assert result.separate and result.animated
    motions = {i.box[1]: i.motion for i in result.items}
    assert motions[min(motions)] is Motion.SWAY  # the moon (largest, crescent)
    assert {m for k, m in motions.items() if k != min(motions)} == {Motion.TWINKLE}
    alpha = np.asarray(result.character.getchannel("A"))
    for item in result.items:
        x0, y0, x1, y1 = item.box
        assert alpha[y0:y1, x0:x1].max() == 0
    assert np.asarray(result.sprites[0].getchannel("A")).max() == 255
    assert np.asarray(result.character.getchannel("A"))[160, 150] == 255  # body untouched
    phases = [i.phase for i in result.items]
    assert len(set(phases)) == len(phases)


def test_plan_together_and_off() -> None:
    together = plan(_scene(), "together")
    assert not together.separate and together.items == []
    assert together.character.tobytes() == _scene().tobytes()
    off = plan(_scene(), "off")
    assert off.separate and not off.animated and len(off.items) == 3


def test_plan_vision_verdicts_rejects_and_size_rule(tmp_path: Path) -> None:
    reply = json.dumps(
        {
            "items": [
                {"id": 1, "decor": True, "what": "star", "motion": "twinkle"},  # the moon
                {"id": 2, "decor": False, "what": "button"},
                {"id": 3, "decor": True, "what": "star", "motion": "pulse"},
            ]
        }
    )
    fake = FakeReplicate(b"", describe_reply=reply)
    client = _client(tmp_path, fake)
    result = plan(_scene(), DecorMode.AUTO, client)
    assert fake.creates(DESCRIBE_MODEL)[0]["input"]["system_prompt"] == DECOR_SYSTEM
    assert [i.index for i in result.items] == [1, 3]
    assert result.rejected[0]["index"] == 2
    # The dominant moon may not twinkle even when the model says so.
    assert result.items[0].motion is Motion.SWAY
    assert any("dominant" in w for w in result.warnings)
    assert result.items[1].motion is Motion.PULSE and result.items[1].source == DESCRIBE_MODEL
    assert result.cost_usd > 0
    assert _journal(tmp_path / "costs.jsonl")[0]["step"] == "decor"


def test_plan_bad_reply_falls_back_to_heuristic(tmp_path: Path) -> None:
    fake = FakeReplicate(b"", describe_reply="no idea")
    result = plan(_scene(), DecorMode.AUTO, _client(tmp_path, fake))
    assert len(result.items) == 3 and all(i.source == "heuristic" for i in result.items)
    assert "heuristic" in result.warnings[0]


def test_parse_reply_and_summary() -> None:
    parsed = decor.parse_reply('ok {"items": [{"id": "2", "decor": "yes", "motion": "spin"}]}')
    assert parsed == {2: (True, "", None)}
    with pytest.raises(ValueError):
        decor.parse_reply('{"x": 1}')
    items = [{"what": "moon"}, {"what": "star"}, {"what": "star"}, {"what": "?"}]
    assert summary_uk(items) == "місяць, зірка ×2, елемент"


# Motion and composition


@pytest.mark.parametrize("motion", list(Motion))
def test_pose_is_periodic(motion: Motion) -> None:
    a, opacity_a = pose(motion, 0.0, 0.3, 50)
    b, opacity_b = pose(motion, 1.0, 0.3, 50)
    assert a.scale_x == pytest.approx(b.scale_x) and a.angle_deg == pytest.approx(b.angle_deg)
    assert a.dy == pytest.approx(b.dy) and opacity_a == pytest.approx(opacity_b)
    moved = {pose(motion, t / 8, 0.3, 50)[0] for t in range(8)}
    assert len(moved) > 1


def test_compose_keeps_decor_in_place_while_character_moves() -> None:
    """The character bobs 30 px; decor stays where the composition put it."""
    scene = _scene()
    split = plan(scene, DecorMode.AUTO)
    frame, framing = frame_for_model(split.character, margins=TIGHT)
    box = framing.pixel_box(framing.placement, frame.size)
    sprite = split.character.crop(split.character.getchannel("A").getbbox())
    sprite = sprite.resize((box[2] - box[0], box[3] - box[1]))
    frames = []
    for i in range(20):
        canvas = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        canvas.alpha_composite(sprite, (box[0], box[1] - round(30 * math.sin(math.pi * i / 19))))
        frames.append(canvas)
    placed = place(split, framing, frame.size)
    result = compose(frames, 20.0, placed, min_side=800)
    master = result.master
    assert master.size == (800, 800) and master.frame_count == 20
    assert result.scale > 1.0  # decor rendered straight at the upscaled master size
    # Yellow decor pixels: their centre barely moves while the body moves by 30 px.
    centres = []
    for f in master.frames:
        a = np.asarray(f).astype(int)
        yellow = (a[..., 3] > 128) & (a[..., 0] > 230) & (a[..., 1] > 190) & (a[..., 2] < 100)
        ys, xs = np.nonzero(yellow)
        centres.append((xs.mean(), ys.mean()))
    spread = np.ptp(np.array(centres), axis=0)
    assert spread.max() < 0.03 * master.size[0], spread
    assert (spread > 0).any()  # but it does move on its own


def test_animate_with_decor_generates_character_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    source = tmp_path / "scene.png"
    _scene().save(source)
    split = plan(_scene(), DecorMode.AUTO)
    frame, _ = frame_for_model(split.character, canvas_for("720p"), TIGHT)
    clip = tmp_path / "clip.mp4"
    write_mp4(clip, [frame] * 12, 16.0)
    reply = json.dumps({"action": "the hero nods", "decorations": "the moon sways"})
    fake = FakeReplicate(clip.read_bytes(), describe_reply=reply)
    result = animate(
        source,
        targets=[ExportTarget.TELEGRAM_STICKER],
        out_dir=tmp_path / "out",
        transport=fake,
        dotenv=None,
        emote_focus="full",
        resolution="720p",
        budget_usd=0.2,
    )
    ai = result.manifest["ai"]
    assert ai["decor"]["mode"] == "auto" and len(ai["decor"]["items"]) == 3
    assert ai["decor"]["composition"]["scale"] >= 1.0
    assert "moon" not in result.description.prompt  # decor sentence dropped
    wan = fake.creates(WAN_MODEL)[0]["input"]
    assert wan["resolution"] == "720p"
    assert ai["costs_usd"]["generate"] == 0.11
    assert list(ai["framing"]["canvas_size"]) == [1280, 720]
    assert ai["framing"]["margins"] == {"top": 0.12, "side": 0.04, "bottom": 0.03,
                                        "full_width": True}  # fmt: skip
    model_input = Image.open(result.root / "model_input.png").convert("RGB")
    assert model_input.size == (1280, 720)
    # No yellow decor pixels in what the video model saw.
    data = np.asarray(model_input).astype(int)
    assert not ((data[..., 0] > 230) & (data[..., 1] > 190) & (data[..., 2] < 100)).any()
    sticker = result.results[0]
    assert sticker.valid and (sticker.width, sticker.height) == (512, 512)


# Framing and Wan parameters


def test_margins_follow_the_action() -> None:
    assert margins_for("The girl yawns behind her fist") is TIGHT
    for action in ("She raises both arms", "he jumps", "stretches upward", "waves hello"):
        assert margins_for(action) is RAISED, action
    assert TIGHT.top < RAISED.top < Margins().top


def test_full_width_framing_makes_wide_character_larger() -> None:
    wide = Image.new("RGBA", (320, 120), (0, 0, 0, 0))
    ImageDraw.Draw(wide).rectangle((10, 10, 309, 109), fill=(255, 0, 0, 255))
    _, narrow = frame_for_model(wide)
    _, full = frame_for_model(wide, margins=TIGHT)
    width = lambda f: f.placement[2] - f.placement[0]  # noqa: E731
    assert width(full) > 1.5 * width(narrow)


def test_wan_resolution_and_price() -> None:
    assert wan_params()["resolution"] == "480p"
    assert wan_price(wan_params()).estimate_usd == 0.05
    assert wan_price(wan_params(resolution="720p")).flat_usd == 0.11
    assert canvas_for("720p") == (1280, 720)
    with pytest.raises(ValueError):
        wan_params(resolution="1080p")
    with pytest.raises(ValueError):
        canvas_for("360p")


# Telegram sticker sharpness


def _detail_frames(count: int = 30, side: int = 400) -> list[Image.Image]:
    rng = np.random.default_rng(1)
    frames = []
    for i in range(count):
        data = np.zeros((side, side, 4), dtype=np.uint8)
        noise = rng.integers(0, 255, (side // 2, side // 2, 3), dtype=np.uint8)
        data[side // 4 : 3 * side // 4, side // 4 : 3 * side // 4, :3] = np.roll(noise, i, axis=1)
        data[side // 4 : 3 * side // 4, side // 4 : 3 * side // 4, 3] = 255
        frames.append(Image.fromarray(data, "RGBA"))
    return frames


def test_sharpen_raises_detail_keeps_alpha() -> None:
    frame = _detail_frames(1)[0].resize((200, 200), Image.Resampling.BILINEAR)
    sharp = sharpen(frame, Unsharp())
    assert np.array_equal(np.asarray(sharp)[..., 3], np.asarray(frame)[..., 3])
    assert laplacian_variance(sharp) > laplacian_variance(frame)
    assert np.asarray(sharp)[0, 0].tolist() == [0, 0, 0, 0]


def test_laplacian_variance_drops_with_blur() -> None:
    frame = _detail_frames(1)[0]
    blurred = frame.resize((100, 100)).resize(frame.size, Image.Resampling.BILINEAR)
    assert laplacian_variance(frame) > 2 * laplacian_variance(blurred)
    assert laplacian_variance(Image.new("RGBA", (10, 10))) == 0.0


def test_two_pass_vp9_keeps_alpha(tmp_path: Path) -> None:
    path = tmp_path / "a.webm"
    encode_webm(path, _detail_frames(10, 128), 1.0, Vp9Rate(20, 300))
    result = validate(path, ExportTarget.OBS_OVERLAY)
    assert result.valid, result.violations
    assert str(Vp9Rate(20, 300)) == "2-pass crf 20 cap 300 kbit/s"


def test_telegram_order_tries_25_fps_before_quality() -> None:
    master = MasterAnimation(frames=_detail_frames(81, 64), fps=30.0)
    timings = [Timing(tuple(range(81)), 2.7)]
    order = telegram_sticker_order(master, timings, 256 * 1024)
    (t0, q0), (t1, q1), (t2, q2) = order[:3]
    assert round(t0.fps) == 30 and round(t1.fps) == 25 and t2 is t1
    assert isinstance(q0, Vp9Rate) and q0 == q1 and q2.crf > q1.crf and q2.kbps < q1.kbps
    assert order[-1][1] == 63  # one-pass CRF ladder stays as the last resort


def test_fit_candidates_stops_at_first_fit(tmp_path: Path) -> None:
    sizes = {"a": 100, "b": 40, "c": 10}
    path = tmp_path / "f.bin"

    def encode(file: Path, _t: Timing, q: Any) -> None:
        file.write_bytes(b"x" * sizes[q])

    timing = Timing((0,), 1.0)
    candidates = [(timing, q) for q in "abc"]
    _, quality, attempts = fit_candidates(path, candidates, encode, 50, lambda _t, q: q)
    assert quality == "b" and len(attempts) == 2


def test_telegram_sticker_export_sharpened_and_valid(tmp_path: Path) -> None:
    master = MasterAnimation(frames=_detail_frames(30, 300), fps=30.0)
    plain = export(master, "telegram_sticker", tmp_path / "plain", unsharp=None)
    sharp = export(master, "telegram_sticker", tmp_path / "sharp")
    assert plain.valid and sharp.valid
    assert "2-pass" in sharp.attempts[0]
    lap = lambda p: laplacian_variance(compare_quality.read_webm(p)[0])  # noqa: E731
    assert lap(sharp.path) > lap(plain.path)


def test_compare_quality_helpers(tmp_path: Path) -> None:
    path = tmp_path / "a.webm"
    encode_webm(path, _detail_frames(6, 64), 0.5, 30)
    frames = compare_quality.read_webm(path)
    assert len(frames) == 6 and frames[0].mode == "RGBA" and frames[0].getpixel((0, 0))[3] < 128
    assert compare_quality.pick(10, 4) == [0, 2, 5, 7]
    sheet = compare_quality.grid([("a", frames)], [[0, 3]], cell=32)
    assert sheet.size == (64, 32 + compare_quality.LABEL_H)
