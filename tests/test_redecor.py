"""Old decor out of an existing clip, procedural decor in (reprocess)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from stickermotion.cli import app
from stickermotion.core.models import ExportTarget
from stickermotion.engine.ai import redecor
from stickermotion.engine.ai.decor import DecorMode, find_candidates, label_components, plan
from stickermotion.engine.ai.framing import GREEN, frame_for_model
from stickermotion.engine.ai.matting import chroma_key
from stickermotion.engine.ai.video import write_mp4
from stickermotion.reprocess import reprocess
from test_decor import _scene

runner = CliRunner()


def _yellow(data: np.ndarray) -> np.ndarray:
    return (data[..., 0] > 220) & (data[..., 1] > 180) & (data[..., 2] < 120) & (data[..., 3] > 128)


def _moving(frame: Image.Image, count: int = 16, travel: int = 14) -> list[Image.Image]:
    """The whole scene (decor included) drifts up and left and comes back, like a head."""
    frames = []
    for i in range(count):
        t = np.sin(np.pi * i / (count - 1))
        dx, dy = -round(travel * t), -round(travel * t)
        canvas = Image.new("RGB", frame.size, GREEN)
        canvas.paste(frame, (dx, dy))
        frames.append(canvas)
    return frames


@pytest.fixture
def together(tmp_path: Path) -> tuple[Path, Path, Path, list[Image.Image]]:
    """Input PNG, model input with the decor in it and a clip where the decor follows."""
    source = tmp_path / "scene.png"
    _scene().save(source)
    frame, _ = frame_for_model(_scene())
    model_input = tmp_path / "model_input.png"
    frame.save(model_input)
    frames = _moving(frame)
    raw = tmp_path / "raw.mp4"
    write_mp4(raw, frames, 16.0)
    return source, model_input, raw, frames


def test_infer_mapping_composition_and_character() -> None:
    scene = _scene()
    split = plan(scene, DecorMode.AUTO)
    with_decor, _ = frame_for_model(scene)
    mapping = redecor.infer_mapping(scene, split.character, with_decor)
    assert mapping.reference == "composition"
    box = mapping.box(redecor.visible_box(scene))
    assert box == pytest.approx(redecor.model_input_box(with_decor), abs=1.5)
    alone, framing = frame_for_model(split.character)
    mapping = redecor.infer_mapping(scene, split.character, alone)
    assert mapping.reference == "character"
    known = redecor.Mapping.from_framing(framing, split.character, alone.size)
    assert known.scale_x == pytest.approx(mapping.scale_x, rel=0.02)


def test_decor_in_clip_tells_present_from_removed() -> None:
    scene = _scene()
    split = plan(scene, DecorMode.AUTO)
    candidates, _ = find_candidates(scene)
    with_decor, _ = frame_for_model(scene)
    mapping = redecor.infer_mapping(scene, split.character, with_decor)
    masks = redecor.candidate_masks(candidates, mapping, with_decor.size)
    assert all(redecor.decor_in_clip(with_decor, masks))
    alone, framing = frame_for_model(split.character)
    mapping = redecor.Mapping.from_framing(framing, split.character, alone.size)
    masks = redecor.candidate_masks(candidates, mapping, alone.size)
    assert not any(redecor.decor_in_clip(alone, masks))


def test_shifted_moves_and_clips() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = mask[4, 4] = True
    moved = redecor.shifted(mask, (1, 2))
    assert moved[2, 1] and moved.sum() == 1


def test_track_and_remove_follow_the_decor(together: tuple) -> None:
    source, model_input, _raw, frames = together
    scene = Image.open(source).convert("RGBA")
    split = plan(scene, DecorMode.AUTO)
    model = Image.open(model_input).convert("RGB")
    mapping = redecor.infer_mapping(scene, split.character, model)
    candidates, _ = find_candidates(scene)
    masks = redecor.candidate_masks(candidates, mapping, model.size)
    shifts = redecor.track(frames, model, masks)
    peak = len(frames) // 2
    assert all(abs(s[0] + 14) <= 1 and abs(s[1] + 14) <= 1 for s in shifts[peak])
    cut = [chroma_key(f) for f in frames]
    removal = redecor.remove(cut, masks, shifts)
    for before, after in zip(cut, removal.frames, strict=True):
        a, b = np.asarray(after), np.asarray(before)
        assert not _yellow(a).any()  # no old decor anywhere
        # The character (red body) is untouched.
        red = (b[..., 0] > 150) & (b[..., 1] < 90) & (b[..., 3] > 128)
        assert (a[..., 3][red] == b[..., 3][red]).all()
    assert max(removal.residual_px) == 0 and not removal.reverted
    assert removal.to_dict()["max_shift_px"][0] == pytest.approx([14, 14], abs=1)


def test_remove_reverts_frame_when_character_would_be_cut() -> None:
    frame = Image.new("RGBA", (60, 40), (0, 0, 0, 0))
    data = np.asarray(frame).copy()
    data[5:35, 2:28] = (200, 40, 60, 255)  # character
    data[5:35, 32:52] = (200, 40, 60, 255)  # a detached part, 60% inside the zone
    zone = np.zeros((40, 60), dtype=bool)
    zone[5:35, 32:44] = True
    removal = redecor.remove([Image.fromarray(data, "RGBA")], [zone], [[(0, 0)]], zone_px=0)
    assert removal.reverted == [0] and removal.warnings
    assert np.asarray(removal.frames[0])[20, 48, 3] == 255


def test_reprocess_removes_old_decor_and_lays_still_decor(together: tuple, tmp_path: Path) -> None:
    source, model_input, raw, _frames = together
    result = reprocess(
        raw,
        source,
        model_input,
        tmp_path / "out",
        targets=[ExportTarget.TELEGRAM_STICKER],
        meta={"prompt": "nods", "params": {"seed": 7}},
    )
    info = result.manifest["ai"]["reprocess"]
    assert info["old_decor_items"] == 3 and info["mapping"]["reference"] == "composition"
    assert result.manifest["ai"]["seed"] == 7 and result.manifest["ai"]["costs_usd"]["total"] == 0
    assert result.removal is not None and max(result.removal.residual_px) == 0
    # The composed decor stands still while the character moves.
    loop = result.results[0]
    assert loop.valid
    centres = []
    from compare_quality import read_webm

    for frame in read_webm(loop.path):
        labels, _count = label_components(_yellow(np.asarray(frame)))
        moon = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1  # largest: the moon
        ys, xs = np.nonzero(moon)
        centres.append((xs.mean(), ys.mean()))
    spread = np.ptp(np.array(centres), axis=0)
    square = result.manifest["ai"]["decor"]["composition"]["video_square"]
    old_travel = 14 * 512 / (square[2] - square[0])  # the old decor at the sticker scale
    assert spread.max() < 0.3 * old_travel  # what is left is the moon's own sway
    assert (result.root / "raw.mp4").is_file() and (result.root / "model_input.png").is_file()
    manifest = json.loads((result.root / "manifest.json").read_text())
    assert manifest["ai"]["prompt"] == "nods"


def test_reprocess_clip_without_decor_removes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "scene.png"
    _scene().save(source)
    split = plan(_scene(), DecorMode.AUTO)
    frame, framing = frame_for_model(split.character)
    frame.save(tmp_path / "model_input.png")
    write_mp4(tmp_path / "raw.mp4", [frame] * 10, 16.0)
    result = reprocess(
        tmp_path / "raw.mp4",
        source,
        tmp_path / "model_input.png",
        tmp_path / "out",
        targets=[ExportTarget.TELEGRAM_STICKER],
        framing=framing,
    )
    assert result.removal is None
    assert result.manifest["ai"]["reprocess"]["framing_known"] is True
    assert len(result.manifest["ai"]["decor"]["items"]) == 3


def test_cli_reprocess(together: tuple, tmp_path: Path) -> None:
    source, model_input, raw, _frames = together
    out = tmp_path / "cli"
    result = runner.invoke(
        app,
        [
            "reprocess",
            str(raw),
            "--input",
            str(source),
            "--model-input",
            str(model_input),
            "--out",
            str(out),
            "--target",
            "telegram_sticker",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "old decor: 3 items tracked" in result.output
    assert (out / "scene" / "ai" / "telegram_sticker.webm").is_file()
    missing = runner.invoke(
        app, ["reprocess", str(raw), "--input", "nope.png", "--model-input", str(model_input)]
    )
    assert missing.exit_code == 2
