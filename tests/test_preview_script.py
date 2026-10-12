from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

from stickermotion.engine.presets import list_presets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import preview_presets
finally:
    sys.path.pop(0)


def test_prepare_canvas_size_and_margin() -> None:
    with Image.open(ROOT / "tests/fixtures/char_tall.png") as image:
        canvas = preview_presets.prepare_canvas(image, canvas=200, margin=0.2)
    assert canvas.size == (200, 200)
    left, top, right, bottom = canvas.getchannel("A").getbbox()  # type: ignore[misc]
    assert top >= 39 and bottom <= 161


def test_prepare_canvas_rejects_empty() -> None:
    with pytest.raises(ValueError):
        preview_presets.prepare_canvas(Image.new("RGBA", (8, 8)))


def test_checkerboard_alternates() -> None:
    board = preview_presets.checkerboard((8, 8), 4)
    assert board.getpixel((0, 0)) != board.getpixel((4, 0))


def test_render_previews_writes_contact_sheets(tmp_path: Path) -> None:
    written = preview_presets.render_previews(ROOT / "tests/fixtures/char_square.png", tmp_path)
    expected = len(list_presets()) * 2 * 2
    assert len(written) == expected
    for path in written:
        assert path.exists()
        with Image.open(path) as sheet:
            assert sheet.mode == "RGBA"
    small = [p for p in written if p.name.endswith("_28px.png")]
    assert len(small) == expected // 2
