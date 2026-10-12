import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent
EXPECTED = ["char_square.png", "char_tall.png", "char_wide_soft.png"]


@pytest.mark.parametrize("name", EXPECTED)
def test_fixture_is_rgba_with_transparency(name: str) -> None:
    with Image.open(FIXTURES / name) as image:
        assert image.mode == "RGBA"
        alpha = np.asarray(image.getchannel("A"))
    assert (alpha == 0).any(), "expected fully transparent pixels"
    assert (alpha == 255).any(), "expected opaque pixels"


def test_fixture_aspect_ratios() -> None:
    sizes = {name: Image.open(FIXTURES / name).size for name in EXPECTED}
    w, h = sizes["char_square.png"]
    assert w == h
    w, h = sizes["char_tall.png"]
    assert h > w
    w, h = sizes["char_wide_soft.png"]
    assert w > h


def test_soft_fixture_has_semi_transparent_edges() -> None:
    alpha = np.asarray(Image.open(FIXTURES / "char_wide_soft.png").getchannel("A"))
    semi = ((alpha > 0) & (alpha < 255)).sum()
    assert semi > 100


def test_committed_fixtures_match_generator() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from make_fixtures import build_fixtures
    finally:
        sys.path.pop(0)
    images = build_fixtures()
    assert sorted(images) == EXPECTED
    for name, generated in images.items():
        stored = np.asarray(Image.open(FIXTURES / name))
        assert np.array_equal(stored, np.asarray(generated)), name
