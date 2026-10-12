"""Test helpers for presets: fixtures prepared with preprocess.normalize (ADR-008)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from stickermotion.preprocess import normalize

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_NAMES = ["char_square.png", "char_tall.png", "char_wide_soft.png"]
CANVAS = 1024
MARGIN = 0.18


def on_canvas(image: Image.Image, canvas: int = CANVAS, margin: float = MARGIN) -> Image.Image:
    """Character on a square transparent canvas with ``margin`` headroom per side."""
    return normalize(image, canvas_size=canvas, headroom=margin).image


def load_fixture(name: str, canvas: int = CANVAS) -> Image.Image:
    with Image.open(FIXTURES / name) as image:
        return on_canvas(image, canvas)
