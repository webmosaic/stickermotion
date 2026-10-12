"""Render every preset in both classes for a PNG and save contact sheets for review.

Usage:
    python scripts/preview_presets.py path/to/character.png [--out out/preview]

For each preset x class two PNG files are written to the output directory:
    <stem>_<preset>_<class>.png      frames in a grid on a checkerboard (large)
    <stem>_<preset>_<class>_28px.png the same frames downscaled to 28 px (readability check)

The input is prepared with preprocess.normalize (1024 px canvas, 18% headroom).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from PIL import Image, ImageDraw

from stickermotion.core.models import AnimationClass, MasterAnimation
from stickermotion.engine.presets import get_preset, list_presets
from stickermotion.preprocess import normalize

CANVAS = 1024
MARGIN = 0.18
CELL = 192
SMALL = 28
SMALL_PAD = 6
CHECKER = 16
CHECKER_SMALL = 4
LIGHT = (236, 236, 236, 255)
DARK = (196, 196, 196, 255)
LABEL = (40, 40, 40, 255)
CLASSES = [AnimationClass.EMOTE, AnimationClass.STICKER]


def prepare_canvas(image: Image.Image, canvas: int = CANVAS, margin: float = MARGIN) -> Image.Image:
    """Character on a square transparent canvas with ``margin`` headroom (preprocess.normalize)."""
    return normalize(image, canvas_size=canvas, headroom=margin).image


def checkerboard(size: tuple[int, int], square: int) -> Image.Image:
    board = Image.new("RGBA", size, LIGHT)
    draw = ImageDraw.Draw(board)
    for y in range(0, size[1], square):
        for x in range(0, size[0], square):
            if (x // square + y // square) % 2:
                draw.rectangle((x, y, x + square - 1, y + square - 1), fill=DARK)
    return board


def contact_sheet(
    animation: MasterAnimation, cell: int, pad: int, square: int, title: str | None
) -> Image.Image:
    """Grid of frames downscaled to ``cell`` px, each on its own checkerboard."""
    count = animation.frame_count
    columns = min(count, max(1, math.ceil(math.sqrt(count * 1.5))))
    rows = math.ceil(count / columns)
    header = 20 if title else 0
    sheet = Image.new(
        "RGBA",
        (columns * (cell + pad) + pad, rows * (cell + pad) + pad + header),
        (255, 255, 255, 255),
    )
    if title:
        ImageDraw.Draw(sheet).text((pad, 4), title, fill=LABEL)
    tile_bg = checkerboard((cell, cell), square)
    for index, frame in enumerate(animation.frames):
        small = frame.resize((cell, cell), Image.Resampling.LANCZOS, reducing_gap=3.0)
        tile = tile_bg.copy()
        tile.alpha_composite(small)
        column, row = index % columns, index // columns
        sheet.alpha_composite(
            tile, (pad + column * (cell + pad), header + pad + row * (cell + pad))
        )
    return sheet


def render_previews(source: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        canvas = prepare_canvas(image)
    written: list[Path] = []
    for info in list_presets():
        for anim_class in CLASSES:
            preset = get_preset(info.name)(anim_class)
            started = time.perf_counter()
            animation = preset.render(canvas)
            elapsed = time.perf_counter() - started
            params = preset.params
            title = (
                f"{info.name} / {anim_class.value}: {animation.frame_count} frames, "
                f"{params.fps:g} fps, {params.duration_s:g} s, render {elapsed:.1f} s"
            )
            stem = f"{source.stem}_{info.name}_{anim_class.value}"
            large = out_dir / f"{stem}.png"
            small = out_dir / f"{stem}_{SMALL}px.png"
            contact_sheet(animation, CELL, 8, CHECKER, title).save(large)
            contact_sheet(animation, SMALL, SMALL_PAD, CHECKER_SMALL, None).save(small)
            written += [large, small]
            print(f"{title} -> {large.name}, {small.name}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="transparent PNG with the character")
    parser.add_argument("--out", type=Path, default=Path("out/preview"))
    args = parser.parse_args()
    render_previews(args.image, args.out)


if __name__ == "__main__":
    main()
