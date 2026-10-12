"""Frame grids and sharpness numbers for Telegram stickers.

    python scripts/compare_quality.py OUT.jpg LABEL=FILE.webm [LABEL=FILE.webm ...]

Every WEBM is decoded with libvpx (keeps the alpha), ``--frames`` evenly spaced frames are
put on a dark chat-like background in one row per file, and the Laplacian variance of
each shown frame (and the mean over all frames) is printed, measured at the file's own
size (512 px for a Telegram sticker).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from stickermotion.core.sharpness import laplacian_variance
from stickermotion.engine.ai.video import probe

BACKGROUND = (24, 24, 27)
LABEL_H = 22


def read_webm(path: Path) -> list[Image.Image]:
    """Every frame of a VP9 / VP8 WEBM as RGBA (libvpx decoder keeps the alpha)."""
    width, height, _ = probe(path)
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-c:v", "libvpx-vp9", "-i", str(path)]
        + ["-f", "rawvideo", "-pix_fmt", "rgba", "-"],
        check=True,
        capture_output=True,
    )
    size = width * height * 4
    count = len(out.stdout) // size
    data = np.frombuffer(out.stdout[: count * size], dtype=np.uint8)
    return [Image.fromarray(f, "RGBA") for f in data.reshape(count, height, width, 4)]


def pick(count: int, shown: int) -> list[int]:
    """``shown`` evenly spaced indices over ``count`` frames, starting at 0."""
    shown = max(1, min(shown, count))
    return [i * count // shown for i in range(shown)]


def grid(
    rows: list[tuple[str, list[Image.Image]]], indices: list[list[int]], cell: int = 256
) -> Image.Image:
    """One row per (label, frames): the picked frames on the dark background, labelled."""
    columns = max(len(i) for i in indices)
    sheet = Image.new("RGB", (columns * cell, len(rows) * (cell + LABEL_H)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, ((label, frames), picked) in enumerate(zip(rows, indices, strict=True)):
        y = r * (cell + LABEL_H)
        draw.text((4, y + 4), label, fill=(0, 0, 0))
        for c, index in enumerate(picked):
            tile = Image.new("RGBA", frames[index].size, (*BACKGROUND, 255))
            tile.alpha_composite(frames[index])
            tile = tile.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
            sheet.paste(tile, (c * cell, y + LABEL_H))
            draw.text((c * cell + 4, y + LABEL_H + 4), f"#{index}", fill=(255, 255, 255))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("files", nargs="+", help="LABEL=path.webm")
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args()
    rows, indices, report = [], [], {}
    for item in args.files:
        label, _, path = item.partition("=")
        frames = read_webm(Path(path))
        picked = pick(len(frames), args.frames)
        values = [laplacian_variance(f) for f in frames]
        report[label] = {
            "file": path,
            "size": list(frames[0].size),
            "frames": len(frames),
            "lapvar_mean": round(float(np.mean(values)), 1),
            "lapvar_shown": {str(i): round(values[i], 1) for i in picked},
        }
        rows.append((label, frames))
        indices.append(picked)
    grid(rows, indices).save(args.out, quality=88)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
