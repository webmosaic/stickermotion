"""Full-alpha fallback preview for browsers that cannot show WebM with alpha (Safari).

The large preview on the result page plays ``telegram_sticker.webm`` (VP9, full alpha).
Safari decodes VP9 but drops the alpha plane, so the page swaps in an animated WebP made
from the same WebM: libvpx decodes the alpha, Pillow writes a lossy WebP with lossless
alpha. It is built once on first request and kept next to the exports.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE_FILE = "telegram_sticker.webm"
PREVIEW_FILE = "preview.webp"
PREVIEW_QUALITY = 85

_lock = threading.Lock()


def _probe(path: Path) -> tuple[int, int, float]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=False,
    )
    stream = (json.loads(proc.stdout or b"{}").get("streams") or [{}])[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1"
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    return int(stream["width"]), int(stream["height"]), fps or 30.0


def decode_rgba(path: Path) -> tuple[list[Image.Image], float]:
    """Frames of a VP9 WebM with alpha (libvpx decoder keeps the alpha plane)."""
    width, height, fps = _probe(path)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-c:v",
            "libvpx-vp9",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace").strip())
    size = width * height * 4
    count = len(proc.stdout) // size
    data = np.frombuffer(proc.stdout[: count * size], dtype=np.uint8)
    frames = [Image.fromarray(f, "RGBA") for f in data.reshape(count, height, width, 4)]
    return frames, fps


def ensure_preview(root: Path) -> Path | None:
    """``preview.webp`` next to the exports, built from the Telegram sticker if needed."""
    source = root / SOURCE_FILE
    target = root / PREVIEW_FILE
    if not source.is_file() or shutil.which("ffmpeg") is None:
        return None
    with _lock:
        if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
            return target
        frames, fps = decode_rgba(source)
        if not frames:
            return None
        delay = max(10, round(1000 / fps))
        partial = target.with_suffix(".tmp")
        frames[0].save(
            partial,
            format="WEBP",
            save_all=True,
            append_images=frames[1:],
            duration=delay,
            loop=0,
            lossless=False,
            quality=PREVIEW_QUALITY,
            alpha_quality=100,
            method=4,
        )
        partial.replace(target)
    return target
