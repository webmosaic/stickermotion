"""Video I/O through ffmpeg: decode a clip to RGB frames, encode RGB frames to H.264 MP4."""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image


def probe(path: Path) -> tuple[int, int, float]:
    """Width, height and frame rate of the first video stream."""
    out = subprocess.run(
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
        check=True,
        capture_output=True,
    )
    stream = json.loads(out.stdout)["streams"][0]
    rate = stream.get("avg_frame_rate") or "0/0"
    if rate in ("0/0", "0/1"):
        rate = stream.get("r_frame_rate", "0/1")
    return int(stream["width"]), int(stream["height"]), float(Fraction(rate))


def read_frames(path: Path) -> tuple[list[Image.Image], float]:
    """Decode every frame of a video as RGB images; returns (frames, fps)."""
    width, height, fps = probe(path)
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path)]
        + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True,
        capture_output=True,
    )
    frame_bytes = width * height * 3
    count = len(out.stdout) // frame_bytes
    data = np.frombuffer(out.stdout[: count * frame_bytes], dtype=np.uint8)
    data = data.reshape(count, height, width, 3)
    return [Image.fromarray(f, "RGB") for f in data], fps


def write_mp4(path: Path, frames: list[Image.Image], fps: float, crf: int = 20) -> None:
    """Encode RGB frames to H.264 yuv420p (how video models deliver their output)."""
    width, height = frames[0].size
    rate = Fraction(fps).limit_denominator(1000)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{rate.numerator}/{rate.denominator}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        str(path),
    ]
    payload = b"".join(np.asarray(f.convert("RGB"), dtype=np.uint8).tobytes() for f in frames)
    subprocess.run(cmd, input=payload, check=True, capture_output=True)
