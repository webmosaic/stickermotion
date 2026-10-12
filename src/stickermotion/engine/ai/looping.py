"""Seamless loop from a generated clip.

The model gets the same image as first and last frame, so the clip returns close to its
start. ``trim_to_match`` cuts at the frame most similar to frame 0 (main strategy);
``seam_score`` measures the wrap jump against a regular step. When the trimmed seam is
not clean, ``ping_pong`` over the start of the clip is the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

MAX_SEAM_RATIO = 2.0
MIN_TRIM_S = 1.5
PINGPONG_S = 1.5


def _premul(image: Image.Image) -> np.ndarray:
    data = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    data[..., :3] *= data[..., 3:4]
    return data


def frame_distance(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute difference of premultiplied RGBA in [0, 1]."""
    return float(np.mean(np.abs(_premul(a) - _premul(b))))


@dataclass(frozen=True)
class SeamReport:
    wrap: float
    """Distance between the last and the first frame."""
    median_step: float
    """Median distance between neighbouring frames."""

    @property
    def ratio(self) -> float:
        if self.wrap == 0:
            return 0.0
        return self.wrap / self.median_step if self.median_step > 0 else float("inf")

    def ok(self, max_ratio: float = MAX_SEAM_RATIO) -> bool:
        return self.ratio <= max_ratio


def seam_score(frames: list[Image.Image]) -> SeamReport:
    if len(frames) < 2:
        return SeamReport(0.0, 0.0)
    steps = [frame_distance(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    return SeamReport(frame_distance(frames[-1], frames[0]), float(np.median(steps)))


def ping_pong(frames: list[Image.Image]) -> list[Image.Image]:
    """Forward then backward without repeating the ends: 0..n-1, n-2..1."""
    if len(frames) < 3:
        return list(frames)
    return list(frames) + list(frames[-2:0:-1])


def trim_to_match(frames: list[Image.Image], search_from: float = 0.5) -> list[Image.Image]:
    """Cut at the frame (in the last part of the clip) most similar to frame 0, dropping it."""
    count = len(frames)
    if count < 4:
        return list(frames)
    start = max(2, int(count * search_from))
    distances = [frame_distance(frames[0], frames[i]) for i in range(start, count)]
    return list(frames[: start + int(np.argmin(distances))])


@dataclass(frozen=True)
class LoopChoice:
    method: str
    """"trim" or "pingpong"."""
    frames: list[Image.Image]
    seam: SeamReport
    reason: str


def make_loop(frames: list[Image.Image], fps: float) -> LoopChoice:
    """Trim when the clip returns to frame 0 cleanly and the loop lasts long enough,
    else ping-pong over the first ``PINGPONG_S`` seconds."""
    trimmed = trim_to_match(frames)
    seam = seam_score(trimmed)
    trim_s = len(trimmed) / fps
    if seam.ok() and trim_s >= MIN_TRIM_S:
        reason = f"trim: back to frame 0 after {trim_s:.1f} s, seam x{seam.ratio:.2f}"
        return LoopChoice("trim", trimmed, seam, reason)
    head = frames[: max(3, round(PINGPONG_S * fps))]
    looped = ping_pong(head)
    reason = f"pingpong: trim seam x{seam.ratio:.2f} at {trim_s:.1f} s is not clean"
    return LoopChoice("pingpong", looped, seam_score(looped), reason)
