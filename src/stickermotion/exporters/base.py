"""Shared exporter machinery: timing plan, downscaling, frame delays and the weight-fit loop.

An exporter turns a MasterAnimation into one platform file. Timing is planned first
(duration and frame count brought within the target limits, loop kept seamless), then the
encoder is called with decreasing quality and, if that is not enough, decreasing fps until
the file fits the weight limit. Every written file is checked with ``validator.validate``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from stickermotion.core.alpha import resize_premultiplied
from stickermotion.core.models import ExportResult, ExportTarget, MasterAnimation
from stickermotion.core.specs import KB, TargetSpec
from stickermotion.validator import validate

# Lowest fps the fit loop may thin down to (below this motion stops reading as animation).
MIN_FPS = 8.0
# Share of the planned frame count tried at each fps step of the fit loop.
FPS_STEPS = (1.0, 0.75, 0.5, 1 / 3)
# The fit loop aims at this share of a weight limit, so files do not sit right at the edge
# (Discord emoji and Telegram sticker came out at 242-254 KB of 256 KB).
WEIGHT_TARGET = 0.95


class ExportFitError(RuntimeError):
    """The file could not be brought within the weight limit.

    ``attempts`` lists every tried setting with the resulting weight; ``result`` is the
    validation of the last (smallest-quality) attempt, kept on disk for inspection.
    """

    def __init__(self, message: str, attempts: list[str], result: ExportResult | None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.result = result


@dataclass(frozen=True)
class Timing:
    """Which master frames to output and over what loop duration."""

    indices: tuple[int, ...]
    duration_s: float

    @property
    def frame_count(self) -> int:
        return len(self.indices)

    @property
    def fps(self) -> float:
        return self.frame_count / self.duration_s


def sample_indices(source_count: int, count: int) -> tuple[int, ...]:
    """Evenly spaced master frame indices for a loop of ``count`` frames.

    Indices are i * n / count rounded down, so frame 0 comes first and the last index is
    below ``source_count``: the first frame is never repeated at the end and the wrap step
    (last -> first) equals a regular step. When ``count`` divides ``source_count`` the
    spacing is exact.
    """
    if not 1 <= count <= source_count:
        raise ValueError(f"count must be in [1, {source_count}], got {count}")
    return tuple((i * source_count) // count for i in range(count))


def plan_timings(
    master: MasterAnimation,
    spec: TargetSpec,
    max_fps: float | None = None,
    min_fps: float = MIN_FPS,
) -> list[Timing]:
    """Timing candidates for the fit loop, best quality first.

    - A loop longer than the target allows is compressed in time (the whole cycle is kept,
      not cut, so it stays seamless) and thinned to stay at or below the master fps.
    - The frame count is thinned to satisfy the target fps / frame limits (and an extra
      format cap such as the GIF delay floor).
    - Further candidates thin frames more, down to ``min_fps``.
    """
    source = master.frame_count
    duration = master.duration_s
    if spec.max_duration_s is not None and duration > spec.max_duration_s:
        duration = spec.max_duration_s

    # A shortened loop keeps the master fps (frames are thinned, playback is not sped up).
    caps = [c for c in (master.fps, spec.max_fps, max_fps) if c is not None]
    count = min(source, math.floor(duration * min(caps) + 1e-9))
    if spec.max_frames is not None:
        count = min(count, spec.max_frames)
    count = max(1, count)

    floor = max(2, math.ceil(duration * min_fps - 1e-9))
    counts: list[int] = []
    for share in FPS_STEPS:
        candidate = max(1, round(count * share))
        if candidate < floor and counts:
            break
        if candidate not in counts:
            counts.append(candidate)
    return [Timing(sample_indices(source, c), duration) for c in counts]


def distribute_delays(count: int, duration_s: float, unit_ms: int, min_units: int = 1) -> list[int]:
    """Per-frame delays in ms, each a multiple of ``unit_ms``, summing to the loop duration.

    Whole units are spread with an error accumulator (e.g. 3,3,4 cs for 30 fps), so total
    duration stays exact up to one unit.
    """
    total_units = max(count * min_units, round(duration_s * 1000 / unit_ms))
    delays: list[int] = []
    previous = 0
    for i in range(1, count + 1):
        boundary = round(total_units * i / count)
        delays.append(max(min_units, boundary - previous) * unit_ms)
        previous = boundary
    return delays


class FrameCache:
    """Downscaled master frames, computed once per (frame index, size)."""

    def __init__(self, master: MasterAnimation) -> None:
        self.master = master
        self._cache: dict[tuple[int, tuple[int, int]], Image.Image] = {}

    def get(self, index: int, size: tuple[int, int]) -> Image.Image:
        key = (index, size)
        frame = self._cache.get(key)
        if frame is None:
            source = self.master.frames[index]
            # Premultiplied resampling: no dark or light rim on semi-transparent edges.
            frame = source.copy() if source.size == size else resize_premultiplied(source, size)
            self._cache[key] = frame
        return frame

    def frames(self, timing: Timing, size: tuple[int, int]) -> list[Image.Image]:
        return [self.get(i, size) for i in timing.indices]


def bleed_color(frame: Image.Image, passes: int = 4) -> Image.Image:
    """Fill RGB of fully transparent pixels with the colour of nearby visible pixels.

    Alpha is unchanged. Lossy codecs with 4:2:0 chroma (VP9, lossy WEBP) average colour
    across blocks; black under transparent pixels would then leak a dark rim into
    semi-transparent edges. Each pass spreads the alpha-weighted mean colour of the
    3x3 neighbourhood one pixel outwards.
    """
    pixels = np.asarray(frame.convert("RGBA")).copy()
    alpha = pixels[..., 3].astype(np.float32) / 255.0
    weight = alpha.copy()
    color = pixels[..., :3].astype(np.float32) * weight[..., None]
    for _ in range(passes):
        w_pad = np.pad(weight, 1)
        c_pad = np.pad(color, ((1, 1), (1, 1), (0, 0)))
        h, w = weight.shape
        w_sum = sum(w_pad[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3))
        c_sum = sum(c_pad[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3))
        grow = (weight == 0) & (w_sum > 0)
        color[grow] = c_sum[grow] / w_sum[grow][:, None]
        weight[grow] = 1.0
    fill = (pixels[..., 3] == 0) & (weight > 0)
    pixels[fill, :3] = np.clip(np.rint(color[fill]), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels, "RGBA")


@dataclass(frozen=True)
class Unsharp:
    """Unsharp mask applied after downscaling: radius px, strength %, threshold."""

    radius: float = 1.0
    percent: int = 60
    threshold: int = 2


def sharpen(frame: Image.Image, unsharp: Unsharp) -> Image.Image:
    """Unsharp mask on premultiplied colour, so edges do not halo into transparent pixels.

    Alpha is unchanged; colour is clipped to alpha before going back to straight RGBA.
    """
    data = np.asarray(frame.convert("RGBA"))
    alpha = data[..., 3:4].astype(np.float32) / 255.0
    premultiplied = np.rint(data[..., :3] * alpha).astype(np.uint8)
    mask = ImageFilter.UnsharpMask(unsharp.radius, unsharp.percent, unsharp.threshold)
    sharp = np.asarray(Image.fromarray(premultiplied, "RGB").filter(mask), dtype=np.float32)
    sharp = np.minimum(sharp, alpha * 255.0)
    safe = np.where(alpha > 0, alpha, 1.0)
    out = data.copy()
    out[..., :3] = np.clip(np.rint(sharp / safe), 0, 255).astype(np.uint8)
    out[data[..., 3] == 0, :3] = 0
    return Image.fromarray(out, "RGBA")


def fit_to_limit(
    path: Path,
    timings: Sequence[Timing],
    qualities: Sequence[Any],
    encode: Callable[[Path, Timing, Any], None],
    max_bytes: int | None,
    describe: Callable[[Timing, Any], str],
    target_share: float = WEIGHT_TARGET,
) -> tuple[Timing, Any, list[str]]:
    """Encode with decreasing quality, then decreasing fps, until the file fits.

    The loop aims at ``target_share`` of ``max_bytes`` (headroom below the platform limit).
    When no setting reaches that, the last (smallest) attempt is accepted if it is still
    within ``max_bytes`` itself. Returns the chosen timing, quality and the list of
    attempts. Raises ExportFitError (with the last attempt left at ``path``) when nothing
    fits the hard limit.
    """
    candidates = [(timing, quality) for timing in timings for quality in qualities]
    return fit_candidates(path, candidates, encode, max_bytes, describe, target_share)


def fit_candidates(
    path: Path,
    candidates: Sequence[tuple[Timing, Any]],
    encode: Callable[[Path, Timing, Any], None],
    max_bytes: int | None,
    describe: Callable[[Timing, Any], str],
    target_share: float = WEIGHT_TARGET,
) -> tuple[Timing, Any, list[str]]:
    """``fit_to_limit`` over an explicit (timing, quality) order, best first."""
    attempts: list[str] = []
    goal = None if max_bytes is None else int(max_bytes * target_share)
    last: tuple[Timing, Any, int] | None = None
    for timing, quality in candidates:
        encode(path, timing, quality)
        size = path.stat().st_size
        attempts.append(f"{describe(timing, quality)}: {size / KB:.1f} KB")
        if goal is None or size <= goal:
            return timing, quality, attempts
        last = (timing, quality, size)
    if last is not None and max_bytes is not None and last[2] <= max_bytes:
        return last[0], last[1], attempts
    limit = f"{max_bytes / KB:.1f} KB" if max_bytes is not None else "no limit"
    raise ExportFitError(
        f"{path.name}: could not fit within {limit}; tried " + "; ".join(attempts),
        attempts,
        None,
    )


def validated(
    path: Path, target: ExportTarget, size_variant: str | None, attempts: list[str]
) -> ExportResult:
    """Validate a written file and attach the variant and fit attempts."""
    result = validate(path, target, size_variant)
    return result.model_copy(update={"size_variant": size_variant, "attempts": attempts})
