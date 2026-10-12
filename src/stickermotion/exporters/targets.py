"""One exporter per ExportTarget behind a common ``export`` function (ADR-001)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stickermotion.core.models import ExportResult, ExportTarget, MasterAnimation
from stickermotion.core.specs import TargetSpec, get_spec
from stickermotion.exporters.base import (
    WEIGHT_TARGET,
    ExportFitError,
    FrameCache,
    Timing,
    Unsharp,
    fit_candidates,
    plan_timings,
    sample_indices,
    sharpen,
    validated,
)
from stickermotion.exporters.formats import (
    APNG_COLOR_STEPS,
    GIF_COLOR_STEPS,
    GIF_MAX_FPS,
    WEBM_CRF_STEPS,
    WEBP_QUALITY_STEPS,
    Color,
    Vp9Rate,
    encode_apng,
    encode_gif,
    encode_webm,
    encode_webp,
)

TWITCH_MANUAL_SIZES = (28, 56, 112)
TWITCH_AUTO_SIZE = 224
TWITCH_DEFAULT_SIZE = 112
DEFAULT_MATTE: Color = (24, 24, 27)
"""Dark grey close to the Twitch dark chat background (#18181B)."""

DISCORD_STICKER_SIZE = 320
DISCORD_EMOJI_SIZE = 128
TELEGRAM_STICKER_SIDE = 512
TELEGRAM_EMOJI_SIZE = 100
# Telegram emoji fit loop thins frames no lower than this (64 KB target).
TELEGRAM_EMOJI_MIN_FPS = 15.0
OBS_CRF = 24
# Telegram sticker: unsharp after the downscale to 512, two-pass VP9 with a bitrate
# cap from the weight limit, fps lowered to 25 before quality.
TELEGRAM_STICKER_UNSHARP = Unsharp(radius=1.0, percent=60, threshold=2)
TELEGRAM_STICKER_FPS_STEP = 25.0
TELEGRAM_STICKER_CRF_STEPS = (20, 26, 32, 38)
# Share of the weight budget given to the colour bitrate cap: the alpha stream shares it.
TELEGRAM_STICKER_RATE_SHARE = 0.85


@dataclass(frozen=True)
class _Job:
    """Resolved settings of a single file export."""

    spec: TargetSpec
    size: tuple[int, int]
    variant: str | None
    filename: str
    max_bytes: int | None
    timings: list[Timing]
    qualities: tuple[Any, ...]
    encode: Callable[[Path, list[Any], float, Any], None]
    quality_label: str
    order: tuple[tuple[Timing, Any], ...] | None = None
    """Explicit (timing, quality) order; default: every quality at each timing in turn."""
    unsharp: Unsharp | None = None
    """Sharpening applied to frames after downscaling."""

    @property
    def candidates(self) -> list[tuple[Timing, Any]]:
        if self.order is not None:
            return list(self.order)
        return [(t, q) for t in self.timings for q in self.qualities]


def _variant_limit(spec: TargetSpec, name: str | None) -> int | None:
    variants = spec.size_variants
    if name is not None:
        variants = tuple(v for v in variants if v.name == name)
    return variants[0].max_bytes


def _scaled_size(master: MasterAnimation, longer: int) -> tuple[int, int]:
    width, height = master.size
    scale = longer / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def telegram_sticker_order(
    master: MasterAnimation, timings: list[Timing], max_bytes: int | None
) -> tuple[tuple[Timing, Any], ...]:
    """Best first: full fps at the capped bitrate, then 25 fps, then lower CRF floors and
    caps at 25 fps; the one-pass CRF ladder of every timing stays as the last resort."""
    first = timings[0]
    order: list[tuple[Timing, Any]] = []
    if max_bytes is not None:
        budget_kbps = max_bytes * WEIGHT_TARGET * 8 / first.duration_s / 1000
        kbps = round(budget_kbps * TELEGRAM_STICKER_RATE_SHARE)
        rates = [
            Vp9Rate(crf, round(kbps * (1 - 0.1 * step)))
            for step, crf in enumerate(TELEGRAM_STICKER_CRF_STEPS)
        ]
        order.append((first, rates[0]))
        lower = first
        if first.fps > TELEGRAM_STICKER_FPS_STEP + 0.5:
            count = round(first.duration_s * TELEGRAM_STICKER_FPS_STEP)
            lower = Timing(sample_indices(master.frame_count, count), first.duration_s)
            order.append((lower, rates[0]))
        order += [(lower, rate) for rate in rates[1:]]
    order += [(t, q) for t in timings for q in WEBM_CRF_STEPS]
    return tuple(order)


def twitch_variant(size: int) -> str:
    """Twitch size variant for a GIF side: manual set or the single auto-mode file."""
    return "manual" if size in TWITCH_MANUAL_SIZES else "auto"


def _job(master: MasterAnimation, target: ExportTarget, options: dict[str, Any]) -> _Job:
    spec = get_spec(target)
    if target is ExportTarget.TWITCH_GIF:
        side = int(options.get("size", TWITCH_DEFAULT_SIZE))
        variant = twitch_variant(side)
        matte: Color = tuple(options.get("matte", DEFAULT_MATTE))  # type: ignore[assignment]

        def gif(path: Path, frames: list[Any], duration: float, colors: Any) -> None:
            encode_gif(path, frames, duration, colors, matte)

        return _Job(
            spec,
            (side, side),
            variant,
            f"twitch_gif_{side}.gif",
            _variant_limit(spec, variant),
            plan_timings(master, spec, max_fps=GIF_MAX_FPS),
            GIF_COLOR_STEPS,
            gif,
            "colors",
        )
    if target is ExportTarget.DISCORD_STICKER:
        size = (DISCORD_STICKER_SIZE, DISCORD_STICKER_SIZE)
        return _Job(
            spec,
            size,
            None,
            "discord_sticker.png",
            _variant_limit(spec, None),
            plan_timings(master, spec),
            APNG_COLOR_STEPS,
            encode_apng,
            "colors",
        )
    if target is ExportTarget.DISCORD_EMOJI:
        size = (DISCORD_EMOJI_SIZE, DISCORD_EMOJI_SIZE)
        return _Job(
            spec,
            size,
            None,
            "discord_emoji.webp",
            _variant_limit(spec, None),
            plan_timings(master, spec),
            WEBP_QUALITY_STEPS,
            encode_webp,
            "quality",
        )
    if target is ExportTarget.TELEGRAM_STICKER:
        timings = plan_timings(master, spec)
        limit = _variant_limit(spec, None)
        return _Job(
            spec,
            _scaled_size(master, TELEGRAM_STICKER_SIDE),
            None,
            f"{target.value}.webm",
            limit,
            timings,
            WEBM_CRF_STEPS,
            encode_webm,
            "crf",
            order=telegram_sticker_order(master, timings, limit),
            unsharp=options.get("unsharp", TELEGRAM_STICKER_UNSHARP),
        )
    if target is ExportTarget.TELEGRAM_EMOJI:
        return _Job(
            spec,
            (TELEGRAM_EMOJI_SIZE, TELEGRAM_EMOJI_SIZE),
            None,
            f"{target.value}.webm",
            _variant_limit(spec, None),
            plan_timings(master, spec, min_fps=TELEGRAM_EMOJI_MIN_FPS),
            WEBM_CRF_STEPS,
            encode_webm,
            "crf",
        )
    if target is ExportTarget.OBS_OVERLAY:
        crf = int(options.get("crf", OBS_CRF))
        return _Job(
            spec,
            master.size,
            None,
            "obs_overlay.webm",
            None,
            plan_timings(master, spec)[:1],
            (crf,),
            encode_webm,
            "crf",
        )
    raise ValueError(f"no exporter for target {target!r}")  # pragma: no cover


def export(
    master: MasterAnimation,
    target: ExportTarget | str,
    out_dir: Path | str,
    **options: Any,
) -> ExportResult:
    """Export ``master`` to one file for ``target`` in ``out_dir`` and validate it.

    Options:
    - ``size`` (twitch_gif): GIF side, 28 / 56 / 112 (manual set) or 224 (auto mode file);
      default 112.
    - ``matte`` (twitch_gif): RGB colour semi-transparent edges are flattened onto before
      binarizing alpha; default dark grey close to the Twitch dark theme.
    - ``crf`` (obs_overlay): VP9 constant quality, default 24.
    - ``unsharp`` (telegram_sticker): ``Unsharp`` after the downscale, None disables it.

    Quality is lowered step by step, then fps, until the file fits the target weight limit;
    raises ExportFitError (with every tried setting) if that never happens.
    """
    target = ExportTarget(target)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    job = _job(master, target, options)
    path = out / job.filename
    cache = options.get("cache") or FrameCache(master)

    sharpened: dict[int, Any] = {}

    def frames(timing: Timing) -> list[Any]:
        if job.unsharp is None:
            return cache.frames(timing, job.size)
        for index in timing.indices:
            if index not in sharpened:
                sharpened[index] = sharpen(cache.get(index, job.size), job.unsharp)
        return [sharpened[i] for i in timing.indices]

    def encode(file: Path, timing: Timing, quality: Any) -> None:
        job.encode(file, frames(timing), timing.duration_s, quality)

    def describe(timing: Timing, quality: Any) -> str:
        if quality is None:
            label = "lossless"
        elif isinstance(quality, Vp9Rate):
            label = str(quality)
        else:
            label = f"{job.quality_label} {quality}"
        return f"{timing.frame_count} frames @ {timing.fps:.1f} fps, {label}"

    try:
        _, _, attempts = fit_candidates(path, job.candidates, encode, job.max_bytes, describe)
    except ExportFitError as exc:
        result = validated(path, target, job.variant, exc.attempts)
        raise ExportFitError(str(exc), exc.attempts, result) from None
    return validated(path, target, job.variant, attempts)


def export_target(
    master: MasterAnimation,
    target: ExportTarget | str,
    out_dir: Path | str,
    **options: Any,
) -> list[ExportResult]:
    """Every file a target needs: the Twitch set (28, 56, 112 + 224 auto) or a single file."""
    target = ExportTarget(target)
    options.setdefault("cache", FrameCache(master))
    if target is ExportTarget.TWITCH_GIF:
        sizes = (*TWITCH_MANUAL_SIZES, TWITCH_AUTO_SIZE)
        return [export(master, target, out_dir, **{**options, "size": s}) for s in sizes]
    return [export(master, target, out_dir, **options)]
