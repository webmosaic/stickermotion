"""Check an exported file against the limits of an ExportTarget (core.specs).

Raster formats (GIF, APNG, PNG, WEBP) are read with Pillow, WEBM with ffprobe/ffmpeg.
Hard limit breaches go to `violations`, recommended sizes to `warnings` (ADR-007).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from stickermotion.core.alpha import TRANSPARENT_BELOW
from stickermotion.core.models import ExportResult, ExportTarget
from stickermotion.core.specs import KB, SizeVariant, TargetSpec, get_spec

_EBML_MAGIC = b"\x1a\x45\xdf\xa3"
_WEBM_DECODERS = {"vp9": "libvpx-vp9", "vp8": "libvpx"}


class UnreadableFileError(ValueError):
    """The file cannot be parsed as any supported format."""


class FfmpegNotFoundError(RuntimeError):
    """ffprobe / ffmpeg are required to validate WEBM files."""


@dataclass
class _Probe:
    """Measured properties of a file, independent of the target."""

    fmt: str
    width: int
    height: int
    frame_count: int
    duration_s: float
    fps: float | None
    has_transparency: bool
    codec: str | None = None
    alpha_pix_fmt: bool = True
    audio_streams: int = 0
    notes: list[str] = field(default_factory=list)


def _fmt_bytes(value: int) -> str:
    return f"{value / KB:.1f} KB"


def _probe_raster(path: Path) -> _Probe:
    try:
        image = Image.open(path)
    except (OSError, Image.UnidentifiedImageError) as exc:
        raise UnreadableFileError(f"{path}: unsupported or corrupt file ({exc})") from exc
    with image:
        kind = (image.format or "").upper()
        frame_count = getattr(image, "n_frames", 1)
        if kind == "GIF":
            fmt = "gif"
        elif kind == "PNG":
            fmt = "apng" if frame_count > 1 else "png"
        elif kind == "WEBP":
            fmt = "webp"
        else:
            fmt = kind.lower() or "unknown"

        duration_ms = 0.0
        has_transparency = False
        for index in range(frame_count):
            image.seek(index)
            # WEBP fills per-frame `duration` only after the frame is decoded.
            image.load()
            if frame_count > 1:
                duration_ms += float(image.info.get("duration", 0) or 0)
            if not has_transparency:
                alpha = np.asarray(image.convert("RGBA").getchannel("A"))
                has_transparency = bool((alpha < TRANSPARENT_BELOW).any())
        width, height = image.size

    duration_s = duration_ms / 1000.0
    fps = frame_count / duration_s if duration_s > 0 else None
    return _Probe(
        fmt=fmt,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_s=duration_s,
        fps=fps,
        has_transparency=has_transparency,
    )


def _run(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    if shutil.which(cmd[0]) is None:
        raise FfmpegNotFoundError(f"{cmd[0]} not found in PATH; install ffmpeg to validate WEBM")
    return subprocess.run(cmd, capture_output=True, check=False)


def _parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "0"}:
        return None
    rate = Fraction(value)
    return float(rate) if rate > 0 else None


def _decode_first_frame_alpha(path: Path, decoder: str | None, width: int, height: int) -> bool:
    """Decode the first frame to RGBA with an alpha-aware decoder; True if any pixel is transparent.

    FFmpeg's native VP9/VP8 decoders ignore the alpha side channel, so libvpx is forced.
    """
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if decoder:
        cmd += ["-c:v", decoder]
    cmd += ["-i", str(path), "-map", "0:v:0", "-frames:v", "1"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgba", "-"]
    proc = _run(cmd)
    expected = width * height * 4
    if proc.returncode != 0 or len(proc.stdout) < expected:
        return False
    rgba = np.frombuffer(proc.stdout[:expected], dtype=np.uint8).reshape(height, width, 4)
    return bool((rgba[..., 3] < TRANSPARENT_BELOW).any())


def _probe_webm(path: Path) -> _Probe:
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    if proc.returncode != 0:
        raise UnreadableFileError(f"{path}: ffprobe failed: {proc.stderr.decode(errors='replace')}")
    info = json.loads(proc.stdout or b"{}")
    streams = info.get("streams", [])
    video = [s for s in streams if s.get("codec_type") == "video"]
    if not video:
        raise UnreadableFileError(f"{path}: no video stream")
    stream = video[0]
    codec = stream.get("codec_name")
    width, height = int(stream["width"]), int(stream["height"])

    fmt_info = info.get("format", {})
    duration = fmt_info.get("duration") or stream.get("duration") or 0
    duration_s = float(duration)
    frame_count = int(stream.get("nb_read_packets") or stream.get("nb_frames") or 0)
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))

    # ffprobe's native decoder reports yuv420p even for VP9 with alpha; the alpha plane
    # is declared by the Matroska `alpha_mode` flag and decoded only by libvpx.
    tags = {k.lower(): v for k, v in (stream.get("tags") or {}).items()}
    alpha_pix_fmt = str(tags.get("alpha_mode", "0")) == "1" or "yuva" in stream.get("pix_fmt", "")
    has_transparency = _decode_first_frame_alpha(path, _WEBM_DECODERS.get(codec), width, height)

    format_name = fmt_info.get("format_name", "")
    fmt = "webm" if "webm" in format_name or "matroska" in format_name else format_name
    return _Probe(
        fmt=fmt,
        width=width,
        height=height,
        frame_count=max(frame_count, 1),
        duration_s=duration_s,
        fps=fps,
        has_transparency=has_transparency,
        codec=codec,
        alpha_pix_fmt=alpha_pix_fmt,
        audio_streams=sum(1 for s in streams if s.get("codec_type") == "audio"),
    )


def _probe_file(path: Path) -> _Probe:
    with path.open("rb") as handle:
        head = handle.read(4)
    if head == _EBML_MAGIC:
        return _probe_webm(path)
    return _probe_raster(path)


def _select_variant(spec: TargetSpec, size_variant: str | None) -> SizeVariant:
    if size_variant is None:
        return spec.size_variants[0]
    for variant in spec.size_variants:
        if variant.name == size_variant:
            return variant
    names = ", ".join(v.name for v in spec.size_variants)
    raise ValueError(f"{spec.target.value} has no size variant {size_variant!r} (have: {names})")


def _check_size(spec: TargetSpec, variant: SizeVariant, p: _Probe) -> tuple[list[str], list[str]]:
    violations: list[str] = []
    warnings: list[str] = []
    size = f"{p.width}x{p.height}"
    if spec.square and p.width != p.height:
        violations.append(f"size {size} is not square")

    side = max(p.width, p.height)
    side_desc = "side" if spec.square else "longer side"
    size_issues: list[str] = []
    if variant.fixed_sizes and side not in variant.fixed_sizes:
        allowed = "/".join(str(s) for s in variant.fixed_sizes)
        size_issues.append(f"{side_desc} {side} px, expected {allowed} px ({variant.name})")
    if variant.min_side is not None and min(p.width, p.height) < variant.min_side:
        size_issues.append(f"size {size} below minimum {variant.min_side} px ({variant.name})")
    if variant.max_side is not None and side > variant.max_side:
        size_issues.append(f"size {size} above maximum {variant.max_side} px ({variant.name})")

    if variant.recommended:
        warnings.extend(f"{issue}; recommended size" for issue in size_issues)
    else:
        violations.extend(size_issues)
    return violations, warnings


def validate(
    path: Path | str, target: ExportTarget | str, size_variant: str | None = None
) -> ExportResult:
    """Validate a file against the limits of `target`.

    `size_variant` picks a size mode for targets that have several (Twitch: "auto" or
    "manual"); by default the first variant of the target is used.
    """
    path = Path(path)
    spec = get_spec(ExportTarget(target))
    variant = _select_variant(spec, size_variant)
    p = _probe_file(path)
    size_bytes = path.stat().st_size

    violations: list[str] = []
    warnings: list[str] = list(p.notes)

    if p.fmt not in spec.formats:
        violations.append(f"format {p.fmt} not allowed (allowed: {', '.join(spec.formats)})")
    if spec.codec and p.codec is not None and p.codec != spec.codec:
        violations.append(f"codec {p.codec}, expected {spec.codec}")
    if spec.codec and p.fmt == "webm" and p.codec is None:
        violations.append(f"unknown codec, expected {spec.codec}")

    size_violations, size_warnings = _check_size(spec, variant, p)
    violations += size_violations
    warnings += size_warnings

    if variant.max_bytes is not None and size_bytes > variant.max_bytes:
        violations.append(
            f"weight {_fmt_bytes(size_bytes)} exceeds {_fmt_bytes(variant.max_bytes)}"
            f" ({variant.name})"
        )
    if spec.max_frames is not None and p.frame_count > spec.max_frames:
        violations.append(f"{p.frame_count} frames exceeds {spec.max_frames}")
    if spec.max_duration_s is not None and p.duration_s > spec.max_duration_s + 1e-3:
        violations.append(f"duration {p.duration_s:.2f} s exceeds {spec.max_duration_s:g} s")
    if spec.max_fps is not None and p.fps is not None and p.fps > spec.max_fps + 1e-3:
        violations.append(f"{p.fps:.2f} fps exceeds {spec.max_fps:g} fps")

    if spec.requires_alpha:
        if p.fmt == "webm" and not p.alpha_pix_fmt:
            violations.append("no alpha channel in pixel format (expected yuva420p)")
        if not p.has_transparency:
            violations.append("no transparent pixels (alpha required)")
    if not spec.audio_allowed and p.audio_streams:
        violations.append(f"audio not allowed ({p.audio_streams} audio stream(s))")

    return ExportResult(
        path=path,
        target=spec.target,
        width=p.width,
        height=p.height,
        size_bytes=size_bytes,
        frame_count=p.frame_count,
        duration_s=p.duration_s,
        valid=not violations,
        violations=violations,
        warnings=warnings,
    )
