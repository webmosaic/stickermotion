"""Format encoders: GIF (binary alpha), APNG, animated WEBP, WEBM VP9 with alpha.

Each encoder writes one file for a given timing and quality step; the weight-fit loop in
``exporters.base`` decides which steps to try.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from stickermotion.core.alpha import GIF_ALPHA_THRESHOLD
from stickermotion.exporters.base import bleed_color, distribute_delays

Color = tuple[int, int, int]

# GIF delays are stored in centiseconds; browsers replace delays below 20 ms with 100 ms.
GIF_DELAY_UNIT_MS = 10
GIF_MIN_DELAY_UNITS = 2
GIF_MAX_FPS = 1000 / (GIF_DELAY_UNIT_MS * GIF_MIN_DELAY_UNITS)
GIF_COLOR_STEPS = (256, 128, 64, 32, 16)

# APNG: lossless RGBA first, then a shared RGBA palette with fewer colours.
APNG_COLOR_STEPS: tuple[int | None, ...] = (None, 256, 128, 64)

# Animated WEBP: lossless first, then lossy quality.
WEBP_QUALITY_STEPS: tuple[int | None, ...] = (None, 90, 80, 70, 60, 50, 40, 30, 20)

# VP9 constant quality (lower is better); bitrate cap 0 means pure CRF mode.
WEBM_CRF_STEPS = (30, 36, 42, 48, 54, 60, 63)


@dataclass(frozen=True)
class Vp9Rate:
    """Two-pass constrained quality: CRF as the quality floor, a bitrate cap.

    The cap comes from the weight limit and the loop duration, so the encoder spends the
    whole allowed weight on detail instead of guessing it with CRF steps alone.
    """

    crf: int
    kbps: int

    def __str__(self) -> str:
        return f"2-pass crf {self.crf} cap {self.kbps} kbit/s"


class FfmpegMissingError(RuntimeError):
    """ffmpeg with libvpx-vp9 is required for WEBM export."""


# GIF


def matte_and_binarize(
    frame: Image.Image, matte: Color, threshold: int = GIF_ALPHA_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten semi-transparent edges onto ``matte`` and cut alpha to on/off.

    Returns (rgb uint8 HxWx3, opaque mask HxW). Pixels with alpha >= threshold become
    opaque with colour ``c * a + matte * (1 - a)``; the rest become transparent. The edge
    colour therefore blends towards the matte, not towards whatever RGB the transparent
    pixels carried, so there is no light rim on a dark chat background.
    """
    data = np.asarray(frame.convert("RGBA"), dtype=np.float32)
    alpha = data[..., 3:4] / 255.0
    rgb = data[..., :3] * alpha + np.asarray(matte, dtype=np.float32) * (1.0 - alpha)
    opaque = data[..., 3] >= threshold
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8), opaque


def _global_palette(
    flattened: list[tuple[np.ndarray, np.ndarray]], colors: int
) -> tuple[Image.Image, int]:
    """One palette for all frames (no colour flicker between frames).

    Returns a P-mode palette image whose last used index is reserved for transparency, and
    that index. Unused entries repeat colour 0, so nearest-colour mapping never picks them
    over index 0.
    """
    opaque_pixels = [rgb[mask] for rgb, mask in flattened if mask.any()]
    pixels = np.concatenate(opaque_pixels) if opaque_pixels else np.zeros((1, 3), dtype=np.uint8)
    strip = Image.fromarray(pixels.reshape(1, -1, 3), "RGB")
    quantized = strip.quantize(
        colors=colors - 1, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    used = len(quantized.getcolors(256) or [])  # type: ignore[arg-type]
    raw = (quantized.getpalette() or [])[: 3 * max(1, used)]
    palette_colors = [tuple(raw[i : i + 3]) for i in range(0, len(raw), 3)]
    transparent_index = len(palette_colors)
    first = palette_colors[0]
    padded = palette_colors + [first] * (256 - len(palette_colors))
    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette([value for color in padded for value in color])
    return palette_image, transparent_index


def encode_gif(
    path: Path,
    frames: list[Image.Image],
    duration_s: float,
    colors: int,
    matte: Color,
) -> None:
    """Write a looping GIF with binary alpha, delays in multiples of 10 ms."""
    flattened = [matte_and_binarize(frame, matte) for frame in frames]
    palette_image, transparent_index = _global_palette(flattened, colors)
    out_frames: list[Image.Image] = []
    for rgb, opaque in flattened:
        mapped = Image.fromarray(rgb, "RGB").quantize(
            palette=palette_image, dither=Image.Dither.NONE
        )
        indices = np.asarray(mapped).copy()
        # Indices past the real palette are duplicates of colour 0.
        indices[indices >= transparent_index] = 0
        indices[~opaque] = transparent_index
        frame = Image.fromarray(indices, "P")
        frame.putpalette(palette_image.getpalette() or [])
        frame.info["transparency"] = transparent_index
        out_frames.append(frame)
    delays = distribute_delays(
        len(out_frames), duration_s, GIF_DELAY_UNIT_MS, min_units=GIF_MIN_DELAY_UNITS
    )
    out_frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=out_frames[1:],
        duration=delays,
        loop=0,
        disposal=2,
        transparency=transparent_index,
        optimize=False,
    )


# APNG


def shared_palette_frames(frames: list[Image.Image], colors: int) -> list[Image.Image]:
    """Quantize RGBA frames to one shared RGBA palette (P mode, palette carries alpha).

    The frames are stacked into one tall image and quantized together, so every frame uses
    the same palette: APNG stores a single PLTE chunk, and a shared palette also avoids
    colour flicker between frames.
    """
    width, height = frames[0].size
    strip = Image.new("RGBA", (width, height * len(frames)))
    for index, frame in enumerate(frames):
        strip.paste(frame, (0, index * height))
    quantized = strip.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    return [
        quantized.crop((0, index * height, width, (index + 1) * height))
        for index in range(len(frames))
    ]


def encode_apng(
    path: Path, frames: list[Image.Image], duration_s: float, colors: int | None
) -> None:
    """Write a looping APNG with full alpha; ``colors`` reduces the palette if set."""
    out_frames = frames if colors is None else shared_palette_frames(frames, colors)
    delays = distribute_delays(len(out_frames), duration_s, 1)
    out_frames[0].save(
        path,
        format="PNG",
        save_all=True,
        append_images=out_frames[1:],
        duration=delays,
        loop=0,
        disposal=PngImagePlugin.Disposal.OP_NONE,
        blend=PngImagePlugin.Blend.OP_SOURCE,
        optimize=True,
    )


# WEBP


def encode_webp(
    path: Path, frames: list[Image.Image], duration_s: float, quality: int | None
) -> None:
    """Write a looping animated WEBP with full alpha; ``quality`` None means lossless."""
    out_frames = frames if quality is None else [bleed_color(frame) for frame in frames]
    delays = distribute_delays(len(out_frames), duration_s, 1)
    options: dict[str, object] = {"lossless": True, "quality": 100, "method": 4}
    if quality is not None:
        options = {"lossless": False, "quality": quality, "alpha_quality": 100, "method": 4}
    out_frames[0].save(
        path,
        format="WEBP",
        save_all=True,
        append_images=out_frames[1:],
        duration=delays,
        loop=0,
        allow_mixed=False,
        **options,
    )


# WEBM


def _frame_rate(count: int, duration_s: float) -> str:
    rate = (Fraction(count) / Fraction(duration_s)).limit_denominator(1000)
    return f"{rate.numerator}/{rate.denominator}"


def encode_webm(
    path: Path,
    frames: list[Image.Image],
    duration_s: float,
    crf: int | Vp9Rate,
    cpu_used: int = 4,
) -> None:
    """Write a VP9 WEBM with alpha (yuva420p), no audio.

    ``crf`` as an int is one-pass constant quality; a ``Vp9Rate`` is two-pass constrained
    quality with a bitrate cap.
    """
    if shutil.which("ffmpeg") is None:
        raise FfmpegMissingError("ffmpeg not found in PATH; it is required for WEBM export")
    with tempfile.TemporaryDirectory(prefix="stickermotion-") as tmp:
        tmp_dir = Path(tmp)
        for index, frame in enumerate(frames):
            # Colour bleed: 4:2:0 chroma would otherwise pull black from transparent pixels.
            bleed_color(frame).save(tmp_dir / f"{index:05d}.png", compress_level=1)
        head = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
        head += ["-framerate", _frame_rate(len(frames), duration_s)]
        head += ["-i", str(tmp_dir / "%05d.png"), "-an", "-c:v", "libvpx-vp9"]
        head += ["-pix_fmt", "yuva420p"]
        tail = ["-deadline", "good", "-cpu-used", str(cpu_used), "-row-mt", "1", "-f", "webm"]
        if isinstance(crf, Vp9Rate):
            rate = ["-b:v", f"{crf.kbps}k", "-crf", str(crf.crf)]
            log = ["-passlogfile", str(tmp_dir / "pass")]
            first = [*head, *rate, "-pass", "1", *log, *tail, os.devnull]
            proc = subprocess.run(first, capture_output=True, check=False)
            if proc.returncode == 0:
                second = [*head, *rate, "-pass", "2", *log, *tail, str(path)]
                proc = subprocess.run(second, capture_output=True, check=False)
        else:
            cmd = [*head, "-b:v", "0", "-crf", str(crf), *tail, str(path)]
            proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()}")
