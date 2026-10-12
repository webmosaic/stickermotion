"""Platform limits for every ExportTarget as structured data (source: docs/PLATFORM_SPECS.md).

No export logic here: exporters and the validator read these limits.
Sizes are in pixels, weights in bytes (1 KB = 1024 B, 1 MB = 1024 KB), durations in seconds.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from stickermotion.core.models import AnimationClass, ExportTarget

KB = 1024
MB = 1024 * KB


class SizeVariant(BaseModel):
    """One allowed size mode of a target with its own weight limit.

    `fixed_sizes` lists exact allowed sizes (square side, or longer side if not square).
    `min_side` / `max_side` bound the side when the size is a range.
    `None` means no limit.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    fixed_sizes: tuple[int, ...] = ()
    min_side: int | None = None
    max_side: int | None = None
    max_bytes: int | None = None
    recommended: bool = False
    """True when the sizes are only a recommendation: other sizes give a warning (ADR-007)."""


class TargetSpec(BaseModel):
    """Limits of a single export target."""

    model_config = ConfigDict(frozen=True)

    target: ExportTarget
    animation_class: AnimationClass
    formats: tuple[str, ...]
    """Allowed container formats, preferred first."""
    codec: str | None = None
    square: bool
    size_variants: tuple[SizeVariant, ...]
    max_frames: int | None = None
    max_duration_s: float | None = None
    max_fps: float | None = None
    max_flashes_per_s: float | None = None
    requires_alpha: bool = True
    binary_alpha: bool = False
    """True when the format only supports on/off transparency (GIF)."""
    audio_allowed: bool = True

    @property
    def max_bytes(self) -> int | None:
        """Largest weight limit across size variants (None if any variant is unlimited)."""
        limits = [v.max_bytes for v in self.size_variants]
        if any(limit is None for limit in limits):
            return None
        return max(limit for limit in limits if limit is not None)


SPECS: dict[ExportTarget, TargetSpec] = {
    ExportTarget.TWITCH_GIF: TargetSpec(
        target=ExportTarget.TWITCH_GIF,
        animation_class=AnimationClass.EMOTE,
        formats=("gif",),
        square=True,
        size_variants=(
            SizeVariant(name="auto", min_side=112, max_side=4096, max_bytes=1 * MB),
            SizeVariant(name="manual", fixed_sizes=(28, 56, 112), max_bytes=512 * KB),
        ),
        max_frames=60,
        max_flashes_per_s=3,
        binary_alpha=True,
    ),
    ExportTarget.DISCORD_STICKER: TargetSpec(
        target=ExportTarget.DISCORD_STICKER,
        animation_class=AnimationClass.STICKER,
        formats=("apng", "gif", "png", "lottie"),
        square=True,
        size_variants=(SizeVariant(name="default", fixed_sizes=(320,), max_bytes=512 * KB),),
        max_duration_s=5,
    ),
    ExportTarget.DISCORD_EMOJI: TargetSpec(
        target=ExportTarget.DISCORD_EMOJI,
        animation_class=AnimationClass.EMOTE,
        formats=("gif", "webp", "apng"),
        square=True,
        size_variants=(
            SizeVariant(name="default", fixed_sizes=(128,), max_bytes=256 * KB, recommended=True),
        ),
    ),
    ExportTarget.TELEGRAM_STICKER: TargetSpec(
        target=ExportTarget.TELEGRAM_STICKER,
        animation_class=AnimationClass.STICKER,
        formats=("webm",),
        codec="vp9",
        square=False,
        size_variants=(SizeVariant(name="default", fixed_sizes=(512,), max_bytes=256 * KB),),
        max_duration_s=3,
        max_fps=30,
        audio_allowed=False,
    ),
    ExportTarget.TELEGRAM_EMOJI: TargetSpec(
        target=ExportTarget.TELEGRAM_EMOJI,
        animation_class=AnimationClass.EMOTE,
        formats=("webm",),
        codec="vp9",
        square=True,
        # The page says 256 KB, but a 173 KB emoji failed custom emoji pack creation on the
        # live server; 64 KB is the working target for video emoji.
        size_variants=(SizeVariant(name="default", fixed_sizes=(100,), max_bytes=64 * KB),),
        max_duration_s=3,
        max_fps=30,
        audio_allowed=False,
    ),
    ExportTarget.OBS_OVERLAY: TargetSpec(
        target=ExportTarget.OBS_OVERLAY,
        animation_class=AnimationClass.OVERLAY,
        formats=("webm", "png_sequence"),
        codec="vp9",
        square=False,
        size_variants=(SizeVariant(name="default"),),
    ),
}


def get_spec(target: ExportTarget) -> TargetSpec:
    """Return limits for a target."""
    return SPECS[target]
