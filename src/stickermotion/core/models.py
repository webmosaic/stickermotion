"""Shared domain types: animation classes, export targets, master animation, export result."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AnimationClass(StrEnum):
    """How large the result is shown; drives amplitude, detail and speed of presets (ADR-002)."""

    EMOTE = "emote"
    STICKER = "sticker"
    OVERLAY = "overlay"


class ExportTarget(StrEnum):
    """Platform output formats, one exporter per target (ADR-001)."""

    TWITCH_GIF = "twitch_gif"
    DISCORD_STICKER = "discord_sticker"
    DISCORD_EMOJI = "discord_emoji"
    TELEGRAM_STICKER = "telegram_sticker"
    TELEGRAM_EMOJI = "telegram_emoji"
    OBS_OVERLAY = "obs_overlay"


class MasterAnimation(BaseModel):
    """Engine output: a sequence of same-sized RGBA frames played at a fixed fps."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    frames: list[Image.Image] = Field(min_length=1)
    fps: float = Field(gt=0)
    loop: bool = True

    @field_validator("frames")
    @classmethod
    def _frames_rgba_same_size(cls, frames: list[Image.Image]) -> list[Image.Image]:
        for index, frame in enumerate(frames):
            if not isinstance(frame, Image.Image):
                raise ValueError(f"frame {index} is not a PIL image")
            if frame.mode != "RGBA":
                raise ValueError(f"frame {index} has mode {frame.mode}, expected RGBA")
        first_size = frames[0].size
        for index, frame in enumerate(frames):
            if frame.size != first_size:
                raise ValueError(f"frame {index} has size {frame.size}, expected {first_size}")
        return frames

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def size(self) -> tuple[int, int]:
        """Frame size as (width, height)."""
        return self.frames[0].size

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps


class ExportResult(BaseModel):
    """Exported file plus its measured metadata and validation outcome."""

    path: Path
    target: ExportTarget
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    size_bytes: int = Field(ge=0)
    frame_count: int = Field(ge=1)
    duration_s: float = Field(ge=0)
    valid: bool
    violations: list[str] = Field(default_factory=list)
    """Hard limit breaches; any violation makes the file invalid."""
    warnings: list[str] = Field(default_factory=list)
    """Soft issues such as non-recommended sizes (ADR-007); do not affect `valid`."""
    size_variant: str | None = None
    """Size variant the file was validated against (Twitch: "auto" or "manual")."""
    attempts: list[str] = Field(default_factory=list)
    """Encoder settings tried by the exporter's weight-fit loop, with resulting weights."""

    @model_validator(mode="after")
    def _valid_matches_violations(self) -> ExportResult:
        if self.valid and self.violations:
            raise ValueError("a valid result must not have violations")
        return self


class PreparedImage(BaseModel):
    """Preprocess output: the character on a square transparent canvas plus placement data.

    Boxes follow the PIL convention (left, top, right, bottom) with exclusive right/bottom.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    image: Image.Image
    bbox: tuple[int, int, int, int]
    """Bounding box of non-transparent pixels on the canvas."""
    anchor: tuple[float, float]
    """Pivot point "bottom centre" of the character on the canvas (x, y)."""
    source_size: tuple[int, int]
    """Size (width, height) of the original input image."""
    scale: float = Field(gt=0)
    """Factor applied to the cropped character when fitting it into the canvas."""

    @field_validator("image")
    @classmethod
    def _image_rgba_square(cls, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise ValueError("image is not a PIL image")
        if image.mode != "RGBA":
            raise ValueError(f"image has mode {image.mode}, expected RGBA")
        if image.width != image.height:
            raise ValueError(f"canvas must be square, got {image.size}")
        return image

    @property
    def canvas_size(self) -> int:
        return self.image.width
