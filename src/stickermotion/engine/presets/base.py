"""Preset interface: parameters, class defaults (ADR-002), seamless-loop rendering.

A preset describes motion as a pose function of the loop phase ``t`` in [0, 1).
The pose function must be periodic with period 1, so the frame after the last one
(phase 1.0) equals the first frame by formula; no frame is duplicated.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from stickermotion.core.models import AnimationClass, MasterAnimation, PreparedImage
from stickermotion.engine.transform import (
    RESAMPLE_PAD,
    Box,
    FrameTransform,
    SpriteTransformer,
    transformed_bbox,
)

# Base timing per class (ADR-002). Emote: short cycle; sticker: slower and smoother.
# Emote 20 frames fits Twitch (<= 60 frames); sticker 2 s x 30 fps fits Telegram (<= 3 s, 30 fps).
CLASS_DEFAULTS: dict[AnimationClass, tuple[float, float]] = {
    AnimationClass.EMOTE: (1.0, 20.0),
    AnimationClass.STICKER: (2.0, 30.0),
}

# Transparent border kept on every frame so canvas edge pixels stay at alpha 0.
EDGE_MARGIN = RESAMPLE_PAD + 1
# Each fitting step shrinks the motion amplitude by this factor.
FIT_STEP = 0.9
FIT_MAX_STEPS = 60


class PresetParams(BaseModel):
    """Resolved render parameters. Build with :meth:`Preset.params` to get class defaults."""

    model_config = ConfigDict(frozen=True)

    animation_class: AnimationClass
    duration_s: float = Field(gt=0)
    fps: float = Field(gt=0)
    intensity: float = Field(ge=0, le=1)

    @property
    def frame_count(self) -> int:
        """Frames in one loop; at least 2 so a loop actually moves."""
        return max(2, round(self.duration_s * self.fps))


@dataclass(frozen=True)
class PresetInfo:
    """Public description of a preset (for listings and the future API)."""

    name: str
    description: str
    supported_classes: tuple[AnimationClass, ...]


class Preset(ABC):
    """Procedural preset. Subclasses define metadata, per-class timing and :meth:`pose`."""

    name: ClassVar[str]
    description: ClassVar[str]
    supported_classes: ClassVar[tuple[AnimationClass, ...]] = (
        AnimationClass.EMOTE,
        AnimationClass.STICKER,
    )
    # Per-class duration override; falls back to CLASS_DEFAULTS.
    default_duration_s: ClassVar[dict[AnimationClass, float]] = {}

    def __init__(
        self,
        animation_class: AnimationClass | str = AnimationClass.STICKER,
        *,
        duration_s: float | None = None,
        fps: float | None = None,
        intensity: float = 1.0,
    ) -> None:
        self.params = self.make_params(
            animation_class, duration_s=duration_s, fps=fps, intensity=intensity
        )

    @classmethod
    def make_params(
        cls,
        animation_class: AnimationClass | str,
        *,
        duration_s: float | None = None,
        fps: float | None = None,
        intensity: float = 1.0,
    ) -> PresetParams:
        """Fill unspecified parameters with defaults for the class (ADR-002)."""
        anim_class = AnimationClass(animation_class)
        if anim_class not in cls.supported_classes:
            raise ValueError(f"preset {cls.name!r} does not support class {anim_class.value!r}")
        base_duration, base_fps = CLASS_DEFAULTS[anim_class]
        return PresetParams(
            animation_class=anim_class,
            duration_s=duration_s
            if duration_s is not None
            else cls.default_duration_s.get(anim_class, base_duration),
            fps=fps if fps is not None else base_fps,
            intensity=intensity,
        )

    @classmethod
    def info(cls) -> PresetInfo:
        return PresetInfo(cls.name, cls.description, cls.supported_classes)

    @abstractmethod
    def pose(self, t: float, height: float, amplitude: float) -> FrameTransform:
        """Pose at loop phase ``t`` (period 1).

        ``height`` is the character's bbox height in pixels (motion scales with it);
        ``amplitude`` is intensity times the fit factor, 0 means the rest pose.
        """

    def phases(self) -> list[float]:
        """Loop phases of the rendered frames: i / n, never reaching 1.0."""
        count = self.params.frame_count
        return [index / count for index in range(count)]

    def fit_amplitude(self, transformer: SpriteTransformer) -> float:
        """Largest amplitude <= intensity that keeps every frame off the canvas edge."""
        height = transformer.bbox[3] - transformer.bbox[1]
        if not self._fits(transformer, height, 0.0):
            raise ValueError(
                "character touches the canvas edge at rest; place it on a canvas with a margin"
            )
        amplitude = self.params.intensity
        for _ in range(FIT_MAX_STEPS):
            if self._fits(transformer, height, amplitude):
                return amplitude
            amplitude *= FIT_STEP
        return 0.0

    def _fits(self, transformer: SpriteTransformer, height: float, amplitude: float) -> bool:
        width_px, height_px = transformer.size
        limit: Box = (EDGE_MARGIN, EDGE_MARGIN, width_px - EDGE_MARGIN, height_px - EDGE_MARGIN)
        for t in self.phases():
            box = transformed_bbox(
                transformer.bbox, self.pose(t, height, amplitude), transformer.pivot
            )
            if box[0] < limit[0] or box[1] < limit[1] or box[2] > limit[2] or box[3] > limit[3]:
                return False
        return True

    def render(self, image: PreparedImage | Image.Image) -> MasterAnimation:
        """Render one seamless loop.

        Takes the output of ``preprocess.normalize`` (its anchor is used as the pivot) or a
        plain square RGBA canvas with the character centred (pivot: bottom centre of bbox).
        """
        pivot = None
        if isinstance(image, PreparedImage):
            pivot = image.anchor
            image = image.image
        if image.mode != "RGBA":
            raise ValueError(f"expected RGBA image, got {image.mode}")
        transformer = SpriteTransformer(image, pivot)
        amplitude = self.fit_amplitude(transformer)
        height = transformer.bbox[3] - transformer.bbox[1]
        frames = [transformer.render(self.pose(t, height, amplitude)) for t in self.phases()]
        return MasterAnimation(frames=frames, fps=self.params.fps, loop=True)


def smooth_bump(u: float) -> float:
    """0 at u=0 and u=1, 1 at u=0.5, zero slope at both ends (sin^2 ease in-out)."""
    return math.sin(math.pi * u) ** 2
