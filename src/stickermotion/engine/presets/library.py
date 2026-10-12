"""First procedural presets: bounce, wiggle, squash_stretch.

Every pose function is periodic in ``t`` with period 1. Amplitudes are fractions of the
character height (translation) or degrees (rotation). Emote variants move more, in a
shorter cycle and with fewer simultaneous components (readable at 28 px); sticker
variants are softer and smoother (ADR-002).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from stickermotion.core.models import AnimationClass
from stickermotion.engine.presets.base import Preset, smooth_bump
from stickermotion.engine.transform import FrameTransform


def _squash(amount: float) -> tuple[float, float]:
    """Area-preserving scale: positive amount squashes vertically, negative stretches."""
    scale_y = 1.0 - amount
    return 1.0 / scale_y, scale_y


def _punchy_sine(t: float, exponent: float) -> float:
    """Sine with longer holds at the extremes when exponent < 1 (snappier, reads at 28 px)."""
    value = math.sin(2.0 * math.pi * (t % 1.0))
    return math.copysign(abs(value) ** exponent, value)


@dataclass(frozen=True)
class BounceStyle:
    jump: float  # jump height as a fraction of character height
    squash: float  # vertical squash at the landing contact
    contact: float  # share of the loop spent on the ground
    air_stretch: float  # vertical stretch while rising/falling, 0 disables


class Bounce(Preset):
    name = "bounce"
    description = "Jump up and land with a short squash on contact."
    default_duration_s: ClassVar[dict[AnimationClass, float]] = {
        AnimationClass.EMOTE: 0.8,
        AnimationClass.STICKER: 1.4,
    }
    styles: ClassVar[dict[AnimationClass, BounceStyle]] = {
        AnimationClass.EMOTE: BounceStyle(jump=0.30, squash=0.22, contact=0.25, air_stretch=0.0),
        AnimationClass.STICKER: BounceStyle(jump=0.18, squash=0.12, contact=0.30, air_stretch=0.05),
    }

    def pose(self, t: float, height: float, amplitude: float) -> FrameTransform:
        style = self.styles[self.params.animation_class]
        t %= 1.0
        if t < style.contact:
            # On the ground: ease into the squash and back out of it.
            squash = style.squash * amplitude * smooth_bump(t / style.contact)
            scale_x, scale_y = _squash(squash)
            return FrameTransform(scale_x=scale_x, scale_y=scale_y)
        # In the air: ballistic parabola gives natural ease-out up and ease-in down.
        u = (t - style.contact) / (1.0 - style.contact)
        lift = 4.0 * u * (1.0 - u)
        stretch = style.air_stretch * amplitude * math.sin(2.0 * math.pi * u) ** 2
        scale_x, scale_y = _squash(-stretch)
        return FrameTransform(
            scale_x=scale_x, scale_y=scale_y, dy=-style.jump * amplitude * height * lift
        )


@dataclass(frozen=True)
class WiggleStyle:
    angle_deg: float  # peak rotation each side
    exponent: float  # waveform shape, < 1 is snappier
    breathe: float  # subtle vertical squash on each swing, 0 disables


class Wiggle(Preset):
    name = "wiggle"
    description = "Sway side to side, rotating around the bottom."
    default_duration_s: ClassVar[dict[AnimationClass, float]] = {
        AnimationClass.EMOTE: 0.8,
        AnimationClass.STICKER: 2.0,
    }
    styles: ClassVar[dict[AnimationClass, WiggleStyle]] = {
        AnimationClass.EMOTE: WiggleStyle(angle_deg=14.0, exponent=0.6, breathe=0.0),
        AnimationClass.STICKER: WiggleStyle(angle_deg=8.0, exponent=1.0, breathe=0.03),
    }

    def pose(self, t: float, height: float, amplitude: float) -> FrameTransform:
        style = self.styles[self.params.animation_class]
        angle = style.angle_deg * amplitude * _punchy_sine(t, style.exponent)
        # Settle down slightly at each extreme (twice per loop), back to rest in the middle.
        squash = style.breathe * amplitude * math.sin(2.0 * math.pi * t) ** 2
        scale_x, scale_y = _squash(squash)
        return FrameTransform(scale_x=scale_x, scale_y=scale_y, angle_deg=angle)


@dataclass(frozen=True)
class SquashStyle:
    amount: float  # peak vertical squash / stretch
    exponent: float  # waveform shape, < 1 is snappier


class SquashStretch(Preset):
    name = "squash_stretch"
    description = "Rhythmic breathing: squash vertically while widening, area preserved."
    default_duration_s: ClassVar[dict[AnimationClass, float]] = {
        AnimationClass.EMOTE: 0.8,
        AnimationClass.STICKER: 2.0,
    }
    styles: ClassVar[dict[AnimationClass, SquashStyle]] = {
        AnimationClass.EMOTE: SquashStyle(amount=0.16, exponent=0.7),
        AnimationClass.STICKER: SquashStyle(amount=0.07, exponent=1.0),
    }

    def pose(self, t: float, height: float, amplitude: float) -> FrameTransform:
        style = self.styles[self.params.animation_class]
        squash = style.amount * amplitude * _punchy_sine(t, style.exponent)
        scale_x, scale_y = _squash(squash)
        return FrameTransform(scale_x=scale_x, scale_y=scale_y)
