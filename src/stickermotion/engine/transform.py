"""Per-frame affine transform of an RGBA sprite around a pivot point.

Coordinates are continuous canvas pixels, x to the right, y down. The pivot defaults to
the bottom centre of the character's alpha bounding box. Resampling is bicubic on
premultiplied float channels, so semi-transparent edges keep their colour (no fringe).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from stickermotion.core.alpha import premultiply, unpremultiply

Point = tuple[float, float]
Box = tuple[float, float, float, float]

# Extra pixels around the transformed bbox that the bicubic kernel can still touch.
RESAMPLE_PAD = 4


@dataclass(frozen=True)
class FrameTransform:
    """Pose of the sprite in one frame.

    ``scale_x`` / ``scale_y`` scale around the pivot, ``angle_deg`` rotates around it
    (positive = clockwise on screen), ``dx`` / ``dy`` move it in pixels (``dy < 0`` is up).
    """

    scale_x: float = 1.0
    scale_y: float = 1.0
    angle_deg: float = 0.0
    dx: float = 0.0
    dy: float = 0.0

    def __post_init__(self) -> None:
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("scale must be positive")

    def _rotation(self) -> tuple[float, float]:
        rad = math.radians(self.angle_deg)
        return math.cos(rad), math.sin(rad)

    def forward(self, point: Point, pivot: Point) -> Point:
        """Map a source point to its position on the output canvas."""
        cos, sin = self._rotation()
        x = (point[0] - pivot[0]) * self.scale_x
        y = (point[1] - pivot[1]) * self.scale_y
        return (
            pivot[0] + self.dx + cos * x - sin * y,
            pivot[1] + self.dy + sin * x + cos * y,
        )

    def inverse_coefficients(self, pivot: Point) -> tuple[float, float, float, float, float, float]:
        """PIL AFFINE coefficients (a, b, c, d, e, f): output (x, y) -> source point."""
        cos, sin = self._rotation()
        a, b = cos / self.scale_x, sin / self.scale_x
        d, e = -sin / self.scale_y, cos / self.scale_y
        tx, ty = pivot[0] + self.dx, pivot[1] + self.dy
        c = pivot[0] - (a * tx + b * ty)
        f = pivot[1] - (d * tx + e * ty)
        return a, b, c, d, e, f


def alpha_bbox(image: Image.Image) -> Box:
    """Bounding box (left, top, right, bottom) of non-transparent pixels."""
    box = image.getchannel("A").getbbox()
    if box is None:
        raise ValueError("image is fully transparent")
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def default_pivot(box: Box) -> Point:
    """Bottom centre of a bounding box."""
    return ((box[0] + box[2]) / 2.0, box[3])


def transformed_bbox(box: Box, transform: FrameTransform, pivot: Point) -> Box:
    """Exact bounding box of a transformed rectangle (an affine map sends corners to extremes)."""
    corners = [
        transform.forward(corner, pivot)
        for corner in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3]))
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (min(xs), min(ys), max(xs), max(ys))


class SpriteTransformer:
    """Keeps a premultiplied copy of the source so each frame only pays for resampling."""

    def __init__(self, image: Image.Image, pivot: Point | None = None) -> None:
        self.size: tuple[int, int] = image.size
        self.bbox: Box = alpha_bbox(image)
        self.pivot: Point = pivot if pivot is not None else default_pivot(self.bbox)
        premultiplied = premultiply(image)
        self._channels = [
            Image.fromarray(np.ascontiguousarray(premultiplied[..., i]), "F") for i in range(4)
        ]

    def render(self, transform: FrameTransform) -> Image.Image:
        """Render the sprite with ``transform`` onto a transparent canvas of the source size."""
        width, height = self.size
        canvas = Image.new("RGBA", self.size, (0, 0, 0, 0))
        left, top, right, bottom = transformed_bbox(self.bbox, transform, self.pivot)
        x0 = max(0, math.floor(left) - RESAMPLE_PAD)
        y0 = max(0, math.floor(top) - RESAMPLE_PAD)
        x1 = min(width, math.ceil(right) + RESAMPLE_PAD)
        y1 = min(height, math.ceil(bottom) + RESAMPLE_PAD)
        if x1 <= x0 or y1 <= y0:
            return canvas
        a, b, c, d, e, f = transform.inverse_coefficients(self.pivot)
        # Render only the affected window: shift the output origin to (x0, y0).
        coeffs = (a, b, c + a * x0 + b * y0, d, e, f + d * x0 + e * y0)
        window = np.empty((y1 - y0, x1 - x0, 4), dtype=np.float32)
        for index, channel in enumerate(self._channels):
            warped = channel.transform(
                (x1 - x0, y1 - y0),
                Image.Transform.AFFINE,
                coeffs,
                resample=Image.Resampling.BICUBIC,
                fillcolor=0.0,
            )
            window[..., index] = np.asarray(warped, dtype=np.float32)
        canvas.paste(unpremultiply(window), (x0, y0))
        return canvas
