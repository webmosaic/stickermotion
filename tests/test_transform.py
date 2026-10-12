from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from stickermotion.core.alpha import premultiply, unpremultiply
from stickermotion.engine.transform import (
    FrameTransform,
    SpriteTransformer,
    alpha_bbox,
    default_pivot,
    transformed_bbox,
)

COLOR = (220, 40, 30)


def _soft_disc(size: int = 256, color: tuple[int, int, int] = COLOR) -> Image.Image:
    """Single-colour disc with a wide blurred alpha edge; RGB outside is black (worst case)."""
    alpha = Image.new("L", (size, size), 0)
    ImageDraw.Draw(alpha).ellipse((64, 64, size - 64, size - 64), fill=255)
    alpha = alpha.filter(ImageFilter.GaussianBlur(8))
    image = Image.new("RGBA", (size, size), (*color, 255))
    image.putalpha(alpha)
    pixels = np.asarray(image).copy()
    pixels[pixels[..., 3] == 0] = 0
    return Image.fromarray(pixels, "RGBA")


def test_premultiply_roundtrip() -> None:
    image = _soft_disc()
    back = np.asarray(unpremultiply(premultiply(image))).astype(int)
    src = np.asarray(image).astype(int)
    assert np.abs(back - src).max() <= 1


def test_premultiply_values() -> None:
    image = Image.new("RGBA", (1, 1), (200, 100, 0, 128))
    value = premultiply(image)[0, 0]
    assert value[3] == pytest.approx(128 / 255)
    assert value[0] == pytest.approx(200 / 255 * 128 / 255)


def test_unpremultiply_clips_overshoot_and_clears_transparent() -> None:
    data = np.array([[[1.5, 0.2, -0.1, 0.5], [0.3, 0.3, 0.3, -0.2]]], dtype=np.float32)
    out = np.asarray(unpremultiply(data))
    assert tuple(out[0, 0]) == (255, 102, 0, 128)
    assert tuple(out[0, 1]) == (0, 0, 0, 0)


def test_forward_and_inverse_agree() -> None:
    transform = FrameTransform(scale_x=1.3, scale_y=0.8, angle_deg=17, dx=5, dy=-12)
    pivot = (100.0, 180.0)
    a, b, c, d, e, f = transform.inverse_coefficients(pivot)
    for point in [(0.0, 0.0), (120.0, 40.0), (100.0, 180.0)]:
        x, y = transform.forward(point, pivot)
        assert a * x + b * y + c == pytest.approx(point[0])
        assert d * x + e * y + f == pytest.approx(point[1])


def test_pivot_stays_fixed_without_translation() -> None:
    transform = FrameTransform(scale_x=2, scale_y=0.5, angle_deg=30)
    assert transform.forward((10.0, 20.0), (10.0, 20.0)) == pytest.approx((10.0, 20.0))


def test_positive_angle_is_clockwise() -> None:
    # A point above the pivot moves right when rotating clockwise on screen (y down).
    x, _ = FrameTransform(angle_deg=10).forward((0.0, -10.0), (0.0, 0.0))
    assert x > 0


def test_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError):
        FrameTransform(scale_x=0)


def test_alpha_bbox_and_default_pivot() -> None:
    image = Image.new("RGBA", (50, 40), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (10, 5, 30, 25))
    box = alpha_bbox(image)
    assert box == (10, 5, 30, 25)
    assert default_pivot(box) == (20, 25)


def test_alpha_bbox_rejects_empty() -> None:
    with pytest.raises(ValueError):
        alpha_bbox(Image.new("RGBA", (4, 4)))


def test_transformed_bbox_rotation() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    out = transformed_bbox(box, FrameTransform(angle_deg=45), (5.0, 5.0))
    half_diag = 5 * math.sqrt(2)
    assert out == pytest.approx((5 - half_diag, 5 - half_diag, 5 + half_diag, 5 + half_diag))


def test_identity_render_matches_source() -> None:
    image = _soft_disc()
    out = SpriteTransformer(image).render(FrameTransform())
    assert np.abs(np.asarray(out).astype(int) - np.asarray(image).astype(int)).max() <= 1


def test_render_keeps_size_and_mode() -> None:
    out = SpriteTransformer(_soft_disc()).render(FrameTransform(scale_x=1.2, dy=-10))
    assert out.mode == "RGBA"
    assert out.size == (256, 256)


def test_translation_moves_pixels() -> None:
    image = _soft_disc()
    out = SpriteTransformer(image).render(FrameTransform(dx=7, dy=-11))
    bbox_src = alpha_bbox(image)
    bbox_out = alpha_bbox(out)
    assert bbox_out[0] - bbox_src[0] == pytest.approx(7, abs=1)
    assert bbox_out[1] - bbox_src[1] == pytest.approx(-11, abs=1)


def test_custom_pivot() -> None:
    sprite = SpriteTransformer(_soft_disc(), pivot=(128.0, 128.0))
    assert sprite.pivot == (128.0, 128.0)


@pytest.mark.parametrize(
    "transform",
    [
        FrameTransform(angle_deg=13.7),
        FrameTransform(scale_x=1.17, scale_y=0.83, dx=0.4, dy=-3.3),
        FrameTransform(scale_x=0.71, scale_y=1.29, angle_deg=-9),
    ],
)
def test_no_fringe_on_soft_edges(transform: FrameTransform) -> None:
    """Semi-transparent edge pixels keep the sprite colour: no dark halo from the black
    transparent background bleeding in during resampling."""
    out = np.asarray(SpriteTransformer(_soft_disc()).render(transform)).astype(int)
    alpha = out[..., 3]
    edge = (alpha >= 8) & (alpha < 255)
    assert edge.sum() > 500
    deviation = np.abs(out[edge][:, :3] - np.array(COLOR)).max()
    assert deviation <= 6, deviation
    # Fully transparent pixels carry no colour.
    assert out[alpha == 0][:, :3].max() == 0
