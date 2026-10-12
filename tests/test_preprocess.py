from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from stickermotion.core.models import PreparedImage
from stickermotion.preprocess import EmptyImageError, NoAlphaError, normalize

FIXTURES = Path(__file__).parent / "fixtures"
NAMES = ["char_square.png", "char_tall.png", "char_wide_soft.png"]


def _load(name: str) -> Image.Image:
    with Image.open(FIXTURES / name) as image:
        return image.copy()


@pytest.mark.parametrize("name", NAMES)
def test_character_inside_square_canvas_with_headroom(name: str) -> None:
    source = _load(name)
    prepared = normalize(source, canvas_size=512, headroom=0.18)
    image = prepared.image
    assert image.mode == "RGBA"
    assert image.size == (512, 512)
    assert prepared.canvas_size == 512
    assert prepared.source_size == source.size

    alpha = np.asarray(image)[..., 3]
    rows = np.flatnonzero((alpha > 0).any(axis=1))
    cols = np.flatnonzero((alpha > 0).any(axis=0))
    assert prepared.bbox == (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)

    margin = 0.18 * 512 - 2  # small slack for resampling ringing
    left, top, right, bottom = prepared.bbox
    assert left >= margin and top >= margin
    assert 512 - right >= margin and 512 - bottom >= margin
    # Longer side actually fills the available area (not shrunk to nothing).
    assert max(right - left, bottom - top) >= 512 * (1 - 2 * 0.18) - 4


@pytest.mark.parametrize("name", NAMES)
def test_anchor_is_bottom_centre(name: str) -> None:
    prepared = normalize(_load(name), canvas_size=256)
    left, _, right, bottom = prepared.bbox
    assert prepared.anchor == ((left + right) / 2, float(bottom))
    assert abs(prepared.anchor[0] - 128) <= 2


@pytest.mark.parametrize("name", NAMES)
def test_transparent_pixels_have_zero_rgb(name: str) -> None:
    data = np.asarray(normalize(_load(name), canvas_size=512).image)
    assert (data[data[..., 3] == 0][:, :3] == 0).all()


def test_alpha_threshold_drops_faint_pixels() -> None:
    arr = np.zeros((40, 40, 4), dtype=np.uint8)
    arr[10:20, 10:20] = [10, 200, 10, 255]
    arr[0:2, 0:40] = [255, 255, 255, 5]  # faint noise above the character
    prepared = normalize(Image.fromarray(arr, "RGBA"), canvas_size=100, headroom=0.2)
    left, top, right, bottom = prepared.bbox
    # Crop ignores the faint row, so the 10x10 square is centred and scaled to 60 px.
    assert right - left == pytest.approx(60, abs=2)
    assert bottom - top == pytest.approx(60, abs=2)


def test_no_fringe_on_semi_transparent_edges() -> None:
    """Edge pixels keep the body colour: no dark/light halo from hidden RGB."""
    body = (120, 150, 255)
    size = 200
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.hypot(xx - size / 2, yy - size / 2)
    alpha = np.clip((70 - dist) / 20, 0, 1)  # soft 20 px edge
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[..., :3] = 255  # hidden colour of transparent pixels is white
    arr[dist < 70, :3] = body
    arr[..., 3] = np.rint(alpha * 255).astype(np.uint8)
    # Black hidden colour on the other half to catch dark fringes too.
    arr[:, : size // 2][arr[:, : size // 2, 3] == 0, :3] = 0

    for canvas in (64, 1024):  # downscale and upscale
        out = np.asarray(normalize(Image.fromarray(arr, "RGBA"), canvas_size=canvas).image)
        a = out[..., 3]
        edge = out[(a > 0) & (a < 255)][:, :3].astype(int)
        core = out[a == 255][:, :3].astype(int)
        assert len(edge) > 0 and len(core) > 0
        assert np.abs(core - body).max() <= 2
        assert np.abs(edge - body).max() <= 6, f"fringe at canvas {canvas}"


@pytest.mark.parametrize("name", ["char_square.png", "char_tall.png"])
@pytest.mark.parametrize("canvas", [512, 1024, 2048])
def test_no_fringe_on_fixture_outline(name: str, canvas: int) -> None:
    # Hard-edged fixtures: the outermost ring is the outline colour (24, 20, 32), so
    # semi-transparent pixels created by resampling must keep that colour.
    out = np.asarray(normalize(_load(name), canvas_size=canvas).image)
    a = out[..., 3]
    edge = out[(a > 0) & (a < 128)][:, :3].astype(int)
    assert len(edge) > 0
    assert np.abs(edge - (24, 20, 32)).max() <= 3


def test_rgb_input_raises_no_alpha() -> None:
    with pytest.raises(NoAlphaError, match="transparent background"):
        normalize(Image.new("RGB", (64, 64), (255, 0, 0)))


def test_opaque_rgba_raises_no_alpha() -> None:
    with pytest.raises(NoAlphaError, match="opaque"):
        normalize(Image.new("RGBA", (64, 64), (255, 0, 0, 255)))


def test_empty_image_raises() -> None:
    with pytest.raises(EmptyImageError):
        normalize(Image.new("RGBA", (64, 64), (255, 0, 0, 3)))


def test_palette_with_transparency_is_accepted() -> None:
    rgba = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    rgba.paste((255, 0, 0, 255), (8, 8, 24, 24))
    paletted = rgba.convert("P")
    paletted.info["transparency"] = paletted.getpixel((0, 0))
    prepared = normalize(paletted, canvas_size=64)
    assert prepared.image.size == (64, 64)


@pytest.mark.parametrize(
    "kwargs",
    [{"canvas_size": 0}, {"headroom": 0.5}, {"headroom": -0.1}, {"alpha_threshold": 256}],
)
def test_bad_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        normalize(_load("char_square.png"), **kwargs)  # type: ignore[arg-type]


def test_prepared_image_rejects_non_square_or_non_rgba() -> None:
    base = {"bbox": (0, 0, 1, 1), "anchor": (0.5, 1.0), "source_size": (1, 1), "scale": 1.0}
    with pytest.raises(ValidationError, match="square"):
        PreparedImage(image=Image.new("RGBA", (10, 12)), **base)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="RGBA"):
        PreparedImage(image=Image.new("RGB", (10, 10)), **base)  # type: ignore[arg-type]
