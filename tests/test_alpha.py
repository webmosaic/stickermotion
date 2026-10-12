import numpy as np
import pytest
from PIL import Image

from stickermotion.core.alpha import premultiply, resize_premultiplied, unpremultiply


def test_premultiply_scales_color_by_alpha() -> None:
    image = Image.new("RGBA", (2, 1), (200, 100, 50, 128))
    data = premultiply(image)
    assert data.dtype == np.float32
    assert data.shape == (1, 2, 4)
    a = 128 / 255
    assert data[0, 0, 0] == pytest.approx(200 / 255 * a, abs=1e-6)
    assert data[0, 0, 3] == pytest.approx(a, abs=1e-6)


def test_round_trip_preserves_visible_pixels() -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(8, 8, 4), dtype=np.uint8)
    arr[..., 3] = rng.integers(64, 256, size=(8, 8), dtype=np.uint8)
    back = np.asarray(unpremultiply(premultiply(Image.fromarray(arr, "RGBA"))))
    assert np.abs(back.astype(int) - arr.astype(int)).max() <= 2


def test_unpremultiply_zeroes_transparent_rgb() -> None:
    data = np.zeros((1, 2, 4), dtype=np.float32)
    data[0, 1] = [0.5, 0.5, 0.5, 0.5]
    out = np.asarray(unpremultiply(data))
    assert out[0, 0].tolist() == [0, 0, 0, 0]
    assert out[0, 1].tolist() == [255, 255, 255, 128]


def test_unpremultiply_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        unpremultiply(np.zeros((2, 2, 3), dtype=np.float32))


def test_resize_premultiplied_has_no_fringe() -> None:
    # Red opaque square on transparent pixels whose hidden colour is white.
    arr = np.zeros((16, 16, 4), dtype=np.uint8)
    arr[..., :3] = 255
    arr[4:12, 4:12] = [255, 0, 0, 255]
    image = Image.fromarray(arr, "RGBA")
    out = np.asarray(resize_premultiplied(image, (7, 7)))
    edge = out[(out[..., 3] > 0) & (out[..., 3] < 255)]
    assert len(edge) > 0
    # Straight-alpha resize would leak white into G/B on the edge.
    assert edge[:, 1].max() <= 2 and edge[:, 2].max() <= 2
    assert (out[out[..., 3] == 0][:, :3] == 0).all()
