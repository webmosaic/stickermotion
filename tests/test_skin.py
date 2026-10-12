"""Skin hue repair after the cut on synthetic frames."""

from __future__ import annotations

import numpy as np
from PIL import Image

from stickermotion.engine.ai.chain import cut, process
from stickermotion.engine.ai.framing import GREEN, flatten
from stickermotion.engine.ai.skin import SkinFix, skin_reference, to_hsv, to_rgb

SKIN = (230, 175, 145)  # hue ~21 deg
PINK = (240, 120, 140)  # hue ~350 deg: pink clothes / sleep mask
SIZE = 120


def _rotate(color: tuple[int, int, int], degrees: float) -> tuple[int, int, int]:
    hsv = to_hsv(np.array([[color]], dtype=np.uint8))
    hsv[..., 0] = (hsv[..., 0] + degrees) % 360
    r, g, b = (int(v) for v in to_rgb(hsv)[0, 0])
    return r, g, b


LILAC = _rotate(SKIN, -50)  # skin drifted to ~331 deg


def _character() -> Image.Image:
    """Skin body with shading, a pink band (mask) on top, transparent around."""
    data = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    shade = np.linspace(0.8, 1.0, SIZE)[None, :, None]
    data[20:110, 10:110, :3] = np.clip(np.array(SKIN) * shade, 0, 255)[0, 10:110]
    data[20:110, 10:110, 3] = 255
    data[20:35, 10:110, :3] = PINK
    return Image.fromarray(data, "RGBA")


def _hue(rgb: tuple[int, int, int] | np.ndarray) -> float:
    return float(to_hsv(np.asarray(rgb, dtype=np.uint8).reshape(1, 1, 3))[0, 0, 0])


def _hue_gap(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def test_hsv_round_trip() -> None:
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert np.abs(to_rgb(to_hsv(rgb)).astype(int) - rgb).max() <= 1
    assert round(_hue((255, 0, 0))) == 0 and round(_hue((0, 0, 255))) == 240


def test_reference_from_input_png() -> None:
    tone = skin_reference(_character())
    assert tone is not None
    assert _hue_gap(tone.hue, _hue(SKIN)) < 3
    # The input's own pink is a protected hue, the lilac drift is not.
    native = {b * 5 for b in tone.native_bins}
    assert int(_hue(PINK)) // 5 * 5 in native
    assert int(_hue(LILAC)) // 5 * 5 not in native


def test_no_reference_without_skin() -> None:
    blue = Image.new("RGBA", (40, 40), (40, 60, 220, 255))
    assert skin_reference(blue) is None
    assert SkinFix.build(blue, flatten(blue, GREEN), (40, 40)) is None


def test_lilac_skin_back_pink_and_protected_area_untouched() -> None:
    character = _character()
    model_input = flatten(character, GREEN)
    fix = SkinFix.build(character, model_input, (SIZE, SIZE))
    assert fix is not None
    frame = np.asarray(character).copy()
    frame[60:80, 40:70, :3] = LILAC  # moving fist turned lilac
    frame[90:100, 80:100, :3] = PINK  # pink sleeve moved to a skin area (hue limit)
    frame[36:40, 40:60, :3] = LILAC  # right next to the input's pink (mask limit)
    fixed = np.asarray(fix(Image.fromarray(frame, "RGBA")))

    fist = fixed[60:80, 40:70, :3].reshape(-1, 3)
    assert max(_hue_gap(_hue(p), fix.tone.hue) for p in fist) < 2
    # Saturation and brightness stay as the model drew them.
    before = to_hsv(frame[60:80, 40:70, :3])
    after = to_hsv(fixed[60:80, 40:70, :3])
    assert np.abs(before[..., 1:] - after[..., 1:]).max() < 0.02
    assert np.array_equal(fixed[90:100, 80:100], frame[90:100, 80:100])
    assert np.array_equal(fixed[20:40], frame[20:40])
    assert np.array_equal(fixed[..., 3], frame[..., 3])
    # Natural skin (not shifted) is not touched at all.
    assert np.array_equal(fixed[85:110, 10:40], frame[85:110, 10:40])


def test_slight_drift_and_transparent_pixels_untouched() -> None:
    character = _character()
    fix = SkinFix.build(character, flatten(character, GREEN), (SIZE, SIZE))
    assert fix is not None
    frame = np.asarray(character).copy()
    frame[60:70, 40:60, :3] = _rotate(SKIN, -5)  # within normal skin variation
    frame[0:10, 0:10] = (*LILAC, 0)  # transparent
    out = np.asarray(fix(Image.fromarray(frame, "RGBA")))
    assert np.array_equal(out, frame)


def test_cut_applies_skin_fix_after_key() -> None:
    character = _character()
    model_input = flatten(character, GREEN)
    fix = SkinFix.build(character, model_input, (SIZE, SIZE))
    tinted = np.asarray(character).copy()
    tinted[60:80, 40:70, :3] = LILAC
    frame = flatten(Image.fromarray(tinted, "RGBA"), GREEN)
    plain = np.asarray(cut([frame])[0])
    fixed = np.asarray(cut([frame], skin=fix)[0])
    assert _hue_gap(_hue(plain[70, 55, :3]), _hue(LILAC)) < 2
    assert _hue_gap(_hue(fixed[70, 55, :3]), _hue(SKIN)) < 3
    result = process([frame] * 4, 8.0, skin=fix)
    assert result.master.frame_count >= 1


def test_manifest_view() -> None:
    character = _character()
    fix = SkinFix.build(character, flatten(character, GREEN), (SIZE, SIZE))
    assert fix is not None
    info = fix.to_dict()
    assert info["protected_px"] > 0 and 15 < info["hue_deg"] < 30
    assert 350 in info["protected_hues_deg"]


def test_pale_lilac_in_pink_hue_is_repaired_saturated_pink_is_not() -> None:
    """Live clip: the lilac fist shares hues with the pink mask (335-345 deg) but is paler."""
    character = _character()
    fix = SkinFix.build(character, flatten(character, GREEN), (SIZE, SIZE))
    assert fix is not None
    pink_hsv = to_hsv(np.array([[PINK]], dtype=np.uint8))
    assert fix.tone.native_sat_low > 0.3
    pale = pink_hsv.copy()
    pale[..., 1] = 0.32  # paler than the pink, within the skin saturation range
    pale_rgb = tuple(int(v) for v in to_rgb(pale)[0, 0])
    frame = np.asarray(character).copy()
    frame[60:80, 40:70, :3] = pale_rgb
    frame[90:100, 80:100, :3] = PINK
    fixed = np.asarray(fix(Image.fromarray(frame, "RGBA")))
    assert _hue_gap(_hue(fixed[70, 55, :3]), fix.tone.hue) < 2
    # The saturated pink and a 1 px rim around it stay as they are.
    assert np.array_equal(fixed[89:101, 79:101], frame[89:101, 79:101])
