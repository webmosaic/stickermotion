"""Generate synthetic RGBA PNG fixtures in tests/fixtures/.

Each fixture is a simple "drawn" character: filled body with a dark outline on a
fully transparent background. Three aspect ratios; one with soft (semi-transparent) edges.
Deterministic: running it again produces identical files.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

OUTLINE = (24, 20, 32, 255)
EYE_WHITE = (255, 255, 255, 255)


def _character(size: tuple[int, int], body: tuple[int, int, int, int]) -> Image.Image:
    """Draw a blob character (body, eyes, mouth) that fills the canvas with a margin."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin_x, margin_y = width // 10, height // 10
    outline_w = max(4, min(width, height) // 40)
    box = (margin_x, margin_y, width - margin_x, height - margin_y)
    draw.ellipse(box, fill=body, outline=OUTLINE, width=outline_w)

    cx, cy = width // 2, int(height * 0.42)
    eye_r = max(6, min(width, height) // 14)
    eye_dx = max(eye_r * 2, (width - 2 * margin_x) // 6)
    for ex in (cx - eye_dx, cx + eye_dx):
        draw.ellipse(
            (ex - eye_r, cy - eye_r, ex + eye_r, cy + eye_r),
            fill=EYE_WHITE,
            outline=OUTLINE,
            width=outline_w // 2,
        )
        pupil = eye_r // 2
        draw.ellipse((ex - pupil, cy - pupil, ex + pupil, cy + pupil), fill=OUTLINE)

    mouth_w = eye_dx
    mouth_y = int(height * 0.6)
    draw.arc(
        (cx - mouth_w, mouth_y - mouth_w // 2, cx + mouth_w, mouth_y + mouth_w // 2),
        start=20,
        end=160,
        fill=OUTLINE,
        width=outline_w,
    )
    return image


def _soften_edges(image: Image.Image, radius: float) -> Image.Image:
    """Blur only the alpha channel to get semi-transparent anti-aliased edges."""
    r, g, b, a = image.split()
    return Image.merge("RGBA", (r, g, b, a.filter(ImageFilter.GaussianBlur(radius))))


def build_fixtures() -> dict[str, Image.Image]:
    """Return fixture images keyed by file name."""
    return {
        "char_square.png": _character((512, 512), (255, 196, 60, 255)),
        "char_tall.png": _character((320, 640), (110, 200, 120, 255)),
        "char_wide_soft.png": _soften_edges(_character((720, 360), (120, 150, 255, 255)), 4),
    }


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for name, image in build_fixtures().items():
        path = FIXTURES_DIR / name
        image.save(path, format="PNG", optimize=True)
        print(f"{path.relative_to(FIXTURES_DIR.parent.parent)}  {image.size[0]}x{image.size[1]}")


if __name__ == "__main__":
    main()
