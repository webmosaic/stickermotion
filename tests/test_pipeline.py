from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from presets_helpers import FIXTURE_NAMES, FIXTURES
from stickermotion import pipeline
from stickermotion.cli import app
from stickermotion.core.models import AnimationClass, ExportTarget
from stickermotion.engine.presets import list_presets
from stickermotion.exporters import ExportFitError
from stickermotion.report import REAL_SIZE_PX, render_index

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
runner = CliRunner()
SQUARE = FIXTURES / "char_square.png"


@pytest.fixture
def small_canvas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preprocess onto a 384 px canvas to keep pipeline tests fast (layout is unaffected)."""
    real = pipeline.normalize
    monkeypatch.setattr(pipeline, "normalize", lambda image: real(image, canvas_size=384))


def test_master_class() -> None:
    assert pipeline.master_class(ExportTarget.TWITCH_GIF) is AnimationClass.EMOTE
    assert pipeline.master_class(ExportTarget.DISCORD_EMOJI) is AnimationClass.EMOTE
    assert pipeline.master_class(ExportTarget.TELEGRAM_EMOJI) is AnimationClass.EMOTE
    assert pipeline.master_class(ExportTarget.DISCORD_STICKER) is AnimationClass.STICKER
    assert pipeline.master_class(ExportTarget.TELEGRAM_STICKER) is AnimationClass.STICKER
    assert pipeline.master_class(ExportTarget.OBS_OVERLAY) is AnimationClass.STICKER


def test_run_renders_master_once_per_class(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[AnimationClass] = []
    real_get = pipeline.get_preset

    def counting_get(name: str):  # type: ignore[no-untyped-def]
        cls = real_get(name)

        class Counting(cls):  # type: ignore[valid-type, misc]
            def render(self, image):  # type: ignore[no-untyped-def]
                calls.append(self.params.animation_class)
                return super().render(image)

        return Counting

    monkeypatch.setattr(pipeline, "get_preset", counting_get)
    results = pipeline.run(
        SQUARE,
        ["squash_stretch"],
        ["twitch_gif", "discord_emoji", "discord_sticker"],
        tmp_path,
    )
    assert calls == [AnimationClass.EMOTE, AnimationClass.STICKER]
    assert len(results) == 6  # 4 Twitch sizes + emoji + sticker
    assert all(r.valid for r in results)


def test_run_layout_manifest_and_page(tmp_path: Path, small_canvas: None) -> None:
    results = pipeline.run(SQUARE, ["wiggle"], ["discord_emoji", "twitch_gif"], tmp_path)
    root = tmp_path / "char_square"
    names = sorted(p.name for p in (root / "wiggle").iterdir())
    assert names == [
        "discord_emoji.webp",
        "twitch_gif_112.gif",
        "twitch_gif_224.gif",
        "twitch_gif_28.gif",
        "twitch_gif_56.gif",
    ]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["input"] == "char_square.png"
    assert manifest["total_s"] > 0
    preset = manifest["presets"][0]
    assert preset["name"] == "wiggle"
    assert preset["renders"][0]["params"]["animation_class"] == "emote"
    assert preset["renders"][0]["render_s"] > 0
    assert len(preset["exports"]) == len(results) == 5
    for entry in preset["exports"]:
        assert entry["path"].startswith("wiggle/")
        assert entry["export_s"] >= 0
        assert entry["valid"] is True

    page = (root / "index.html").read_text()
    for entry in preset["exports"]:
        assert f'src="{entry["path"]}"' in page
    assert "http://" not in page and "https://" not in page
    assert "#18181b" in page.lower()


def test_run_unknown_preset(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown preset"):
        pipeline.run(SQUARE, ["spin"], None, tmp_path)


def _entry(target: str, path: str, valid: bool = True, **extra: object) -> dict[str, object]:
    return {
        "target": target,
        "path": path,
        "width": 128,
        "height": 128,
        "size_bytes": 2048,
        "frame_count": 16,
        "duration_s": 0.8,
        "valid": valid,
        "violations": [] if valid else ["weight 300.0 KB exceeds 256.0 KB (default)"],
        "warnings": [],
        **extra,
    }


def test_render_index_shows_all_views() -> None:
    manifest = {
        "input": "x.png",
        "total_s": 1.0,
        "presets": [
            {
                "name": "bounce",
                "renders": [{"class": "emote", "frames": 16, "fps": 20, "render_s": 0.5}],
                "exports": [
                    _entry("discord_emoji", "bounce/discord_emoji.webp", valid=False),
                    _entry("telegram_sticker", "bounce/telegram_sticker.webm"),
                    _entry("obs_overlay", "bounce/obs_overlay.webm"),
                ],
            }
        ],
    }
    page = render_index(manifest)
    # Three backgrounds per file in the large view.
    assert page.count('class="stage big dark"') == 3
    assert page.count('class="stage big light"') == 3
    assert page.count('class="stage big checker"') == 3
    # Real size rows: emoji in chat at 28 px, Telegram sticker at 256 px, none for OBS.
    assert 'width="28" height="28"' in page
    assert f'width="{REAL_SIZE_PX[ExportTarget.TELEGRAM_STICKER]}"' in page
    assert page.count("Real size") == 2
    # WEBM as looping muted inline video.
    assert re.search(r"<video [^>]*autoplay loop muted playsinline", page)
    # Violations in red, summary counts them.
    assert '<li class="v">weight 300.0 KB exceeds 256.0 KB (default)</li>' in page
    assert "1 invalid" in page
    assert "128×128 px, 2.0 KB, 16 frames, 0.80 s, 20.0 fps" in page


# CLI


def test_cli_presets() -> None:
    result = runner.invoke(app, ["presets"])
    assert result.exit_code == 0
    for info in list_presets():
        assert info.name in result.stdout
        assert info.description in result.stdout


def test_cli_render_subset(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "render",
            str(SQUARE),
            "--preset",
            "bounce",
            "--target",
            "discord_emoji",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "1 files, 0 invalid" in result.stdout
    assert (tmp_path / "char_square" / "index.html").exists()
    assert (tmp_path / "char_square" / "bounce" / "discord_emoji.webp").exists()


def test_cli_render_exit_code_on_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from stickermotion.validator import validate

    def failing_export(master, target, out_dir, **options):  # type: ignore[no-untyped-def]
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "discord_emoji.webp"
        frames = [f.resize((300, 200)) for f in master.frames]  # not square
        frames[0].save(path, save_all=True, append_images=frames[1:], lossless=True)
        raise ExportFitError("could not fit", ["x"], validate(path, target))

    monkeypatch.setattr(pipeline, "export_target", failing_export)
    result = runner.invoke(
        app,
        ["render", str(SQUARE), "--preset", "bounce", "--target", "discord_emoji"]
        + ["--out", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "INVALID" in result.stdout
    assert "1 invalid" in result.stdout
    assert (tmp_path / "char_square" / "index.html").exists()


def test_cli_render_errors(tmp_path: Path) -> None:
    missing = runner.invoke(app, ["render", str(tmp_path / "nope.png")])
    assert missing.exit_code == 2
    bad_preset = runner.invoke(
        app, ["render", str(SQUARE), "--preset", "spin", "--out", str(tmp_path)]
    )
    assert bad_preset.exit_code == 2
    bad_target = runner.invoke(app, ["render", str(SQUARE), "--target", "myspace"])
    assert bad_target.exit_code != 0


def test_cli_validate(tmp_path: Path) -> None:
    frames = [Image.new("RGBA", (28, 28), (0, 0, 0, 0)) for _ in range(2)]
    frames[0].paste((255, 0, 0, 255), (4, 4, 20, 20))
    frames[1].paste((255, 0, 0, 255), (6, 6, 22, 22))
    good = tmp_path / "ok.gif"
    frames[0].save(good, save_all=True, append_images=frames[1:], duration=100, loop=0)
    ok = runner.invoke(app, ["validate", str(good), "twitch_gif", "--variant", "manual"])
    assert ok.exit_code == 0, ok.stdout
    assert ok.stdout.startswith("ok")

    auto = runner.invoke(app, ["validate", str(good), "twitch_gif", "--variant", "auto"])
    assert auto.exit_code == 1
    assert "below minimum 112 px" in auto.stdout

    bad_variant = runner.invoke(app, ["validate", str(good), "twitch_gif", "--variant", "x"])
    assert bad_variant.exit_code == 2
    missing = runner.invoke(app, ["validate", str(tmp_path / "none.gif"), "twitch_gif"])
    assert missing.exit_code == 2


# Full matrix


@pytest.mark.slow
@needs_ffmpeg
def test_full_matrix_via_cli(tmp_path: Path) -> None:
    """3 fixtures x 3 presets x 6 targets: every file passes the validator, exit code 0."""
    for name in FIXTURE_NAMES:
        result = runner.invoke(app, ["render", str(FIXTURES / name), "--out", str(tmp_path)])
        assert result.exit_code == 0, result.stdout
        root = tmp_path / Path(name).stem
        manifest = json.loads((root / "manifest.json").read_text())
        exports = [e for p in manifest["presets"] for e in p["exports"]]
        # Per preset: 4 Twitch files + 5 single-file targets.
        assert len(exports) == 3 * 9
        assert all(e["valid"] for e in exports), [e for e in exports if not e["valid"]]
        assert {e["target"] for e in exports} == {t.value for t in ExportTarget}
        assert (root / "index.html").exists()
