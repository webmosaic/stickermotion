from typer.testing import CliRunner

from stickermotion import __version__
from stickermotion.cli import app
from stickermotion.core.models import ExportTarget

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"stickermotion {__version__}"


def test_targets_lists_all_targets() -> None:
    result = runner.invoke(app, ["targets"])
    assert result.exit_code == 0
    for target in ExportTarget:
        assert target.value in result.stdout


def test_targets_shows_limits() -> None:
    out = runner.invoke(app, ["targets"]).stdout
    twitch = next(line for line in out.splitlines() if line.startswith("twitch_gif"))
    assert "1 MB" in twitch and "512 KB" in twitch and "60" in twitch
    tg = next(line for line in out.splitlines() if line.startswith("telegram_sticker"))
    assert "vp9" in tg and "256 KB" in tg and "3s" in tg and "30" in tg
