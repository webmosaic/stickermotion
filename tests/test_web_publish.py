"""Publish block, settings page, Twitch archive and preview in the web UI."""

from __future__ import annotations

import io
import logging
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fake_platforms import DC_TOKEN, GUILD, TG_TOKEN, Platforms
from stickermotion.core.models import ExportTarget
from stickermotion.publish.config import PublishConfig
from stickermotion.web.app import create_app
from stickermotion.web.preview import PREVIEW_FILE
from stickermotion.web.publishing import Publishing
from stickermotion.web.security import CSRF_FIELD
from stickermotion.web.store import PubStatus, Store
from stickermotion.web.worker import Worker
from test_ai import TOKEN
from test_web import AUTO_UK, Env, WebFake, _png, _settings

TARGETS = [
    ExportTarget.TELEGRAM_STICKER,
    ExportTarget.TELEGRAM_EMOJI,
    ExportTarget.DISCORD_EMOJI,
    ExportTarget.DISCORD_STICKER,
]
SECRETS = (TG_TOKEN, DC_TOKEN, TOKEN)


def _publish_config(**kwargs: Any) -> PublishConfig:
    values: dict[str, Any] = {
        "telegram_token": TG_TOKEN,
        "telegram_owner_id": 4242,
        "telegram_pack_title": "My Pack",
        "telegram_pack_slug": "mypack",
        "discord_token": DC_TOKEN,
        "discord_guild_id": GUILD,
        "twitch_channel": "MyChannel",
    }
    return PublishConfig(**{**values, **kwargs})


@dataclass
class PubEnv(Env):
    platforms: Platforms = None  # type: ignore[assignment]

    @property
    def publishing(self) -> Publishing:
        return self.client.app.state.publishing  # type: ignore[attr-defined]

    def ready_sticker(self, character: bytes) -> str:
        sticker_id = self.new_sticker(character)
        self.generate(sticker_id, AUTO_UK)
        return sticker_id


def _make(
    tmp_path: Path, clip: bytes, publish: PublishConfig, targets: list[ExportTarget] = TARGETS
) -> Iterator[PubEnv]:
    fake = WebFake(clip)
    platforms = Platforms()
    config = _settings(tmp_path, publish=publish, daily_budget_usd=5.0)

    def worker(store: Store, cfg: Any) -> Worker:
        return Worker(
            store,
            cfg,
            client_options={"transport": fake, "sleep": lambda _s: None},
            targets=targets,
        )

    def pub(store: Store, cfg: Any) -> Publishing:
        return Publishing(store, cfg.publish, platforms)

    app = create_app(config, worker_factory=worker, publishing_factory=pub)
    with TestClient(app) as client:
        yield PubEnv(client, fake, config, platforms)


@pytest.fixture(scope="module")
def green_clip(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Same green H.264 clip as in test_web (Wan output)."""
    from stickermotion.engine.ai.video import write_mp4
    from test_ai import _clip_frames

    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    write_mp4(path, _clip_frames(), 16.0)
    return path.read_bytes()


@pytest.fixture
def penv(tmp_path: Path, green_clip: bytes) -> Iterator[PubEnv]:
    yield from _make(tmp_path, green_clip, _publish_config())


@pytest.fixture
def character() -> bytes:
    from test_ai import _character

    return _png(_character(200))


def _publish(env: PubEnv, sticker_id: str, channel: str, **data: str) -> Any:
    token = env.csrf(f"/s/{sticker_id}")
    return env.client.post(
        f"/s/{sticker_id}/publish/{channel}",
        data={CSRF_FIELD: token, **data},
        follow_redirects=False,
    )


def test_publish_block_and_telegram_flow(penv: PubEnv, character: bytes) -> None:
    penv.login()
    sticker_id = penv.ready_sticker(character)
    page = penv.client.get(f"/s/{sticker_id}").text
    for text in (
        "Опублікувати",
        "Telegram, стікерпак",
        "Telegram, емодзі",
        "Discord, емодзі",
        "Discord, стікер",
        "Twitch, емоут",
        "Не опубліковано",
    ):
        assert text in page, text
    # Automatic emoji from the action ("hops up and lands") and a valid Discord name.
    assert 'name="emoji" value="🎉"' in page
    assert 'name="name" value="hero"' in page

    response = _publish(penv, sticker_id, "telegram_sticker", emoji="😴 💤", also_emoji="1")
    assert response.status_code == 303 and response.headers["location"].endswith("#publish")
    rows = penv.store.publications(sticker_id)
    sticker_row = rows["telegram_sticker"]
    assert sticker_row.state is PubStatus.PUBLISHED and sticker_row.emoji == "😴 💤"
    assert sticker_row.url == "https://t.me/addstickers/mypack_by_ExampleBot"
    assert sticker_row.published_at and sticker_row.remote_id
    assert rows["telegram_emoji"].state is PubStatus.PUBLISHED
    assert rows["telegram_emoji"].url.startswith("https://t.me/addemoji/")
    page = penv.client.get(f"/s/{sticker_id}").text
    assert "https://t.me/addstickers/mypack_by_ExampleBot" in page
    assert "Опубліковано" in page and "Опублікувати ще раз" in page

    # Publishing again does not create a duplicate.
    _publish(penv, sticker_id, "telegram_sticker", emoji="😴")
    tg = penv.platforms.telegram
    assert len(tg.sets["mypack_by_ExampleBot"]["stickers"]) == 1
    assert tg.methods().count("createNewStickerSet") == 2  # sticker pack + emoji pack
    assert tg.methods().count("addStickerToSet") == 0
    state = penv.client.get(f"/s/{sticker_id}/publish.json").json()
    assert state["telegram_sticker"]["status"] == "published"


def test_discord_publish_idempotent_and_named(penv: PubEnv, character: bytes) -> None:
    penv.login()
    sticker_id = penv.ready_sticker(character)
    _publish(penv, sticker_id, "discord_emoji", name="example_emote")
    _publish(penv, sticker_id, "discord_sticker", name="example_emote", emoji="😴")
    dc = penv.platforms.discord
    assert [e["name"] for e in dc.emojis] == ["example_emote"]
    assert dc.stickers[0]["tags"] == "😴"
    _publish(penv, sticker_id, "discord_emoji", name="example_emote")
    assert len(dc.emojis) == 1 and len(dc.requests("POST", "/emojis")) == 1
    row = penv.store.publication(sticker_id, "discord_emoji")
    assert row.state is PubStatus.PUBLISHED and row.url.endswith("?animated=true")
    # Removed on the platform by hand: the next publish uploads again.
    dc.emojis.clear()
    _publish(penv, sticker_id, "discord_emoji", name="example_emote")
    assert len(dc.emojis) == 1


def test_publish_errors_are_shown(penv: PubEnv, character: bytes) -> None:
    penv.login()
    sticker_id = penv.ready_sticker(character)
    penv.platforms.telegram.owner_started = False
    _publish(penv, sticker_id, "telegram_sticker", emoji="😴")
    row = penv.store.publication(sticker_id, "telegram_sticker")
    assert row.state is PubStatus.ERROR and "/start" in row.error
    penv.platforms.discord.emojis.extend(
        {"id": str(i), "name": f"e{i}", "animated": True} for i in range(50)
    )
    _publish(penv, sticker_id, "discord_emoji", name="example")
    page = penv.client.get(f"/s/{sticker_id}").text
    assert "напишіть боту /start" in page.lower() and "немає вільних слотів" in page
    assert "Повторити" in page
    _publish(penv, sticker_id, "discord_sticker", name="погане ім'я", emoji="😴")
    assert "латинські" in penv.store.publication(sticker_id, "discord_sticker").error


def test_unconfigured_platform_shows_what_is_missing(
    tmp_path: Path, green_clip: bytes, character: bytes
) -> None:
    for env in _make(tmp_path, green_clip, PublishConfig()):
        env.login()
        sticker_id = env.ready_sticker(character)
        page = env.client.get(f"/s/{sticker_id}").text
        assert page.count("Не налаштовано: бракує") == 4
        assert "TELEGRAM_BOT_TOKEN, TELEGRAM_OWNER_ID" in page
        assert "DISCORD_BOT_TOKEN, DISCORD_GUILD_ID" in page
        response = _publish(env, sticker_id, "discord_emoji", name="abc")
        assert response.status_code == 303
        row = env.store.publication(sticker_id, "discord_emoji")
        assert row.state is PubStatus.NONE and "не налаштовано" in row.error
        assert env.platforms.discord.calls == [] and env.platforms.telegram.calls == []
        settings = env.client.get("/settings").text
        assert settings.count("Не налаштовано") >= 2 and "Бракує TELEGRAM_BOT_TOKEN" in settings


def test_settings_page_without_tokens(penv: PubEnv, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    penv.login()
    page = penv.client.get("/settings")
    assert page.status_code == 200
    text = page.text
    for expected in (
        "@ExampleBot",
        "mypack_by_ExampleBot",
        "My Pack",
        "4242",
        "Підключено",
        "0 з 50",
        "0 з 5",
        "dashboard.twitch.tv/u/mychannel",
    ):
        assert expected in text, expected
    for secret in SECRETS:
        assert secret not in text
    assert all(c[0] == "GET" for c in penv.platforms.discord.calls)
    assert set(penv.platforms.telegram.methods()) == {"getMe"}  # read only
    assert penv.client.get("/settings", headers={"accept": "application/json"}).status_code == 200
    penv.client.cookies.clear()
    assert penv.client.get("/settings", headers={"accept": "application/json"}).status_code == 401


def test_tokens_never_in_html_logs_or_data(
    penv: PubEnv, character: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    penv.login()
    sticker_id = penv.ready_sticker(character)
    tg_fail = {"createNewStickerSet": None}
    from fake_platforms import _err

    tg_fail["createNewStickerSet"] = _err(400, f"Bad Request: echo {TG_TOKEN}")
    penv.platforms.telegram.fail.update(tg_fail)  # type: ignore[arg-type]
    from stickermotion.publish.http import Response

    penv.platforms.discord.fail["POST emojis"] = Response(
        500, f'{{"code": 0, "message": "echo {DC_TOKEN}"}}'.encode()
    )
    _publish(penv, sticker_id, "telegram_sticker", emoji="😴")
    _publish(penv, sticker_id, "discord_emoji", name="abc")
    _publish(penv, sticker_id, "discord_sticker", name="abc", emoji="😴")
    bodies = [
        penv.client.get(path).content
        for path in (
            "/",
            "/history",
            "/settings",
            f"/s/{sticker_id}",
            f"/s/{sticker_id}/publish.json",
            f"/s/{sticker_id}/status",
            f"/s/{sticker_id}/twitch.zip",
            "/static/app.js",
        )
    ]
    rows = penv.store.publications(sticker_id)
    assert "***" in rows["telegram_sticker"].error and "***" in rows["discord_emoji"].error
    for secret in SECRETS:
        assert all(secret.encode() not in body for body in bodies), secret
        assert secret not in caplog.text
        for path in penv.settings.data_dir.rglob("*"):
            if path.is_file():
                assert secret.encode() not in path.read_bytes(), path
    assert TG_TOKEN not in repr(penv.settings) and DC_TOKEN not in repr(penv.settings)


def test_publish_requires_login_csrf_and_ready_sticker(penv: PubEnv, character: bytes) -> None:
    assert penv.client.post("/s/abc/publish/telegram_sticker").status_code == 401
    penv.login()
    sticker_id = penv.new_sticker(character)  # described, not generated
    no_csrf = penv.client.post(f"/s/{sticker_id}/publish/telegram_sticker", data={"emoji": "😴"})
    assert no_csrf.status_code == 403
    assert _publish(penv, sticker_id, "telegram_sticker").status_code == 409
    assert _publish(penv, sticker_id, "nope").status_code == 404
    assert penv.platforms.telegram.calls == []


def test_delete_offers_platform_removal(penv: PubEnv, character: bytes) -> None:
    penv.login()
    keep = penv.ready_sticker(character)
    drop = penv.ready_sticker(character)
    for sticker_id in (keep, drop):
        _publish(penv, sticker_id, "discord_emoji", name=f"e_{sticker_id}")
        _publish(penv, sticker_id, "telegram_sticker", emoji="😴")
    page = penv.client.get(f"/s/{keep}").text
    assert "Прибрати його також з платформ" in page and 'name="remove_discord_emoji"' in page
    assert 'name="remove_discord_emoji" value="1" checked' not in page  # off by default
    # Default: deleted here, stays on the platforms.
    penv.client.post(f"/s/{keep}/delete", data={CSRF_FIELD: penv.csrf(f"/s/{keep}")})
    assert penv.store.get(keep) is None
    assert len(penv.platforms.discord.emojis) == 2
    # With the checkbox: removed from Discord only.
    response = penv.client.post(
        f"/s/{drop}/delete",
        data={CSRF_FIELD: penv.csrf(f"/s/{drop}"), "remove_discord_emoji": "1"},
        follow_redirects=False,
    )
    assert response.headers["location"] == "/history"
    assert [e["name"] for e in penv.platforms.discord.emojis] == [f"e_{keep}"]
    assert "deleteStickerFromSet" not in penv.platforms.telegram.methods()
    assert penv.store.publications(drop) == {}


def test_unpublish_button(penv: PubEnv, character: bytes) -> None:
    penv.login()
    sticker_id = penv.ready_sticker(character)
    _publish(penv, sticker_id, "telegram_sticker", emoji="😴")
    token = penv.csrf(f"/s/{sticker_id}")
    penv.client.post(f"/s/{sticker_id}/unpublish/telegram_sticker", data={CSRF_FIELD: token})
    assert "deleteStickerFromSet" in penv.platforms.telegram.methods()
    assert penv.store.publication(sticker_id, "telegram_sticker").state is PubStatus.NONE
    assert penv.platforms.telegram.sets["mypack_by_ExampleBot"]["stickers"] == []


def test_publishing_interrupted_by_restart(
    tmp_path: Path, green_clip: bytes, character: bytes
) -> None:
    store = Store(tmp_path / "data")
    sticker = store.create("a.png", character, 200, 200)
    store.save_publication(sticker.id, "discord_emoji", status=PubStatus.PUBLISHING)
    for env in _make(tmp_path, green_clip, _publish_config()):
        row = env.store.publication(sticker.id, "discord_emoji")
        assert row.state is PubStatus.ERROR and "перезапустився" in row.error


def test_twitch_archive_route(tmp_path: Path, green_clip: bytes, character: bytes) -> None:
    targets = [ExportTarget.TWITCH_GIF, ExportTarget.TELEGRAM_STICKER]
    for env in _make(tmp_path, green_clip, _publish_config(), targets):
        env.login()
        sticker_id = env.ready_sticker(character)
        page = env.client.get(f"/s/{sticker_id}").text
        assert "Підготувати для Twitch" in page
        assert "https://dashboard.twitch.tv/u/mychannel/viewer-rewards/emotes" in page
        response = env.client.get(f"/s/{sticker_id}/twitch.zip")
        assert (
            response.status_code == 200 and "attachment" in response.headers["content-disposition"]
        )
        names = sorted(zipfile.ZipFile(io.BytesIO(response.content)).namelist())
        assert names == [
            "README.txt",
            "auto/twitch_gif_224.gif",
            "manual/twitch_gif_112.gif",
            "manual/twitch_gif_28.gif",
            "manual/twitch_gif_56.gif",
        ]


def test_hero_preview_is_full_alpha_webm_with_webp_fallback(penv: PubEnv, character: bytes) -> None:
    penv.login()
    sticker_id = penv.ready_sticker(character)
    page = penv.client.get(f"/s/{sticker_id}").text
    hero = re.search(r"<video data-hero[^>]*>", page, re.S)
    assert hero, page
    assert f"/s/{sticker_id}/files/telegram_sticker.webm" in hero.group(0)
    assert f'data-fallback="/s/{sticker_id}/preview.webp"' in hero.group(0)
    response = penv.client.get(f"/s/{sticker_id}/preview.webp")
    assert response.status_code == 200 and response.headers["content-type"] == "image/webp"
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.format == "WEBP" and getattr(image, "n_frames", 1) > 1
        frame = image.convert("RGBA")
        alpha = frame.getchannel("A")
        low, high = alpha.getextrema()
        assert low == 0 and high == 255
        # Full alpha: soft edges keep intermediate values (a GIF/APNG palette would not).
        assert ((np.asarray(alpha) > 0) & (np.asarray(alpha) < 255)).any()
    assert (penv.store.output_dir(sticker_id) / PREVIEW_FILE).is_file()
    script = penv.client.get("/static/app.js").text
    assert "canPlayType" in script and "userAgent" not in script


def test_publish_log_has_raw_telegram_error(
    penv: PubEnv, character: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    """The real Bot API reason reaches the log, never the token."""
    from fake_platforms import _err

    caplog.set_level(logging.DEBUG)
    penv.login()
    sticker_id = penv.ready_sticker(character)
    penv.platforms.telegram.fail["createNewStickerSet"] = _err(
        400, f"Bad Request: STICKER_VIDEO_BIG {TG_TOKEN}"
    )
    _publish(penv, sticker_id, "telegram_emoji", emoji="😴")
    row = penv.store.publication(sticker_id, "telegram_emoji")
    assert row.state is PubStatus.ERROR and "завеликий" in row.error
    assert "raw error: createNewStickerSet HTTP 400" in caplog.text
    assert "STICKER_VIDEO_BIG" in caplog.text and TG_TOKEN not in caplog.text
