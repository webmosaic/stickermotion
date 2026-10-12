"""Publishers with fake Bot API and Discord API: no network, no real bots."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from fake_platforms import (
    DC_TOKEN,
    GUILD,
    TG_TOKEN,
    FakeDiscord,
    FakeTelegram,
    _err,
    parse_multipart,
)
from stickermotion.publish import Channel, NotConfiguredError, PublishError, PublishRequest
from stickermotion.publish.config import PublishConfig
from stickermotion.publish.discord import DiscordPublisher, discord_info, webp_to_gif
from stickermotion.publish.http import Response, body_snippet, multipart, redact
from stickermotion.publish.naming import (
    discord_name,
    discord_name_problem,
    split_emoji,
    suggest_emoji,
    telegram_set_name,
    telegram_set_name_problem,
    unique_name,
)
from stickermotion.publish.telegram import (
    OwnerNotStartedError,
    TelegramPublisher,
    scrub,
    telegram_info,
)
from stickermotion.publish.twitch import build_archive, emotes_url


def _config(**kwargs: object) -> PublishConfig:
    values: dict[str, object] = {
        "telegram_token": TG_TOKEN,
        "telegram_owner_id": 4242,
        "telegram_pack_title": "My Pack",
        "telegram_pack_slug": "mypack",
        "discord_token": DC_TOKEN,
        "discord_guild_id": GUILD,
    }
    return PublishConfig(**{**values, **kwargs})  # type: ignore[arg-type]


def _animated(path: Path, fmt: str, frames: int = 4) -> Path:
    images = []
    for index in range(frames):
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        image.paste((200, 40 * index, 90, 255), (10 + index * 4, 10, 40 + index * 4, 50))
        images.append(image)
    images[0].save(path, fmt, save_all=True, append_images=images[1:], duration=80, loop=0)
    return path


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    webm_a = tmp_path / "a" / "telegram_sticker.webm"
    webm_b = tmp_path / "b" / "telegram_sticker.webm"
    for path, content in ((webm_a, b"\x1a\x45\xdf\xa3 sticker A"), (webm_b, b"\x1a\x45\xdf\xa3 B")):
        path.parent.mkdir()
        path.write_bytes(content)
    return {
        "a": webm_a,
        "b": webm_b,
        "webp": _animated(tmp_path / "discord_emoji.webp", "WEBP"),
        "png": _animated(tmp_path / "discord_sticker.png", "PNG"),
    }


def _request(
    path: Path, name: str = "example_emote", emoji: tuple[str, ...] = ("😴",)
) -> PublishRequest:
    return PublishRequest(path, name, emoji, "дівчина позіхає")


# Telegram


def test_telegram_first_sticker_creates_pack_then_adds(files: dict[str, Path]) -> None:
    fake = FakeTelegram()
    publisher = TelegramPublisher(_config(), Channel.TELEGRAM_STICKER, fake)
    first = publisher.publish(_request(files["a"]))
    assert "createNewStickerSet" in fake.methods() and "addStickerToSet" not in fake.methods()
    created = dict(fake.calls)["createNewStickerSet"]
    assert created["name"] == "mypack_by_ExampleBot"
    assert created["user_id"] == "4242" and created["title"] == "My Pack"
    assert created["sticker_type"] == "regular"
    sticker = json.loads(created["stickers"])[0]
    assert sticker == {"sticker": "attach://sticker0", "format": "video", "emoji_list": ["😴"]}
    assert created["sticker0"]["filename"] == "telegram_sticker.webm"
    assert first.url == "https://t.me/addstickers/mypack_by_ExampleBot"
    assert first.remote_id and first.extra["set_name"] == "mypack_by_ExampleBot"

    second = publisher.publish(_request(files["b"]))
    assert fake.methods().count("createNewStickerSet") == 1
    assert fake.methods().count("addStickerToSet") == 1
    assert second.remote_id != first.remote_id
    assert len(fake.sets["mypack_by_ExampleBot"]["stickers"]) == 2
    assert publisher.exists(first) and publisher.exists(second)


def test_telegram_custom_emoji_pack(files: dict[str, Path]) -> None:
    fake = FakeTelegram()
    publisher = TelegramPublisher(_config(), Channel.TELEGRAM_EMOJI, fake)
    remote = publisher.publish(_request(files["a"], emoji=("😴", "💤")))
    created = dict(fake.calls)["createNewStickerSet"]
    assert created["name"] == "mypack_emoji_by_ExampleBot"
    assert created["sticker_type"] == "custom_emoji" and created["title"] == "My Pack емодзі"
    assert json.loads(created["stickers"])[0]["emoji_list"] == ["😴", "💤"]
    assert remote.url == "https://t.me/addemoji/mypack_emoji_by_ExampleBot"
    assert remote.extra["custom_emoji_id"]


def test_telegram_same_sticker_twice_does_not_duplicate(files: dict[str, Path]) -> None:
    fake = FakeTelegram()
    publisher = TelegramPublisher(_config(), Channel.TELEGRAM_STICKER, fake)
    first = publisher.publish(_request(files["a"]))
    again = publisher.publish(_request(files["a"]))
    assert len(fake.sets["mypack_by_ExampleBot"]["stickers"]) == 1
    assert again.extra["file_unique_id"] == first.extra["file_unique_id"]


def test_telegram_owner_never_wrote_to_bot(files: dict[str, Path]) -> None:
    fake = FakeTelegram(owner_started=False)
    publisher = TelegramPublisher(_config(), Channel.TELEGRAM_STICKER, fake)
    with pytest.raises(OwnerNotStartedError, match="/start"):
        publisher.publish(_request(files["a"]))
    assert fake.sets == {}


def test_telegram_errors_are_explained_without_token(files: dict[str, Path]) -> None:
    fake = FakeTelegram()
    publisher = TelegramPublisher(_config(), Channel.TELEGRAM_STICKER, fake)
    fake.fail["createNewStickerSet"] = _err(400, "Bad Request: STICKER_VIDEO_LONG")
    with pytest.raises(PublishError, match="3 секунди"):
        publisher.publish(_request(files["a"]))
    fake.fail["createNewStickerSet"] = _err(429, "Too Many Requests", parameters={"retry_after": 7})
    with pytest.raises(PublishError, match="за 7 с"):
        publisher.publish(_request(files["a"]))
    fake.fail["createNewStickerSet"] = _err(400, f"Bad Request: weird {TG_TOKEN}")
    with pytest.raises(PublishError) as caught:
        publisher.publish(_request(files["a"]))
    assert TG_TOKEN not in str(caught.value) and "***" in str(caught.value)
    wrong = TelegramPublisher(
        _config(telegram_token="1:" + "x" * 30), Channel.TELEGRAM_STICKER, fake
    )
    with pytest.raises(PublishError, match="TELEGRAM_BOT_TOKEN"):
        wrong.publish(_request(files["a"]))
    with pytest.raises(PublishError, match="емодзі"):
        publisher.publish(_request(files["a"], emoji=()))


def test_telegram_remove(files: dict[str, Path]) -> None:
    fake = FakeTelegram()
    publisher = TelegramPublisher(_config(), Channel.TELEGRAM_STICKER, fake)
    remote = publisher.publish(_request(files["a"]))
    publisher.remove(remote)
    assert "deleteStickerFromSet" in fake.methods()
    assert not publisher.exists(remote)
    publisher.remove(remote)  # already gone: no error
    assert fake.methods().count("deleteStickerFromSet") == 1


def test_telegram_not_configured() -> None:
    with pytest.raises(NotConfiguredError) as caught:
        TelegramPublisher(PublishConfig(), Channel.TELEGRAM_STICKER, FakeTelegram())
    assert caught.value.missing == ("TELEGRAM_BOT_TOKEN", "TELEGRAM_OWNER_ID")


def test_telegram_info_has_no_token() -> None:
    info = telegram_info(_config(), FakeTelegram())
    assert info["bot"] == "ExampleBot" and info["pack"] == "mypack_by_ExampleBot"
    assert info["emoji_pack_url"].endswith("mypack_emoji_by_ExampleBot")
    assert TG_TOKEN not in json.dumps(info, ensure_ascii=False)
    assert telegram_info(PublishConfig())["configured"] is False


class StrictTelegram(FakeTelegram):
    """Like the live Bot API on 2026-10-09: a multipart body with no parts gets a bare 400."""

    requests: list[tuple[str, str, dict[str, str], bytes | None]]

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> Response:
        if not hasattr(self, "requests"):
            self.requests = []
        self.requests.append((method, url, headers, body))
        if headers.get("Content-Type", "").startswith("multipart") and not parse_multipart(
            headers, body
        ):
            return Response(400, b"Bad Request")
        return super().__call__(method, url, headers, body)


def test_telegram_get_me_without_parameters_sends_no_body() -> None:
    fake = StrictTelegram()
    info = telegram_info(_config(), fake)
    assert "error" not in info and info["bot"] == "ExampleBot"
    method, url, headers, body = fake.requests[0]
    assert url.endswith("/getMe") and body is None and "Content-Type" not in headers
    # Methods with fields keep the multipart body.
    TelegramPublisher(_config(), Channel.TELEGRAM_STICKER, fake).get_set()
    method, url, headers, body = fake.requests[-1]
    assert method == "POST" and url.endswith("/getStickerSet")
    assert headers["Content-Type"].startswith("multipart/form-data") and body


def test_telegram_empty_multipart_is_what_fails() -> None:
    fake = StrictTelegram()
    body, kind = multipart({})
    response = fake(
        "POST", f"https://api.telegram.org/bot{TG_TOKEN}/getMe", {"Content-Type": kind}, body
    )
    assert response.status == 400 and response.json() is None


def test_telegram_non_json_error_shows_part_of_body_without_token() -> None:
    def transport(method: str, url: str, headers: dict[str, str], body: bytes | None) -> Response:
        return Response(502, f"<html> Bad   Gateway\n{TG_TOKEN} {'x' * 500}</html>".encode())

    info = telegram_info(_config(), transport)
    error = info["error"]
    assert "HTTP 502" in error and "Bad Gateway" in error
    assert TG_TOKEN not in error and len(error) < 300


def test_body_snippet() -> None:
    assert body_snippet(b"  a\n\n b  ") == "a b"
    assert len(body_snippet(b"y" * 1000, 50)) == 50
    assert body_snippet(b"\xff ok") == "� ok"


# Discord


def test_discord_emoji_and_sticker(files: dict[str, Path]) -> None:
    fake = FakeDiscord()
    emoji = DiscordPublisher(_config(), Channel.DISCORD_EMOJI, fake)
    remote = emoji.publish(_request(files["webp"]))
    post = fake.requests("POST", "/emojis")[0][2]
    assert post["name"] == "example_emote" and post["image"].startswith("data:image/webp;base64,")
    assert post["roles"] == []
    assert remote.url.endswith(f"{remote.remote_id}.webp?animated=true")
    assert emoji.exists(remote)

    sticker = DiscordPublisher(_config(), Channel.DISCORD_STICKER, fake)
    made = sticker.publish(_request(files["png"]))
    form = fake.requests("POST", "/stickers")[0][2]
    assert form["name"] == "example_emote" and form["tags"] == "😴"
    assert form["description"] == "дівчина позіхає"
    assert form["file"]["filename"] == "discord_sticker.png"
    assert made.remote_id == fake.stickers[0]["id"]


def test_discord_webp_refused_falls_back_to_gif(files: dict[str, Path]) -> None:
    fake = FakeDiscord(accept_webp=False)
    emoji = DiscordPublisher(_config(), Channel.DISCORD_EMOJI, fake)
    emoji.publish(_request(files["webp"]))
    kinds = [c[2]["image"].split(";")[0] for c in fake.requests("POST", "/emojis")]
    assert kinds == ["data:image/webp", "data:image/gif"]
    assert fake.emojis[0]["kind"] == "image/gif"
    gif = webp_to_gif(files["webp"].read_bytes())
    with Image.open(io.BytesIO(gif)) as image:
        assert image.format == "GIF" and image.n_frames == 4


def test_discord_no_free_slots_is_explained(files: dict[str, Path]) -> None:
    full = [{"id": str(i), "name": f"e{i}", "animated": True} for i in range(50)]
    fake = FakeDiscord(emojis=full)
    emoji = DiscordPublisher(_config(), Channel.DISCORD_EMOJI, fake)
    with pytest.raises(
        PublishError, match="немає вільних слотів анімованих емодзі: зайнято 50 з 50"
    ):
        emoji.publish(_request(files["webp"]))
    assert fake.requests("POST") == []  # refused before the upload
    stickers = [{"id": str(i), "name": f"s{i}"} for i in range(15)]
    boosted = FakeDiscord(premium_tier=1, stickers=stickers)
    sticker = DiscordPublisher(_config(), Channel.DISCORD_STICKER, boosted)
    with pytest.raises(PublishError, match="стікерів: зайнято 15 з 15 \\(рівень бусту 1\\)"):
        sticker.publish(_request(files["png"]))
    # Static emoji do not take animated slots.
    statics = [{"id": str(i), "name": f"e{i}", "animated": False} for i in range(50)]
    DiscordPublisher(_config(), Channel.DISCORD_EMOJI, FakeDiscord(emojis=statics)).publish(
        _request(files["webp"])
    )


def test_discord_slot_error_from_api_is_explained(files: dict[str, Path]) -> None:
    fake = FakeDiscord()
    fake.fail["POST emojis"] = Response(400, b'{"code": 30008, "message": "Maximum emojis"}')
    emoji = DiscordPublisher(_config(), Channel.DISCORD_EMOJI, fake)
    with pytest.raises(PublishError, match="закінчились слоти емодзі"):
        emoji.publish(_request(files["webp"]))
    fake.fail["GET " + GUILD] = Response(403, b'{"code": 50013, "message": "Missing Permissions"}')
    with pytest.raises(PublishError, match="Create/Manage Expressions"):
        emoji.publish(_request(files["webp"]))


def test_discord_invalid_names(files: dict[str, Path]) -> None:
    fake = FakeDiscord()
    emoji = DiscordPublisher(_config(), Channel.DISCORD_EMOJI, fake)
    for bad, reason in (
        ("сонна", "латинські"),
        ("a", "від 2 до 32"),
        ("x" * 33, "від 2 до 32"),
        ("has space", "латинські"),
    ):
        with pytest.raises(PublishError, match=reason):
            emoji.publish(_request(files["webp"], name=bad))
    sticker = DiscordPublisher(_config(), Channel.DISCORD_STICKER, fake)
    with pytest.raises(PublishError, match="від 2 до 30"):
        sticker.publish(_request(files["png"], name="x" * 31))
    assert fake.requests("POST") == []


def test_discord_taken_name_gets_suffix(files: dict[str, Path]) -> None:
    fake = FakeDiscord(emojis=[{"id": "1", "name": "example_emote", "animated": True}])
    remote = DiscordPublisher(_config(), Channel.DISCORD_EMOJI, fake).publish(
        _request(files["webp"])
    )
    assert remote.extra["name"] == "example_emote_2"


def test_discord_remove_and_gone(files: dict[str, Path]) -> None:
    fake = FakeDiscord()
    emoji = DiscordPublisher(_config(), Channel.DISCORD_EMOJI, fake)
    remote = emoji.publish(_request(files["webp"]))
    emoji.remove(remote)
    assert fake.emojis == [] and not emoji.exists(remote)
    emoji.remove(remote)  # 404 is fine
    sticker = DiscordPublisher(_config(), Channel.DISCORD_STICKER, fake)
    made = sticker.publish(_request(files["png"]))
    sticker.remove(made)
    assert fake.stickers == []


def test_discord_errors_hide_token(files: dict[str, Path]) -> None:
    fake = FakeDiscord()
    fake.fail["POST emojis"] = Response(
        500, json.dumps({"code": 0, "message": f"x {DC_TOKEN}"}).encode()
    )
    with pytest.raises(PublishError) as caught:
        DiscordPublisher(_config(), Channel.DISCORD_EMOJI, fake).publish(_request(files["webp"]))
    assert DC_TOKEN not in str(caught.value)
    bad = DiscordPublisher(_config(discord_token="wrong"), Channel.DISCORD_EMOJI, fake)
    with pytest.raises(PublishError, match="DISCORD_BOT_TOKEN"):
        bad.publish(_request(files["webp"]))
    with pytest.raises(NotConfiguredError, match="DISCORD_GUILD_ID"):
        DiscordPublisher(_config(discord_guild_id=""), Channel.DISCORD_EMOJI, fake)


def test_discord_info() -> None:
    fake = FakeDiscord(premium_tier=2, emojis=[{"id": "1", "name": "a", "animated": True}])
    info = discord_info(_config(), fake)
    assert info["guild"] == "My Server" and info["tier"] == 2
    assert info["emoji_slots"] == (1, 150) and info["sticker_slots"] == (0, 30)
    assert DC_TOKEN not in json.dumps(info)
    assert all(c[0] == "GET" for c in fake.calls)  # read only


# Twitch


def test_twitch_archive(tmp_path: Path) -> None:
    for side in (28, 56, 112, 224):
        _animated(tmp_path / f"twitch_gif_{side}.gif", "GIF")
    data = build_archive(tmp_path, "MyChannel", "Example")
    archive = zipfile.ZipFile(io.BytesIO(data))
    assert sorted(archive.namelist()) == [
        "README.txt",
        "auto/twitch_gif_224.gif",
        "manual/twitch_gif_112.gif",
        "manual/twitch_gif_28.gif",
        "manual/twitch_gif_56.gif",
    ]
    readme = archive.read("README.txt").decode()
    assert "https://dashboard.twitch.tv/u/mychannel/viewer-rewards/emotes" in readme
    assert "Example" in readme and "twitch_gif_224.gif" in readme
    assert emotes_url() == "https://dashboard.twitch.tv/viewer-rewards/emotes"
    with pytest.raises(FileNotFoundError):
        build_archive(tmp_path / "missing")


# Names, emoji, helpers


def test_names() -> None:
    assert discord_name("Сонна дівчина!.png") == "Sonna_divchyna_png"
    assert discord_name("a") == "a_sticker"
    assert discord_name("💤") == "sticker"
    assert len(discord_name("x" * 80)) == 32
    assert discord_name_problem("ok_name") is None
    assert unique_name("cat", {"Cat", "cat_2"}, 32) == "cat_3"
    assert telegram_set_name("My Pack", "ExampleBot") == "My_Pack_by_ExampleBot"
    assert telegram_set_name("1pack", "Bot").startswith("s1pack")
    assert telegram_set_name("a__b", "Bot") == "a_b_by_Bot"
    assert len(telegram_set_name("x" * 100, "ExampleBot")) == 64
    assert telegram_set_name_problem("a__b_by_Bot", "Bot")
    assert telegram_set_name_problem("ok_by_bot", "Bot") is None


def test_emoji_rules() -> None:
    assert suggest_emoji("The girl yawns slowly, covering her mouth with her fist") == ("😴",)
    assert suggest_emoji("the character waves the right hand") == ("👋",)
    assert suggest_emoji("something unusual") == ("🙂",)
    assert split_emoji("😴 💤, 👋🏽 ok ❤️ 😴") == ("😴", "💤", "👋🏽", "❤️")
    assert split_emoji("no emoji here") == ()
    assert split_emoji("🇺🇦") == ("🇺🇦",)


def test_helpers_hide_secrets() -> None:
    assert redact(f"a {TG_TOKEN} b", TG_TOKEN, None) == "a *** b"
    assert TG_TOKEN not in scrub(f"https://api.telegram.org/bot{TG_TOKEN}/getMe")
    body, kind = multipart({"a": "1"})
    assert kind.startswith("multipart/form-data; boundary=") and b'name="a"' in body
    assert TG_TOKEN not in repr(_config()) and DC_TOKEN not in repr(_config())


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TG_TOKEN)
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "not-a-number")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", DC_TOKEN)
    monkeypatch.setenv("DISCORD_GUILD_ID", GUILD)
    monkeypatch.delenv("TELEGRAM_PACK_SLUG", raising=False)
    config = PublishConfig.from_env()
    assert config.telegram_missing() == ["TELEGRAM_OWNER_ID (число)"]
    assert config.discord_missing() == [] and config.telegram_pack_slug == "stickermotion"
    monkeypatch.setenv("TELEGRAM_OWNER_ID", "4242")
    assert PublishConfig.from_env().telegram_owner_id == 4242


def test_telegram_raw_error_kept_for_log_without_token(files: dict[str, Path]) -> None:
    """The user sees a mapped message, the log gets the full Bot API answer."""
    fake = FakeTelegram()
    publisher = TelegramPublisher(_config(), Channel.TELEGRAM_EMOJI, fake)
    fake.fail["createNewStickerSet"] = _err(
        400, f"Bad Request: STICKER_VIDEO_BIG emoji too big {TG_TOKEN}"
    )
    with pytest.raises(PublishError, match="64 КБ") as caught:
        publisher.publish(_request(files["a"]))
    raw = caught.value.raw
    assert raw.startswith("createNewStickerSet HTTP 400: ")
    assert "STICKER_VIDEO_BIG emoji too big" in raw and '"error_code": 400' in raw
    assert TG_TOKEN not in raw and "***" in raw
    assert PublishError("x").raw == ""
