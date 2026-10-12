"""Telegram: a bot fills a video sticker pack and a custom emoji pack through the Bot API
(ADR-017). The pack belongs to the user ``TELEGRAM_OWNER_ID``.

Methods (Bot API, checked 2026-10-09): ``getMe``, ``getStickerSet``,
``createNewStickerSet`` (``user_id``, ``name``, ``title``, ``stickers`` as InputSticker
list, ``sticker_type``), ``addStickerToSet`` (``user_id``, ``name``, ``sticker``),
``deleteStickerFromSet`` (``sticker`` = file_id). Files go as multipart parts referenced by
``attach://<name>``. A set name ends with ``_by_<bot username>``; the first sticker creates
the set, the next ones are added. Only outgoing calls: the bot's updates are never read,
so another application that already runs the same bot is not disturbed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from stickermotion.publish.base import Channel, PublishRequest, Remote
from stickermotion.publish.config import PublishConfig
from stickermotion.publish.http import (
    FilePart,
    NotConfiguredError,
    PublishError,
    Transport,
    body_snippet,
    multipart,
    raw_text,
    redact,
    urllib_transport,
)
from stickermotion.publish.naming import telegram_set_name

API = "https://api.telegram.org"
REGULAR = "regular"
CUSTOM_EMOJI = "custom_emoji"
MAX_STICKERS = {REGULAR: 120, CUSTOM_EMOJI: 200}


class OwnerNotStartedError(PublishError):
    """The pack owner never wrote to the bot, so Telegram does not know them for this bot."""


class SetMissingError(PublishError):
    """The sticker set does not exist (yet)."""


_OWNER_UNKNOWN = (
    "PEER_ID_INVALID",
    "USER_ID_INVALID",
    "user not found",
    "chat not found",
    "bot was blocked",
    "user is deactivated",
)
_SET_MISSING = ("STICKERSET_INVALID", "sticker set not found", "STICKERSET_NOT_FOUND")
_STICKER_GONE = ("STICKER_INVALID", "sticker not found", "STICKERSET_INVALID")


def _has(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in needles)


@dataclass
class TelegramClient:
    token: str = field(repr=False)
    transport: Transport = urllib_transport

    def call(
        self, method: str, fields: dict[str, Any] | None = None, files: list[FilePart] | None = None
    ) -> Any:
        values = {
            k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            for k, v in (fields or {}).items()
        }
        url = f"{API}/bot{self.token}/{method}"
        if values or files:
            body, content_type = multipart(values, files or [])
            response = self.transport("POST", url, {"Content-Type": content_type}, body)
        else:
            # A multipart body with no parts (only the closing boundary) gets a bare
            # HTTP 400 from Telegram, so parameterless methods such as getMe go bodiless.
            response = self.transport("GET", url, {}, None)
        data = response.json()
        if isinstance(data, dict) and data.get("ok"):
            return data.get("result")
        description = ""
        code = response.status
        retry_after = None
        raw = ""
        if isinstance(data, dict):
            description = str(data.get("description") or "")
            code = int(data.get("error_code") or code)
            retry_after = (data.get("parameters") or {}).get("retry_after")
        else:
            raw = scrub(redact(body_snippet(response.body), self.token))
        error = self.explain(method, code, redact(description, self.token), retry_after, raw)
        # The full Bot API answer goes to the publication log: the user-facing
        # message maps several errors to one text and hides the real cause.
        full = scrub(redact(raw_text(response.body), self.token))
        error.raw = f"{method} HTTP {response.status}: {full}"
        raise error

    def explain(
        self,
        method: str,
        code: int,
        description: str,
        retry_after: Any = None,
        raw: str = "",
    ) -> PublishError:
        """Bot API error to a user-facing message (Ukrainian)."""
        if code == 401 or (code == 404 and not description):
            return PublishError(
                "Telegram не прийняв токен бота (401). Перевірте TELEGRAM_BOT_TOKEN."
            )
        if code == 429:
            wait = f" за {int(retry_after)} с" if retry_after else " за хвилину"
            return PublishError(f"Telegram тимчасово обмежив запити. Спробуйте{wait}.")
        if method == "getStickerSet" and _has(description, _SET_MISSING):
            return SetMissingError(description)
        if _has(description, _OWNER_UNKNOWN):
            return OwnerNotStartedError(
                "Telegram не знає власника для цього бота. Напишіть боту /start з акаунта "
                "власника (TELEGRAM_OWNER_ID) і повторіть публікацію."
            )
        known = (
            (("STICKERS_TOO_MUCH", "too many stickers"), "У паку вже максимум стікерів."),
            (
                ("SHORT_NAME_OCCUPIED", "name is already occupied"),
                "Ім'я пака вже зайняте. Змініть TELEGRAM_PACK_SLUG.",
            ),
            (
                ("STICKER_VIDEO_LONG",),
                "Відео стікера довше за 3 секунди, Telegram його не приймає.",
            ),
            (
                ("STICKER_VIDEO_BIG", "file is too big", "STICKER_FILE_BIG"),
                "Файл стікера завеликий для Telegram (стікер до 256 КБ, емодзі до 64 КБ).",
            ),
            (
                (
                    "STICKER_VIDEO_NOWEBM",
                    "STICKER_PNG_DIMENSIONS",
                    "wrong file type",
                    "STICKER_FILE_INVALID",
                    "STICKER_VIDEO_DIMENSIONS",
                ),
                "Telegram не прийняв файл стікера (формат або розмір кадру).",
            ),
            (
                ("STICKER_EMOJI_INVALID", "invalid sticker emojis", "EMOJI_INVALID"),
                "Telegram не прийняв емодзі. Вкажіть звичайні емодзі, від 1 до 20.",
            ),
            (
                ("STICKERSET_INVALID",),
                "Пак не знайдено в Telegram. Можливо, його видалили; повторіть публікацію.",
            ),
        )
        for needles, message in known:
            if _has(description, needles):
                return PublishError(message)
        if code == 403:
            return OwnerNotStartedError(
                "Бот не може звернутися до власника. Напишіть боту /start і повторіть."
            )
        detail = description or f"HTTP {code}"
        if raw and not description:
            detail = f"{detail} ({raw})"
        return PublishError(f"Telegram відповів помилкою: {detail}")


class TelegramPublisher:
    """One pack kind: the video sticker pack or the custom emoji pack."""

    def __init__(
        self,
        config: PublishConfig,
        channel: Channel = Channel.TELEGRAM_STICKER,
        transport: Transport = urllib_transport,
    ) -> None:
        missing = config.telegram_missing()
        if missing:
            raise NotConfiguredError("Telegram", missing)
        if channel not in (Channel.TELEGRAM_STICKER, Channel.TELEGRAM_EMOJI):
            raise ValueError(channel)
        self.channel = channel
        self.config = config
        self.client = TelegramClient(config.telegram_token, transport)
        self.kind = CUSTOM_EMOJI if channel is Channel.TELEGRAM_EMOJI else REGULAR
        self._username: str | None = None

    # Names

    @property
    def bot_username(self) -> str:
        if self._username is None:
            me = self.client.call("getMe") or {}
            username = str(me.get("username") or "")
            if not username:
                raise PublishError("Telegram не повернув ім'я бота (getMe).")
            self._username = username
        return self._username

    @property
    def set_name(self) -> str:
        suffix = "emoji" if self.kind == CUSTOM_EMOJI else ""
        try:
            return telegram_set_name(self.config.telegram_pack_slug, self.bot_username, suffix)
        except ValueError as exc:
            raise PublishError(f"{exc} Змініть TELEGRAM_PACK_SLUG.") from None

    @property
    def title(self) -> str:
        title = self.config.telegram_pack_title
        if self.kind == CUSTOM_EMOJI:
            title = f"{title} емодзі"
        return title[:64]

    def set_url(self, name: str | None = None) -> str:
        path = "addemoji" if self.kind == CUSTOM_EMOJI else "addstickers"
        return f"https://t.me/{path}/{name or self.set_name}"

    # State

    def get_set(self) -> dict[str, Any] | None:
        try:
            result = self.client.call("getStickerSet", {"name": self.set_name})
        except SetMissingError:
            return None
        return result if isinstance(result, dict) else None

    def _input_sticker(self, request: PublishRequest, attach: str) -> dict[str, Any]:
        emoji = list(request.emoji[:20])
        if not emoji:
            raise PublishError("Вкажіть хоча б один емодзі для стікера.")
        return {"sticker": f"attach://{attach}", "format": "video", "emoji_list": emoji}

    def publish(self, request: PublishRequest) -> Remote:
        if not request.file.is_file():
            raise PublishError(f"Немає файлу {request.file.name}. Перегенеруйте стікер.")
        name = self.set_name
        part = FilePart("sticker0", request.file.name, request.file.read_bytes(), "video/webm")
        sticker = self._input_sticker(request, part.field)
        owner = self.config.telegram_owner_id
        existing = self.get_set()
        before = {s.get("file_unique_id") for s in (existing or {}).get("stickers", [])}
        if existing is None:
            self.client.call(
                "createNewStickerSet",
                {
                    "user_id": str(owner),
                    "name": name,
                    "title": self.title,
                    "stickers": [sticker],
                    "sticker_type": self.kind,
                },
                [part],
            )
        else:
            if len(before) >= MAX_STICKERS[self.kind]:
                raise PublishError(
                    f"У паку вже {len(before)} стікерів, це максимум. Видаліть зайві або "
                    "задайте інший TELEGRAM_PACK_SLUG."
                )
            self.client.call(
                "addStickerToSet", {"user_id": str(owner), "name": name, "sticker": sticker}, [part]
            )
        after = self.get_set() or {}
        stickers = after.get("stickers", [])
        added = [s for s in stickers if s.get("file_unique_id") not in before]
        # The Bot API leaves the set unchanged when exactly the same sticker is added again.
        item = added[-1] if added else (stickers[-1] if stickers else None)
        if item is None:
            raise PublishError("Telegram прийняв стікер, але пак порожній. Спробуйте ще раз.")
        extra = {
            "set_name": name,
            "file_unique_id": item.get("file_unique_id", ""),
            "custom_emoji_id": item.get("custom_emoji_id", ""),
        }
        return Remote(str(item.get("file_id", "")), self.set_url(name), extra)

    def exists(self, remote: Remote) -> bool:
        found = self.get_set()
        if found is None:
            return False
        unique = remote.extra.get("file_unique_id")
        return any(
            s.get("file_unique_id") == unique or s.get("file_id") == remote.remote_id
            for s in found.get("stickers", [])
        )

    def remove(self, remote: Remote) -> None:
        found = self.get_set()
        if found is None:
            return
        unique = remote.extra.get("file_unique_id")
        match = next(
            (
                s
                for s in found.get("stickers", [])
                if s.get("file_unique_id") == unique or s.get("file_id") == remote.remote_id
            ),
            None,
        )
        if match is None:
            return
        try:
            self.client.call("deleteStickerFromSet", {"sticker": str(match.get("file_id"))})
        except PublishError as exc:
            if not _has(str(exc), _STICKER_GONE) and "не знайдено" not in str(exc):
                raise


def telegram_info(config: PublishConfig, transport: Transport = urllib_transport) -> dict[str, Any]:
    """Read-only view for the settings page: bot name and pack links, never the token."""
    missing = config.telegram_missing()
    info: dict[str, Any] = {
        "configured": not missing,
        "missing": missing,
        "title": config.telegram_pack_title,
        "slug": config.telegram_pack_slug,
        "owner_id": config.telegram_owner_id,
    }
    if missing:
        return info
    publisher = TelegramPublisher(config, Channel.TELEGRAM_STICKER, transport)
    try:
        info["bot"] = publisher.bot_username
        info["pack"] = publisher.set_name
        info["pack_url"] = publisher.set_url()
        emoji = TelegramPublisher(config, Channel.TELEGRAM_EMOJI, transport)
        emoji._username = publisher.bot_username
        info["emoji_pack"] = emoji.set_name
        info["emoji_pack_url"] = emoji.set_url()
    except PublishError as exc:
        info["error"] = redact(str(exc), *config.secrets)
    return info


_TOKEN_SHAPE = re.compile(r"\d{5,}:[A-Za-z0-9_-]{20,}")


def scrub(text: str) -> str:
    """Remove anything shaped like a bot token (``123456:ABC...``) from free text."""
    return _TOKEN_SHAPE.sub("***", text)
