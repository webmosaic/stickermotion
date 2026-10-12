"""Discord: a bot uploads an animated emoji and a sticker to one server (guild)
over REST, without a gateway connection (ADR-017).

Endpoints (Discord API v10, checked 2026-10-09): ``GET /guilds/{id}`` (``premium_tier``,
``emojis``, ``stickers``), ``POST /guilds/{id}/emojis`` (JSON ``name``, ``image`` as a
data URI, ``roles``), ``DELETE /guilds/{id}/emojis/{emoji}``, ``POST /guilds/{id}/stickers``
(multipart ``name`` 2-30, ``description`` empty or 2-100, ``tags`` up to 200, ``file`` PNG /
APNG / GIF up to 512 KiB), ``DELETE /guilds/{id}/stickers/{sticker}``. The emoji page lists
WebP among accepted upload formats, while the general image data section names only
JPG / GIF / PNG, so an animated WebP that is refused is retried once as a GIF.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageSequence

from stickermotion.publish.base import Channel, PublishRequest, Remote
from stickermotion.publish.config import PublishConfig
from stickermotion.publish.http import (
    FilePart,
    NotConfiguredError,
    PublishError,
    Transport,
    multipart,
    redact,
    urllib_transport,
)
from stickermotion.publish.naming import (
    DISCORD_EMOJI_NAME,
    DISCORD_STICKER_NAME,
    discord_name_problem,
    unique_name,
)

API = "https://discord.com/api/v10"
USER_AGENT = "DiscordBot (https://github.com/stickermotion, 0.1)"
EMOJI_MAX_BYTES = 256 * 1024
STICKER_MAX_BYTES = 512 * 1024

# Slots per Server Boost level (premium_tier 0-3): static and animated emoji are separate
# pools of the same size; stickers have their own pool.
EMOJI_SLOTS = {0: 50, 1: 100, 2: 150, 3: 250}
STICKER_SLOTS = {0: 5, 1: 15, 2: 30, 3: 60}

_INVALID_IMAGE = {50035, 50046, 50110, 50123}


class DiscordApiError(PublishError):
    """Discord REST error; ``str(exc)`` is the user-facing message."""

    def __init__(self, status: int, code: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code


@dataclass
class DiscordClient:
    token: str = field(repr=False)
    transport: Transport = urllib_transport

    def call(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        body: bytes | None = None,
        content_type: str = "",
    ) -> Any:
        headers = {"Authorization": f"Bot {self.token}", "User-Agent": USER_AGENT}
        if payload is not None:
            body = json.dumps(payload).encode()
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        response = self.transport(method, f"{API}{path}", headers, body)
        if 200 <= response.status < 300:
            return response.json()
        data = response.json()
        code = int(data.get("code") or 0) if isinstance(data, dict) else 0
        message = str(data.get("message") or "") if isinstance(data, dict) else ""
        if isinstance(data, dict) and data.get("errors"):
            message += " " + json.dumps(data["errors"], ensure_ascii=False)[:400]
        retry_after = data.get("retry_after") if isinstance(data, dict) else None
        raise DiscordApiError(
            response.status,
            code,
            self.explain(response.status, code, redact(message, self.token), retry_after),
        )

    @staticmethod
    def explain(status: int, code: int, message: str, retry_after: Any = None) -> str:
        """Discord error to a user-facing message (Ukrainian)."""
        if status == 401:
            return "Discord не прийняв токен бота (401). Перевірте DISCORD_BOT_TOKEN."
        if status == 429:
            wait = f" за {float(retry_after):.0f} с" if retry_after is not None else " за хвилину"
            return f"Discord тимчасово обмежив запити. Спробуйте{wait}."
        known = {
            30008: "На сервері закінчились слоти емодзі.",
            30039: "На сервері закінчились слоти стікерів.",
            50001: "Бот не має доступу до сервера. Перевірте DISCORD_GUILD_ID і чи бот на сервері.",
            50013: "Боту бракує права Create/Manage Expressions на сервері.",
            50045: "Файл завеликий для Discord.",
            50138: "Discord не зміг стиснути емодзі до 256 КБ.",
            10004: "Сервер не знайдено. Перевірте DISCORD_GUILD_ID.",
            170003: "Discord: частота кадрів стікера завелика.",
            170006: "Discord: частота кадрів стікера поза допустимими межами.",
            170007: "Discord: стікер довший за 5 секунд.",
        }
        if code in known:
            return known[code]
        if status == 403:
            return "Боту бракує прав на сервері Discord (403)."
        if status == 404:
            return "Discord не знайшов сервер або елемент (404). Перевірте DISCORD_GUILD_ID."
        detail = message.strip() or f"HTTP {status}"
        return f"Discord відповів помилкою {code or status}: {detail}"


def webp_to_gif(data: bytes, max_bytes: int = EMOJI_MAX_BYTES) -> bytes:
    """Animated WebP to a looping GIF with binary alpha that fits ``max_bytes``."""
    frames: list[Image.Image] = []
    durations: list[int] = []
    with Image.open(io.BytesIO(data)) as image:
        for frame in ImageSequence.Iterator(image):
            frames.append(frame.convert("RGBA"))
            durations.append(int(frame.info.get("duration") or 40))
    for colors in (256, 128, 64, 32):
        out: list[Image.Image] = []
        for frame in frames:
            alpha = frame.getchannel("A").point(lambda a: 255 if a >= 128 else 0)
            paletted = frame.convert("RGB").quantize(colors - 1, dither=Image.Dither.NONE)
            mask = Image.eval(alpha, lambda a: 255 - a)
            paletted.paste(colors - 1, mask=mask)
            paletted.info["transparency"] = colors - 1
            out.append(paletted)
        buffer = io.BytesIO()
        out[0].save(
            buffer,
            "GIF",
            save_all=True,
            append_images=out[1:],
            duration=durations,
            loop=0,
            disposal=2,
            transparency=colors - 1,
        )
        if buffer.tell() <= max_bytes:
            return buffer.getvalue()
    raise PublishError("Не вдалося вмістити GIF-версію емодзі в 256 КБ.")


def _data_uri(data: bytes, kind: str) -> str:
    return f"data:{kind};base64,{base64.b64encode(data).decode()}"


class DiscordPublisher:
    """Animated emoji (``discord_emoji.webp``) or sticker (``discord_sticker.png``)."""

    def __init__(
        self,
        config: PublishConfig,
        channel: Channel = Channel.DISCORD_EMOJI,
        transport: Transport = urllib_transport,
    ) -> None:
        missing = config.discord_missing()
        if missing:
            raise NotConfiguredError("Discord", missing)
        if channel not in (Channel.DISCORD_EMOJI, Channel.DISCORD_STICKER):
            raise ValueError(channel)
        self.channel = channel
        self.guild_id = config.discord_guild_id
        self.client = DiscordClient(config.discord_token, transport)

    @property
    def is_emoji(self) -> bool:
        return self.channel is Channel.DISCORD_EMOJI

    @property
    def limits(self) -> tuple[int, int]:
        return DISCORD_EMOJI_NAME if self.is_emoji else DISCORD_STICKER_NAME

    def guild(self) -> dict[str, Any]:
        data = self.client.call("GET", f"/guilds/{self.guild_id}")
        return data if isinstance(data, dict) else {}

    def slots(self, guild: dict[str, Any]) -> tuple[int, int]:
        """(used, total) for this channel's pool on the server."""
        tier = int(guild.get("premium_tier") or 0)
        if self.is_emoji:
            used = sum(1 for e in guild.get("emojis") or [] if e.get("animated"))
            return used, EMOJI_SLOTS.get(tier, EMOJI_SLOTS[3])
        stickers = guild.get("stickers")
        if stickers is None:
            stickers = self.client.call("GET", f"/guilds/{self.guild_id}/stickers") or []
        return len(stickers), STICKER_SLOTS.get(tier, STICKER_SLOTS[3])

    def _taken(self, guild: dict[str, Any]) -> set[str]:
        items = guild.get("emojis") if self.is_emoji else guild.get("stickers")
        return {str(i.get("name") or "") for i in items or []}

    def publish(self, request: PublishRequest) -> Remote:
        problem = discord_name_problem(request.name, self.limits)
        if problem:
            raise PublishError(f"Ім'я для Discord: {problem}")
        if not request.file.is_file():
            raise PublishError(f"Немає файлу {request.file.name}. Перегенеруйте стікер.")
        guild = self.guild()
        used, total = self.slots(guild)
        if used >= total:
            what = "анімованих емодзі" if self.is_emoji else "стікерів"
            tier = int(guild.get("premium_tier") or 0)
            raise PublishError(
                f"На сервері немає вільних слотів {what}: зайнято {used} з {total} (рівень "
                f"бусту {tier}). Видаліть старі або підніміть рівень бусту."
            )
        name = unique_name(request.name, self._taken(guild), self.limits[1])
        if self.is_emoji:
            item = self._emoji(request.file, name)
            url = f"https://cdn.discordapp.com/emojis/{item['id']}.webp?animated=true"
        else:
            item = self._sticker(request, name)
            url = f"https://media.discordapp.net/stickers/{item['id']}.png"
        return Remote(
            str(item["id"]), url, {"name": item.get("name", name), "guild": self.guild_id}
        )

    def _emoji(self, path: Path, name: str) -> dict[str, Any]:
        data = path.read_bytes()
        kind = "image/webp" if path.suffix == ".webp" else "image/gif"
        endpoint = f"/guilds/{self.guild_id}/emojis"
        try:
            item = self.client.call(
                "POST", endpoint, {"name": name, "image": _data_uri(data, kind), "roles": []}
            )
        except DiscordApiError as exc:
            if kind != "image/webp" or exc.code not in _INVALID_IMAGE:
                raise
            # WebP refused: the same animation as GIF (binary alpha).
            gif = webp_to_gif(data)
            item = self.client.call(
                "POST", endpoint, {"name": name, "image": _data_uri(gif, "image/gif"), "roles": []}
            )
            item = {**item, "format": "gif"}
        return item

    def _sticker(self, request: PublishRequest, name: str) -> dict[str, Any]:
        data = request.file.read_bytes()
        if len(data) > STICKER_MAX_BYTES:
            raise PublishError("Файл стікера більший за 512 КБ, Discord його не прийме.")
        description = request.description.strip()[:100]
        if len(description) < 2:
            description = ""
        tag = request.emoji[0] if request.emoji else "🙂"
        body, content_type = multipart(
            {"name": name, "description": description, "tags": tag},
            [FilePart("file", request.file.name, data, "image/png")],
        )
        return self.client.call(
            "POST", f"/guilds/{self.guild_id}/stickers", body=body, content_type=content_type
        )

    def exists(self, remote: Remote) -> bool:
        kind = "emojis" if self.is_emoji else "stickers"
        try:
            self.client.call("GET", f"/guilds/{self.guild_id}/{kind}/{remote.remote_id}")
        except DiscordApiError as exc:
            if exc.status == 404 or exc.code in (10014, 10060):
                return False
            raise
        return True

    def remove(self, remote: Remote) -> None:
        kind = "emojis" if self.is_emoji else "stickers"
        try:
            self.client.call("DELETE", f"/guilds/{self.guild_id}/{kind}/{remote.remote_id}")
        except DiscordApiError as exc:
            if exc.status == 404 or exc.code in (10014, 10060):
                return  # already gone
            raise


def discord_info(config: PublishConfig, transport: Transport = urllib_transport) -> dict[str, Any]:
    """Read-only view for the settings page: server name and free slots, never the token."""
    missing = config.discord_missing()
    info: dict[str, Any] = {
        "configured": not missing,
        "missing": missing,
        "guild_id": config.discord_guild_id,
    }
    if missing:
        return info
    try:
        emoji = DiscordPublisher(config, Channel.DISCORD_EMOJI, transport)
        guild = emoji.guild()
        sticker = DiscordPublisher(config, Channel.DISCORD_STICKER, transport)
        info["guild"] = guild.get("name") or config.discord_guild_id
        info["tier"] = int(guild.get("premium_tier") or 0)
        info["emoji_slots"] = emoji.slots(guild)
        info["sticker_slots"] = sticker.slots(guild)
    except PublishError as exc:
        info["error"] = redact(str(exc), *config.secrets)
    return info
