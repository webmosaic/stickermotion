"""Publication service of the web UI: per sticker and channel state, idempotent publish,
removal, and the read-only platform overview (ADR-017).

The platform calls run in a thread (``asyncio.to_thread`` in the routes); a channel that
is already ``publishing`` is not started twice, and a ``published`` one that still exists
on the platform is not uploaded again.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stickermotion.publish import (
    CHANNEL_FILES,
    CHANNEL_TITLES,
    PLATFORM_OF,
    Channel,
    NotConfiguredError,
    Publisher,
    PublishError,
    PublishRequest,
    Remote,
    Transport,
)
from stickermotion.publish.config import PublishConfig
from stickermotion.publish.discord import DiscordPublisher, discord_info
from stickermotion.publish.http import redact, urllib_transport
from stickermotion.publish.naming import (
    DISCORD_EMOJI_NAME,
    DISCORD_STICKER_NAME,
    discord_name,
    split_emoji,
    suggest_emoji,
)
from stickermotion.publish.telegram import TelegramPublisher, scrub, telegram_info
from stickermotion.publish.twitch import available as twitch_files
from stickermotion.publish.twitch import emotes_url
from stickermotion.web.store import Publication, PubStatus, Sticker, Store, now

log = logging.getLogger("stickermotion.web")

PUB_LABELS = {
    PubStatus.NONE: "Не опубліковано",
    PubStatus.PUBLISHING: "Публікується",
    PubStatus.PUBLISHED: "Опубліковано",
    PubStatus.ERROR: "Помилка",
}
CHANNELS = (
    Channel.TELEGRAM_STICKER,
    Channel.TELEGRAM_EMOJI,
    Channel.DISCORD_EMOJI,
    Channel.DISCORD_STICKER,
)
OPTIONAL = {Channel.TELEGRAM_EMOJI}
"""Shown behind a checkbox: a second (custom emoji) pack is optional."""


@dataclass(frozen=True)
class ChannelView:
    channel: Channel
    title: str
    configured: bool
    missing: tuple[str, ...]
    has_file: bool
    publication: Publication
    label: str
    name: str
    emoji: str
    needs_name: bool
    needs_emoji: bool
    name_max: int
    optional: bool


class Publishing:
    def __init__(
        self, store: Store, config: PublishConfig, transport: Transport = urllib_transport
    ) -> None:
        self.store = store
        self.config = config
        self.transport = transport
        self._lock = threading.Lock()

    # Configuration

    def missing(self, channel: Channel) -> list[str]:
        if PLATFORM_OF[channel] == "telegram":
            return self.config.telegram_missing()
        return self.config.discord_missing()

    def publisher(self, channel: Channel) -> Publisher:
        if PLATFORM_OF[channel] == "telegram":
            return TelegramPublisher(self.config, channel, self.transport)
        return DiscordPublisher(self.config, channel, self.transport)

    def clean(self, message: str) -> str:
        return scrub(redact(message, *self.config.secrets))

    # Defaults

    def default_name(self, sticker: Sticker, channel: Channel) -> str:
        limits = DISCORD_STICKER_NAME if channel is Channel.DISCORD_STICKER else DISCORD_EMOJI_NAME
        return discord_name(Path(sticker.filename).stem or "sticker", limits)

    def default_emoji(self, sticker: Sticker) -> str:
        return " ".join(suggest_emoji(sticker.prompt_en, sticker.text))

    # View

    def views(self, sticker: Sticker) -> list[ChannelView]:
        saved = self.store.publications(sticker.id)
        root = self.store.output_dir(sticker.id)
        emoji = self.default_emoji(sticker)
        items = []
        for channel in CHANNELS:
            row = saved.get(channel.value) or Publication(sticker.id, channel.value)
            missing = tuple(self.missing(channel))
            is_discord = PLATFORM_OF[channel] == "discord"
            limits = (
                DISCORD_STICKER_NAME if channel is Channel.DISCORD_STICKER else DISCORD_EMOJI_NAME
            )
            items.append(
                ChannelView(
                    channel=channel,
                    title=CHANNEL_TITLES[channel],
                    configured=not missing,
                    missing=missing,
                    has_file=(root / CHANNEL_FILES[channel]).is_file(),
                    publication=row,
                    label=PUB_LABELS[row.state],
                    name=row.name or self.default_name(sticker, channel),
                    emoji=row.emoji or emoji,
                    needs_name=is_discord,
                    needs_emoji=channel is not Channel.DISCORD_EMOJI,
                    name_max=limits[1],
                    optional=channel in OPTIONAL,
                )
            )
        return items

    def twitch_view(self, sticker: Sticker) -> dict[str, Any]:
        return {
            "files": twitch_files(self.store.output_dir(sticker.id)),
            "url": emotes_url(self.config.twitch_channel),
        }

    # Actions

    def publish(
        self, sticker: Sticker, channel: Channel, name: str, emoji_text: str
    ) -> Publication:
        """Publish once; returns the saved row (``error`` holds the user-facing message)."""
        sid = sticker.id
        with self._lock:
            current = self.store.publication(sid, channel.value)
            if current.state is PubStatus.PUBLISHING:
                return current
            emoji = split_emoji(emoji_text) or split_emoji(self.default_emoji(sticker))
            name = name.strip() or self.default_name(sticker, channel)
            self.store.save_publication(
                sid,
                channel.value,
                status=PubStatus.PUBLISHING,
                error="",
                name=name,
                emoji=" ".join(emoji),
            )
        try:
            publisher = self.publisher(channel)
            if current.state is PubStatus.PUBLISHED and current.remote_id:
                remote = Remote(current.remote_id, current.url, current.details)
                if publisher.exists(remote):
                    # Already there: nothing is uploaded again.
                    return self.store.save_publication(
                        sid, channel.value, status=PubStatus.PUBLISHED, error=""
                    )
            request = PublishRequest(
                file=self.store.output_dir(sid) / CHANNEL_FILES[channel],
                name=name,
                emoji=emoji,
                description=sticker.text,
            )
            remote = publisher.publish(request)
        except PublishError as exc:
            message = self.clean(str(exc))
            log.warning("publish %s %s failed: %s", sid, channel.value, message)
            if exc.raw:
                log.warning("publish %s %s raw error: %s", sid, channel.value, self.clean(exc.raw))
            status = (
                PubStatus.PUBLISHED if current.state is PubStatus.PUBLISHED else PubStatus.ERROR
            )
            if isinstance(exc, NotConfiguredError):
                status = (
                    current.state if current.state is not PubStatus.PUBLISHING else PubStatus.NONE
                )
            return self.store.save_publication(sid, channel.value, status=status, error=message)
        except Exception as exc:  # the user sees the reason, the service keeps running
            message = self.clean(f"Внутрішня помилка: {type(exc).__name__}: {exc}")
            log.error("publish %s %s crashed: %s", sid, channel.value, message)
            return self.store.save_publication(
                sid, channel.value, status=PubStatus.ERROR, error=message
            )
        log.info("published %s to %s", sid, channel.value)
        return self.store.save_publication(
            sid,
            channel.value,
            status=PubStatus.PUBLISHED,
            remote_id=remote.remote_id,
            url=remote.url,
            extra=json.dumps(remote.extra, ensure_ascii=False),
            name=str(remote.extra.get("name") or name),
            published_at=now(),
            error="",
        )

    def remove(self, sticker_id: str, channel: Channel) -> str | None:
        """Remove from the platform; returns a user-facing error or ``None``."""
        row = self.store.publication(sticker_id, channel.value)
        if row.state is not PubStatus.PUBLISHED or not row.remote_id:
            if row.state is not PubStatus.NONE:
                self.store.save_publication(
                    sticker_id, channel.value, status=PubStatus.NONE, error="", remote_id="", url=""
                )
            return None
        try:
            self.publisher(channel).remove(Remote(row.remote_id, row.url, row.details))
        except PublishError as exc:
            message = self.clean(str(exc))
            self.store.save_publication(sticker_id, channel.value, error=message)
            return message
        self.store.save_publication(
            sticker_id,
            channel.value,
            status=PubStatus.NONE,
            remote_id="",
            url="",
            extra="{}",
            published_at="",
            error="",
        )
        return None

    def published(self, sticker_id: str) -> list[Channel]:
        rows = self.store.publications(sticker_id)
        return [
            c for c in CHANNELS if c.value in rows and rows[c.value].state is PubStatus.PUBLISHED
        ]

    # Settings page

    def overview(self) -> dict[str, Any]:
        telegram = telegram_info(self.config, self.transport)
        discord = discord_info(self.config, self.transport)
        return {
            "telegram": telegram,
            "discord": discord,
            "twitch": {
                "url": emotes_url(self.config.twitch_channel),
                "channel": self.config.twitch_channel,
            },
        }
