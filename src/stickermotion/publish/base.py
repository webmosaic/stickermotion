"""Common publisher interface: publish, check state, remove."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class Channel(StrEnum):
    """One place a sticker can be published to (platform plus kind)."""

    TELEGRAM_STICKER = "telegram_sticker"
    TELEGRAM_EMOJI = "telegram_emoji"
    DISCORD_EMOJI = "discord_emoji"
    DISCORD_STICKER = "discord_sticker"


CHANNEL_TITLES = {
    Channel.TELEGRAM_STICKER: "Telegram, стікерпак",
    Channel.TELEGRAM_EMOJI: "Telegram, емодзі",
    Channel.DISCORD_EMOJI: "Discord, емодзі",
    Channel.DISCORD_STICKER: "Discord, стікер",
}

CHANNEL_FILES = {
    Channel.TELEGRAM_STICKER: "telegram_sticker.webm",
    Channel.TELEGRAM_EMOJI: "telegram_emoji.webm",
    Channel.DISCORD_EMOJI: "discord_emoji.webp",
    Channel.DISCORD_STICKER: "discord_sticker.png",
}
"""Export file each channel uploads (names from ``exporters.targets``)."""

PLATFORM_OF = {
    Channel.TELEGRAM_STICKER: "telegram",
    Channel.TELEGRAM_EMOJI: "telegram",
    Channel.DISCORD_EMOJI: "discord",
    Channel.DISCORD_STICKER: "discord",
}


@dataclass(frozen=True)
class PublishRequest:
    file: Path
    name: str
    """Discord name of the emoji or sticker (already valid)."""
    emoji: tuple[str, ...]
    """Emoji associated with the sticker (Telegram ``emoji_list``, Discord sticker tag)."""
    description: str = ""


@dataclass(frozen=True)
class Remote:
    """Where a published item lives on the platform."""

    remote_id: str
    url: str
    extra: dict[str, Any] = field(default_factory=dict)


class Publisher(Protocol):
    channel: Channel

    def publish(self, request: PublishRequest) -> Remote:
        """Upload one item; raises ``PublishError`` with a user-facing message."""
        ...

    def exists(self, remote: Remote) -> bool:
        """Whether the item is still on the platform."""
        ...

    def remove(self, remote: Remote) -> None:
        """Remove the item; an item that is already gone is not an error."""
        ...
