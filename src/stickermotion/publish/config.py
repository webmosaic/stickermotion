"""Publishing settings from environment variables. Tokens never leave this object."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_PACK_TITLE = "StickerMotion"
DEFAULT_PACK_SLUG = "stickermotion"


def _int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.lstrip("-").isdigit() else None


@dataclass
class PublishConfig:
    """``TELEGRAM_BOT_TOKEN``, ``TELEGRAM_OWNER_ID`` (numeric), ``TELEGRAM_PACK_TITLE``,
    ``TELEGRAM_PACK_SLUG``; ``DISCORD_BOT_TOKEN``, ``DISCORD_GUILD_ID``; optional
    ``TWITCH_CHANNEL`` for the link to the emote page of the creator dashboard."""

    telegram_token: str = field(default="", repr=False)
    telegram_owner_id: int | None = None
    telegram_owner_raw: str = field(default="", repr=False)
    telegram_pack_title: str = DEFAULT_PACK_TITLE
    telegram_pack_slug: str = DEFAULT_PACK_SLUG
    discord_token: str = field(default="", repr=False)
    discord_guild_id: str = ""
    twitch_channel: str = ""

    @classmethod
    def from_env(cls) -> PublishConfig:
        get = os.environ.get
        owner = get("TELEGRAM_OWNER_ID", "")
        return cls(
            telegram_token=get("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_owner_id=_int(owner),
            telegram_owner_raw=owner.strip(),
            telegram_pack_title=get("TELEGRAM_PACK_TITLE", "").strip() or DEFAULT_PACK_TITLE,
            telegram_pack_slug=get("TELEGRAM_PACK_SLUG", "").strip() or DEFAULT_PACK_SLUG,
            discord_token=get("DISCORD_BOT_TOKEN", "").strip(),
            discord_guild_id=get("DISCORD_GUILD_ID", "").strip(),
            twitch_channel=get("TWITCH_CHANNEL", "").strip().lstrip("@"),
        )

    def telegram_missing(self) -> list[str]:
        missing = []
        if not self.telegram_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if self.telegram_owner_id is None:
            missing.append(
                "TELEGRAM_OWNER_ID (число)" if self.telegram_owner_raw else "TELEGRAM_OWNER_ID"
            )
        return missing

    def discord_missing(self) -> list[str]:
        missing = []
        if not self.discord_token:
            missing.append("DISCORD_BOT_TOKEN")
        if not self.discord_guild_id.isdigit():
            missing.append("DISCORD_GUILD_ID")
        return missing

    @property
    def secrets(self) -> tuple[str, ...]:
        return tuple(s for s in (self.telegram_token, self.discord_token) if s)
