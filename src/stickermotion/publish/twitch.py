"""Twitch: no public API for emote uploads (ADR-017), so the service prepares an archive
with the manual set (28/56/112), the single auto-mode file and a short instruction.
The upload itself happens in the creator dashboard."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

MANUAL_FILES = ("twitch_gif_28.gif", "twitch_gif_56.gif", "twitch_gif_112.gif")
AUTO_FILE = "twitch_gif_224.gif"
DASHBOARD = "https://dashboard.twitch.tv"


def emotes_url(channel: str = "") -> str:
    """Emote page of the creator dashboard (the channel is part of the path)."""
    channel = channel.strip().lstrip("@").lower()
    if channel:
        return f"{DASHBOARD}/u/{channel}/viewer-rewards/emotes"
    return f"{DASHBOARD}/viewer-rewards/emotes"


def instruction(channel: str = "", name: str = "") -> str:
    url = emotes_url(channel)
    code = f" Код емоута (після префікса каналу): {name}." if name else ""
    return (
        "Анімований емоут для Twitch\n"
        "===========================\n\n"
        f"1. Відкрийте сторінку емоутів у кабінеті автора: {url}\n"
        "2. Натисніть «Додати емоут» і виберіть анімований емоут.\n"
        f"3. Автоматичний режим: завантажте один файл {AUTO_FILE}.\n"
        "   Ручний режим: завантажте окремо twitch_gif_28.gif, twitch_gif_56.gif і "
        "twitch_gif_112.gif у поля 28, 56 і 112.\n"
        f"4. Вкажіть код емоута, рівень і збережіть.{code}\n\n"
        "Анімовані емоути доступні афіліатам і партнерам. Файли вже пройшли перевірку "
        "лімітів Twitch (до 60 кадрів, 1 МБ в авто-режимі, 512 КБ на файл у ручному).\n"
    )


def available(root: Path) -> list[str]:
    return [name for name in (*MANUAL_FILES, AUTO_FILE) if (root / name).is_file()]


def build_archive(root: Path, channel: str = "", name: str = "") -> bytes:
    """ZIP: ``manual/`` with 28/56/112, ``auto/`` with the single file, ``README.txt``."""
    files = available(root)
    if not files:
        raise FileNotFoundError("no Twitch GIF files")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for item in files:
            folder = "auto" if item == AUTO_FILE else "manual"
            archive.write(root / item, f"{folder}/{item}")
        archive.writestr("README.txt", instruction(channel, name))
    return buffer.getvalue()
