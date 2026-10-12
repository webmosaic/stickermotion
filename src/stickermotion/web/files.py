"""Uploaded PNG checks and the per-platform view of a finished sticker (files, archives)."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from stickermotion.core.models import ExportTarget
from stickermotion.core.specs import KB

MIN_SIDE = 64
MAX_SIDE = 4096
# Pillow refuses images above this many pixels (decompression bombs).
Image.MAX_IMAGE_PIXELS = MAX_SIDE * MAX_SIDE


class UploadError(ValueError):
    """The uploaded file is not a usable PNG; the message is shown to the user."""


@dataclass(frozen=True)
class CheckedPng:
    data: bytes
    width: int
    height: int


def check_png(data: bytes, max_bytes: int) -> CheckedPng:
    """PNG with real transparency and a sensible size, else ``UploadError`` (Ukrainian)."""
    if not data:
        raise UploadError("Файл порожній. Виберіть PNG з прозорим фоном.")
    if len(data) > max_bytes:
        raise UploadError(
            f"Файл завеликий: {len(data) / 1024 / 1024:.1f} МБ, "
            f"можна до {max_bytes / 1024 / 1024:.0f} МБ."
        )
    try:
        with Image.open(io.BytesIO(data)) as image:
            fmt = image.format
            width, height = image.size
            if fmt != "PNG":
                raise UploadError("Це не PNG. Потрібен PNG з прозорим фоном.")
            image.load()
            if image.mode not in ("RGBA", "LA", "PA") and "transparency" not in image.info:
                raise UploadError("У PNG немає прозорості. Експортуйте персонажа з прозорим фоном.")
            alpha = image.convert("RGBA").getchannel("A")
    except UploadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise UploadError("Не вдалося прочитати файл. Потрібен PNG з прозорим фоном.") from None
    except Image.DecompressionBombError:
        raise UploadError(f"Зображення завелике, максимум {MAX_SIDE}×{MAX_SIDE} px.") from None
    if min(width, height) < MIN_SIDE:
        raise UploadError(f"Зображення замале: {width}×{height} px, потрібно від {MIN_SIDE} px.")
    if max(width, height) > MAX_SIDE:
        raise UploadError(f"Зображення завелике: {width}×{height} px, максимум {MAX_SIDE} px.")
    low, high = alpha.getextrema()
    if low == 255:
        raise UploadError("Фон не прозорий: усі пікселі непрозорі. Потрібен прозорий фон.")
    if high < 128:
        raise UploadError("Зображення майже повністю прозоре, персонажа не видно.")
    return CheckedPng(data, width, height)


# Platforms


@dataclass(frozen=True)
class Platform:
    key: str
    title: str
    targets: tuple[ExportTarget, ...]


PLATFORMS = (
    Platform("twitch", "Twitch", (ExportTarget.TWITCH_GIF,)),
    Platform("discord", "Discord", (ExportTarget.DISCORD_STICKER, ExportTarget.DISCORD_EMOJI)),
    Platform("telegram", "Telegram", (ExportTarget.TELEGRAM_STICKER, ExportTarget.TELEGRAM_EMOJI)),
    Platform("obs", "OBS", (ExportTarget.OBS_OVERLAY,)),
)
ARCHIVES = {p.key: p for p in PLATFORMS}
ALL = "all"

TARGET_TITLES = {
    ExportTarget.TWITCH_GIF: "Емоут",
    ExportTarget.DISCORD_STICKER: "Стікер",
    ExportTarget.DISCORD_EMOJI: "Емодзі",
    ExportTarget.TELEGRAM_STICKER: "Відеостікер",
    ExportTarget.TELEGRAM_EMOJI: "Емодзі",
    ExportTarget.OBS_OVERLAY: "Оверлей",
}


@dataclass(frozen=True)
class OutputFile:
    path: str
    """Relative to the sticker's output directory."""
    target: ExportTarget
    width: int
    height: int
    size_bytes: int
    frames: int
    duration_s: float
    valid: bool
    problems: tuple[str, ...]

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def title(self) -> str:
        return TARGET_TITLES[self.target]

    @property
    def weight(self) -> str:
        return weight(self.size_bytes)


def weight(size_bytes: int) -> str:
    if size_bytes >= 1024 * KB:
        return f"{size_bytes / 1024 / KB:.1f} МБ"
    return f"{max(1, round(size_bytes / KB))} КБ"


def output_files(manifest: dict[str, Any]) -> list[OutputFile]:
    """Main exports (not the full-composition comparison emotes in ``full/``)."""
    if not manifest.get("presets"):
        return []
    files = []
    for entry in manifest["presets"][0]["exports"]:
        files.append(
            OutputFile(
                entry["path"],
                ExportTarget(entry["target"]),
                entry["width"],
                entry["height"],
                entry["size_bytes"],
                entry["frame_count"],
                entry["duration_s"],
                entry["valid"],
                tuple(entry.get("violations") or ()),
            )
        )
    return files


def by_platform(files: list[OutputFile]) -> list[tuple[Platform, list[OutputFile]]]:
    groups = []
    for platform in PLATFORMS:
        items = [f for f in files if f.target in platform.targets]
        items.sort(key=lambda f: (platform.targets.index(f.target), f.width))
        if items:
            groups.append((platform, items))
    return groups


def find(files: list[OutputFile], name: str) -> OutputFile | None:
    return next((f for f in files if f.name == name), None)


def build_zip(root: Path, files: list[OutputFile], group: str) -> bytes:
    """Archive of one platform (flat) or of everything (one folder per platform)."""
    if group != ALL and group not in ARCHIVES:
        raise KeyError(group)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for platform, items in by_platform(files):
            if group != ALL and platform.key != group:
                continue
            for item in items:
                arcname = item.name if group != ALL else f"{platform.key}/{item.name}"
                archive.write(root / item.path, arcname)
    return buffer.getvalue()
