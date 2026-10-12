"""SQLite store of stickers (jobs and history) plus the spend read from the cost journal.

Layout under ``DATA_DIR``: ``app.db``, ``costs.jsonl`` (cost journal), ``.cache/ai/``
(raw video cache) and ``stickers/<id>/`` with ``input.png``, the ``ai/`` output of the
AI pipeline and ``raw/`` with the last raw clips (``raws``).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, fields
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from stickermotion import raws
from stickermotion.ai_pipeline import AI_DIR, COSTS_FILE

DB_FILE = "app.db"
STICKERS_DIR = "stickers"
INPUT_FILE = "input.png"


class Status(StrEnum):
    DESCRIBE_QUEUED = "describe_queued"
    """Waiting for the paid-call lock before the description starts."""
    DESCRIBING = "describing"
    DESCRIBED = "described"
    """Description ready, waiting for the user to generate."""
    QUEUED = "queued"
    GENERATING = "generating"
    CUTTING = "cutting"
    EXPORTING = "exporting"
    DONE = "done"
    ERROR = "error"
    INTERRUPTED = "interrupted"
    """The service restarted mid-generation; the user may retry (paid) by hand."""
    CHOOSING = "choosing"
    """Variants are ready: waiting for the user to pick one."""


ACTIVE = (Status.QUEUED, Status.GENERATING, Status.CUTTING, Status.EXPORTING)
DESCRIBE_STATES = (Status.DESCRIBE_QUEUED, Status.DESCRIBING)
BUSY = (*DESCRIBE_STATES, *ACTIVE)


@dataclass
class Sticker:
    id: str
    created_at: str
    updated_at: str
    filename: str
    status: str
    width: int = 0
    height: int = 0
    error: str = ""
    text: str = ""
    """What the user sees and edits (Ukrainian by default)."""
    auto_uk: str = ""
    """Last automatic description, translated to Ukrainian."""
    auto_en: str = ""
    """Last automatic description as the vision model wrote it."""
    prompt_en: str = ""
    """English action text of the last generation."""
    regenerate: int = 0
    """1 when the queued generation must skip the video cache."""
    generations: int = 0
    cost_usd: float = 0.0
    decor: str = "auto"
    """Decor mode of the next generation: ``auto`` or ``together``."""
    decor_found: str = ""
    """JSON list of decor items found at the description step (empty: none)."""
    variants: int = 1
    """Clips the next generation makes (1..3)."""
    job: str = ""
    """Queued work: ``generate`` (paid) or ``reprocess`` (a stored clip, free)."""
    job_raw: str = ""
    """Raw clip id the queued ``reprocess`` uses."""
    raw_id: str = ""
    """Raw clip the current result was made from."""

    @property
    def decor_items(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.decor_found or "[]")
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []

    @property
    def state(self) -> Status:
        return Status(self.status)

    @property
    def busy(self) -> bool:
        return self.state in BUSY

    @property
    def active(self) -> bool:
        return self.state in ACTIVE


class PubStatus(StrEnum):
    """State of one sticker on one publication channel."""

    NONE = "none"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    ERROR = "error"


@dataclass
class Publication:
    sticker_id: str
    channel: str
    status: str = PubStatus.NONE.value
    remote_id: str = ""
    url: str = ""
    extra: str = "{}"
    """JSON with platform details (set name, file_unique_id, emoji name...)."""
    name: str = ""
    emoji: str = ""
    published_at: str = ""
    error: str = ""
    updated_at: str = ""

    @property
    def state(self) -> PubStatus:
        return PubStatus(self.status)

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.extra or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


_COLUMNS = [f.name for f in fields(Sticker)]
_PUB_COLUMNS = [f.name for f in fields(Publication)]
_SCHEMA = """
CREATE TABLE IF NOT EXISTS stickers (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    status TEXT NOT NULL,
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    auto_uk TEXT NOT NULL DEFAULT '',
    auto_en TEXT NOT NULL DEFAULT '',
    prompt_en TEXT NOT NULL DEFAULT '',
    regenerate INTEGER NOT NULL DEFAULT 0,
    generations INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    decor TEXT NOT NULL DEFAULT 'auto',
    decor_found TEXT NOT NULL DEFAULT '',
    variants INTEGER NOT NULL DEFAULT 1,
    job TEXT NOT NULL DEFAULT '',
    job_raw TEXT NOT NULL DEFAULT '',
    raw_id TEXT NOT NULL DEFAULT ''
)
"""
# Columns added after the first release, for databases created before them.
_ADDED_COLUMNS = {
    "decor": "TEXT NOT NULL DEFAULT 'auto'",
    "decor_found": "TEXT NOT NULL DEFAULT ''",
    "variants": "INTEGER NOT NULL DEFAULT 1",
    "job": "TEXT NOT NULL DEFAULT ''",
    "job_raw": "TEXT NOT NULL DEFAULT ''",
    "raw_id": "TEXT NOT NULL DEFAULT ''",
}
_PUB_SCHEMA = """
CREATE TABLE IF NOT EXISTS publications (
    sticker_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'none',
    remote_id TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    extra TEXT NOT NULL DEFAULT '{}',
    name TEXT NOT NULL DEFAULT '',
    emoji TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (sticker_id, channel)
)
"""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / STICKERS_DIR).mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.data_dir / DB_FILE, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock, self._db:
            self._db.execute(_SCHEMA)
            self._db.execute(_PUB_SCHEMA)
            present = {row[1] for row in self._db.execute("PRAGMA table_info(stickers)")}
            for name, definition in _ADDED_COLUMNS.items():
                if name not in present:
                    self._db.execute(f"ALTER TABLE stickers ADD COLUMN {name} {definition}")

    # Paths

    @property
    def journal(self) -> Path:
        return self.data_dir / COSTS_FILE

    def sticker_dir(self, sticker_id: str) -> Path:
        return self.data_dir / STICKERS_DIR / sticker_id

    def input_path(self, sticker_id: str) -> Path:
        return self.sticker_dir(sticker_id) / INPUT_FILE

    def output_dir(self, sticker_id: str) -> Path:
        return self.sticker_dir(sticker_id) / AI_DIR

    def raw_dir(self, sticker_id: str) -> Path:
        """Stored raw clips of one sticker, newest ``raws.KEEP`` kept."""
        return self.sticker_dir(sticker_id) / raws.RAW_DIR

    def clips(self, sticker_id: str) -> list[raws.RawClip]:
        return raws.list_clips(self.raw_dir(sticker_id))

    def migrate_raws(self) -> list[str]:
        """Stickers made before the raw clip library keep only ``ai/raw.mp4``: it becomes
        their first stored clip and the current one. Runs once per sticker (skipped when clips
        exist)."""
        migrated = []
        for sticker in self.list():
            clip = raws.import_legacy(self.raw_dir(sticker.id), self.output_dir(sticker.id))
            if clip is not None:
                self.update(sticker.id, raw_id=clip.id)
                migrated.append(sticker.id)
        return migrated

    def manifest(self, sticker_id: str) -> dict[str, Any] | None:
        path = self.output_dir(sticker_id) / "manifest.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text())

    # Rows

    def create(self, filename: str, png: bytes, width: int, height: int) -> Sticker:
        sticker_id = uuid.uuid4().hex[:12]
        folder = self.sticker_dir(sticker_id)
        folder.mkdir(parents=True)
        (folder / INPUT_FILE).write_bytes(png)
        stamp = now()
        sticker = Sticker(
            sticker_id, stamp, stamp, filename, Status.DESCRIBING.value, width, height
        )
        values = [getattr(sticker, c) for c in _COLUMNS]
        with self._lock, self._db:
            self._db.execute(
                f"INSERT INTO stickers ({', '.join(_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
                values,
            )
        return sticker

    def get(self, sticker_id: str) -> Sticker | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM stickers WHERE id = ?", (sticker_id,)).fetchone()
        return Sticker(**dict(row)) if row else None

    def list(self) -> list[Sticker]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM stickers ORDER BY created_at DESC").fetchall()
        return [Sticker(**dict(r)) for r in rows]

    def update(self, sticker_id: str, **values: Any) -> Sticker:
        unknown = set(values) - set(_COLUMNS)
        if unknown:
            raise KeyError(f"unknown columns {sorted(unknown)}")
        values = {k: (v.value if isinstance(v, StrEnum) else v) for k, v in values.items()}
        values["updated_at"] = now()
        sets = ", ".join(f"{k} = ?" for k in values)
        with self._lock, self._db:
            self._db.execute(
                f"UPDATE stickers SET {sets} WHERE id = ?", [*values.values(), sticker_id]
            )
        sticker = self.get(sticker_id)
        if sticker is None:
            raise KeyError(sticker_id)
        return sticker

    def add_cost(self, sticker_id: str, usd: float) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE stickers SET cost_usd = ROUND(cost_usd + ?, 6) WHERE id = ?",
                (usd, sticker_id),
            )

    def delete(self, sticker_id: str) -> None:
        import shutil

        with self._lock, self._db:
            self._db.execute("DELETE FROM stickers WHERE id = ?", (sticker_id,))
            self._db.execute("DELETE FROM publications WHERE sticker_id = ?", (sticker_id,))
        shutil.rmtree(self.sticker_dir(sticker_id), ignore_errors=True)

    def mark_interrupted(self) -> list[str]:
        """After a restart: unfinished generations become ``interrupted`` (no automatic paid
        retry), unfinished descriptions become errors with a hint to describe again."""
        changed = []
        for sticker in self.list():
            if sticker.active:
                self.update(
                    sticker.id,
                    status=Status.INTERRUPTED,
                    error="Сервіс перезапустився під час генерації. Можна повторити.",
                )
                changed.append(sticker.id)
            elif sticker.state in DESCRIBE_STATES:
                self.update(
                    sticker.id,
                    status=Status.ERROR,
                    error="Сервіс перезапустився під час опису. Натисніть «Описати ще раз».",
                )
                changed.append(sticker.id)
        return changed

    # Publications

    def publications(self, sticker_id: str) -> dict[str, Publication]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM publications WHERE sticker_id = ?", (sticker_id,)
            ).fetchall()
        return {row["channel"]: Publication(**dict(row)) for row in rows}

    def publication(self, sticker_id: str, channel: str) -> Publication:
        """The row, or an unsaved ``none`` row when the channel was never touched."""
        return self.publications(sticker_id).get(channel) or Publication(sticker_id, channel)

    def save_publication(self, sticker_id: str, channel: str, **values: Any) -> Publication:
        unknown = set(values) - set(_PUB_COLUMNS)
        if unknown:
            raise KeyError(f"unknown columns {sorted(unknown)}")
        values = {k: (v.value if isinstance(v, StrEnum) else v) for k, v in values.items()}
        values["updated_at"] = now()
        row = {**vars(self.publication(sticker_id, channel)), **values}
        with self._lock, self._db:
            self._db.execute(
                f"INSERT OR REPLACE INTO publications ({', '.join(_PUB_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _PUB_COLUMNS)})",
                [row[c] for c in _PUB_COLUMNS],
            )
        return self.publication(sticker_id, channel)

    def mark_publishing_interrupted(self) -> list[tuple[str, str]]:
        """After a restart a ``publishing`` row cannot be finished: it becomes an error."""
        with self._lock:
            rows = self._db.execute(
                "SELECT sticker_id, channel FROM publications WHERE status = ?",
                (PubStatus.PUBLISHING.value,),
            ).fetchall()
        changed = [(r["sticker_id"], r["channel"]) for r in rows]
        for sticker_id, channel in changed:
            self.save_publication(
                sticker_id,
                channel,
                status=PubStatus.ERROR,
                error="Сервіс перезапустився під час публікації. Перевірте платформу і повторіть.",
            )
        return changed

    # Spend

    def journal_entries(self) -> list[dict[str, Any]]:
        if not self.journal.is_file():
            return []
        entries = []
        for line in self.journal.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def spent(self, day: str | None = None) -> float:
        """Total spend from the journal, or the spend of ``day`` (``YYYY-MM-DD``, local)."""
        total = 0.0
        for entry in self.journal_entries():
            if day is None or str(entry.get("time", "")).startswith(day):
                total += float(entry.get("cost_usd") or 0.0)
        return round(total, 6)

    def spent_today(self) -> float:
        return self.spent(datetime.now().astimezone().strftime("%Y-%m-%d"))
