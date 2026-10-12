"""Library of raw generated clips, so a liked motion can be reused later.

Layout: ``<raw dir>/<clip id>/`` with ``raw.mp4`` (the clip as Wan delivered it),
``model_input.png`` (the frame it was generated from), ``meta.json`` (prompt, Wan
parameters with the seed, framing, emote focus box, cost...) and, once made, a Telegram
sticker preview (``telegram_sticker.webm``). The newest ``KEEP`` clips are kept.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from stickermotion.engine.ai.framing import Framing
from stickermotion.engine.ai.redecor import framing_from_dict

RAW_DIR = "raw"
VIDEO = "raw.mp4"
MODEL_INPUT = "model_input.png"
META = "meta.json"
PREVIEW = "telegram_sticker.webm"
KEEP = 5
_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9]{6}(-[a-z0-9]+)?$")


@dataclass
class RawClip:
    id: str
    folder: Path
    meta: dict[str, Any]

    @property
    def video(self) -> Path:
        return self.folder / VIDEO

    @property
    def model_input(self) -> Path:
        return self.folder / MODEL_INPUT

    @property
    def preview(self) -> Path | None:
        path = self.folder / PREVIEW
        return path if path.is_file() else None

    @property
    def seed(self) -> int | None:
        value = (self.meta.get("params") or {}).get("seed", self.meta.get("seed"))
        return int(value) if isinstance(value, int | float) else None

    @property
    def prompt(self) -> str:
        return str(self.meta.get("prompt") or "")

    @property
    def created(self) -> str:
        return str(self.meta.get("created") or "")

    @property
    def batch(self) -> str:
        """Generation batch (variants made together share it)."""
        return str(self.meta.get("batch") or self.id)

    @property
    def framing(self) -> Framing | None:
        data = self.meta.get("framing")
        if not isinstance(data, dict):
            return None
        try:
            return framing_from_dict(data)
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    @property
    def framing_subject(self) -> str:
        """What the framing placed: ``character`` (decor out) or ``composition``."""
        return str(self.meta.get("framing_subject") or "character")

    @property
    def focus_box(self) -> list[int] | None:
        box = self.meta.get("focus_box")
        if isinstance(box, list) and len(box) == 4:
            return [int(v) for v in box]
        return None

    @property
    def focus_spec(self) -> str:
        """Emote focus for a free reprocess: the stored box, or the full composition."""
        box = self.focus_box
        if box is None:
            return "full"
        x0, y0, x1, y1 = box
        return f"{x0},{y0},{x1 - x0},{y1 - y0}"


def new_id(seed: int | None = None, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S-%f")
    return f"{stamp}-{seed}" if seed is not None else stamp


def valid_id(clip_id: str) -> bool:
    return bool(_ID.match(clip_id))


def load(folder: Path) -> RawClip | None:
    if not (folder / VIDEO).is_file():
        return None
    try:
        meta = json.loads((folder / META).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        meta = {}
    return RawClip(folder.name, folder, meta if isinstance(meta, dict) else {})


def get(raw_dir: Path, clip_id: str) -> RawClip | None:
    if not valid_id(clip_id):
        return None
    return load(Path(raw_dir) / clip_id)


def list_clips(raw_dir: Path) -> list[RawClip]:
    """Newest first."""
    root = Path(raw_dir)
    if not root.is_dir():
        return []
    clips = [c for c in (load(p) for p in root.iterdir() if valid_id(p.name)) if c]
    return sorted(clips, key=lambda c: c.id, reverse=True)


def prune(raw_dir: Path, keep: int = KEEP, protect: tuple[str, ...] = ()) -> list[str]:
    """Delete all but the newest ``keep`` clips (``protect`` ids are never deleted)."""
    removed = []
    for clip in list_clips(raw_dir)[keep:]:
        if clip.id in protect:
            continue
        shutil.rmtree(clip.folder, ignore_errors=True)
        removed.append(clip.id)
    return removed


def save(
    raw_dir: Path,
    video: Path,
    model_input: Path | Image.Image,
    meta: dict[str, Any],
    preview: Path | None = None,
    keep: int = KEEP,
    protect: tuple[str, ...] = (),
    clip_id: str | None = None,
) -> RawClip:
    """Copy a clip with its model input and metadata into the library, then prune."""
    seed = (meta.get("params") or {}).get("seed", meta.get("seed"))
    clip_id = clip_id or new_id(int(seed) if isinstance(seed, int) else None)
    folder = Path(raw_dir) / clip_id
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(video, folder / VIDEO)
    if isinstance(model_input, Image.Image):
        model_input.save(folder / MODEL_INPUT)
    else:
        shutil.copyfile(model_input, folder / MODEL_INPUT)
    if preview is not None and preview.is_file():
        shutil.copyfile(preview, folder / PREVIEW)
    data = {"created": datetime.now().astimezone().isoformat(timespec="seconds"), **meta}
    data["id"] = clip_id
    (folder / META).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    prune(raw_dir, keep, (*protect, clip_id))
    clip = load(folder)
    assert clip is not None
    return clip


def meta_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Clip metadata from an ``animate`` / ``reprocess`` manifest."""
    ai = manifest.get("ai") or {}
    params = dict(ai.get("params") or {})
    decor = ai.get("decor") or {}
    separate = decor.get("mode") in ("auto", "off") and bool(decor.get("items"))
    focus = ai.get("emote_focus") or {}
    meta: dict[str, Any] = {
        "prompt": ai.get("prompt", ""),
        "params": params,
        "seed": params.get("seed", ai.get("seed")),
        "prediction_id": ai.get("prediction_id"),
        "cost_usd": (ai.get("costs_usd") or {}).get("generate", 0.0),
        "decor": decor.get("mode"),
        "focus_box": focus.get("source_box"),
    }
    # The framing is reusable only when it placed the character alone; a clip made with
    # the decor in the frame is mapped from its model input instead.
    if ai.get("framing") and separate and not ai.get("reprocess"):
        meta["framing"] = ai["framing"]
        meta["framing_subject"] = "character"
    if ai.get("reprocess") and ai["reprocess"].get("framing_known"):
        meta["framing"] = ai.get("framing")
        meta["framing_subject"] = "character"
    return meta


def from_output(
    raw_dir: Path, ai_dir: Path, protect: tuple[str, ...] = (), **extra: Any
) -> RawClip | None:
    """Keep the clip of a finished ``animate`` output (``ai/``) in the library."""
    video = ai_dir / "raw.mp4"
    model_input = ai_dir / "model_input.png"
    manifest_path = ai_dir / "manifest.json"
    if not (video.is_file() and model_input.is_file() and manifest_path.is_file()):
        return None
    manifest = json.loads(manifest_path.read_text())
    meta = {**meta_from_manifest(manifest), **extra}
    preview = ai_dir / PREVIEW
    return save(raw_dir, video, model_input, meta, preview, protect=protect)


def import_legacy(raw_dir: Path, ai_dir: Path) -> RawClip | None:
    """Migration: a sticker made before the raw clip library has only ``ai/raw.mp4``.

    It is kept as a clip.
    """
    if list_clips(raw_dir):
        return None
    video = ai_dir / "raw.mp4"
    model_input = ai_dir / "model_input.png"
    manifest_path = ai_dir / "manifest.json"
    if not (video.is_file() and model_input.is_file() and manifest_path.is_file()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    created = datetime.fromtimestamp(manifest_path.stat().st_mtime).astimezone()
    meta = {
        **meta_from_manifest(manifest),
        "legacy": True,
        "created": created.isoformat(timespec="seconds"),
    }
    clip_id = new_id(None, created)
    return save(raw_dir, video, model_input, meta, ai_dir / PREVIEW, clip_id=clip_id)
