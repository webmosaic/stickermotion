"""Reprocess an existing raw clip: cut, old decor out, new decor in, exports.

No new generation: a liked motion is reused. Output layout is the one of
``ai_pipeline.animate`` (``out_dir/<name>/ai/`` with the targets, ``raw.mp4``,
``model_input.png``, ``manifest.json`` and ``index.html``), so the web UI shows it the same.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from stickermotion import __version__
from stickermotion.ai_pipeline import AI_DIR, FULL_EMOTES_DIR, _export_all, is_emote
from stickermotion.core.models import ExportResult, ExportTarget, MasterAnimation
from stickermotion.engine.ai import decor as decor_layers
from stickermotion.engine.ai import redecor
from stickermotion.engine.ai.chain import CutMethod, process
from stickermotion.engine.ai.decor import DecorMode
from stickermotion.engine.ai.focus import (
    FULL,
    Focus,
    FocusSpec,
    crop_frames,
    motion_square,
    parse_spec,
    resolve,
    to_video_box,
)
from stickermotion.engine.ai.framing import Framing
from stickermotion.engine.ai.generate import WAN_MODEL
from stickermotion.engine.ai.matting import RembgMatter
from stickermotion.engine.ai.provider import ReplicateClient
from stickermotion.engine.ai.skin import SkinFix
from stickermotion.engine.ai.video import read_frames
from stickermotion.report import write_index


@dataclass
class ReprocessResult:
    root: Path
    results: list[ExportResult]
    compare: list[ExportResult]
    manifest: dict[str, Any]
    removal: redecor.Removal | None
    """Old decor removal (None when the clip had no decor in it)."""

    @property
    def page(self) -> Path:
        return self.root / "index.html"


def _exports(results: Sequence[ExportResult], root: Path) -> list[dict[str, Any]]:
    return [
        {**r.model_dump(mode="json"), "path": r.path.relative_to(root).as_posix()} for r in results
    ]


def old_decor_masks(
    original: Image.Image,
    plan: decor_layers.DecorPlan,
    mapping: redecor.Mapping,
    model_frame: Image.Image,
) -> list[Any]:
    """Masks (video pixels) of the planned decor items the model input frame still shows."""
    if plan.requested is DecorMode.TOGETHER or not plan.items:
        return []
    wanted = {i.index for i in plan.items}
    candidates, _ = decor_layers.find_candidates(original)
    masks = [mapping.mask(c.mask, model_frame.size) for c in candidates if c.index in wanted]
    masks = [m for m in masks if m.any()]
    present = redecor.decor_in_clip(model_frame, masks)
    return [m for m, shown in zip(masks, present, strict=True) if shown]


def reprocess(
    raw_path: Path | str,
    input_path: Path | str,
    model_input_path: Path | str,
    out_dir: Path | str = "out",
    name: str | None = None,
    targets: Sequence[ExportTarget | str] | None = None,
    emote_focus: str | FocusSpec = FULL,
    client: ReplicateClient | None = None,
    decor: DecorMode | str = DecorMode.AUTO,
    cut: CutMethod = CutMethod.CHROMA,
    matter: RembgMatter | None = None,
    framing: Framing | None = None,
    framing_subject: str = "character",
    meta: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    root: Path | None = None,
) -> ReprocessResult:
    """Cut and export an existing raw clip again, without a new generation.

    The decor of ``input_path`` is found by its shape (no paid call). When the model input
    of the clip still shows it (clips made with the decor in the frame), the old decor is
    tracked and removed from every cut frame (``engine.ai.redecor``); then the procedural
    decor from the input PNG is laid back, standing on its place. ``framing``: the framing
    the clip was generated with (from its metadata); when None it is inferred from the
    model input. ``emote_focus``: ``full`` (default), ``X,Y,W,H`` in input pixels or
    ``auto`` (needs ``client``, one cheap vision call). ``meta`` (prompt, params, seed...)
    goes into the manifest. ``progress`` gets ``cut`` and ``export``. ``framing_subject``:
    ``character`` (the framing placed the character alone) or ``composition``. ``root``
    writes the output straight into that folder instead of ``out_dir/<name>/ai/``.
    """
    started = time.perf_counter()
    timings: dict[str, float] = {}
    step = progress or (lambda _step: None)
    source = Path(input_path)
    raw = Path(raw_path)
    target_list = [ExportTarget(t) for t in targets] if targets else list(ExportTarget)
    focus_spec = parse_spec(emote_focus) if isinstance(emote_focus, str) else emote_focus
    root = Path(root) if root is not None else Path(out_dir) / (name or source.stem) / AI_DIR
    root.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})

    t0 = time.perf_counter()
    with Image.open(source) as opened:
        original = opened.convert("RGBA")
    # Free: the shape heuristic picks the motions, no vision call.
    decor_plan = decor_layers.plan(original, decor, None)
    image = decor_plan.character
    with Image.open(model_input_path) as opened:
        model_frame = opened.convert("RGB")
    clip, clip_fps = read_frames(raw)
    size = clip[0].size
    if model_frame.size != size:
        model_frame = model_frame.resize(size, Image.Resampling.LANCZOS)
    if framing is not None:
        subject = original if framing_subject == "composition" else image
        mapping = redecor.Mapping.from_framing(framing, subject, size, framing_subject)
    else:
        mapping = redecor.infer_mapping(original, image, model_frame)
    used_framing = mapping.framing(image, size)
    timings["load_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    masks = old_decor_masks(original, decor_plan, mapping, model_frame)
    removal: redecor.Removal | None = None
    clean: Callable[[list[Image.Image]], list[Image.Image]] | None = None
    if masks:
        shifts = redecor.track(clip, model_frame, masks)

        def clean(frames: list[Image.Image]) -> list[Image.Image]:
            nonlocal removal
            removal = redecor.remove(frames, masks, shifts)
            return removal.frames

    timings["track_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    step("cut")
    skin = SkinFix.build(image, model_frame, size)
    chain = process(clip, clip_fps, cut, used_framing.background, matter, skin, clean)
    master = chain.master
    composition: decor_layers.Composition | None = None
    if decor_plan.animated:
        placed = decor_layers.place(decor_plan, used_framing, size)
        composition = decor_layers.compose(chain.loop.frames, clip_fps, placed)
        master = composition.master
    focus = (
        resolve(focus_spec, image, client)
        if any(map(is_emote, target_list))
        else Focus(focus_spec.mode, FULL)
    )
    focus_info: dict[str, Any] = focus.to_dict()
    emote_master: MasterAnimation | None = None
    if focus.active:
        assert focus.source_box is not None
        zone = to_video_box(focus.source_box, image, used_framing, size)
        square = motion_square(zone, chain.loop.frames)
        emote_master = MasterAnimation(
            frames=crop_frames(chain.loop.frames, square), fps=clip_fps, loop=True
        )
        focus_info.update(video_zone=list(zone), video_square=list(square))
    timings["cut_loop_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    step("export")
    emote_targets = [t for t in target_list if is_emote(t)]
    full_targets = [t for t in target_list if not is_emote(t)]
    compare: list[ExportResult] = []
    if emote_master is None:
        results = _export_all(master, target_list, root)
    else:
        results = _export_all(master, full_targets, root)
        results += _export_all(emote_master, emote_targets, root)
        compare = _export_all(master, emote_targets, root / FULL_EMOTES_DIR)
    timings["export_s"] = time.perf_counter() - t0

    if raw.resolve() != (root / "raw.mp4").resolve():
        shutil.copyfile(raw, root / "raw.mp4")
    model_frame.save(root / "model_input.png")
    spent = round(client.ledger.spent_usd, 6) if client is not None else 0.0
    decor_info = decor_plan.to_dict()
    decor_info["composition"] = composition.to_dict() if composition else None
    params = dict(meta.get("params") or {})
    prompt = str(meta.get("prompt") or "")
    manifest: dict[str, Any] = {
        "stickermotion": __version__,
        "mode": "ai",
        "input": source.name,
        "total_s": round(time.perf_counter() - started, 3),
        "targets": [t.value for t in target_list],
        "ai": {
            "prompt": prompt,
            "prompt_source": "reprocess",
            "action": prompt,
            "model": WAN_MODEL,
            "params": params,
            "seed": params.get("seed", meta.get("seed")),
            "cached": True,
            "prediction_id": meta.get("prediction_id"),
            "costs_usd": {"generate": 0.0, "focus": spent, "total": spent},
            "reprocess": {
                "raw": raw.name,
                "raw_id": meta.get("id"),
                "mapping": mapping.to_dict(),
                "framing_known": framing is not None,
                "old_decor_items": len(masks),
                "removal": removal.to_dict() if removal is not None else None,
            },
            "framing": used_framing.to_dict(),
            "decor": decor_info,
            "clip": {
                "frames": len(clip),
                "fps": clip_fps,
                "duration_s": round(len(clip) / clip_fps, 3),
                "size": list(size),
            },
            "emote_focus": focus_info,
            "cut": chain.cut.value,
            "skin_fix": skin.to_dict() if skin is not None else None,
            "loop": {
                "method": chain.loop.method,
                "frames": master.frame_count,
                "duration_s": round(master.duration_s, 3),
                "seam_ratio": round(chain.loop.seam.ratio, 3),
                "reason": chain.loop.reason,
            },
            "edge_touch": chain.edge_touch,
            "master_size": list(master.size),
            "timings_s": {k: round(v, 3) for k, v in timings.items()},
            "raw": "raw.mp4",
            "model_input": "model_input.png",
        },
        "presets": [
            {
                "name": "ai (reprocessed)",
                "renders": [
                    {
                        "class": f"{chain.loop.method} loop",
                        "frames": master.frame_count,
                        "fps": master.fps,
                        "render_s": round(timings["cut_loop_s"], 3),
                    }
                ],
                "exports": _exports(results, root),
            }
        ],
    }
    if compare:
        focused = [r for r in results if is_emote(r.target)]
        manifest["ai"]["emote_focus"]["pairs"] = [
            {
                "target": f.target.value,
                "name": f.path.name,
                "focus": f.path.relative_to(root).as_posix(),
                "full": c.path.relative_to(root).as_posix(),
            }
            for f, c in zip(focused, compare, strict=True)
        ]
        manifest["presets"].append(
            {
                "name": "full composition emotes (comparison)",
                "renders": [],
                "exports": _exports(compare, root),
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    write_index(root, manifest)
    return ReprocessResult(root, results, compare, manifest, removal)
