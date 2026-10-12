"""AI mode: PNG -> description -> Wan clip -> transparent loop -> exports, manifest, page.

Output layout: ``out_dir/<input stem>/ai/`` holds the target files, ``raw.mp4``,
``model_input.png``, ``prompt.txt``, ``manifest.json`` and ``index.html``. The cost
journal is ``out_dir/costs.jsonl``; raw videos are cached in ``out_dir/.cache/ai/``.

With an emote focus (``emote_focus`` other than ``full``) the emote targets are cut to the
key zone and the full-composition emotes go to ``full/`` next to them for comparison;
sticker targets and OBS always get the full composition.

Decor (``decor``): with ``auto`` the stars, moon and other detached decor are taken
out of the model input, the clip is generated for the character alone and the decor is
animated procedurally and laid back into the master (``engine.ai.decor``). Focused emotes
are cut from the character alone.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from stickermotion import __version__
from stickermotion.core.models import AnimationClass, ExportResult, ExportTarget, MasterAnimation
from stickermotion.core.specs import get_spec
from stickermotion.engine.ai import decor as decor_layers
from stickermotion.engine.ai.chain import CutMethod, process
from stickermotion.engine.ai.costs import DEFAULT_BUDGET_USD, CostLedger
from stickermotion.engine.ai.decor import DecorMode
from stickermotion.engine.ai.describe import Description, build_prompt, describe, manual
from stickermotion.engine.ai.focus import (
    AUTO,
    FULL,
    Focus,
    FocusSpec,
    crop_frames,
    motion_square,
    parse_spec,
    resolve,
    to_video_box,
)
from stickermotion.engine.ai.framing import Margins, canvas_for, frame_for_model, margins_for
from stickermotion.engine.ai.generate import WAN_MODEL, generate, wan_params
from stickermotion.engine.ai.matting import RembgMatter
from stickermotion.engine.ai.provider import ReplicateClient, Transport, read_token
from stickermotion.engine.ai.skin import SkinFix
from stickermotion.engine.ai.video import read_frames
from stickermotion.exporters import ExportFitError, export_target
from stickermotion.report import write_index

AI_DIR = "ai"
COSTS_FILE = "costs.jsonl"
CACHE_DIR = ".cache/ai"
FULL_EMOTES_DIR = "full"


@dataclass
class AnimateResult:
    root: Path
    description: Description
    results: list[ExportResult]
    spent_usd: float
    cached: bool
    timings: dict[str, float] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    compare: list[ExportResult] = field(default_factory=list)
    """Full-composition emotes exported next to the focused ones (``full/``)."""

    @property
    def page(self) -> Path:
        return self.root / "index.html"


def is_emote(target: ExportTarget) -> bool:
    return get_spec(target).animation_class is AnimationClass.EMOTE


def _export_all(master: Any, targets: Sequence[ExportTarget], root: Path) -> list[ExportResult]:
    results: list[ExportResult] = []
    for target in targets:
        try:
            results.extend(export_target(master, target, root))
        except ExportFitError as exc:
            if exc.result is None:
                raise
            results.append(exc.result)
    return results


def animate(
    input_path: Path | str,
    prompt: str | None = None,
    targets: Sequence[ExportTarget | str] | None = None,
    out_dir: Path | str = "out",
    budget_usd: float = DEFAULT_BUDGET_USD,
    regenerate: bool = False,
    margins: Margins | None = None,
    cut: CutMethod = CutMethod.CHROMA,
    client: ReplicateClient | None = None,
    transport: Transport | None = None,
    dotenv: Path | None = Path(".env"),
    matter: RembgMatter | None = None,
    frames: int | None = None,
    fps: int | None = None,
    emote_focus: str | FocusSpec = AUTO,
    name: str | None = None,
    progress: Callable[[str], None] | None = None,
    decor: DecorMode | str = DecorMode.AUTO,
    resolution: str | None = None,
    seed: int | None = None,
) -> AnimateResult:
    """Run the AI chain for one PNG. ``client`` / ``transport`` replace the network in tests.

    ``frames`` / ``fps``: Wan ``num_frames`` / ``frames_per_second`` (ADR-015 defaults when
    None). ``emote_focus``: ``auto`` (default, ADR-013), ``full`` or ``X,Y,W,H`` in input PNG
    pixels. ``name``: output subdirectory under ``out_dir`` instead of the input stem.
    ``progress`` is called with ``generate``, ``cut`` and ``export`` as each step starts.
    ``decor``: ``auto`` (separate decor layers when found), ``off`` (decor removed) or
    ``together`` (decor stays in the model input). ``resolution``: Wan ``480p`` (default)
    or ``720p``. ``margins``: fixed framing margins; by default they follow the action
    (``framing.margins_for``). ``seed``: Wan seed, recorded in the manifest; None
    leaves it to the model and keeps the cache keys of earlier runs.
    """
    params = wan_params(frames, fps, resolution, seed)
    canvas = canvas_for(params["resolution"])
    focus_spec = parse_spec(emote_focus) if isinstance(emote_focus, str) else emote_focus
    source = Path(input_path)
    out = Path(out_dir)
    target_list = [ExportTarget(t) for t in targets] if targets else list(ExportTarget)
    root = out / (name or source.stem) / AI_DIR
    root.mkdir(parents=True, exist_ok=True)
    if client is None:
        ledger = CostLedger(out / COSTS_FILE, budget_usd)
        kwargs: dict[str, Any] = {"transport": transport} if transport else {}
        client = ReplicateClient(read_token(dotenv), ledger, **kwargs)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    t0 = time.perf_counter()
    with Image.open(source) as opened:
        original = opened.convert("RGBA")
    decor_plan = decor_layers.plan(original, decor, client)
    # Everything mapped through the framing works on the character alone.
    image = decor_plan.character
    timings["decor_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if prompt:
        description = manual(prompt)
    else:
        description = describe(client, frame_for_model(image, canvas, margins)[0], image)
        if decor_plan.separate and description.decorations:
            # The decor is animated on its own: the video prompt says nothing about it.
            description = Description(
                build_prompt(description.action), description.source, description.action
            )
    timings["describe_s"] = time.perf_counter() - t0
    (root / "prompt.txt").write_text(description.prompt + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    used_margins = margins or margins_for(description.action)
    frame, framing = frame_for_model(image, canvas, used_margins)
    frame.save(root / "model_input.png")
    timings["frame_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    focus = (
        resolve(focus_spec, image, client)
        if any(map(is_emote, target_list))
        else Focus(focus_spec.mode, FULL)
    )
    timings["focus_s"] = time.perf_counter() - t0

    step = progress or (lambda _step: None)
    t0 = time.perf_counter()
    step("generate")
    generation = generate(
        client, frame, description.prompt, out / CACHE_DIR, regenerate, params=params
    )
    timings["generate_s"] = time.perf_counter() - t0
    shutil.copyfile(generation.video, root / "raw.mp4")

    t0 = time.perf_counter()
    step("cut")
    clip, clip_fps = read_frames(generation.video)
    skin = SkinFix.build(image, frame, clip[0].size)
    chain = process(clip, clip_fps, cut, framing.background, matter, skin)
    master = chain.master
    composition: decor_layers.Composition | None = None
    if decor_plan.animated:
        placed = decor_layers.place(decor_plan, framing, clip[0].size)
        composition = decor_layers.compose(chain.loop.frames, clip_fps, placed)
        master = composition.master
    focus_info: dict[str, Any] = focus.to_dict()
    emote_master: MasterAnimation | None = None
    if focus.active:
        assert focus.source_box is not None
        zone = to_video_box(focus.source_box, image, framing, clip[0].size)
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
    if emote_master is None:
        results = _export_all(master, target_list, root)
        compare: list[ExportResult] = []
    else:
        results = _export_all(master, full_targets, root)
        results += _export_all(emote_master, emote_targets, root)
        compare = _export_all(master, emote_targets, root / FULL_EMOTES_DIR)
    timings["export_s"] = time.perf_counter() - t0
    total_s = time.perf_counter() - started

    run_entries = client.ledger.entries
    describe_cost = sum(e["cost_usd"] for e in run_entries if e.get("step") == "describe")
    focus_cost = sum(e["cost_usd"] for e in run_entries if e.get("step") == "focus")
    decor_cost = sum(e["cost_usd"] for e in run_entries if e.get("step") == "decor")
    decor_info = decor_plan.to_dict()
    decor_info["composition"] = composition.to_dict() if composition else None
    manifest: dict[str, Any] = {
        "stickermotion": __version__,
        "mode": "ai",
        "input": source.name,
        "total_s": round(total_s, 3),
        "targets": [t.value for t in target_list],
        "ai": {
            "prompt": description.prompt,
            "prompt_source": description.source,
            "action": description.action,
            "decorations": description.decorations,
            "model": WAN_MODEL,
            "params": params,
            "cache_key": generation.cache_key,
            "cached": generation.cached,
            "prediction_id": generation.prediction_id,
            "costs_usd": {
                "describe": round(describe_cost, 6),
                "focus": round(focus_cost, 6),
                "decor": round(decor_cost, 6),
                "generate": generation.cost_usd,
                "total": client.ledger.spent_usd,
            },
            "budget_usd": client.ledger.budget_usd,
            "run_id": client.ledger.run_id,
            "framing": framing.to_dict(),
            "decor": decor_info,
            "clip": {
                "frames": len(clip),
                "fps": clip_fps,
                "duration_s": round(len(clip) / clip_fps, 3),
                "size": list(clip[0].size),
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
                "name": "ai",
                "renders": [
                    {
                        "class": f"{chain.loop.method} loop",
                        "frames": master.frame_count,
                        "fps": master.fps,
                        "render_s": round(timings["cut_loop_s"], 3),
                    }
                ],
                "exports": [
                    {**r.model_dump(mode="json"), "path": r.path.relative_to(root).as_posix()}
                    for r in results
                ],
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
                "exports": [
                    {**r.model_dump(mode="json"), "path": r.path.relative_to(root).as_posix()}
                    for r in compare
                ],
            }
        )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    write_index(root, manifest)
    return AnimateResult(
        root,
        description,
        results,
        client.ledger.spent_usd,
        generation.cached,
        timings,
        manifest,
        compare,
    )
