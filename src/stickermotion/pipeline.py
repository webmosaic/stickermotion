"""Full stage-1 cycle: PNG -> preprocess -> preset master -> exports -> manifest and page.

A master is rendered once per (preset, class) and reused by every target of that class.
Output layout: ``out_dir/<input stem>/<preset>/<target files>`` plus ``manifest.json`` and
``index.html`` in ``out_dir/<input stem>/``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from stickermotion import __version__
from stickermotion.core.models import AnimationClass, ExportResult, ExportTarget
from stickermotion.core.specs import get_spec
from stickermotion.engine.presets import get_preset, list_presets
from stickermotion.exporters import ExportFitError, export_target
from stickermotion.preprocess import normalize
from stickermotion.report import write_index


def master_class(target: ExportTarget) -> AnimationClass:
    """Preset class for a target: emote targets get emotes, stickers and overlay stickers."""
    if get_spec(target).animation_class is AnimationClass.EMOTE:
        return AnimationClass.EMOTE
    return AnimationClass.STICKER


def _result_entry(result: ExportResult, root: Path, seconds: float) -> dict[str, Any]:
    entry = result.model_dump(mode="json")
    entry["path"] = result.path.relative_to(root).as_posix()
    entry["export_s"] = round(seconds, 3)
    return entry


def run(
    input_path: Path | str,
    presets: Sequence[str] | None = None,
    targets: Sequence[ExportTarget | str] | None = None,
    out_dir: Path | str = "out",
) -> list[ExportResult]:
    """Render every preset for the input and export it to every target.

    ``presets`` / ``targets`` default to all. Files that fail the weight fit are still
    returned (invalid, with the violation) so the caller and the page can show them.
    """
    source = Path(input_path)
    preset_names = list(presets) if presets else [info.name for info in list_presets()]
    target_list = [ExportTarget(t) for t in targets] if targets else list(ExportTarget)
    for name in preset_names:
        get_preset(name)  # fail fast on unknown names

    root = Path(out_dir) / source.stem
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with Image.open(source) as image:
        prepared = normalize(image)
    preprocess_s = time.perf_counter() - started

    results: list[ExportResult] = []
    manifest_presets: list[dict[str, Any]] = []
    for name in preset_names:
        preset_cls = get_preset(name)
        preset_dir = root / name
        renders: list[dict[str, Any]] = []
        exports: list[dict[str, Any]] = []
        classes = sorted({master_class(t) for t in target_list}, key=list(AnimationClass).index)
        for anim_class in classes:
            preset = preset_cls(anim_class)
            t0 = time.perf_counter()
            master = preset.render(prepared)
            render_s = time.perf_counter() - t0
            renders.append(
                {
                    "class": anim_class.value,
                    "params": preset.params.model_dump(mode="json"),
                    "frames": master.frame_count,
                    "fps": master.fps,
                    "size": list(master.size),
                    "render_s": round(render_s, 3),
                }
            )
            for target in (t for t in target_list if master_class(t) is anim_class):
                t1 = time.perf_counter()
                try:
                    produced = export_target(master, target, preset_dir)
                except ExportFitError as exc:
                    if exc.result is None:
                        raise
                    produced = [exc.result]
                seconds = (time.perf_counter() - t1) / len(produced)
                for result in produced:
                    results.append(result)
                    exports.append(
                        {
                            "preset": name,
                            "class": anim_class.value,
                            **_result_entry(result, root, seconds),
                        }
                    )
        manifest_presets.append({"name": name, "renders": renders, "exports": exports})

    manifest = {
        "stickermotion": __version__,
        "input": source.name,
        "source_size": list(prepared.source_size),
        "preprocess_s": round(preprocess_s, 3),
        "total_s": round(time.perf_counter() - started, 3),
        "targets": [t.value for t in target_list],
        "presets": manifest_presets,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    write_index(root, manifest)
    return results
