"""Several clips of one description at once, each with its own seed, to choose from.

Wan 2.2 fast has no negative prompt, so the same prompt gives a calm yawn one time and a
wide "singing" mouth the next. ``animate_variants`` describes and frames the character
once, generates ``count`` clips with different seeds and gives each a quick Telegram
sticker preview (cut, loop, decor, one export). Every clip goes to the raw library
(``raws``); the full export of the chosen one is ``reprocess``, which costs nothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from stickermotion import raws
from stickermotion.ai_pipeline import CACHE_DIR, COSTS_FILE
from stickermotion.core.models import ExportTarget
from stickermotion.engine.ai import decor as decor_layers
from stickermotion.engine.ai.costs import DEFAULT_BUDGET_USD, CostLedger
from stickermotion.engine.ai.decor import DecorMode
from stickermotion.engine.ai.describe import Description, build_prompt, describe, manual
from stickermotion.engine.ai.focus import AUTO, FULL, parse_spec, resolve
from stickermotion.engine.ai.framing import canvas_for, frame_for_model, margins_for
from stickermotion.engine.ai.generate import (
    WAN_MODEL,
    generate,
    random_seed,
    wan_params,
    wan_price,
)
from stickermotion.engine.ai.provider import ReplicateClient, Transport, read_token
from stickermotion.reprocess import reprocess

MAX_VARIANTS = 3
VARIANTS_DIR = "variants"


@dataclass
class VariantsResult:
    description: Description
    clips: list[raws.RawClip]
    spent_usd: float
    batch: str
    previews: list[Path] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


def check_count(count: int) -> int:
    if not 1 <= count <= MAX_VARIANTS:
        raise ValueError(f"variants must be in 1..{MAX_VARIANTS}, got {count}")
    return count


def batch_estimate_usd(count: int, resolution: str | None = None) -> float:
    """Upper price of ``count`` Wan clips (the cheap vision calls are on top)."""
    return round(count * wan_price(wan_params(resolution=resolution)).estimate_usd, 6)


def animate_variants(
    input_path: Path | str,
    prompt: str | None = None,
    count: int = 2,
    out_dir: Path | str = "out",
    name: str | None = None,
    raw_dir: Path | None = None,
    budget_usd: float = DEFAULT_BUDGET_USD,
    client: ReplicateClient | None = None,
    transport: Transport | None = None,
    dotenv: Path | None = Path(".env"),
    decor: DecorMode | str = DecorMode.AUTO,
    resolution: str | None = None,
    seeds: Sequence[int] | None = None,
    emote_focus: str = AUTO,
    progress: Callable[[str], None] | None = None,
) -> VariantsResult:
    """Generate ``count`` clips (1..3) with different seeds and a preview of each.

    One ledger holds the whole batch, so the run budget (``budget_usd``) covers every
    variant and the batch is refused before any paid call when the clips alone would
    exceed it. Clips go to ``raw_dir`` (default ``out_dir/<name>/raw``); the emote focus
    zone is found once and kept with each clip for the full export later.
    """
    check_count(count)
    seeds = list(seeds) if seeds is not None else [random_seed() for _ in range(count)]
    if len(seeds) != count:
        raise ValueError("one seed per variant")
    source = Path(input_path)
    base = Path(out_dir) / (name or source.stem)
    raw_root = raw_dir or base / raws.RAW_DIR
    if client is None:
        ledger = CostLedger(Path(out_dir) / COSTS_FILE, budget_usd)
        kwargs: dict[str, Any] = {"transport": transport} if transport else {}
        client = ReplicateClient(read_token(dotenv), ledger, **kwargs)
    client.ledger.check(batch_estimate_usd(count, resolution), f"{count} variants")
    step = progress or (lambda _step: None)
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    with Image.open(source) as opened:
        original = opened.convert("RGBA")
    decor_plan = decor_layers.plan(original, decor, client)
    image = decor_plan.character
    params0 = wan_params(resolution=resolution)
    canvas = canvas_for(params0["resolution"])
    if prompt:
        description = manual(prompt)
    else:
        description = describe(client, frame_for_model(image, canvas)[0], image)
        if decor_plan.separate and description.decorations:
            description = Description(
                build_prompt(description.action), description.source, description.action
            )
    frame, framing = frame_for_model(image, canvas, margins_for(description.action))
    focus = resolve(parse_spec(emote_focus), image, client) if emote_focus != FULL else None
    focus_box = list(focus.source_box) if focus is not None and focus.source_box else None
    timings["prepare_s"] = time.perf_counter() - t0

    batch = raws.new_id()
    clips: list[raws.RawClip] = []
    previews: list[Path] = []
    work = base / VARIANTS_DIR
    work.mkdir(parents=True, exist_ok=True)
    model_input = work / "model_input.png"
    frame.save(model_input)
    for index, seed in enumerate(seeds):
        t0 = time.perf_counter()
        step("generate")
        params = wan_params(resolution=resolution, seed=seed)
        generation = generate(
            client, frame, description.prompt, Path(out_dir) / CACHE_DIR, False, params=params
        )
        timings[f"generate_{index + 1}_s"] = time.perf_counter() - t0
        meta = {
            "prompt": description.prompt,
            "params": params,
            "seed": seed,
            "model": WAN_MODEL,
            "prediction_id": generation.prediction_id,
            "cost_usd": generation.cost_usd,
            "cached": generation.cached,
            "framing": framing.to_dict(),
            "framing_subject": "character" if decor_plan.separate else "composition",
            "decor": decor_plan.mode.value,
            "focus_box": focus_box,
            "batch": batch,
            "variant": index + 1,
        }
        protect = tuple(c.id for c in clips)
        clip = raws.save(raw_root, generation.video, model_input, meta, protect=protect)
        clips.append(clip)
        t0 = time.perf_counter()
        step("cut")
        result = reprocess(
            clip.video,
            source,
            clip.model_input,
            targets=[ExportTarget.TELEGRAM_STICKER],
            emote_focus=FULL,
            decor=decor,
            framing=framing,
            framing_subject=meta["framing_subject"],
            meta=clip.meta,
            root=clip.folder,
        )
        previews.append(result.results[0].path)
        timings[f"preview_{index + 1}_s"] = time.perf_counter() - t0
    return VariantsResult(description, clips, client.ledger.spent_usd, batch, previews, timings)
