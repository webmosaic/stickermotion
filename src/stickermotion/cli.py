"""Command-line interface (typer): render, validate, presets and export targets."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer

from stickermotion import __version__
from stickermotion.core.models import ExportResult, ExportTarget
from stickermotion.core.specs import KB, MB, SPECS, SizeVariant, TargetSpec

app = typer.Typer(help="StickerMotion: animated stickers and emotes from a raster image.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"stickermotion {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """StickerMotion CLI."""


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    if value % MB == 0:
        return f"{value // MB} MB"
    return f"{value // KB} KB"


def _fmt_variant(variant: SizeVariant) -> str:
    if variant.fixed_sizes:
        size = "/".join(str(s) for s in variant.fixed_sizes)
    elif variant.min_side is not None or variant.max_side is not None:
        size = f"{variant.min_side or 0}-{variant.max_side or 'inf'}"
    else:
        size = "any"
    return f"{variant.name}: {size}, {_fmt_bytes(variant.max_bytes)}"


def _fmt_num(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:g}{suffix}"


def _row(spec: TargetSpec) -> list[str]:
    fmt = ",".join(spec.formats)
    if spec.codec:
        fmt += f" ({spec.codec})"
    return [
        spec.target.value,
        spec.animation_class.value,
        fmt,
        "; ".join(_fmt_variant(v) for v in spec.size_variants),
        _fmt_num(spec.max_frames),
        _fmt_num(spec.max_duration_s, "s"),
        _fmt_num(spec.max_fps),
        "binary" if spec.binary_alpha else ("yes" if spec.requires_alpha else "no"),
    ]


HEADERS = ["target", "class", "format", "size: px, max weight", "frames", "dur", "fps", "alpha"]


def render_targets_table() -> str:
    """Render all target limits as a plain-text table."""
    rows = [HEADERS, *(_row(spec) for spec in SPECS.values())]
    widths = [max(len(row[i]) for row in rows) for i in range(len(HEADERS))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


@app.command()
def targets() -> None:
    """Print export targets and their platform limits."""
    typer.echo(render_targets_table())


def _result_line(result: ExportResult, base: Path | None = None) -> str:
    path = result.path.relative_to(base) if base else result.path
    status = "ok" if result.valid else "INVALID"
    line = (
        f"{status:7} {path}  {result.width}x{result.height}  {result.size_bytes / KB:.1f} KB  "
        f"{result.frame_count} frames  {result.duration_s:.2f} s"
    )
    details = [*result.violations, *(f"warning: {w}" for w in result.warnings)]
    return "\n".join([line, *(f"        - {d}" for d in details)])


@app.command()
def render(
    input_path: Annotated[Path, typer.Argument(metavar="INPUT.png", help="Transparent PNG.")],
    preset: Annotated[
        list[str] | None, typer.Option("--preset", help="Preset name; repeat for several.")
    ] = None,
    target: Annotated[
        list[ExportTarget] | None, typer.Option("--target", help="Target; repeat for several.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("out"),
) -> None:
    """Render presets for INPUT and export them to targets (default: all presets, all targets)."""
    from stickermotion.pipeline import run

    if not input_path.is_file():
        typer.echo(f"error: {input_path} not found", err=True)
        raise typer.Exit(2)
    started = time.perf_counter()
    try:
        results = run(input_path, preset or None, target or None, out)
    except (KeyError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    root = out / input_path.stem
    for result in results:
        typer.echo(_result_line(result, root))
    invalid = sum(1 for r in results if not r.valid)
    typer.echo(
        f"{len(results)} files, {invalid} invalid, {time.perf_counter() - started:.1f} s; "
        f"page: {root / 'index.html'}"
    )
    if invalid:
        raise typer.Exit(1)


@app.command()
def animate(
    input_path: Annotated[Path, typer.Argument(metavar="INPUT.png", help="Transparent PNG.")],
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", help="Action for the video model; skips the auto description."),
    ] = None,
    target: Annotated[
        list[ExportTarget] | None, typer.Option("--target", help="Target; repeat for several.")
    ] = None,
    budget: Annotated[
        float, typer.Option("--budget", help="Spend limit for this run, USD.", min=0.0)
    ] = 0.10,
    regenerate: Annotated[
        bool, typer.Option("--regenerate", help="Ignore the cached clip and pay for a new one.")
    ] = False,
    cut: Annotated[
        str, typer.Option("--cut", help="chroma (default) or rembg (needs the rembg extra).")
    ] = "chroma",
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("out"),
    frames: Annotated[
        int | None,
        typer.Option(
            "--frames",
            help="Wan num_frames, 81..121 (default 81). The price follows the frame count.",
        ),
    ] = None,
    fps: Annotated[
        int | None,
        typer.Option(
            "--fps", help="Wan frames_per_second, 5..30 (default 30). Does not change the price."
        ),
    ] = None,
    emote_focus: Annotated[
        str,
        typer.Option(
            "--emote-focus",
            help="Emote crop: auto (default, vision model), full or X,Y,W,H in input pixels.",
        ),
    ] = "auto",
    decor: Annotated[
        str,
        typer.Option(
            "--decor",
            help="Detached decor (stars, moon...): auto (default, animated separately), "
            "off (removed) or together (kept in the video model input).",
        ),
    ] = "auto",
    resolution: Annotated[
        str,
        typer.Option("--resolution", help="Wan resolution: 480p (default, 0.05$) or 720p (0.11$)."),
    ] = "480p",
    variants: Annotated[
        int,
        typer.Option(
            "--variants",
            help="Clips to generate with different seeds (1-3, default 1). With 2-3 only "
            "Telegram sticker previews are made; reprocess the chosen clip for all files.",
            min=1,
            max=3,
        ),
    ] = 1,
    seed: Annotated[
        int | None, typer.Option("--seed", help="Wan seed (repeat a clip); one variant only.")
    ] = None,
) -> None:
    """Animate INPUT with the AI chain (description, Wan 2.2, cut, loop) and export it."""
    from stickermotion.ai_pipeline import COSTS_FILE
    from stickermotion.ai_pipeline import animate as run_animate
    from stickermotion.engine.ai import BudgetExceededError, CutMethod, ProviderError
    from stickermotion.engine.ai.decor import DecorMode

    if not input_path.is_file():
        typer.echo(f"error: {input_path} not found", err=True)
        raise typer.Exit(2)
    try:
        method = CutMethod(cut)
    except ValueError as exc:
        typer.echo(f"error: unknown cut {cut!r} (chroma, rembg)", err=True)
        raise typer.Exit(2) from exc
    try:
        decor_mode = DecorMode(decor)
    except ValueError as exc:
        typer.echo(f"error: unknown decor mode {decor!r} (auto, off, together)", err=True)
        raise typer.Exit(2) from exc
    if variants > 1:
        if seed is not None:
            typer.echo("error: --seed repeats one clip; use it without --variants", err=True)
            raise typer.Exit(2)
        _variants(input_path, prompt, variants, out, budget, decor_mode, resolution)
        return
    try:
        result = run_animate(
            input_path,
            prompt,
            target or None,
            out,
            budget,
            regenerate,
            cut=method,
            frames=frames,
            fps=fps,
            emote_focus=emote_focus,
            decor=decor_mode,
            resolution=resolution,
            seed=seed,
        )
    except (BudgetExceededError, ProviderError) as exc:
        typer.echo(f"error: {exc}", err=True)
        typer.echo(f"cost journal: {out / COSTS_FILE}", err=True)
        raise typer.Exit(3) from exc
    except (KeyError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    ai = result.manifest["ai"]
    costs = ai["costs_usd"]
    typer.echo(f"prompt ({result.description.source}): {result.description.prompt}")
    typer.echo(
        f"cost: description {costs['describe']:.4f}$, focus {costs['focus']:.4f}$, "
        f"decor {costs['decor']:.4f}$, "
        f"generation {costs['generate']:.4f}$"
        f"{' (cached)' if result.cached else ''}, total {costs['total']:.4f}$ "
        f"of {ai['budget_usd']:.2f}$"
    )
    typer.echo(f"loop: {ai['loop']['reason']}")
    found = ai["decor"]
    if found["items"]:
        names = ", ".join(f"{i['what'] or 'item'} ({i['motion']})" for i in found["items"])
        typer.echo(f"decor: {found['mode']}, {names}")
    for warning in found["warnings"]:
        typer.echo(f"warning: {warning}")
    focus = ai["emote_focus"]
    if focus["requested"] != "full":
        typer.echo(
            f"emote focus: {focus['requested']} -> {focus['mode']}, zone {focus['source_box']}"
        )
    for warning in focus["warnings"]:
        typer.echo(f"warning: {warning}")
    for item in [*result.results, *result.compare]:
        typer.echo(_result_line(item, result.root))
    every = [*result.results, *result.compare]
    invalid = sum(1 for r in every if not r.valid)
    typer.echo(
        f"{len(every)} files, {invalid} invalid, {result.manifest['total_s']:.1f} s; "
        f"raw: {result.root / 'raw.mp4'}; page: {result.page}; costs: {out / COSTS_FILE}"
    )
    if invalid:
        raise typer.Exit(1)


def _variants(
    input_path: Path,
    prompt: str | None,
    count: int,
    out: Path,
    budget: float,
    decor: object,
    resolution: str,
) -> None:
    from stickermotion.ai_pipeline import COSTS_FILE
    from stickermotion.engine.ai import BudgetExceededError, ProviderError
    from stickermotion.variants import animate_variants

    try:
        result = animate_variants(
            input_path,
            prompt,
            count,
            out,
            budget_usd=budget,
            decor=str(decor),
            resolution=resolution,
        )
    except (BudgetExceededError, ProviderError) as exc:
        typer.echo(f"error: {exc}", err=True)
        typer.echo(f"cost journal: {out / COSTS_FILE}", err=True)
        raise typer.Exit(3) from exc
    except (KeyError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"prompt ({result.description.source}): {result.description.prompt}")
    for clip, preview in zip(result.clips, result.previews, strict=True):
        typer.echo(f"variant {clip.meta.get('variant')}: seed {clip.seed}, preview {preview}")
    typer.echo(f"total {result.spent_usd:.4f}$ of {budget:.2f}$; costs: {out / COSTS_FILE}")
    first = result.clips[0]
    typer.echo(
        "full export of the chosen clip (free): stickermotion reprocess "
        f"{first.video} --input {input_path} --model-input {first.model_input}"
    )


@app.command()
def reprocess(
    raw_path: Annotated[Path, typer.Argument(metavar="RAW.mp4", help="Generated raw clip.")],
    input_path: Annotated[
        Path, typer.Option("--input", help="The transparent PNG the clip was made from.")
    ],
    model_input: Annotated[
        Path, typer.Option("--model-input", help="model_input.png the clip was generated from.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("out"),
    target: Annotated[
        list[ExportTarget] | None, typer.Option("--target", help="Target; repeat for several.")
    ] = None,
    emote_focus: Annotated[
        str,
        typer.Option(
            "--emote-focus",
            help="Emote crop: stored (default: the zone kept in the clip's meta.json, else "
            "full), full, X,Y,W,H in input pixels or auto (one cheap vision call).",
        ),
    ] = "stored",
    decor: Annotated[
        str,
        typer.Option(
            "--decor",
            help="auto (default: old decor out of the clip, procedural decor in), "
            "off (old decor out, none back) or together (clip as it is).",
        ),
    ] = "auto",
    name: Annotated[
        str | None, typer.Option("--name", help="Output subdirectory (default: input stem).")
    ] = None,
) -> None:
    """Cut and export an existing clip again with the decor as separate layers (no generation)."""
    from stickermotion.engine.ai.decor import DecorMode
    from stickermotion.reprocess import reprocess as run_reprocess

    for path in (raw_path, input_path, model_input):
        if not path.is_file():
            typer.echo(f"error: {path} not found", err=True)
            raise typer.Exit(2)
    try:
        decor_mode = DecorMode(decor)
    except ValueError as exc:
        typer.echo(f"error: unknown decor mode {decor!r} (auto, off, together)", err=True)
        raise typer.Exit(2) from exc
    client = None
    if emote_focus == "auto":
        from stickermotion.ai_pipeline import COSTS_FILE
        from stickermotion.engine.ai.costs import CostLedger
        from stickermotion.engine.ai.provider import ReplicateClient, read_token

        client = ReplicateClient(read_token(Path(".env")), CostLedger(out / COSTS_FILE, 0.01))
    # A clip from the raw library carries its framing and prompt in meta.json.
    from stickermotion.raws import META, load

    clip = load(raw_path.parent) if (raw_path.parent / META).is_file() else None
    if emote_focus == "stored":
        emote_focus = clip.focus_spec if clip else "full"
    try:
        result = run_reprocess(
            raw_path,
            input_path,
            model_input,
            out,
            name=name,
            targets=target or None,
            emote_focus=emote_focus,
            client=client,
            decor=decor_mode,
            framing=clip.framing if clip else None,
            framing_subject=clip.framing_subject if clip else "character",
            meta=clip.meta if clip else None,
        )
    except (KeyError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    info = result.manifest["ai"]["reprocess"]
    removal = info["removal"]
    if removal is None:
        typer.echo("old decor: none in the clip")
    else:
        typer.echo(
            f"old decor: {info['old_decor_items']} items tracked, removed "
            f"{removal['removed_px'][0]}..{removal['removed_px'][1]} px per frame, "
            f"left {removal['residual_px_max']} px, character loss "
            f"{removal['character_loss_max']:.2%}"
        )
        for warning in removal["warnings"]:
            typer.echo(f"warning: {warning}")
    found = result.manifest["ai"]["decor"]
    if found["items"]:
        names = ", ".join(f"item {i['index']} ({i['motion']})" for i in found["items"])
        typer.echo(f"decor: {found['mode']}, {names}")
    every = [*result.results, *result.compare]
    for item in every:
        typer.echo(_result_line(item, result.root))
    invalid = sum(1 for r in every if not r.valid)
    typer.echo(
        f"{len(every)} files, {invalid} invalid, {result.manifest['total_s']:.1f} s; "
        f"page: {result.page}; cost: 0$ for the clip"
    )
    if invalid:
        raise typer.Exit(1)


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host", help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port.")] = 8000,
) -> None:
    """Run the web UI (needs APP_PASSWORD; DATA_DIR, DAILY_BUDGET_USD optional)."""
    import uvicorn

    from stickermotion.web.app import create_app
    from stickermotion.web.config import Settings

    try:
        settings = Settings.from_env()
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"StickerMotion web UI on http://{host}:{port}, data in {settings.data_dir}")
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")


@app.command()
def validate(
    file: Annotated[Path, typer.Argument(help="Exported file.")],
    target: Annotated[ExportTarget, typer.Argument(help="Export target.")],
    variant: Annotated[
        str | None, typer.Option("--variant", help="Size variant (twitch_gif: auto|manual).")
    ] = None,
) -> None:
    """Check FILE against the limits of TARGET."""
    from stickermotion.validator import UnreadableFileError
    from stickermotion.validator import validate as validate_file

    try:
        result = validate_file(file, target, variant)
    except (FileNotFoundError, UnreadableFileError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(_result_line(result))
    if not result.valid:
        raise typer.Exit(1)


@app.command()
def presets() -> None:
    """List animation presets with descriptions."""
    from stickermotion.engine.presets import list_presets

    for info in list_presets():
        classes = "/".join(c.value for c in info.supported_classes)
        typer.echo(f"{info.name:16} {classes:14} {info.description}")


if __name__ == "__main__":  # pragma: no cover
    app()
