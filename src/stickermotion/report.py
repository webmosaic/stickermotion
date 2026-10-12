"""Self-contained comparison page (index.html) for reviewing a render.

Every exported file is shown on three backgrounds (Twitch dark #18181B, white, checker),
large for inspection and at its real display size. Links are relative, no external assets.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from stickermotion.core.models import ExportTarget
from stickermotion.core.specs import KB

PREVIEW_PX = 160
# Size at which the platform actually shows the file (emotes in chat, stickers in a message).
REAL_SIZE_PX: dict[ExportTarget, int] = {
    ExportTarget.TWITCH_GIF: 28,
    ExportTarget.DISCORD_EMOJI: 28,
    ExportTarget.TELEGRAM_EMOJI: 28,
    ExportTarget.DISCORD_STICKER: 160,
    ExportTarget.TELEGRAM_STICKER: 256,
}
BACKGROUNDS = (("dark", "Twitch dark"), ("light", "White"), ("checker", "Checker"))

_CSS = """
:root {
  --page: #f3f3f5; --ink: #1d1d22; --muted: #6a6a75; --line: #d9d9de; --panel: #ffffff;
  --bad: #c62828; --good: #2e7d32; --warn: #a15c00;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --page: #101013; --ink: #e8e8ec; --muted: #9a9aa5; --line: #2c2c33; --panel: #18181d;
    --bad: #ff6b6b; --good: #7bd88f; --warn: #f0b35a;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; font-weight: 650; }
h2 { font-size: 20px; margin: 40px 0 4px; font-weight: 650; }
.lede, .render { color: var(--muted); margin: 0 0 12px; }
.file { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 14px; margin: 12px 0; }
.file h3 { font-size: 15px; margin: 0 0 2px; font-weight: 600; }
.meta { color: var(--muted); font-size: 13px; margin: 0 0 10px; }
.meta .ok { color: var(--good); } .meta .fail { color: var(--bad); font-weight: 600; }
.problems { margin: 0 0 10px; padding-left: 18px; font-size: 13px; }
.problems .v { color: var(--bad); } .problems .w { color: var(--warn); }
.row { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; }
.tile { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--muted); }
.stage { display: grid; place-items: center; border-radius: 6px; overflow: hidden; }
.stage.big { width: 172px; height: 172px; }
.stage.real { padding: 10px; min-width: 48px; min-height: 48px; }
.stage img, .stage video { display: block; }
.stage.big img, .stage.big video { width: 160px; height: 160px; object-fit: contain; }
.pixelated { image-rendering: pixelated; }
.dark { background: #18181b; } .light { background: #ffffff; outline: 1px solid var(--line); }
.checker { background-color: #ffffff; background-image:
  linear-gradient(45deg, #c8c8c8 25%, transparent 25%, transparent 75%, #c8c8c8 75%),
  linear-gradient(45deg, #c8c8c8 25%, transparent 25%, transparent 75%, #c8c8c8 75%);
  background-size: 16px 16px; background-position: 0 0, 8px 8px; }
.label { margin: 10px 0 6px; font-size: 13px; color: var(--muted); }
.chat { display: flex; align-items: center; gap: 6px; padding: 8px 10px; border-radius: 6px;
  font-size: 13px; }
.chat.dark { color: #efeff1; } .chat.light { color: #1d1d22; }
.ai video, .ai img { max-width: 100%; height: auto; border-radius: 6px; display: block; }
.ai .prompt { white-space: pre-wrap; background: var(--page); border: 1px solid var(--line);
  border-radius: 6px; padding: 10px; font-size: 14px; margin: 0 0 10px; }
"""


def _media(path: str, size: int | None, pixelated: bool) -> str:
    style = f' width="{size}" height="{size}"' if size else ""
    cls = ' class="pixelated"' if pixelated else ""
    src = escape(path, quote=True)
    if path.endswith(".webm"):
        return f'<video src="{src}"{style}{cls} autoplay loop muted playsinline></video>'
    return f'<img src="{src}"{style}{cls} alt="">'


def _fmt_weight(size_bytes: int) -> str:
    return f"{size_bytes / KB:.1f} KB"


def _file_block(entry: dict[str, Any]) -> str:
    target = ExportTarget(entry["target"])
    path = entry["path"]
    width, height = entry["width"], entry["height"]
    fps = entry["frame_count"] / entry["duration_s"] if entry["duration_s"] else 0
    status = (
        '<span class="ok">valid</span>' if entry["valid"] else '<span class="fail">invalid</span>'
    )
    variant = f", {entry['size_variant']}" if entry.get("size_variant") else ""
    meta = (
        f"{width}×{height} px, {_fmt_weight(entry['size_bytes'])}, "
        f"{entry['frame_count']} frames, {entry['duration_s']:.2f} s, {fps:.1f} fps{variant}"
        f"; {status}"
    )
    problems = "".join(f'<li class="v">{escape(v)}</li>' for v in entry["violations"])
    problems += "".join(f'<li class="w">{escape(w)}</li>' for w in entry.get("warnings", []))
    problems_html = f'<ul class="problems">{problems}</ul>' if problems else ""

    small = max(width, height) < PREVIEW_PX
    big = "".join(
        f'<div class="tile"><div class="stage big {bg}">{_media(path, None, small)}</div>'
        f"{label}</div>"
        for bg, label in BACKGROUNDS
    )
    real_html = ""
    real = REAL_SIZE_PX.get(target)
    if real is not None:
        if real <= 32:
            # Emotes: inside a line of chat text, as viewers see them.
            tiles = "".join(
                f'<div class="tile"><div class="chat {bg}"><span>viewer:</span>'
                f"{_media(path, real, False)}<span>nice</span></div>{label}</div>"
                for bg, label in BACKGROUNDS
            )
        else:
            tiles = "".join(
                f'<div class="tile"><div class="stage real {bg}">{_media(path, real, False)}</div>'
                f"{label}</div>"
                for bg, label in BACKGROUNDS
            )
        real_html = f'<p class="label">Real size, {real} px</p><div class="row">{tiles}</div>'
    name = escape(Path(path).name)
    return (
        f'<section class="file"><h3>{escape(target.value)} <span class="meta">{name}</span></h3>'
        f'<p class="meta">{meta}</p>{problems_html}<div class="row">{big}</div>{real_html}'
        "</section>"
    )


def _ai_block(ai: dict[str, Any]) -> str:
    """Raw model clip, model input frame, prompt and costs of an AI run."""
    costs = ai.get("costs_usd", {})
    loop = ai.get("loop", {})
    cached = " (from cache, not paid again)" if ai.get("cached") else ""
    meta = (
        f"{escape(ai.get('model', ''))}{cached}; prompt: {escape(ai.get('prompt_source', ''))}; "
        f"cost: description {costs.get('describe', 0):.4f}$, generation "
        f"{costs.get('generate', 0):.4f}$, total {costs.get('total', 0):.4f}$; "
        f"loop: {escape(str(loop.get('reason', '')))}"
    )
    raw = escape(ai.get("raw", "raw.mp4"), quote=True)
    frame = escape(ai.get("model_input", "model_input.png"), quote=True)
    return (
        '<section class="file ai"><h3>Raw clip and prompt</h3>'
        f'<p class="meta">{meta}</p>'
        f'<p class="prompt">{escape(ai.get("prompt", ""))}</p>'
        '<div class="row">'
        f'<div class="tile" style="flex:1 1 320px"><video src="{raw}" autoplay loop muted '
        "playsinline controls></video>Raw model output</div>"
        f'<div class="tile" style="flex:1 1 320px"><img src="{frame}" alt="">'
        "Model input (first and last frame)</div></div></section>"
    )


def _focus_block(focus: dict[str, Any]) -> str:
    """Emotes at real size: full composition next to the focused key zone."""
    pairs = focus.get("pairs") or []
    box = focus.get("source_box")
    how = f"{escape(str(focus.get('requested')))} &rarr; {escape(str(focus.get('mode')))}"
    meta = f"emote focus {how}; zone in the input PNG (x0, y0, x1, y1): {box}"
    if focus.get("cost_usd"):
        meta += f"; vision call {focus['cost_usd']:.6f}$"
    warnings = "".join(f'<li class="w">{escape(w)}</li>' for w in focus.get("warnings", []))
    real = REAL_SIZE_PX[ExportTarget.TWITCH_GIF]
    rows = []
    for pair in pairs:
        tiles = []
        for key, label in (("full", "Full composition"), ("focus", "Focus")):
            chats = "".join(
                f'<div class="chat {bg}"><span>viewer:</span>'
                f"{_media(pair[key], real, False)}<span>nice</span></div>"
                for bg, _ in BACKGROUNDS[:2]
            )
            big = _media(pair[key], None, True)
            tiles.append(
                f'<div class="tile"><div class="stage big checker">{big}</div>{label}'
                f'<div class="row">{chats}</div></div>'
            )
        rows.append(
            f'<p class="label">{escape(pair["target"])} {escape(pair["name"])}, {real} px</p>'
            f'<div class="row">{"".join(tiles)}</div>'
        )
    return (
        f'<section class="file"><h3>Emotes at {real} px: full composition vs focus</h3>'
        f'<p class="meta">{meta}</p>'
        + (f'<ul class="problems">{warnings}</ul>' if warnings else "")
        + "".join(rows)
        + "</section>"
    )


def render_index(manifest: dict[str, Any]) -> str:
    """HTML of the comparison page for a manifest (paths relative to the page)."""
    exports = [e for p in manifest["presets"] for e in p["exports"]]
    invalid = sum(1 for e in exports if not e["valid"])
    summary = (
        f"{len(exports)} files, all valid"
        if not invalid
        else f'{len(exports)} files, <span style="color:var(--bad)">{invalid} invalid</span>'
    )
    sections = []
    for preset in manifest["presets"]:
        renders = "; ".join(
            f"{r['class']}: {r['frames']} frames at {r['fps']:g} fps, "
            f"rendered in {r['render_s']:.1f} s"
            for r in preset["renders"]
        )
        files = "".join(_file_block(e) for e in preset["exports"])
        sections.append(
            f'<h2 id="{escape(preset["name"])}">{escape(preset["name"])}</h2>'
            f'<p class="render">{escape(renders)}</p>{files}'
        )
    focus = (manifest.get("ai") or {}).get("emote_focus")
    title = f"{manifest['input']} - StickerMotion"
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>{escape(manifest['input'])}</h1>"
        f'<p class="lede">{summary}. Total time {manifest["total_s"]:.1f} s. '
        "Each file on the Twitch dark background, white and checker, then at real size.</p>"
        + (_ai_block(manifest["ai"]) if manifest.get("ai") else "")
        + (_focus_block(focus) if focus and focus.get("requested") != "full" else "")
        + "".join(sections)
        + "</main></body></html>\n"
    )


def write_index(root: Path, manifest: dict[str, Any]) -> Path:
    """Write ``root/index.html`` for the manifest and return its path."""
    path = root / "index.html"
    path.write_text(render_index(manifest), encoding="utf-8")
    return path
