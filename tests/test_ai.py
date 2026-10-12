"""AI engine (engine/ai) and `stickermotion animate`. No network: a fake Replicate transport."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from stickermotion.ai_pipeline import COSTS_FILE, animate
from stickermotion.cli import app
from stickermotion.core.models import ExportTarget
from stickermotion.engine.ai import (
    BudgetExceededError,
    CostLedger,
    CutMethod,
    Margins,
    PaymentRequiredError,
    RateLimitedError,
    ReplicateClient,
    frame_for_model,
    manual,
    read_token,
)
from stickermotion.engine.ai.chain import (
    clean_alpha,
    edge_touch,
    process,
    to_master,
    union_square,
)
from stickermotion.engine.ai.costs import Price, summarize_input
from stickermotion.engine.ai.decor import DECOR_SYSTEM
from stickermotion.engine.ai.describe import (
    DESCRIBE_MODEL,
    FIXED_REQUIREMENTS,
    build_prompt,
    describe,
    parse_reply,
)
from stickermotion.engine.ai.framing import GREEN, tight_crop
from stickermotion.engine.ai.generate import WAN_MODEL, WAN_PARAMS, cache_key, generate
from stickermotion.engine.ai.looping import make_loop, ping_pong, seam_score, trim_to_match
from stickermotion.engine.ai.matting import (
    RembgMatter,
    RembgUnavailableError,
    border_color,
    chroma_key,
    combine_masks,
    frame_key,
    key_alpha,
)
from stickermotion.engine.ai.provider import API, ApiError
from stickermotion.engine.ai.video import read_frames, write_mp4
from stickermotion.validator import validate

TOKEN = "fake-replicate-token-0123456789abcdef"
runner = CliRunner()


def _character(size: int = 160) -> Image.Image:
    """Red disc with a white stripe and a yellow star on a transparent canvas."""
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 30, size - 20, size - 10), fill=(200, 40, 60, 255))
    draw.rectangle((size // 2 - 6, 40, size // 2 + 6, size - 20), fill=(250, 250, 250, 255))
    draw.polygon([(140, 4), (146, 18), (134, 18)], fill=(250, 210, 40, 255))
    return image


def _clip_frames(count: int = 40) -> list[Image.Image]:
    """Character on green that rises and returns to the start (first = last), 832x480."""
    frame, framing = frame_for_model(_character())
    box = framing.pixel_box(framing.placement, frame.size)
    sprite = _character().crop(_character().getchannel("A").getbbox())
    sprite = sprite.resize((box[2] - box[0], box[3] - box[1]))
    frames = []
    for i in range(count):
        lift = round(40 * math.sin(math.pi * i / (count - 1)))
        canvas = Image.new("RGBA", frame.size, (*GREEN, 255))
        canvas.alpha_composite(sprite, (box[0], box[1] - lift))
        frames.append(canvas.convert("RGB"))
    return frames


@pytest.fixture(scope="session")
def green_clip(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Short H.264 clip on a green background, as Wan delivers it."""
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    write_mp4(path, _clip_frames(), 16.0)
    return path.read_bytes()


@pytest.fixture
def character_png(tmp_path: Path) -> Path:
    path = tmp_path / "hero.png"
    _character().save(path)
    return path


# Fake Replicate


@dataclass
class FakeReplicate:
    """Answers the Replicate endpoints the client uses; records every call."""

    video: bytes
    describe_reply: str = '{"action": "the character hops up and lands", "decorations": ""}'
    create_errors: list[ApiError] = field(default_factory=list)
    orphans: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, str, dict[str, str], bytes | None]] = field(default_factory=list)
    created: list[dict[str, Any]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)

    def creates(self, model: str | None = None) -> list[dict[str, Any]]:
        return [c for c in self.created if model is None or c["model"] == model]

    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> bytes:
        self.calls.append((method, url, headers, body))
        if url == "https://example.test/out.mp4":
            return self.video
        if method == "POST" and url.endswith("/predictions") and "/models/" in url:
            if self.create_errors:
                raise self.create_errors.pop(0)
            model = url.split("/models/")[1].rsplit("/predictions", 1)[0]
            payload = json.loads(body or b"{}")
            pid = f"p{len(self.created) + 1}"
            self.created.append({"id": pid, "model": model, "input": payload["input"]})
            return json.dumps(
                {"id": pid, "status": "processing", "urls": {"get": f"{API}/predictions/{pid}"}}
            ).encode()
        if method == "GET" and "/predictions/" in url:
            pid = url.rsplit("/", 1)[1]
            item = next(c for c in self.created if c["id"] == pid)
            is_text = item["model"] == DESCRIBE_MODEL
            return json.dumps(
                {
                    "id": pid,
                    "status": "succeeded",
                    "output": [self.describe_reply] if is_text else "https://example.test/out.mp4",
                    # Metric names as Replicate reports them for openai/gpt-5-nano.
                    "metrics": {"token_input_count": 1200, "token_output_count": 300}
                    if is_text
                    else {"predict_time": 23.0, "video_output_count": 1},
                    "urls": {"get": url},
                }
            ).encode()
        if method == "GET" and url == f"{API}/predictions":
            return json.dumps({"results": self.orphans}).encode()
        if method == "POST" and url.endswith("/cancel"):
            self.cancelled.append(url.split("/predictions/")[1].split("/")[0])
            return b"{}"
        raise AssertionError(f"unexpected call {method} {url}")


def _client(
    tmp_path: Path, fake: FakeReplicate, budget: float = 0.10, token: str | None = TOKEN
) -> ReplicateClient:
    ledger = CostLedger(tmp_path / "costs.jsonl", budget)
    return ReplicateClient(token, ledger, transport=fake, sleep=lambda _s: None)


def _journal(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# Framing


def test_frame_has_more_headroom_above_than_below() -> None:
    frame, framing = frame_for_model(_character(), margins=Margins(top=0.3, bottom=0.04))
    assert frame.size == (832, 480) and frame.mode == "RGB"
    assert frame.getpixel((0, 0)) == GREEN
    left, top, right, bottom = framing.pixel_box(framing.placement, frame.size)
    assert top >= round(0.3 * 480) - 1
    assert 480 - bottom <= round(0.04 * 480) + 1
    assert top > 480 - bottom
    # Centred horizontally, inside the central square with the side margin.
    assert abs((left + right) / 2 - 416) <= 1
    assert right - left <= 480 * (1 - 2 * 0.08) + 1


def test_frame_margins_are_parameters() -> None:
    _, small = frame_for_model(_character(), margins=Margins(top=0.1, side=0.02, bottom=0.02))
    _, big = frame_for_model(_character(), margins=Margins(top=0.4, side=0.1, bottom=0.05))
    small_h = small.placement[3] - small.placement[1]
    big_h = big.placement[3] - big.placement[1]
    assert small_h > big_h
    with pytest.raises(ValueError):
        Margins(top=0.6)


def test_tight_crop_drops_transparent_border() -> None:
    crop = tight_crop(_character())
    assert crop.getchannel("A").getbbox() == (0, 0, *crop.size)


# Matting


def test_chroma_key_cuts_green_and_keeps_character() -> None:
    frame = _clip_frames(2)[0]
    rgba = chroma_key(frame)
    alpha = np.asarray(rgba.getchannel("A"))
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0
    # The red body is opaque and keeps its colour (no green left after despill).
    data = np.asarray(rgba)
    body = data[alpha == 255]
    assert len(body) > 1000
    assert (body[:, 1].astype(int) <= np.maximum(body[:, 0], body[:, 2])).all()


def test_key_alpha_is_linear_on_soft_edges() -> None:
    rgb = np.array([[[0, 1, 0], [0.5, 0.75, 0.5], [1, 0.2, 0.2]]], dtype=np.float32)
    alpha = key_alpha(rgb, GREEN)
    assert alpha[0, 0] == 0 and alpha[0, 2] == 1
    assert 0.3 < alpha[0, 1] < 0.9


def test_frame_key_follows_drifted_green_but_not_other_colours() -> None:
    drifted = Image.new("RGB", (64, 64), (40, 236, 36))
    assert border_color(drifted) == (40, 236, 36)
    assert frame_key(drifted) == (40, 236, 36)
    assert frame_key(Image.new("RGB", (64, 64), (128, 0, 128))) == GREEN


def test_combine_masks_removes_shadow_keeps_detached_prop() -> None:
    model = np.zeros((10, 10), dtype=np.float32)
    model[2:5, 2:5] = 1
    keyed = np.zeros((10, 10), dtype=np.float32)
    keyed[2:5, 2:5] = 1
    keyed[8, 8] = 1.0  # detached prop the model ignored
    keyed[8, 1] = 0.4  # shadow: darkened key colour
    out = combine_masks(model, keyed, grow_px=0)
    assert out[3, 3] == 1 and out[8, 8] == 1 and out[8, 1] == 0


def test_rembg_matter_uses_mask_fn_and_key() -> None:
    frame = _clip_frames(2)[0]
    matter = RembgMatter(mask_fn=lambda f: Image.new("L", f.size, 255))
    out = matter(frame)
    assert out.mode == "RGBA" and out.getpixel((0, 0))[3] == 0


def test_rembg_missing_gives_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("rembg", "onnxruntime"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RembgUnavailableError, match="rembg"):
        RembgMatter()(Image.new("RGB", (8, 8), GREEN))


# Looping


def _moving(count: int, back: bool) -> list[Image.Image]:
    frames = []
    for i in range(count):
        # back: the last frame equals the first one (as Wan does with first = last image).
        x = round(20 * math.sin(math.pi * i / (count - 1))) if back else i * 2
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        ImageDraw.Draw(img).rectangle((5 + x, 20, 25 + x, 40), fill=(255, 0, 0, 255))
        frames.append(img)
    return frames


def test_trim_finds_return_to_first_frame() -> None:
    frames = _moving(32, back=True)  # frame 31 equals frame 0
    trimmed = trim_to_match(frames + frames[1:5])
    assert len(trimmed) == 31
    assert seam_score(trimmed).ok()


def test_make_loop_prefers_trim_and_falls_back_to_pingpong() -> None:
    returning = make_loop(_moving(40, back=True), fps=16)
    assert returning.method == "trim" and returning.seam.ok()
    drifting = make_loop(_moving(40, back=False), fps=16)
    assert drifting.method == "pingpong"
    assert drifting.seam.ok()


def test_ping_pong_has_no_repeated_ends() -> None:
    frames = _moving(5, back=False)
    assert len(ping_pong(frames)) == 8


# Chain


def test_process_gives_square_master_and_edge_report() -> None:
    frames = _clip_frames()
    result = process(frames, 16.0)
    assert result.master.size[0] == result.master.size[1]
    assert result.loop.method == "trim"
    assert not any(result.edge_touch.values())
    assert result.cut is CutMethod.CHROMA


def test_edge_touch_flags_cropped_limb() -> None:
    img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle((20, 0, 30, 20), fill=(255, 0, 0, 255))
    assert edge_touch([img]) == {"top": True, "left": False, "right": False, "bottom": False}


def test_clean_alpha_drops_edge_band_and_specks_keeps_small_props() -> None:
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle((30, 30, 60, 60), fill=(255, 0, 0, 255))  # body
    draw.rectangle((80, 10, 87, 17), fill=(255, 220, 0, 255))  # small star, 8 px
    draw.point((10, 90), fill=(255, 255, 255, 255))  # codec speck
    draw.line((0, 0, 99, 0), fill=(0, 255, 0, 40))  # encoder edge row
    out = clean_alpha(img)
    alpha = np.asarray(out.getchannel("A"))
    assert alpha[45, 45] == 255 and alpha[13, 83] == 255
    assert alpha[90, 10] == 0 and alpha[0].max() == 0


def test_union_square_and_master() -> None:
    frames = _moving(4, back=False)
    box = union_square(frames, margin=0)
    assert box[2] - box[0] == box[3] - box[1]
    assert to_master(frames, 8).frame_count == 4


# Description


def test_parse_reply_json_and_plain() -> None:
    assert parse_reply('```json\n{"action": "waves", "decorations": "stars blink"}\n```') == (
        "waves",
        "stars blink",
    )
    assert parse_reply("  she waves  hello ") == ("she waves hello", "")
    with pytest.raises(ValueError):
        parse_reply("   ")


def test_build_prompt_appends_fixed_requirements() -> None:
    prompt = build_prompt("The cat yawns.", "The moon sways")
    assert prompt.startswith("The cat yawns. The moon sways.")
    assert prompt.endswith(FIXED_REQUIREMENTS)


def test_describe_calls_vision_model_cheaply(tmp_path: Path, green_clip: bytes) -> None:
    reply = '{"action": "The girl stretches", "decorations": "stars blink"}'
    fake = FakeReplicate(green_clip, reply)
    client = _client(tmp_path, fake)
    frame, _ = frame_for_model(_character())
    result = describe(client, frame)
    assert result.source == DESCRIBE_MODEL
    assert result.prompt.startswith("The girl stretches. Stars blink.")
    sent = fake.creates(DESCRIBE_MODEL)[0]["input"]
    assert sent["image_input"][0].startswith("data:image/png;base64,")
    entry = _journal(tmp_path / "costs.jsonl")[0]
    # 1200 input + 300 output tokens at 0.05 / 0.40 $ per million.
    assert entry["cost_usd"] == pytest.approx(0.00018)
    assert entry["cost_usd"] * 50 < 0.05  # far below one generation


# Spend control


def test_budget_blocks_call_before_any_request(tmp_path: Path, green_clip: bytes) -> None:
    fake = FakeReplicate(green_clip)
    client = _client(tmp_path, fake, budget=0.04)
    frame, _ = frame_for_model(_character())
    with pytest.raises(BudgetExceededError, match="budget"):
        generate(client, frame, "p", tmp_path / "cache")
    assert not any("/predictions" in c[1] for c in fake.calls)


def test_402_is_not_retried(tmp_path: Path, green_clip: bytes) -> None:
    fake = FakeReplicate(green_clip, create_errors=[ApiError(402, '{"title":"Insufficient"}')])
    client = _client(tmp_path, fake)
    frame, _ = frame_for_model(_character())
    with pytest.raises(PaymentRequiredError, match="402"):
        generate(client, frame, "p", tmp_path / "cache")
    posts = [c for c in fake.calls if c[0] == "POST" and "/models/" in c[1]]
    assert len(posts) == 1
    entries = _journal(tmp_path / "costs.jsonl")
    assert entries[-1]["status"] == "402" and entries[-1]["cost_usd"] == 0


def test_429_cancels_own_orphans_then_retries(tmp_path: Path, green_clip: bytes) -> None:
    fake = FakeReplicate(green_clip, create_errors=[ApiError(429, '{"retry_after": 7}')])
    fake.orphans = [
        # Created by this run after the 429: cancelled.
        {
            "id": "orphan1",
            "model": WAN_MODEL,
            "status": "starting",
            "created_at": "2999-01-01T00:00:00Z",
            "urls": {"cancel": f"{API}/predictions/orphan1/cancel"},
        },
        # Older prediction (not this run): left alone.
        {
            "id": "old",
            "model": WAN_MODEL,
            "status": "processing",
            "created_at": "2000-01-01T00:00:00Z",
        },
        # Other model: left alone.
        {
            "id": "other",
            "model": "someone/else",
            "status": "starting",
            "created_at": "2999-01-01T00:00:00Z",
        },
    ]
    slept: list[float] = []
    ledger = CostLedger(tmp_path / "costs.jsonl", 0.10)
    client = ReplicateClient(TOKEN, ledger, transport=fake, sleep=slept.append)
    frame, _ = frame_for_model(_character())
    gen = generate(client, frame, "p", tmp_path / "cache")
    assert fake.cancelled == ["orphan1"]
    assert 7.0 in slept
    assert len(fake.creates(WAN_MODEL)) == 1
    assert gen.cost_usd == 0.05
    statuses = [e["status"] for e in _journal(tmp_path / "costs.jsonl")]
    assert statuses == ["429", "succeeded"]
    assert _journal(tmp_path / "costs.jsonl")[0]["cancelled"] == ["orphan1"]


def test_429_gives_up_after_retries(tmp_path: Path, green_clip: bytes) -> None:
    errors = [ApiError(429, "slow down") for _ in range(5)]
    fake = FakeReplicate(green_clip, create_errors=errors)
    client = _client(tmp_path, fake)
    client.max_retries = 2
    frame, _ = frame_for_model(_character())
    with pytest.raises(RateLimitedError):
        generate(client, frame, "p", tmp_path / "cache")
    assert len([c for c in fake.calls if c[0] == "POST" and "/models/" in c[1]]) == 3


def test_cache_does_not_pay_twice_and_regenerate_does(tmp_path: Path, green_clip: bytes) -> None:
    fake = FakeReplicate(green_clip)
    frame, _ = frame_for_model(_character())
    first = generate(_client(tmp_path, fake), frame, "p", tmp_path / "cache")
    second = generate(_client(tmp_path, fake), frame, "p", tmp_path / "cache")
    assert not first.cached and second.cached
    assert first.video == second.video and second.cost_usd == 0
    assert len(fake.creates(WAN_MODEL)) == 1
    other = generate(_client(tmp_path, fake), frame, "another prompt", tmp_path / "cache")
    assert not other.cached and other.video != first.video
    forced = generate(_client(tmp_path, fake), frame, "p", tmp_path / "cache", regenerate=True)
    assert not forced.cached and len(fake.creates(WAN_MODEL)) == 3


def test_cache_key_depends_on_every_input() -> None:
    base = cache_key(b"png", "p", WAN_MODEL, WAN_PARAMS)
    assert base == cache_key(b"png", "p", WAN_MODEL, dict(WAN_PARAMS))
    assert base != cache_key(b"png2", "p", WAN_MODEL, WAN_PARAMS)
    assert base != cache_key(b"png", "q", WAN_MODEL, WAN_PARAMS)
    assert base != cache_key(b"png", "p", "other/model", WAN_PARAMS)
    assert base != cache_key(b"png", "p", WAN_MODEL, {**WAN_PARAMS, "num_frames": 121})


def test_wan_input_uses_same_first_and_last_frame(tmp_path: Path, green_clip: bytes) -> None:
    fake = FakeReplicate(green_clip)
    frame, _ = frame_for_model(_character())
    generate(_client(tmp_path, fake), frame, "p", tmp_path / "cache")
    sent = fake.creates(WAN_MODEL)[0]["input"]
    assert sent["image"] == sent["last_image"]
    assert sent["resolution"] == "480p" and sent["prompt"] == "p"


def test_summarize_input_hides_image_data() -> None:
    summary = summarize_input({"image": "data:image/png;base64,AAAA", "x": ["data:image/png;b"]})
    assert summary["image"].startswith("<image/png")
    assert "AAAA" not in json.dumps(summary)


def test_flat_and_token_prices() -> None:
    assert Price(0.05, flat_usd=0.05).actual({}, True) == 0.05
    assert Price(0.05, flat_usd=0.05).actual({}, False) == 0
    token = Price(0.002, input_per_m=0.05, output_per_m=0.4)
    assert token.actual({"token_input_count": 1e6, "token_output_count": 0}, True) == 0.05
    assert token.actual({"input_token_count": 0, "output_token_count": 1e6}, True) == 0.4
    assert token.actual({}, True) == 0.002  # no metrics: book the upper bound


# Key safety


def test_read_token_env_then_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(f'OTHER=1\nREPLICATE_API_TOKEN="{TOKEN}"\n')
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    assert read_token(dotenv) == TOKEN
    monkeypatch.setenv("REPLICATE_API_TOKEN", "env-token")
    assert read_token(dotenv) == "env-token"
    monkeypatch.delenv("REPLICATE_API_TOKEN")
    assert read_token(tmp_path / "missing") is None


def test_key_never_in_journal_errors_or_output(
    tmp_path: Path, green_clip: bytes, character_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    # The API echoes the key back in an error body; it must be redacted everywhere.
    fake = FakeReplicate(green_clip, create_errors=[ApiError(402, f"bad credit for {TOKEN}")])
    real = animate
    monkeypatch.setattr(
        "stickermotion.ai_pipeline.animate", lambda *a, **k: real(*a, **k, transport=fake)
    )
    out = tmp_path / "out"
    result = runner.invoke(
        app, ["animate", str(character_png), "--prompt", "hop", "--out", str(out)]
    )
    assert result.exit_code == 3
    assert "402" in result.output and TOKEN not in result.output
    journal = (out / COSTS_FILE).read_text()
    assert TOKEN not in journal and "***" in journal
    # Successful run: the token is sent only in the Authorization header.
    fake2 = FakeReplicate(green_clip)
    done = animate(
        character_png,
        prompt="hop",
        targets=[ExportTarget.TELEGRAM_EMOJI],
        out_dir=out,
        transport=fake2,
        dotenv=None,
    )
    for _method, _url, headers, body in fake2.calls:
        assert TOKEN not in (body or b"").decode("utf-8", "replace")
        assert TOKEN not in _url
        if "Authorization" in headers:
            assert headers["Authorization"] == f"Bearer {TOKEN}"
    texts = [(out / COSTS_FILE).read_text()]
    texts += [p.read_text() for p in done.root.glob("*.json")]
    texts += [p.read_text() for p in done.root.glob("*.txt")]
    texts += [p.read_text() for p in done.root.glob("*.html")]
    assert all(TOKEN not in t for t in texts)


def test_missing_token_fails_before_any_call(
    tmp_path: Path, green_clip: bytes, character_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    fake = FakeReplicate(green_clip)
    from stickermotion.engine.ai import MissingTokenError

    with pytest.raises(MissingTokenError):
        animate(character_png, prompt="hop", out_dir=tmp_path, transport=fake, dotenv=None)
    assert fake.calls == []


# Full offline run


def test_manual_prompt_overrides_automatic(
    tmp_path: Path, green_clip: bytes, character_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    fake = FakeReplicate(green_clip)
    result = animate(
        character_png,
        prompt="The hero waves",
        targets=[ExportTarget.TELEGRAM_EMOJI],
        out_dir=tmp_path,
        transport=fake,
        dotenv=None,
        emote_focus="full",
    )
    assert result.description == manual("The hero waves")
    # The only cheap-model call is the decor check (the star next to the hero).
    assert [c["input"]["system_prompt"] for c in fake.creates(DESCRIBE_MODEL)] == [DECOR_SYSTEM]
    assert fake.creates(WAN_MODEL)[0]["input"]["prompt"].startswith("The hero waves.")
    assert (result.root / "prompt.txt").read_text().startswith("The hero waves.")


def test_animate_cli_offline_all_targets_valid(
    tmp_path: Path, green_clip: bytes, character_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`stickermotion animate` end to end with the fake provider: every target valid."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    fake = FakeReplicate(green_clip)
    real = animate

    def patched(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **kwargs, transport=fake)

    monkeypatch.setattr("stickermotion.ai_pipeline.animate", patched)
    out = tmp_path / "out"
    args = ["animate", str(character_png), "--out", str(out), "--emote-focus", "full"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert TOKEN not in result.output
    assert "prompt (openai/gpt-5-nano): The character hops up and lands." in result.output
    # Description, decor check (the hero has a detached star) and the clip.
    assert "total 0.0504$" in result.output
    assert "decor: auto, item (float)" in result.output
    root = out / "hero" / "ai"
    for name in ("raw.mp4", "prompt.txt", "manifest.json", "index.html", "model_input.png"):
        assert (root / name).is_file(), name
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["ai"]["costs_usd"]["generate"] == 0.05
    assert manifest["ai"]["loop"]["method"] == "trim"
    files = [e for p in manifest["presets"] for e in p["exports"]]
    assert {e["target"] for e in files} == {t.value for t in ExportTarget}
    for entry in files:
        check = validate(root / entry["path"], ExportTarget(entry["target"]), entry["size_variant"])
        assert check.valid, (entry["path"], check.violations)
    page = (root / "index.html").read_text()
    assert 'src="raw.mp4"' in page and "The character hops up and lands." in page
    journal = _journal(out / COSTS_FILE)
    assert [e["step"] for e in journal] == ["decor", "describe", "generate"]
    assert all(e["run_id"] == journal[0]["run_id"] for e in journal)

    # Second run: the clip comes from the cache (the description is still paid).
    again = runner.invoke(app, args)
    assert again.exit_code == 0, again.output
    assert "(cached)" in again.output
    assert len(fake.creates(WAN_MODEL)) == 1


def test_animate_cli_budget_error_exit_code(
    tmp_path: Path, green_clip: bytes, character_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    fake = FakeReplicate(green_clip)
    real = animate
    monkeypatch.setattr(
        "stickermotion.ai_pipeline.animate",
        lambda *a, **k: real(*a, **k, transport=fake),
    )
    result = runner.invoke(
        app, ["animate", str(character_png), "--budget", "0.01", "--out", str(tmp_path)]
    )
    assert result.exit_code == 3
    assert "budget" in result.output
    assert fake.creates(WAN_MODEL) == []


def test_read_frames_roundtrip(tmp_path: Path) -> None:
    frames = _clip_frames(8)
    path = tmp_path / "c.mp4"
    write_mp4(path, frames, 16)
    decoded, fps = read_frames(path)
    assert len(decoded) == 8 and fps == 16 and decoded[0].size == (832, 480)


def test_prompt_keeps_face_proportions() -> None:
    """Mouth opened too wide on automatic descriptions."""
    from stickermotion.engine.ai.describe import INSTRUCTION

    prompt = build_prompt("The girl yawns")
    assert "Natural face, jaw and head proportions in every frame" in prompt
    assert FIXED_REQUIREMENTS in manual("She laughs").prompt
    for word in ('"wide"', '"huge"', '"enormous"'):
        assert word in INSTRUCTION


MOUTH_TRIGGERS = ("yawn", "laugh", "shout", "scream", "open", "never")


def test_fixed_requirements_have_no_mouth_triggers() -> None:
    """Wan 2.2 fast reads named mouth actions (even negated) as a cue."""
    words = re.findall(r"[a-z]+", FIXED_REQUIREMENTS.lower())
    for trigger in MOUTH_TRIGGERS:
        assert not any(word.startswith(trigger) for word in words), trigger


def test_fixed_requirements_keep_mouth_and_hand_size() -> None:
    """Mouth and hands keep the size and shape of the original drawing."""
    assert "The mouth keeps the size and shape of the original drawing" in FIXED_REQUIREMENTS
    assert "only as much as the described action itself requires" in FIXED_REQUIREMENTS
    assert "Hands and fingers keep their original size, shape" in FIXED_REQUIREMENTS
    assert "Hands and skin keep exactly their original colors" in FIXED_REQUIREMENTS
    assert "returns exactly to the starting pose" in FIXED_REQUIREMENTS
    assert "Static camera, flat plain green background" in FIXED_REQUIREMENTS


def test_instruction_asks_for_gentle_mouth_without_naming_open_mouth_actions() -> None:
    """The describe instruction states the mouth rule positively."""
    from stickermotion.engine.ai.describe import INSTRUCTION

    assert "small, gentle or slight" in INSTRUCTION
    assert "mention the mouth only when the action itself needs it" in INSTRUCTION
    words = re.findall(r"[a-z]+", INSTRUCTION.lower())
    for trigger in ("yawn", "laugh", "shout", "scream"):
        assert not any(word.startswith(trigger) for word in words), trigger
