"""Replicate's moderation flag (E005) on the focus and decor steps.

Fake transport only: no network, no spend.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from PIL import Image

from stickermotion.ai_pipeline import animate
from stickermotion.core.models import ExportTarget
from stickermotion.engine.ai.costs import BudgetExceededError
from stickermotion.engine.ai.decor import DecorMode, plan
from stickermotion.engine.ai.describe import (
    DESCRIBE_MODEL,
    DESCRIBE_PRICE,
    NEUTRAL,
    ask_vision,
    describe,
)
from stickermotion.engine.ai.focus import AUTO, FULL, FocusSpec, resolve
from stickermotion.engine.ai.generate import WAN_MODEL
from stickermotion.engine.ai.provider import PredictionFailedError
from stickermotion.engine.ai.video import write_mp4
from test_ai import TOKEN, _character, _client, _clip_frames, _journal
from test_decor import _scene
from test_web import WebFake


@pytest.fixture(scope="module")
def green_clip(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    write_mp4(path, _clip_frames(), 16.0)
    return path.read_bytes()


def _fallbacks(path: Path) -> list[dict[str, object]]:
    return [e for e in _journal(path) if e["status"] == "fallback"]


# Focus


def test_focus_flag_once_is_retried_and_used(tmp_path: Path) -> None:
    fake = WebFake(b"", flag_tasks={"focus": 1})
    focus = resolve(FocusSpec(AUTO), _character(200), _client(tmp_path, fake))
    assert focus.mode == AUTO and focus.active and not focus.warnings
    assert fake.llm_tasks() == ["focus", "focus"]
    statuses = [e["status"] for e in _journal(tmp_path / "costs.jsonl")]
    assert statuses == ["failed", "succeeded"]


def test_focus_flag_twice_falls_back_to_full(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    fake = WebFake(b"", flag_tasks={"focus": 2})
    with caplog.at_level(logging.WARNING, logger="stickermotion.ai"):
        focus = resolve(FocusSpec(AUTO), _character(200), _client(tmp_path, fake))
    assert focus.requested == AUTO and focus.mode == FULL and not focus.active
    assert focus.cost_usd == 0.0 and "E005" in focus.warnings[0]
    assert fake.llm_tasks() == ["focus", "focus"]
    [entry] = _fallbacks(tmp_path / "costs.jsonl")
    assert entry["step"] == "focus" and entry["fallback"] == FULL and entry["cost_usd"] == 0.0
    assert "focus: vision model failed" in caplog.text and "fallback: full" in caplog.text


def test_focus_other_error_falls_back_without_retry(tmp_path: Path) -> None:
    fake = WebFake(b"", fail_tasks={"focus": 1})
    focus = resolve(FocusSpec(AUTO), _character(200), _client(tmp_path, fake))
    assert focus.mode == FULL and "out of memory" in focus.warnings[0]
    assert fake.llm_tasks() == ["focus"]


def test_focus_budget_error_is_not_swallowed(tmp_path: Path) -> None:
    fake = WebFake(b"")
    with pytest.raises(BudgetExceededError):
        resolve(FocusSpec(AUTO), _character(200), _client(tmp_path, fake, budget=0.0001))
    assert fake.created == []


# Decor


def test_decor_flag_once_is_retried_and_used(tmp_path: Path) -> None:
    fake = WebFake(b"", flag_tasks={"decor": 1})
    result = plan(_scene(), DecorMode.AUTO, _client(tmp_path, fake))
    assert fake.llm_tasks() == ["decor", "decor"]
    assert result.reply is not None and any(i.source == DESCRIBE_MODEL for i in result.items)


def test_decor_flag_twice_falls_back_to_shape_heuristic(tmp_path: Path) -> None:
    fake = WebFake(b"", flag_tasks={"decor": 2})
    result = plan(_scene(), DecorMode.AUTO, _client(tmp_path, fake))
    assert result.mode is DecorMode.AUTO and result.animated
    assert len(result.items) == 3 and all(i.source == "heuristic" for i in result.items)
    assert result.reply is None and result.cost_usd == 0.0
    assert "fallback: shape heuristic" in result.warnings[0]
    [entry] = _fallbacks(tmp_path / "costs.jsonl")
    assert entry["step"] == "decor" and entry["model"] == DESCRIBE_MODEL


# Describe (CLI path): retried on grey


def test_describe_flag_is_retried_on_neutral_grey(tmp_path: Path) -> None:
    fake = WebFake(b"", flag_describes=1)
    client = _client(tmp_path, fake)
    character = _character(200)
    green = Image.new("RGBA", character.size, (0, 255, 0, 255))
    green.alpha_composite(character)
    found = describe(client, green, character)
    assert found.source == DESCRIBE_MODEL and found.action
    assert fake.llm_tasks() == ["describe", "describe"]
    first, second = (c["input"]["image_input"][0] for c in fake.creates(DESCRIBE_MODEL))
    assert first != second  # the retry sends the character on grey, not the green frame


def test_ask_vision_repeats_the_same_picture_without_retry_picture(tmp_path: Path) -> None:
    fake = WebFake(b"", flag_tasks={"focus": 1})
    client = _client(tmp_path, fake)
    picture = Image.new("RGB", (32, 32), NEUTRAL[:3])

    def model_input(ref: str) -> dict[str, object]:
        return {"system_prompt": "You locate regions", "prompt": "", "image_input": [ref]}

    ask_vision(client, model_input, picture, DESCRIBE_PRICE, "focus")
    refs = [c["input"]["image_input"][0] for c in fake.creates(DESCRIBE_MODEL)]
    assert len(refs) == 2 and refs[0] == refs[1]  # the same grey picture once more
    fake.flag_tasks = {"focus": 2}
    with pytest.raises(PredictionFailedError, match="flagged as sensitive"):
        ask_vision(client, model_input, picture, DESCRIBE_PRICE, "focus")


# The whole chain


def test_animate_finishes_when_focus_and_decor_are_flagged_twice(
    tmp_path: Path, green_clip: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("REPLICATE_API_TOKEN", TOKEN)
    source = tmp_path / "hero.png"
    _character(200).save(source)
    fake = WebFake(green_clip, flag_tasks={"decor": 2, "focus": 2}, flag_describes=1)
    result = animate(
        source,
        targets=[ExportTarget.TELEGRAM_EMOJI, ExportTarget.DISCORD_STICKER],
        out_dir=tmp_path / "out",
        transport=fake,
        dotenv=None,
        budget_usd=0.2,
    )
    assert len(fake.creates(WAN_MODEL)) == 1
    assert fake.llm_tasks() == ["decor", "decor", "describe", "describe", "focus", "focus"]
    ai = result.manifest["ai"]
    assert ai["emote_focus"]["mode"] == FULL and ai["emote_focus"]["requested"] == AUTO
    assert all(i["source"] == "heuristic" for i in ai["decor"]["items"])
    assert all(r.valid for r in result.results)
    steps = [e["step"] for e in _fallbacks(tmp_path / "out" / "costs.jsonl")]
    assert steps == ["decor", "focus"]
