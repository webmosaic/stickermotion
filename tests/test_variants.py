"""Variants with seeds, stored raw clips, "use this motion", migration."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from stickermotion import raws
from stickermotion.cli import app as cli_app
from stickermotion.core.models import ExportTarget
from stickermotion.engine.ai.costs import BudgetExceededError
from stickermotion.engine.ai.generate import WAN_MODEL, cache_key, wan_input, wan_params
from stickermotion.variants import animate_variants, batch_estimate_usd, check_count
from stickermotion.web.security import CSRF_FIELD
from stickermotion.web.store import Status, Store
from test_ai import TOKEN, FakeReplicate, _client
from test_web import AUTO_UK, Env, _make_env, _png

runner = CliRunner()


@pytest.fixture
def env(tmp_path: Path, green_clip: bytes) -> Iterator[Env]:
    yield from _make_env(tmp_path, green_clip)


@pytest.fixture(scope="module")
def green_clip(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    from stickermotion.engine.ai.video import write_mp4
    from test_ai import _clip_frames

    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    write_mp4(path, _clip_frames(), 16.0)
    return path.read_bytes()


@pytest.fixture
def character() -> bytes:
    from test_ai import _character

    return _png(_character(200))


def _generate(env: Env, sticker_id: str, variants: int) -> Any:
    data = {
        "text": AUTO_UK,
        "decor_separate": "1",
        "variants": str(variants),
        CSRF_FIELD: env.csrf(f"/s/{sticker_id}"),
    }
    response = env.client.post(f"/s/{sticker_id}/generate", data=data, follow_redirects=False)
    env.worker.wait(timeout=600)
    return response


def _use(env: Env, sticker_id: str, clip_id: str) -> Any:
    data = {CSRF_FIELD: env.csrf(f"/s/{sticker_id}")}
    response = env.client.post(
        f"/s/{sticker_id}/raw/{clip_id}/use", data=data, follow_redirects=False
    )
    env.worker.wait(timeout=600)
    return response


# Seeds and counts


def test_seed_goes_to_wan_and_cache_key_only_when_given() -> None:
    assert "seed" not in wan_params()
    params = wan_params(seed=42)
    assert params["seed"] == 42 and wan_input("ref", "p", params)["seed"] == 42
    assert cache_key(b"x", "p", WAN_MODEL, params) != cache_key(
        b"x", "p", WAN_MODEL, wan_params(seed=43)
    )
    with pytest.raises(ValueError):
        wan_params(seed=-1)
    for bad in (0, 4):
        with pytest.raises(ValueError):
            check_count(bad)
    assert batch_estimate_usd(2) == pytest.approx(0.10)


def test_variants_budget_is_checked_before_any_call(tmp_path: Path, green_clip: bytes) -> None:
    from test_ai import _character

    source = tmp_path / "hero.png"
    _character().save(source)
    fake = FakeReplicate(green_clip)
    with pytest.raises(BudgetExceededError):
        animate_variants(
            source, "hops", 3, tmp_path / "out", client=_client(tmp_path, fake, budget=0.12)
        )
    assert fake.created == []


def test_animate_variants_seeds_previews_and_raw_library(tmp_path: Path, green_clip: bytes) -> None:
    from test_ai import _character

    source = tmp_path / "hero.png"
    _character().save(source)
    fake = FakeReplicate(green_clip)
    result = animate_variants(
        source,
        "the hero hops",
        2,
        tmp_path / "out",
        client=_client(tmp_path, fake, budget=0.12),
        seeds=[11, 22],
        emote_focus="full",
    )
    wan = fake.creates(WAN_MODEL)
    assert [w["input"]["seed"] for w in wan] == [11, 22]
    assert result.spent_usd == pytest.approx(0.10, abs=0.001)
    assert [c.seed for c in result.clips] == [11, 22]
    assert len({c.batch for c in result.clips}) == 1
    for clip, preview in zip(result.clips, result.previews, strict=True):
        assert preview == clip.folder / raws.PREVIEW and preview.is_file()
        assert clip.framing is not None and clip.meta["variant"] in (1, 2)
        manifest = json.loads((clip.folder / "manifest.json").read_text())
        assert manifest["targets"] == [ExportTarget.TELEGRAM_STICKER.value]
    assert raws.list_clips(tmp_path / "out" / "hero" / raws.RAW_DIR)[0].seed == 22


# Raw library


def test_raw_library_keeps_newest_five(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"mp4")
    frame = tmp_path / "m.png"
    from PIL import Image

    Image.new("RGB", (8, 8)).save(frame)
    ids = [raws.save(tmp_path / "raw", video, frame, {"params": {"seed": i}}).id for i in range(7)]
    kept = [c.id for c in raws.list_clips(tmp_path / "raw")]
    assert kept == list(reversed(ids))[:5]
    assert raws.get(tmp_path / "raw", "../etc") is None
    assert raws.get(tmp_path / "raw", ids[-1]).seed == 6  # type: ignore[union-attr]


def test_clip_focus_spec_and_framing_round_trip(tmp_path: Path) -> None:
    from stickermotion.engine.ai.framing import frame_for_model
    from test_ai import _character

    _frame, framing = frame_for_model(_character())
    clip = raws.RawClip(
        "20260101-000000-000000",
        tmp_path,
        {"focus_box": [10, 20, 50, 80], "framing": framing.to_dict()},
    )
    assert clip.focus_spec == "10,20,40,60"
    assert clip.framing == framing


# Web


def test_web_variants_choose_and_budget(env: Env, character: bytes) -> None:
    env.login()
    sticker_id = env.new_sticker(character)
    page = env.client.get(f"/s/{sticker_id}").text
    assert "2 варіанти · 0.10$" in page and "3 варіанти · 0.15$" in page
    _generate(env, sticker_id, 2)
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.CHOOSING, sticker
    wan = env.fake.creates(WAN_MODEL)
    assert len(wan) == 2 and wan[0]["input"]["seed"] != wan[1]["input"]["seed"]
    assert sticker.generations == 2 and sticker.cost_usd == pytest.approx(env.store.spent())
    assert 0.10 <= sticker.cost_usd < 0.102
    clips = env.store.clips(sticker_id)
    assert len(clips) == 2 and all(c.preview for c in clips)
    page = env.client.get(f"/s/{sticker_id}").text
    assert "Виберіть варіант" in page and "Вибрати" in page
    preview = env.client.get(f"/s/{sticker_id}/raw/{clips[1].id}/preview.webp")
    assert preview.status_code == 200 and preview.headers["content-type"] == "image/webp"
    assert env.client.get(f"/s/{sticker_id}/raw/nope/preview.webp").status_code == 404

    chosen = clips[1]
    _use(env, sticker_id, chosen.id)
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DONE, sticker
    assert sticker.raw_id == chosen.id
    assert len(env.fake.creates(WAN_MODEL)) == 2  # choosing is free
    manifest = env.store.manifest(sticker_id)
    assert manifest is not None and manifest["ai"]["seed"] == chosen.seed
    assert manifest["ai"]["emote_focus"]["mode"] == "manual"  # zone kept from the batch
    page = env.client.get(f"/s/{sticker_id}").text
    assert "Попередні рухи" in page and "Зараз використовується" in page


def test_web_daily_budget_counts_every_variant(
    tmp_path: Path, green_clip: bytes, character: bytes
) -> None:
    for env in _make_env(tmp_path, green_clip, daily_budget_usd=0.12):
        env.login()
        sticker_id = env.new_sticker(character)
        _generate(env, sticker_id, 3)
        sticker = env.store.get(sticker_id)
        assert sticker is not None and "Денний ліміт" in sticker.error
        assert env.fake.creates(WAN_MODEL) == []


def test_web_single_generation_keeps_clip_and_reuses_motion(env: Env, character: bytes) -> None:
    env.login()
    sticker_id = env.new_sticker(character)
    _generate(env, sticker_id, 1)
    _generate(env, sticker_id, 1)
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DONE
    clips = env.store.clips(sticker_id)
    assert len(clips) == 2 and clips[0].id == sticker.raw_id
    assert all(c.seed is not None for c in clips)  # no preview: the test targets skip it
    page = env.client.get(f"/s/{sticker_id}").text
    assert "Використати цей рух" in page
    _use(env, sticker_id, clips[1].id)
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DONE and sticker.raw_id == clips[1].id
    assert len(env.fake.creates(WAN_MODEL)) == 2
    manifest = env.store.manifest(sticker_id)
    assert manifest is not None and manifest["ai"]["seed"] == clips[1].seed
    assert manifest["ai"]["reprocess"]["framing_known"] is True


def test_migration_keeps_clip_of_an_old_sticker(
    tmp_path: Path, green_clip: bytes, character: bytes
) -> None:
    """An old database (before the raw clip library) gets the new columns.

    Its ai/raw.mp4 becomes a clip.
    """
    data = tmp_path / "data"
    data.mkdir()
    db = sqlite3.connect(data / "app.db")
    db.execute(
        "CREATE TABLE stickers (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL, filename TEXT NOT NULL, status TEXT NOT NULL, "
        "width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0, "
        "error TEXT NOT NULL DEFAULT '', text TEXT NOT NULL DEFAULT '', "
        "auto_uk TEXT NOT NULL DEFAULT '', auto_en TEXT NOT NULL DEFAULT '', "
        "prompt_en TEXT NOT NULL DEFAULT '', regenerate INTEGER NOT NULL DEFAULT 0, "
        "generations INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0, "
        "decor TEXT NOT NULL DEFAULT 'auto', decor_found TEXT NOT NULL DEFAULT '')"
    )
    db.execute(
        "INSERT INTO stickers (id, created_at, updated_at, filename, status) "
        "VALUES ('abc123', 't', 't', 'a.png', 'done')"
    )
    db.commit()
    db.close()
    ai = data / "stickers" / "abc123" / "ai"
    ai.mkdir(parents=True)
    (data / "stickers" / "abc123" / "input.png").write_bytes(character)
    (ai / "raw.mp4").write_bytes(green_clip)
    from PIL import Image

    Image.new("RGB", (832, 480), (0, 255, 0)).save(ai / "model_input.png")
    manifest = {"ai": {"prompt": "hops", "params": {"num_frames": 81}, "decor": {}}}
    (ai / "manifest.json").write_text(json.dumps(manifest))
    store = Store(data)
    old = store.get("abc123")
    assert old is not None and old.variants == 1 and old.job == "" and old.raw_id == ""
    assert store.migrate_raws() == ["abc123"]
    clips = store.clips("abc123")
    assert len(clips) == 1 and clips[0].meta["legacy"] and clips[0].prompt == "hops"
    assert clips[0].framing is None and clips[0].seed is None
    assert store.get("abc123").raw_id == clips[0].id  # type: ignore[union-attr]
    assert store.migrate_raws() == []  # once


def test_cli_seed_and_variants_conflict(tmp_path: Path) -> None:
    from test_ai import _character

    source = tmp_path / "hero.png"
    _character().save(source)
    result = runner.invoke(
        cli_app, ["animate", str(source), "--variants", "2", "--seed", "5", "--prompt", "x"]
    )
    assert result.exit_code == 2 and "--seed" in result.output
    result = runner.invoke(cli_app, ["animate", str(source), "--variants", "4"])
    assert result.exit_code == 2


def test_token_constant_is_fake() -> None:
    assert TOKEN.startswith("fake-")
