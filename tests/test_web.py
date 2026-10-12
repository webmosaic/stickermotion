"""Web UI with a fake Replicate transport: no network, no spend."""

from __future__ import annotations

import io
import json
import re
import threading
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from stickermotion.core.models import ExportTarget
from stickermotion.engine.ai.describe import DESCRIBE_MODEL
from stickermotion.engine.ai.generate import WAN_MODEL
from stickermotion.engine.ai.provider import ApiError
from stickermotion.engine.ai.video import write_mp4
from stickermotion.web.app import create_app
from stickermotion.web.config import Settings, load_secret
from stickermotion.web.files import UploadError, check_png
from stickermotion.web.security import CSRF_FIELD
from stickermotion.web.store import Status, Store
from stickermotion.web.text import has_cyrillic, to_english
from stickermotion.web.worker import Worker
from test_ai import TOKEN, FakeReplicate, _character, _clip_frames

PASSWORD = "correct horse battery"
TARGETS = [ExportTarget.DISCORD_STICKER, ExportTarget.TELEGRAM_EMOJI, ExportTarget.OBS_OVERLAY]
AUTO_EN = "the character hops up and lands"
AUTO_UK = "персонаж підстрибує і приземляється"
FLAGGED = "The input or output was flagged as sensitive. (E005)"


@dataclass
class WebFake(FakeReplicate):
    """FakeReplicate that answers the cheap LLM by task: describe, focus or translation."""

    to_en: str = "the character waves with the right hand"
    flag_describes: int = 0
    """How many describe calls fail with Replicate's moderation error E005."""
    flag_tasks: dict[str, int] = field(default_factory=dict)
    """How many calls of a task (focus, decor, ...) fail with E005."""
    fail_tasks: dict[str, int] = field(default_factory=dict)
    """How many calls of a task fail with some other model error."""

    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> bytes:
        if method == "GET" and "/predictions/" in url and not url.endswith("/cancel"):
            pid = url.rsplit("/", 1)[1]
            item = next((c for c in self.created if c["id"] == pid), None)
            if item is not None and item["model"] == DESCRIBE_MODEL:
                self.calls.append((method, url, headers, body))
                task = self._task(item["input"])
                error = None
                if task == "describe" and self.flag_describes:
                    self.flag_describes -= 1
                    error = FLAGGED
                elif self.flag_tasks.get(task):
                    self.flag_tasks[task] -= 1
                    error = FLAGGED
                elif self.fail_tasks.get(task):
                    self.fail_tasks[task] -= 1
                    error = "CUDA out of memory"
                if error is not None:
                    return json.dumps(
                        {"id": pid, "status": "failed", "error": error, "urls": {"get": url}}
                    ).encode()
                return json.dumps(
                    {
                        "id": pid,
                        "status": "succeeded",
                        "output": [self._reply(item["input"])],
                        "metrics": {"token_input_count": 1200, "token_output_count": 300},
                        "urls": {"get": url},
                    }
                ).encode()
        return super().__call__(method, url, headers, body)

    def _reply(self, model_input: dict[str, Any]) -> str:
        prompt = model_input.get("prompt", "")
        if "Translate into Ukrainian" in prompt:
            return AUTO_UK
        if "Translate into English" in prompt:
            return self.to_en
        if "locate regions" in model_input.get("system_prompt", ""):
            return '{"x": 0.2, "y": 0.15, "w": 0.6, "h": 0.55, "what": "face"}'
        if "decorations apart" in model_input.get("system_prompt", ""):
            return '{"items": [{"id": 1, "decor": true, "what": "star", "motion": "twinkle"}]}'
        return json.dumps({"action": AUTO_EN, "decorations": ""})

    @staticmethod
    def _task(model_input: dict[str, Any]) -> str:
        prompt = model_input.get("prompt", "")
        if "Translate into" in prompt:
            return "translate_" + ("uk" if "Ukrainian" in prompt else "en")
        if "locate regions" in model_input.get("system_prompt", ""):
            return "focus"
        if "decorations apart" in model_input.get("system_prompt", ""):
            return "decor"
        return "describe"

    def llm_tasks(self) -> list[str]:
        return [self._task(item["input"]) for item in self.creates(DESCRIBE_MODEL)]


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def green_clip(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Short H.264 clip on a green background, as Wan delivers it (same as in test_ai)."""
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    write_mp4(path, _clip_frames(), 16.0)
    return path.read_bytes()


@pytest.fixture
def character() -> bytes:
    return _png(_character(200))


def _settings(tmp_path: Path, **kwargs: Any) -> Settings:
    defaults: dict[str, Any] = {
        "password": PASSWORD,
        "data_dir": tmp_path / "data",
        "token": TOKEN,
        "daily_budget_usd": 1.0,
    }
    return Settings(**{**defaults, **kwargs})


@dataclass
class Env:
    client: TestClient
    fake: WebFake
    settings: Settings

    @property
    def store(self) -> Store:
        return self.client.app.state.store  # type: ignore[attr-defined]

    @property
    def worker(self) -> Worker:
        return self.client.app.state.worker  # type: ignore[attr-defined]

    def csrf(self, path: str = "/") -> str:
        page = self.client.get(path).text
        match = re.search(rf'name="{CSRF_FIELD}" value="([^"]+)"', page)
        assert match, page[:500]
        return match.group(1)

    def login(self) -> None:
        token = self.csrf("/login")
        response = self.client.post(
            "/login", data={"password": PASSWORD, CSRF_FIELD: token}, follow_redirects=False
        )
        assert response.status_code == 303, response.text

    def upload(self, data: bytes, name: str = "hero.png") -> Any:
        return self.client.post(
            "/upload",
            data={CSRF_FIELD: self.csrf()},
            files={"file": (name, data, "image/png")},
            follow_redirects=False,
        )

    def new_sticker(self, data: bytes) -> str:
        response = self.upload(data)
        assert response.status_code == 303, response.text
        sticker_id = response.headers["location"].rsplit("/", 1)[1]
        self.worker.wait()
        return sticker_id

    def generate(self, sticker_id: str, text: str, decor_separate: str | None = "1") -> Any:
        """Submit the form as the browser does (the decor box is ticked by default)."""
        data = {"text": text, CSRF_FIELD: self.csrf(f"/s/{sticker_id}")}
        if decor_separate is not None:
            data["decor_separate"] = decor_separate
        response = self.client.post(
            f"/s/{sticker_id}/generate",
            data=data,
            follow_redirects=False,
        )
        self.worker.wait(timeout=300)
        return response


def _make_env(tmp_path: Path, green_clip: bytes, **settings: Any) -> Iterator[Env]:
    fake = WebFake(green_clip)
    config = _settings(tmp_path, **settings)

    def factory(store: Store, cfg: Settings) -> Worker:
        options = {"transport": fake, "sleep": lambda _s: None}
        return Worker(store, cfg, client_options=options, targets=TARGETS)

    app = create_app(config, worker_factory=factory)
    with TestClient(app) as client:
        yield Env(client, fake, config)


@pytest.fixture
def env(tmp_path: Path, green_clip: bytes) -> Iterator[Env]:
    yield from _make_env(tmp_path, green_clip)


# Login and protection


def test_pages_and_files_require_login(env: Env) -> None:
    for path in ("/", "/history", "/s/abc123", "/s/abc123/input.png", "/s/abc123/zip/all"):
        page = env.client.get(path, headers={"accept": "text/html"}, follow_redirects=False)
        assert page.status_code == 303 and page.headers["location"] == "/login", path
        api = env.client.get(path, headers={"accept": "application/json"})
        assert api.status_code == 401, path
    assert env.client.get("/healthz").json() == {"status": "ok"}
    login = env.client.get("/login")
    assert login.status_code == 200 and 'type="password"' in login.text


def test_login_with_wrong_and_right_password(env: Env) -> None:
    token = env.csrf("/login")
    wrong = env.client.post("/login", data={"password": "nope", CSRF_FIELD: token})
    assert wrong.status_code == 401 and "Неправильний пароль" in wrong.text
    env.login()
    home = env.client.get("/")
    assert home.status_code == 200 and "Новий стікер" in home.text
    logout = env.client.post("/logout", data={CSRF_FIELD: env.csrf()}, follow_redirects=False)
    assert logout.status_code == 303
    assert env.client.get("/", follow_redirects=False).status_code in (303, 401)


def test_csrf_is_required_for_forms(env: Env, character: bytes) -> None:
    no_token = env.client.post("/login", data={"password": PASSWORD})
    assert no_token.status_code == 403
    env.login()
    upload = env.client.post("/upload", files={"file": ("a.png", character, "image/png")})
    assert upload.status_code == 403
    forged = env.client.post(
        "/upload",
        data={CSRF_FIELD: "forged"},
        files={"file": ("a.png", character, "image/png")},
    )
    assert forged.status_code == 403
    assert env.store.list() == []
    sticker_id = env.new_sticker(character)
    for action in ("generate", "describe", "delete"):
        response = env.client.post(f"/s/{sticker_id}/{action}", data={"text": "hop"})
        assert response.status_code == 403, action
    assert env.store.get(sticker_id) is not None
    assert env.fake.creates(WAN_MODEL) == []


def test_session_cookie_is_signed_and_http_only(env: Env) -> None:
    token = env.csrf("/login")
    response = env.client.post(
        "/login", data={"password": PASSWORD, CSRF_FIELD: token}, follow_redirects=False
    )
    cookie = response.headers["set-cookie"]
    assert "httponly" in cookie.lower() and "samesite=lax" in cookie.lower()
    env.client.cookies.set("sm_session", "forged.value")
    assert env.client.get("/", follow_redirects=False).status_code in (303, 401)


def test_security_headers(env: Env) -> None:
    env.login()
    page = env.client.get("/")
    assert "default-src 'self'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    assert "cdn" not in page.text.lower() and "https://" not in page.text


# Upload validation


def test_upload_rejects_invalid_files(env: Env) -> None:
    env.login()
    opaque = _png(Image.new("RGB", (200, 200), (10, 120, 200)))
    cases = {
        b"not an image at all": "Не вдалося прочитати",
        _png(Image.new("RGBA", (200, 200), (1, 2, 3, 255))): "Фон не прозорий",
        opaque: "немає прозорості",
        _png(Image.new("RGBA", (200, 200), (0, 0, 0, 0))): "майже повністю прозоре",
        _png(_character().resize((40, 40))): "замале",
    }
    jpeg = io.BytesIO()
    Image.new("RGB", (200, 200)).save(jpeg, "JPEG")
    cases[jpeg.getvalue()] = "Це не PNG"
    for data, message in cases.items():
        response = env.upload(data)
        assert response.status_code == 400, message
        assert message in response.text
    assert env.store.list() == []


def test_upload_size_limit(tmp_path: Path, green_clip: bytes) -> None:
    for env in _make_env(tmp_path, green_clip, max_upload_bytes=50 * 1024):
        env.login()
        noise = np.random.default_rng(1).integers(0, 256, (150, 150, 4), dtype=np.uint8)
        data = _png(Image.fromarray(noise, "RGBA"))
        assert 50 * 1024 < len(data) < 50 * 1024 + 64 * 1024
        response = env.upload(data)
        assert response.status_code == 400 and "завеликий" in response.text
        huge = env.upload(b"\x89PNG" + b"0" * (200 * 1024))
        assert huge.status_code == 413
    with pytest.raises(UploadError, match="завеликий"):
        check_png(b"x" * 11, 10)


def test_check_png_accepts_character(character: bytes) -> None:
    checked = check_png(character, 10 * 1024 * 1024)
    assert (checked.width, checked.height) == (200, 200)


# Description


def test_description_is_shown_in_ukrainian(env: Env, character: bytes) -> None:
    env.login()
    sticker_id = env.new_sticker(character)
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DESCRIBED
    assert sticker.text == AUTO_UK and sticker.auto_en == AUTO_EN
    # The decor check runs first (the test character has a detached star).
    assert env.fake.llm_tasks() == ["decor", "describe", "translate_uk"]
    page = env.client.get(f"/s/{sticker_id}")
    assert AUTO_UK in page.text and "Згенерувати" in page.text and "0.05$" in page.text
    assert env.fake.creates(WAN_MODEL) == []
    status = env.client.get(f"/s/{sticker_id}/status").json()
    assert status["status"] == "described" and not status["busy"]


def test_moderation_flag_falls_back_to_neutral_background(env: Env, character: bytes) -> None:
    env.login()
    env.fake.flag_describes = 1
    sticker_id = env.new_sticker(character)
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DESCRIBED and sticker.text == AUTO_UK
    assert env.fake.llm_tasks() == ["decor", "describe", "describe", "translate_uk"]
    journal = [json.loads(line) for line in env.store.journal.read_text().splitlines()]
    describes = [e for e in journal if e["step"] == "describe"]
    assert [e["status"] for e in describes][:2] == ["failed", "succeeded"]
    assert describes[0]["cost_usd"] == 0.0
    # Flagged twice: a clear message, nothing billed.
    env.fake.flag_describes = 2
    flagged = env.new_sticker(character)
    failed = env.store.get(flagged)
    assert failed is not None and failed.state is Status.ERROR
    assert "E005" in failed.error and "не списано" in failed.error


def test_moderation_flag_on_decor_and_focus_does_not_stop_the_chain(
    env: Env, character: bytes
) -> None:
    """E005 on the decor and focus steps is retried once on grey, then the step
    falls back (shape heuristic, full composition); description and generation finish."""
    env.login()
    env.fake.flag_tasks = {"decor": 2}
    sticker_id = env.new_sticker(character)
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DESCRIBED, sticker
    assert sticker.text == AUTO_UK
    assert [i["source"] for i in sticker.decor_items] == ["heuristic"]
    assert env.fake.llm_tasks() == ["decor", "decor", "describe", "translate_uk"]

    env.fake.flag_tasks = {"decor": 2, "focus": 2}
    env.generate(sticker_id, AUTO_UK)
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DONE, sticker
    assert len(env.fake.creates(WAN_MODEL)) == 1
    assert env.fake.llm_tasks()[4:] == ["decor", "decor", "focus", "focus"]
    manifest = env.store.manifest(sticker_id)
    assert manifest is not None
    focus = manifest["ai"]["emote_focus"]
    assert focus["requested"] == "auto" and focus["mode"] == "full"
    assert "fallback" in focus["warnings"][0] and "E005" in focus["warnings"][0]
    decor = manifest["ai"]["decor"]
    assert decor["mode"] == "auto" and [i["source"] for i in decor["items"]] == ["heuristic"]
    assert any("fallback" in w for w in decor["warnings"])
    journal = [json.loads(line) for line in env.store.journal.read_text().splitlines()]
    fallbacks = [(e["step"], e["fallback"]) for e in journal if e["status"] == "fallback"]
    assert fallbacks == [
        ("decor", "shape heuristic"),
        ("decor", "shape heuristic"),
        ("focus", "full"),
    ]
    flagged = [e for e in journal if e["status"] == "failed"]
    assert len(flagged) == 6 and all(e["cost_usd"] == 0.0 for e in flagged)


def test_describe_again(env: Env, character: bytes) -> None:
    env.login()
    sticker_id = env.new_sticker(character)
    response = env.client.post(
        f"/s/{sticker_id}/describe",
        data={CSRF_FIELD: env.csrf(f"/s/{sticker_id}")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    env.worker.wait()
    assert env.fake.llm_tasks() == ["decor", "describe", "translate_uk"] * 2
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DESCRIBED and sticker.cost_usd > 0


def test_to_english_rules(tmp_path: Path, green_clip: bytes) -> None:
    for env in _make_env(tmp_path, green_clip):
        client = env.worker.client()
        assert to_english(client, AUTO_UK, AUTO_UK, AUTO_EN) == AUTO_EN  # unchanged: no call
        assert to_english(client, "She waves", AUTO_UK, AUTO_EN) == "She waves"  # English
        assert env.fake.creates(DESCRIBE_MODEL) == []
        assert to_english(client, "вона махає рукою", AUTO_UK, AUTO_EN) == env.fake.to_en
        assert env.fake.llm_tasks() == ["translate_en"]
    assert has_cyrillic("Привіт") and not has_cyrillic("hello")


# Generation, result, archives


def test_full_flow_with_edited_description(env: Env, character: bytes) -> None:
    env.login()
    sticker_id = env.new_sticker(character)
    edited = "дівчина махає правою рукою"
    response = env.generate(sticker_id, edited)
    assert response.status_code == 303
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DONE, sticker
    assert sticker.text == edited and sticker.prompt_en == env.fake.to_en
    assert sticker.generations == 1
    wan = env.fake.creates(WAN_MODEL)
    assert len(wan) == 1
    assert wan[0]["input"]["prompt"].startswith("The character waves with the right hand.")
    assert wan[0]["input"]["frames_per_second"] == 30 and wan[0]["input"]["num_frames"] == 81
    assert "translate_en" in env.fake.llm_tasks() and "focus" in env.fake.llm_tasks()
    assert sticker.cost_usd == pytest.approx(env.store.spent(), abs=1e-6)
    assert 0.05 < sticker.cost_usd < 0.052

    page = env.client.get(f"/s/{sticker_id}").text
    for text in ("Емоути в чаті", "Discord", "Telegram", "OBS", "Перегенерувати", "Шаховий"):
        assert text in page, text
    assert "discord_sticker.png" in page and "telegram_emoji.webm" in page
    sticker_png = env.client.get(f"/s/{sticker_id}/files/discord_sticker.png?download=1")
    assert sticker_png.status_code == 200
    assert "attachment" in sticker_png.headers["content-disposition"]
    assert env.client.get(f"/s/{sticker_id}/files/manifest.json").status_code == 404
    traversal = env.client.get(f"/s/{sticker_id}/files/..%2Finput.png")
    assert traversal.status_code == 404
    assert env.client.get(f"/s/{sticker_id}/files/%2E%2E%2F%2E%2E%2Fapp.db").status_code == 404

    telegram = env.client.get(f"/s/{sticker_id}/zip/telegram")
    assert telegram.status_code == 200 and telegram.headers["content-type"] == "application/zip"
    assert zipfile.ZipFile(io.BytesIO(telegram.content)).namelist() == ["telegram_emoji.webm"]
    everything = zipfile.ZipFile(io.BytesIO(env.client.get(f"/s/{sticker_id}/zip/all").content))
    assert sorted(everything.namelist()) == [
        "discord/discord_sticker.png",
        "obs/obs_overlay.webm",
        "telegram/telegram_emoji.webm",
    ]
    assert env.client.get(f"/s/{sticker_id}/zip/twitch").status_code == 200
    assert env.client.get(f"/s/{sticker_id}/zip/nope").status_code == 404


def test_decor_toggle_found_and_passed_to_generation(env: Env, character: bytes) -> None:
    """Decor found at the description step, shown with a checkbox (on by default);
    unticking it keeps the decor in the model input (together)."""
    env.login()
    sticker_id = env.new_sticker(character)
    sticker = env.store.get(sticker_id)
    assert sticker is not None
    assert [i["what"] for i in sticker.decor_items] == ["star"]
    page = env.client.get(f"/s/{sticker_id}").text
    assert 'name="decor_separate" value="1" checked' in page
    assert "Анімувати декор окремо" in page and "знайдено: зірка" in page

    env.generate(sticker_id, AUTO_UK)
    manifest = env.store.manifest(sticker_id)
    assert manifest is not None
    assert manifest["ai"]["decor"]["mode"] == "auto"
    assert manifest["ai"]["decor"]["composition"] is not None

    env.generate(sticker_id, AUTO_UK, decor_separate=None)
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.decor == "together"
    manifest = env.store.manifest(sticker_id)
    assert manifest is not None and manifest["ai"]["decor"]["mode"] == "together"
    page = env.client.get(f"/s/{sticker_id}").text
    assert 'name="decor_separate" value="1" >' in page


def test_store_adds_decor_columns_to_an_old_database(tmp_path: Path) -> None:
    import sqlite3

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
        "generations INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0)"
    )
    db.execute(
        "INSERT INTO stickers (id, created_at, updated_at, filename, status) "
        "VALUES ('abc', 't', 't', 'a.png', 'done')"
    )
    db.commit()
    db.close()
    store = Store(data)
    old = store.get("abc")
    assert old is not None and old.decor == "auto" and old.decor_items == []


def test_regenerate_is_a_new_paid_generation(env: Env, character: bytes) -> None:
    env.login()
    sticker_id = env.new_sticker(character)
    env.generate(sticker_id, AUTO_UK)
    env.generate(sticker_id, AUTO_UK)
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DONE and sticker.generations == 2
    wan = env.fake.creates(WAN_MODEL)
    assert len(wan) == 2  # the cache is skipped on purpose
    assert wan[0]["input"]["prompt"].startswith("The character hops up and lands.")
    assert "translate_en" not in env.fake.llm_tasks()  # unchanged text reuses the English one


def test_empty_description_is_rejected(env: Env, character: bytes) -> None:
    env.login()
    sticker_id = env.new_sticker(character)
    env.generate(sticker_id, "   ")
    sticker = env.store.get(sticker_id)
    assert sticker is not None and "порожній" in sticker.error
    assert env.fake.creates(WAN_MODEL) == []


def test_provider_error_is_shown_without_key(env: Env, character: bytes) -> None:
    from stickermotion.engine.ai.provider import ApiError

    env.login()
    sticker_id = env.new_sticker(character)
    # Decor and focus fall back on their errors; the Wan call (third) fails the run.
    env.fake.create_errors.extend(ApiError(500, f"boom with {TOKEN} inside") for _ in range(3))
    env.generate(sticker_id, "she waves")
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.ERROR
    assert env.fake.create_errors == []
    assert TOKEN not in sticker.error and "***" in sticker.error
    page = env.client.get(f"/s/{sticker_id}").text
    assert "Не вдалося" in page and TOKEN not in page


# History and deletion


def test_history_lists_stickers_and_total(env: Env, character: bytes) -> None:
    env.login()
    empty = env.client.get("/history").text
    assert "Тут з'являться ваші стікери" in empty
    first = env.new_sticker(character)
    env.generate(first, AUTO_UK)
    second = env.new_sticker(character)
    page = env.client.get("/history").text
    assert page.count('class="row"') == 2
    assert f"/s/{first}" in page and f"/s/{second}" in page
    assert AUTO_UK in page and "Готово" in page and "Опис готовий" in page
    total = env.store.spent()
    assert f"{total:.4f}$" in page or f"{total:.2f}$" in page
    assert f"/s/{first}/files/discord_sticker.png" in page  # preview of the result


def test_delete_removes_files(env: Env, character: bytes) -> None:
    env.login()
    sticker_id = env.new_sticker(character)
    folder = env.store.sticker_dir(sticker_id)
    assert folder.is_dir()
    response = env.client.post(
        f"/s/{sticker_id}/delete",
        data={CSRF_FIELD: env.csrf(f"/s/{sticker_id}")},
        follow_redirects=False,
    )
    assert response.status_code == 303 and response.headers["location"] == "/history"
    assert env.store.get(sticker_id) is None and not folder.exists()
    assert env.client.get(f"/s/{sticker_id}").status_code == 404


# Restart and budget


def test_interrupted_after_restart_is_not_retried(
    tmp_path: Path, green_clip: bytes, character: bytes
) -> None:
    store = Store(tmp_path / "data")
    generating = store.create("a.png", character, 200, 200)
    store.update(generating.id, status=Status.CUTTING, text="hop")
    queued = store.create("b.png", character, 200, 200)
    store.update(queued.id, status=Status.QUEUED, text="hop")
    describing = store.create("c.png", character, 200, 200)
    done = store.create("d.png", character, 200, 200)
    store.update(done.id, status=Status.DONE)
    for env in _make_env(tmp_path, green_clip):
        env.login()
        for sticker_id in (generating.id, queued.id):
            sticker = env.store.get(sticker_id)
            assert sticker is not None and sticker.state is Status.INTERRUPTED
            page = env.client.get(f"/s/{sticker_id}").text
            assert "Генерацію перервано" in page and "Перегенерувати" in page
        assert env.store.get(describing.id).state is Status.ERROR  # type: ignore[union-attr]
        assert env.store.get(done.id).state is Status.DONE  # type: ignore[union-attr]
        env.worker.wait()
        assert env.fake.created == []  # nothing paid automatically
        env.generate(generating.id, "hop")  # manual retry works
        assert env.store.get(generating.id).state is Status.DONE  # type: ignore[union-attr]


def test_daily_budget_blocks_paid_calls(
    tmp_path: Path, green_clip: bytes, character: bytes
) -> None:
    for env in _make_env(tmp_path, green_clip, daily_budget_usd=0.06):
        env.login()
        sticker_id = env.new_sticker(character)
        env.generate(sticker_id, AUTO_UK)
        assert env.store.get(sticker_id).state is Status.DONE  # type: ignore[union-attr]
        spent = env.store.spent_today()
        assert 0.05 < spent <= 0.06
        # Not enough left for another generation: refused before any call.
        before = len(env.fake.created)
        page = env.client.get(f"/s/{sticker_id}").text
        assert "Денний ліміт" in page and "disabled" in page
        env.generate(sticker_id, AUTO_UK)
        sticker = env.store.get(sticker_id)
        assert sticker is not None and "Денний ліміт" in sticker.error
        assert len(env.fake.created) == before
        assert "Денний ліміт" in env.client.get(f"/s/{sticker_id}").text
    # The run ledger also gets the daily remainder as its budget (second guard).
    for env in _make_env(tmp_path / "b", green_clip, daily_budget_usd=0.03):
        assert env.worker.client().ledger.budget_usd == pytest.approx(0.03)


def test_daily_budget_blocks_upload_description(
    tmp_path: Path, green_clip: bytes, character: bytes
) -> None:
    for env in _make_env(tmp_path, green_clip, daily_budget_usd=0.0):
        env.login()
        assert "Денний ліміт вичерпано" in env.client.get("/").text
        response = env.upload(character)
        assert response.status_code == 402 and "Денний ліміт" in response.text
        assert env.fake.created == [] and env.store.list() == []


# Key never leaks


def test_key_never_in_responses_or_logs(
    env: Env, character: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    env.login()
    sticker_id = env.new_sticker(character)
    env.generate(sticker_id, "she waves")
    bodies = [
        env.client.get(path).content
        for path in (
            "/",
            "/history",
            f"/s/{sticker_id}",
            f"/s/{sticker_id}/status",
            f"/s/{sticker_id}/zip/all",
            "/healthz",
            "/static/app.js",
        )
    ]
    assert all(TOKEN.encode() not in body for body in bodies)
    assert TOKEN not in caplog.text
    journal = env.store.journal.read_text()
    assert TOKEN not in journal and journal.strip()
    for path in env.settings.data_dir.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".txt", ".html", ".db"):
            assert TOKEN.encode() not in path.read_bytes(), path
    assert TOKEN not in repr(env.settings)


# Settings


def test_settings_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"REPLICATE_API_TOKEN={TOKEN}\nAPP_PASSWORD=from-file\n")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("DAILY_BUDGET_USD", "2.5")
    settings = Settings.from_env(dotenv)
    assert settings.password == "from-file" and settings.token == TOKEN
    assert settings.daily_budget_usd == 2.5 and settings.data_dir == tmp_path / "d"
    key = (tmp_path / "d" / ".secret_key").read_text().strip()
    assert settings.secret_key == key and load_secret(tmp_path / "d" / ".secret_key") == key
    monkeypatch.setenv("APP_PASSWORD", "")
    with pytest.raises(ValueError, match="APP_PASSWORD"):
        Settings(password="", data_dir=tmp_path / "e")


def test_status_endpoint_unknown_id(env: Env) -> None:
    env.login()
    assert env.client.get("/s/doesnotexist/status").status_code == 404
    assert env.client.get("/s/..%2F..%2Fetc/status").status_code == 404


# Paid steps one at a time


@dataclass
class RateLimitedFake(WebFake):
    """Answers 429 to a create while another prediction is still in flight, as Replicate
    does on a low-balance account, and remembers the highest concurrency it saw."""

    active: set[str] = field(default_factory=set)
    max_active: int = 0
    rate_limited: int = 0
    gate: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, method: str, url: str, headers: dict[str, str], body: bytes | None) -> bytes:
        if method == "POST" and url.endswith("/predictions") and "/models/" in url:
            with self.gate:
                if self.active:
                    self.rate_limited += 1
                    raise ApiError(429, '{"retry_after": 1}')
                reply = super().__call__(method, url, headers, body)
                self.active.add(json.loads(reply)["id"])
                self.max_active = max(self.max_active, len(self.active))
            time.sleep(0.02)  # widen the window a parallel create would hit
            return reply
        reply = super().__call__(method, url, headers, body)
        if method == "GET" and "/predictions/" in url and not url.endswith("/cancel"):
            with self.gate:
                self.active.discard(url.rsplit("/", 1)[1])
        return reply


def test_mass_upload_is_described_without_rate_limit_or_cancels(
    tmp_path: Path, green_clip: bytes, character: bytes
) -> None:
    fake = RateLimitedFake(green_clip)
    config = _settings(tmp_path, daily_budget_usd=5.0)

    def factory(store: Store, cfg: Settings) -> Worker:
        options = {"transport": fake, "sleep": lambda _s: None}
        return Worker(store, cfg, client_options=options, targets=TARGETS)

    app = create_app(config, worker_factory=factory)
    with TestClient(app) as client:
        env = Env(client, fake, config)
        env.login()
        ids = []
        for index in range(10):
            response = env.upload(character, name=f"hero{index}.png")
            assert response.status_code == 303, response.text
            ids.append(response.headers["location"].rsplit("/", 1)[1])
        env.worker.wait(timeout=300)
        stickers = [env.store.get(sticker_id) for sticker_id in ids]
        assert [s.state if s else None for s in stickers] == [Status.DESCRIBED] * 10
        assert fake.rate_limited == 0 and fake.cancelled == []
        assert fake.max_active == 1
        assert fake.llm_tasks().count("describe") == 10


def test_description_waits_in_queue_while_a_paid_step_runs(env: Env, character: bytes) -> None:
    env.login()
    with env.worker._paid:  # a generation or another description is being paid for
        response = env.upload(character)
        sticker_id = response.headers["location"].rsplit("/", 1)[1]
        sticker = env.store.get(sticker_id)
        assert sticker is not None and sticker.state is Status.DESCRIBE_QUEUED and sticker.busy
        assert "У черзі на опис" in env.client.get(f"/s/{sticker_id}").text
        status = env.client.get(f"/s/{sticker_id}/status").json()
        assert status["status"] == "describe_queued" and status["busy"]
        assert env.fake.creates() == []
    env.worker.wait()
    sticker = env.store.get(sticker_id)
    assert sticker is not None and sticker.state is Status.DESCRIBED


def test_queued_description_becomes_error_after_restart(tmp_path: Path, character: bytes) -> None:
    store = Store(tmp_path / "data")
    sticker = store.create("hero.png", character, 200, 200)
    store.update(sticker.id, status=Status.DESCRIBE_QUEUED)
    assert store.mark_interrupted() == [sticker.id]
    after = store.get(sticker.id)
    assert after is not None and after.state is Status.ERROR
