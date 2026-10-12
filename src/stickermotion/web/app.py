"""FastAPI app: login, upload, description, generation, result, history.

Server-rendered Jinja2 pages with a little local JS (drag and drop, progress polling,
background switch). No CDN: fonts, CSS and JS are served from ``static/``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from stickermotion import raws
from stickermotion.core.models import ExportTarget
from stickermotion.engine.ai.decor import summary_uk
from stickermotion.publish import CHANNEL_TITLES, Channel
from stickermotion.publish.twitch import build_archive as twitch_archive
from stickermotion.web import files as outputs
from stickermotion.web.config import Settings
from stickermotion.web.files import ALL, ARCHIVES, UploadError, check_png
from stickermotion.web.preview import ensure_preview
from stickermotion.web.publishing import CHANNELS, Publishing
from stickermotion.web.security import (
    CSRF_FIELD,
    SESSION_USER,
    UPLOAD_OVERHEAD,
    BodyLimitMiddleware,
    csrf_ok,
    csrf_token,
    logged_in,
    password_ok,
)
from stickermotion.web.store import Status, Sticker, Store
from stickermotion.web.text import MAX_TEXT, clean
from stickermotion.web.worker import (
    DESCRIBE_ESTIMATE_USD,
    GENERATE_ESTIMATE_USD,
    Worker,
    generate_estimate_usd,
    generation_price_usd,
)

HERE = Path(__file__).parent
log = logging.getLogger("stickermotion.web")

STATUS_LABELS = {
    Status.DESCRIBE_QUEUED: "У черзі на опис",
    Status.DESCRIBING: "Описуємо дію",
    Status.DESCRIBED: "Опис готовий",
    Status.QUEUED: "У черзі",
    Status.GENERATING: "Генерація відео",
    Status.CUTTING: "Вирізання персонажа",
    Status.EXPORTING: "Експорт файлів",
    Status.DONE: "Готово",
    Status.ERROR: "Помилка",
    Status.INTERRUPTED: "Перервано",
    Status.CHOOSING: "Вибір варіанта",
}
VARIANT_CHOICES = (1, 2, 3)
PROGRESS_STEPS = (
    (Status.GENERATING, "Генерація", "Модель анімує персонажа, зазвичай 1-2 хвилини"),
    (Status.CUTTING, "Вирізання", "Прибираємо зелений фон і робимо безшовний луп"),
    (Status.EXPORTING, "Експорт", "Готуємо файли під ліміти кожної платформи"),
)
EMOTE_PREVIEWS = (
    (ExportTarget.TWITCH_GIF, "twitch_gif_28.gif", "Twitch"),
    (ExportTarget.DISCORD_EMOJI, "discord_emoji.webp", "Discord"),
    (ExportTarget.TELEGRAM_EMOJI, "telegram_emoji.webm", "Telegram"),
)
CSP = (
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
    "style-src 'self'; script-src 'self'; font-src 'self'; frame-ancestors 'none'; "
    "base-uri 'none'; form-action 'self'"
)


class LoginRequiredError(Exception):
    pass


def _date(value: str) -> str:
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return value
    return stamp.strftime("%d.%m.%Y, %H:%M")


def _usd(value: float) -> str:
    if value == 0:
        return "0$"
    if value < 0.01:
        return f"{value:.4f}$"
    return f"{value:.2f}$" if round(value, 2) == round(value, 4) else f"{value:.4f}$"


def create_app(
    settings: Settings | None = None,
    worker_factory: Callable[[Store, Settings], Worker] | None = None,
    publishing_factory: Callable[[Store, Settings], Publishing] | None = None,
) -> FastAPI:
    """``publishing_factory`` swaps the platform transport (fakes in tests)."""
    settings = settings or Settings.from_env()
    store = Store(settings.data_dir)
    worker = (worker_factory or Worker)(store, settings)
    publishing = (
        publishing_factory(store, settings)
        if publishing_factory
        else Publishing(store, settings.publish)
    )
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters["date"] = _date
    templates.env.filters["usd"] = _usd

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        worker.start()
        store.mark_publishing_interrupted()
        try:
            yield
        finally:
            worker.stop()

    app = FastAPI(
        title="StickerMotion", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.state.store = store
    app.state.worker = worker
    app.state.publishing = publishing
    app.state.settings = settings
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie="sm_session",
        max_age=30 * 24 * 3600,
        same_site="lax",
        https_only=settings.cookie_secure,
    )
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_upload_bytes + UPLOAD_OVERHEAD)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.middleware("http")
    async def headers(request: Request, call_next: Callable[[Request], Any]) -> Response:
        response: Response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if not request.url.path.startswith("/static"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(LoginRequiredError)
    async def to_login(request: Request, _exc: LoginRequiredError) -> Response:
        if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/login", status_code=303)
        return JSONResponse({"error": "login required"}, status_code=401)

    # Helpers

    def require_login(request: Request) -> None:
        if not logged_in(request):
            raise LoginRequiredError

    async def require_csrf(request: Request) -> Any:
        form = await request.form()
        if not csrf_ok(request, form):
            raise HTTPException(403, "CSRF token missing or wrong; reload the page.")
        return form

    def sticker_or_404(sticker_id: str) -> Sticker:
        sticker = store.get(sticker_id) if sticker_id.isalnum() else None
        if sticker is None:
            raise HTTPException(404, "Стікер не знайдено")
        return sticker

    def budget() -> dict[str, Any]:
        return {
            "today": store.spent_today(),
            "total": store.spent(),
            "daily": settings.daily_budget_usd,
            "left": worker.remaining_today(),
            "can_generate": worker.fits_today(GENERATE_ESTIMATE_USD),
            "can_describe": worker.fits_today(DESCRIBE_ESTIMATE_USD),
        }

    def render(request: Request, name: str, status_code: int = 200, **context: Any) -> Response:
        context.update(
            csrf=csrf_token(request),
            csrf_field=CSRF_FIELD,
            budget=budget(),
            price=generation_price_usd(),
            prices={n: generation_price_usd(n) for n in VARIANT_CHOICES},
            path=request.url.path,
            max_mb=round(settings.max_upload_bytes / 1024 / 1024),
        )
        return templates.TemplateResponse(request, name, context, status_code=status_code)

    def result_view(sticker: Sticker) -> dict[str, Any] | None:
        manifest = store.manifest(sticker.id)
        if sticker.state is not Status.DONE or manifest is None:
            return None
        items = outputs.output_files(manifest)
        names = {f.name for f in items}
        ai = manifest.get("ai", {})
        return {
            "groups": outputs.by_platform(items),
            "emotes": [e for e in EMOTE_PREVIEWS if e[1] in names],
            "hero": "telegram_sticker.webm" if "telegram_sticker.webm" in names else None,
            "hero_fallback": "discord_sticker.png" if "discord_sticker.png" in names else None,
            "invalid": [f for f in items if not f.valid],
            "duration": ai.get("loop", {}).get("duration_s"),
            "frames": ai.get("loop", {}).get("frames"),
            "focus": ai.get("emote_focus", {}).get("mode"),
            "prompt": ai.get("prompt", ""),
            "archives": [(p.key, p.title) for p, _ in outputs.by_platform(items)],
        }

    def clip_views(sticker: Sticker) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """(variants to choose from, earlier clips) for the sticker page."""
        clips = store.clips(sticker.id)
        views = [
            {
                "id": c.id,
                "created": c.created,
                "seed": c.seed,
                "prompt": c.prompt,
                "variant": c.meta.get("variant"),
                "legacy": bool(c.meta.get("legacy")),
                "preview": bool(c.preview),
                "current": c.id == sticker.raw_id,
                "batch": c.batch,
            }
            for c in clips
        ]
        if sticker.state is Status.CHOOSING and views:
            batch = views[0]["batch"]
            choose = sorted(
                (v for v in views if v["batch"] == batch), key=lambda v: v["variant"] or 0
            )
            return choose, [v for v in views if v["batch"] != batch]
        return [], views

    def thumb(sticker: Sticker) -> str:
        if (
            sticker.state is Status.DONE
            and (store.output_dir(sticker.id) / "discord_sticker.png").is_file()
        ):
            return f"/s/{sticker.id}/files/discord_sticker.png"
        return f"/s/{sticker.id}/input.png"

    # Public

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/login")
    async def login_page(request: Request) -> Response:
        if logged_in(request):
            return RedirectResponse("/", status_code=303)
        return render(request, "login.html")

    @app.post("/login")
    async def login(request: Request) -> Response:
        form = await request.form()
        if not csrf_ok(request, form):
            return render(request, "login.html", 403, error="Сторінка застаріла. Спробуйте ще раз.")
        if not password_ok(str(form.get("password", "")), settings.password):
            await asyncio.sleep(1.0)  # slows down guessing
            return render(request, "login.html", 401, error="Неправильний пароль.")
        request.session.clear()
        request.session[SESSION_USER] = True
        csrf_token(request)
        return RedirectResponse("/", status_code=303)

    @app.post("/logout")
    async def logout(request: Request) -> Response:
        await require_csrf(request)
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    # Pages

    @app.get("/")
    async def new_sticker(request: Request) -> Response:
        require_login(request)
        return render(request, "new.html")

    @app.post("/upload")
    async def upload(request: Request) -> Response:
        require_login(request)
        form = await require_csrf(request)
        item = form.get("file")
        data = await item.read() if hasattr(item, "read") else b""
        filename = Path(getattr(item, "filename", "") or "sticker.png").name[:120]
        try:
            checked = check_png(data, settings.max_upload_bytes)
        except UploadError as exc:
            return render(request, "new.html", 400, error=str(exc))
        if not worker.fits_today(DESCRIBE_ESTIMATE_USD):
            return render(request, "new.html", 402, error=_daily_message(settings, store))
        sticker = store.create(filename, checked.data, checked.width, checked.height)
        worker.submit_description(sticker.id)
        return RedirectResponse(f"/s/{sticker.id}", status_code=303)

    @app.get("/s/{sticker_id}")
    async def sticker_page(request: Request, sticker_id: str) -> Response:
        require_login(request)
        sticker = sticker_or_404(sticker_id)
        result = result_view(sticker)
        choose, earlier = clip_views(sticker)
        return render(
            request,
            "sticker.html",
            sticker=sticker,
            label=STATUS_LABELS[sticker.state],
            steps=PROGRESS_STEPS,
            result=result,
            max_text=MAX_TEXT,
            channels=publishing.views(sticker) if result else [],
            twitch=publishing.twitch_view(sticker) if result else None,
            published=publishing.published(sticker.id),
            channel_titles=CHANNEL_TITLES,
            decor_summary=summary_uk(sticker.decor_items),
            choose=choose,
            clips=earlier,
            variant_choices=VARIANT_CHOICES,
        )

    @app.get("/s/{sticker_id}/status")
    async def sticker_status(request: Request, sticker_id: str) -> dict[str, Any]:
        require_login(request)
        sticker = sticker_or_404(sticker_id)
        return {
            "status": sticker.status,
            "label": STATUS_LABELS[sticker.state],
            "busy": sticker.busy,
            "error": sticker.error,
            "text": sticker.text,
            "cost_usd": sticker.cost_usd,
        }

    @app.post("/s/{sticker_id}/describe")
    async def describe_again(request: Request, sticker_id: str) -> Response:
        require_login(request)
        await require_csrf(request)
        sticker = sticker_or_404(sticker_id)
        if sticker.busy:
            return RedirectResponse(f"/s/{sticker_id}", status_code=303)
        if not worker.fits_today(DESCRIBE_ESTIMATE_USD):
            store.update(sticker_id, error=_daily_message(settings, store))
            return RedirectResponse(f"/s/{sticker_id}", status_code=303)
        worker.submit_description(sticker_id)
        return RedirectResponse(f"/s/{sticker_id}", status_code=303)

    @app.post("/s/{sticker_id}/generate")
    async def generate(request: Request, sticker_id: str) -> Response:
        require_login(request)
        form = await require_csrf(request)
        sticker = sticker_or_404(sticker_id)
        if sticker.busy:
            return RedirectResponse(f"/s/{sticker_id}", status_code=303)
        text = clean(str(form.get("text", "")))
        if not text:
            store.update(sticker_id, error="Опис порожній. Напишіть, що має робити персонаж.")
            return RedirectResponse(f"/s/{sticker_id}", status_code=303)
        try:
            variants = int(str(form.get("variants", "1")))
        except ValueError:
            variants = 1
        variants = variants if variants in VARIANT_CHOICES else 1
        if not worker.fits_today(generate_estimate_usd(variants)):
            store.update(sticker_id, text=text, error=_daily_message(settings, store))
            return RedirectResponse(f"/s/{sticker_id}", status_code=303)
        # The "animate decor separately" box is only shown when decor was found.
        decor = "auto" if not sticker.decor_items or form.get("decor_separate") else "together"
        store.update(sticker_id, text=text, decor=decor)
        # After a finished generation the button means "a new clip": skip the video cache.
        worker.submit_generation(sticker_id, regenerate=sticker.generations > 0, variants=variants)
        return RedirectResponse(f"/s/{sticker_id}", status_code=303)

    # Stored raw clips

    def clip_or_404(sticker_id: str, clip_id: str) -> raws.RawClip:
        clip = raws.get(store.raw_dir(sticker_id), clip_id)
        if clip is None:
            raise HTTPException(404, "Ролик не знайдено")
        return clip

    @app.post("/s/{sticker_id}/raw/{clip_id}/use")
    async def use_clip(request: Request, sticker_id: str, clip_id: str) -> Response:
        """Full export of a stored clip: a chosen variant or "use this motion" (free)."""
        require_login(request)
        await require_csrf(request)
        sticker = sticker_or_404(sticker_id)
        clip_or_404(sticker_id, clip_id)
        if not sticker.busy:
            worker.submit_reprocess(sticker_id, clip_id)
        return RedirectResponse(f"/s/{sticker_id}", status_code=303)

    @app.get("/s/{sticker_id}/raw/{clip_id}/preview.webp")
    async def clip_preview(request: Request, sticker_id: str, clip_id: str) -> Response:
        require_login(request)
        sticker_or_404(sticker_id)
        clip = clip_or_404(sticker_id, clip_id)
        path = await asyncio.to_thread(ensure_preview, clip.folder)
        if path is None:
            raise HTTPException(404, "Превʼю немає")
        return FileResponse(path, media_type="image/webp")

    @app.post("/s/{sticker_id}/delete")
    async def delete(request: Request, sticker_id: str) -> Response:
        require_login(request)
        await require_csrf(request)
        sticker = sticker_or_404(sticker_id)
        if sticker.active and sticker.state is not Status.QUEUED:
            store.update(sticker_id, error="Не можна видалити під час генерації. Зачекайте.")
            return RedirectResponse(f"/s/{sticker_id}", status_code=303)
        form = await request.form()
        errors = []
        for channel in publishing.published(sticker_id):
            if form.get(f"remove_{channel.value}") == "1":
                error = await asyncio.to_thread(publishing.remove, sticker_id, channel)
                if error:
                    errors.append(f"{channel.value}: {error}")
        if errors:
            store.update(
                sticker_id,
                error="Стікер не видалено, бо не вдалося прибрати його з платформ: "
                + "; ".join(errors),
            )
            return RedirectResponse(f"/s/{sticker_id}", status_code=303)
        store.delete(sticker_id)
        return RedirectResponse("/history", status_code=303)

    # Publishing

    def channel_or_404(value: str) -> Channel:
        try:
            return Channel(value)
        except ValueError:
            raise HTTPException(404, "Невідома платформа") from None

    @app.post("/s/{sticker_id}/publish/{channel}")
    async def publish(request: Request, sticker_id: str, channel: str) -> Response:
        require_login(request)
        form = await require_csrf(request)
        sticker = sticker_or_404(sticker_id)
        target = channel_or_404(channel)
        if result_view(sticker) is None:
            raise HTTPException(409, "Стікер ще не готовий")
        name = str(form.get("name", ""))[:64]
        emoji = str(form.get("emoji", ""))[:200]
        targets = [target]
        if target is Channel.TELEGRAM_STICKER and form.get("also_emoji") == "1":
            targets.append(Channel.TELEGRAM_EMOJI)
        for item in targets:
            await asyncio.to_thread(publishing.publish, sticker, item, name, emoji)
        return RedirectResponse(f"/s/{sticker_id}#publish", status_code=303)

    @app.post("/s/{sticker_id}/unpublish/{channel}")
    async def unpublish(request: Request, sticker_id: str, channel: str) -> Response:
        require_login(request)
        await require_csrf(request)
        sticker_or_404(sticker_id)
        await asyncio.to_thread(publishing.remove, sticker_id, channel_or_404(channel))
        return RedirectResponse(f"/s/{sticker_id}#publish", status_code=303)

    @app.get("/s/{sticker_id}/publish.json")
    async def publish_state(request: Request, sticker_id: str) -> dict[str, Any]:
        require_login(request)
        sticker_or_404(sticker_id)
        rows = store.publications(sticker_id)
        return {
            c.value: {
                "status": rows[c.value].status if c.value in rows else "none",
                "url": rows[c.value].url if c.value in rows else "",
                "error": rows[c.value].error if c.value in rows else "",
            }
            for c in CHANNELS
        }

    @app.get("/s/{sticker_id}/twitch.zip")
    async def twitch_zip(request: Request, sticker_id: str) -> Response:
        require_login(request)
        sticker = sticker_or_404(sticker_id)
        stem = Path(sticker.filename).stem or "sticker"
        name = "".join(c if c.isalnum() else "" for c in stem.title())[:20]
        try:
            data = twitch_archive(
                store.output_dir(sticker_id), settings.publish.twitch_channel, name
            )
        except FileNotFoundError:
            raise HTTPException(404, "Файлів для Twitch немає") from None
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60] or "sticker"
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe}_twitch_upload.zip"'},
        )

    @app.get("/s/{sticker_id}/preview.webp")
    async def preview(request: Request, sticker_id: str) -> Response:
        """Full-alpha animated WebP of the Telegram sticker for browsers without WebM alpha."""
        require_login(request)
        sticker_or_404(sticker_id)
        path = await asyncio.to_thread(ensure_preview, store.output_dir(sticker_id))
        if path is None:
            raise HTTPException(404, "Превʼю немає")
        return FileResponse(path, media_type="image/webp")

    @app.get("/settings")
    async def settings_page(request: Request) -> Response:
        require_login(request)
        overview = await asyncio.to_thread(publishing.overview)
        return render(request, "settings.html", overview=overview)

    @app.get("/history")
    async def history(request: Request) -> Response:
        require_login(request)
        items = [(s, thumb(s), STATUS_LABELS[s.state]) for s in store.list()]
        return render(request, "history.html", items=items)

    # Files

    @app.get("/s/{sticker_id}/input.png")
    async def input_png(request: Request, sticker_id: str) -> Response:
        require_login(request)
        sticker_or_404(sticker_id)
        return FileResponse(store.input_path(sticker_id), media_type="image/png")

    @app.get("/s/{sticker_id}/files/{name:path}")
    async def output_file(request: Request, sticker_id: str, name: str) -> Response:
        require_login(request)
        sticker_or_404(sticker_id)
        root = store.output_dir(sticker_id).resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.suffix == ".json":
            raise HTTPException(404, "Файл не знайдено")
        download = request.query_params.get("download") == "1"
        return FileResponse(
            path,
            filename=path.name if download else None,
            content_disposition_type="attachment" if download else "inline",
        )

    @app.get("/s/{sticker_id}/zip/{group}")
    async def archive(request: Request, sticker_id: str, group: str) -> Response:
        require_login(request)
        sticker = sticker_or_404(sticker_id)
        manifest = store.manifest(sticker_id)
        if group != ALL and group not in ARCHIVES or manifest is None:
            raise HTTPException(404, "Архів не знайдено")
        items = outputs.output_files(manifest)
        data = outputs.build_zip(store.output_dir(sticker_id), items, group)
        stem = Path(sticker.filename).stem or "sticker"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60] or "sticker"
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe}_{group}.zip"'},
        )

    return app


def _daily_message(settings: Settings, store: Store) -> str:
    return (
        f"Денний ліміт {settings.daily_budget_usd:.2f}$ вичерпано: сьогодні витрачено "
        f"{store.spent_today():.4f}$. Платних викликів не було. Ліміт оновиться завтра "
        "або його можна підняти в DAILY_BUDGET_USD."
    )


def app_from_env() -> FastAPI:
    """Factory for ``uvicorn --factory stickermotion.web.app:app_from_env``."""
    return create_app(Settings.from_env())
