"""Background work in the web process: descriptions (cheap) and one generation at a time.

Generations go through one worker thread and a FIFO queue (one paid generation at a time,
no Redis for a single user). Descriptions run on a small separate pool. Paid Replicate
steps (describe, decor, focus, generate) of all stickers are serialized by one lock,
so a mass upload does not hit Replicate's rate limit: a waiting description shows
"in queue", a waiting generation stays queued. Every paid call goes through
``ReplicateClient`` with a ledger whose budget is the smaller of the per-run budget
and what is left of the daily budget, so neither guard can be passed.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from PIL import Image

from stickermotion import raws
from stickermotion.ai_pipeline import animate
from stickermotion.core.models import ExportTarget
from stickermotion.engine.ai import decor as decor_layers
from stickermotion.engine.ai.costs import BudgetExceededError, CostLedger
from stickermotion.engine.ai.describe import DESCRIBE_PRICE
from stickermotion.engine.ai.generate import random_seed, wan_params, wan_price
from stickermotion.engine.ai.provider import (
    MissingTokenError,
    PaymentRequiredError,
    ProviderError,
    RateLimitedError,
    ReplicateClient,
)
from stickermotion.reprocess import reprocess
from stickermotion.variants import animate_variants, check_count
from stickermotion.web import text
from stickermotion.web.config import Settings
from stickermotion.web.store import STICKERS_DIR, Status, Store

log = logging.getLogger("stickermotion.web")

STEP_STATUS = {"generate": Status.GENERATING, "cut": Status.CUTTING, "export": Status.EXPORTING}
# Upper bounds used to decide whether an action still fits the daily budget.
# describe + translation + decor check
DESCRIBE_ESTIMATE_USD = round(DESCRIBE_PRICE.estimate_usd * 3, 6)
GENERATE_ESTIMATE_USD = round(
    wan_price(wan_params()).estimate_usd + DESCRIBE_PRICE.estimate_usd * 3, 6
)  # Wan + emote focus + translation + decor check


def generation_price_usd(variants: int = 1) -> float:
    """Price shown on the button: Wan clips with the ADR-015 parameters (per variant)."""
    return round(variants * wan_price(wan_params()).estimate_usd, 6)


def generate_estimate_usd(variants: int = 1) -> float:
    """Upper bound of a generation of ``variants`` clips for the daily budget check."""
    return round(GENERATE_ESTIMATE_USD + (variants - 1) * wan_price(wan_params()).estimate_usd, 6)


class Worker:
    def __init__(
        self,
        store: Store,
        settings: Settings,
        client_options: dict[str, Any] | None = None,
        targets: list[ExportTarget] | None = None,
    ) -> None:
        """``client_options`` go to ``ReplicateClient`` (a fake transport in tests);
        ``targets`` default to every export target."""
        self.store = store
        self.settings = settings
        self.client_options = client_options or {}
        self.targets = targets
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="describe")
        self._futures: set[Future[None]] = set()
        self._lock = threading.Lock()
        self._paid = threading.Lock()
        """Held for the paid part of a description or a generation."""

    # Lifecycle

    def start(self) -> None:
        migrated = self.store.migrate_raws()
        if migrated:
            log.info("kept the raw clip of %d earlier stickers", len(migrated))
        interrupted = self.store.mark_interrupted()
        if interrupted:
            log.warning("marked %d unfinished stickers as interrupted", len(interrupted))
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="generate", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=5)
            self._thread = None
        self._pool.shutdown(wait=False, cancel_futures=True)

    def wait(self, timeout: float = 120.0) -> None:
        """Block until queued work is finished (tests)."""
        with self._lock:
            futures = list(self._futures)
        for future in futures:
            future.result(timeout=timeout)
        self._queue.join()

    # Budget

    def remaining_today(self) -> float:
        return round(max(0.0, self.settings.daily_budget_usd - self.store.spent_today()), 6)

    def fits_today(self, estimate_usd: float) -> bool:
        return estimate_usd <= self.remaining_today() + 1e-9

    def client(self, variants: int = 1) -> ReplicateClient:
        """Ledger budget: the run budget per variant (a batch of N variants is one
        run worth N generations), never more than what is left of the day."""
        budget = min(self.settings.run_budget_usd * variants, self.remaining_today())
        ledger = CostLedger(self.store.journal, budget)
        return ReplicateClient(self.settings.token, ledger, **self.client_options)

    def redact(self, message: str) -> str:
        token = self.settings.token
        return message.replace(token, "***") if token else message

    def explain(self, exc: BaseException) -> str:
        """User-facing message (Ukrainian) without the key."""
        if isinstance(exc, BudgetExceededError):
            if self.remaining_today() < self.settings.run_budget_usd:
                return (
                    f"Денний ліміт витрат {self.settings.daily_budget_usd:.2f}$ не дозволяє цей "
                    f"крок: сьогодні вже витрачено {self.store.spent_today():.4f}$. Платний виклик "
                    "не робився. Ліміт оновиться завтра або його можна змінити в DAILY_BUDGET_USD."
                )
            return (
                f"Ліміт на одну генерацію {self.settings.run_budget_usd:.2f}$ не дозволяє цей "
                "крок. Платний виклик не робився."
            )
        if isinstance(exc, MissingTokenError):
            return "На сервері не задано ключ REPLICATE_API_TOKEN."
        if isinstance(exc, PaymentRequiredError):
            return "На рахунку Replicate закінчились кошти. Поповніть баланс і повторіть."
        if isinstance(exc, RateLimitedError):
            return "Replicate тимчасово обмежив частоту запитів. Спробуйте за хвилину."
        if isinstance(exc, ProviderError) and "flagged as sensitive" in str(exc):
            return (
                "Модерація Replicate помилково позначила запит як чутливий (E005), гроші не "
                "списано. Повторіть ще раз або напишіть опис дії самостійно."
            )
        if isinstance(exc, ProviderError):
            return self.redact(f"Помилка сервісу генерації: {exc}")
        return self.redact(f"Внутрішня помилка: {type(exc).__name__}: {exc}")

    # Descriptions

    def submit_description(self, sticker_id: str) -> None:
        self.store.update(sticker_id, status=Status.DESCRIBE_QUEUED, error="")
        future = self._pool.submit(self._describe_in_turn, sticker_id)
        with self._lock:
            self._futures.add(future)
        future.add_done_callback(self._forget)

    def _forget(self, future: Future[None]) -> None:
        with self._lock:
            self._futures.discard(future)

    def _describe_in_turn(self, sticker_id: str) -> None:
        with self._paid:
            sticker = self.store.get(sticker_id)
            if sticker is None or sticker.state is not Status.DESCRIBE_QUEUED:
                return  # deleted while waiting
            self.store.update(sticker_id, status=Status.DESCRIBING)
            self._describe(sticker_id)

    def _describe(self, sticker_id: str) -> None:
        client: ReplicateClient | None = None
        try:
            client = self.client()
            with Image.open(self.store.input_path(sticker_id)) as opened:
                image = opened.convert("RGBA")
            # Detached decor is animated on its own, so the description is made
            # for the character alone and does not mention it.
            decor = decor_layers.plan(image, decor_layers.DecorMode.AUTO, client)
            found = text.auto_description(client, decor.character)
            self.store.update(
                sticker_id,
                status=Status.DESCRIBED,
                text=found.ukrainian,
                auto_uk=found.ukrainian,
                auto_en=found.english,
                decor_found=json.dumps([i.to_dict() for i in decor.items]),
                error="",
            )
        except Exception as exc:  # the user sees the reason, the service keeps running
            log.error("describe %s failed: %s", sticker_id, self.redact(repr(exc)))
            self.store.update(sticker_id, status=Status.ERROR, error=self.explain(exc))
        finally:
            if client is not None:
                self.store.add_cost(sticker_id, client.ledger.spent_usd)

    # Generations

    def submit_generation(
        self, sticker_id: str, regenerate: bool = False, variants: int = 1
    ) -> None:
        self.store.update(
            sticker_id,
            status=Status.QUEUED,
            error="",
            regenerate=int(regenerate),
            variants=check_count(variants),
            job="generate",
            job_raw="",
        )
        self._queue.put(sticker_id)

    def submit_reprocess(self, sticker_id: str, clip_id: str) -> None:
        """Full export of a stored clip (free): "use this motion" or a chosen variant."""
        self.store.update(
            sticker_id, status=Status.QUEUED, error="", job="reprocess", job_raw=clip_id
        )
        self._queue.put(sticker_id)

    def _loop(self) -> None:
        while True:
            sticker_id = self._queue.get()
            try:
                if sticker_id is None:
                    return
                sticker = self.store.get(sticker_id)
                if sticker is not None and sticker.job == "reprocess":
                    self._reprocess(sticker_id)
                else:
                    with self._paid:  # stays "queued" while a description is paid for
                        self._generate(sticker_id)
            finally:
                self._queue.task_done()

    def _generate(self, sticker_id: str) -> None:
        sticker = self.store.get(sticker_id)
        if sticker is None or sticker.state is not Status.QUEUED:
            return  # deleted or changed while waiting
        client: ReplicateClient | None = None
        variants = max(1, min(3, sticker.variants or 1))
        try:
            self.store.update(sticker_id, status=Status.GENERATING)
            client = self.client(variants)
            prompt = text.to_english(client, sticker.text, sticker.auto_uk, sticker.auto_en)
            self.store.update(sticker_id, prompt_en=prompt)

            def progress(step: str) -> None:
                if variants == 1 or step == "generate":
                    self.store.update(sticker_id, status=STEP_STATUS[step])

            decor = sticker.decor or decor_layers.DecorMode.AUTO
            if variants > 1:
                # Previews only; the current result (if any) stays until one is chosen.
                animate_variants(
                    self.store.input_path(sticker_id),
                    prompt,
                    variants,
                    out_dir=self.store.data_dir,
                    name=f"{STICKERS_DIR}/{sticker_id}",
                    raw_dir=self.store.raw_dir(sticker_id),
                    client=client,
                    decor=decor,
                    progress=progress,
                )
                if self.store.get(sticker_id) is None:
                    return
                self.store.update(
                    sticker_id,
                    status=Status.CHOOSING,
                    error="",
                    regenerate=0,
                    job="",
                    generations=sticker.generations + variants,
                )
                return
            shutil.rmtree(self.store.output_dir(sticker_id), ignore_errors=True)
            animate(
                self.store.input_path(sticker_id),
                prompt=prompt,
                targets=self.targets,
                out_dir=self.store.data_dir,
                name=f"{STICKERS_DIR}/{sticker_id}",
                regenerate=bool(sticker.regenerate),
                client=client,
                progress=progress,
                decor=decor,
                seed=random_seed(),
            )
            if self.store.get(sticker_id) is None:
                return  # deleted while running
            clip = raws.from_output(
                self.store.raw_dir(sticker_id), self.store.output_dir(sticker_id)
            )
            self.store.update(
                sticker_id,
                status=Status.DONE,
                error="",
                regenerate=0,
                job="",
                raw_id=clip.id if clip else "",
                generations=sticker.generations + 1,
            )
        except Exception as exc:
            log.error("generate %s failed: %s", sticker_id, self.redact(repr(exc)))
            if self.store.get(sticker_id) is not None:
                self.store.update(sticker_id, status=Status.ERROR, error=self.explain(exc))
        finally:
            if client is not None and self.store.get(sticker_id) is not None:
                self.store.add_cost(sticker_id, client.ledger.spent_usd)

    def _reprocess(self, sticker_id: str) -> None:
        sticker = self.store.get(sticker_id)
        if sticker is None or sticker.state is not Status.QUEUED:
            return
        try:
            clip = raws.get(self.store.raw_dir(sticker_id), sticker.job_raw)
            if clip is None:
                raise FileNotFoundError("цей ролик більше не зберігається")
            self.store.update(sticker_id, status=Status.CUTTING)
            focus = clip.focus_spec
            previous = self.store.manifest(sticker_id) or {}
            box = (previous.get("ai") or {}).get("emote_focus", {}).get("source_box")
            if focus == "full" and isinstance(box, list) and len(box) == 4:
                focus = f"{box[0]},{box[1]},{box[2] - box[0]},{box[3] - box[1]}"
            shutil.rmtree(self.store.output_dir(sticker_id), ignore_errors=True)

            def progress(step: str) -> None:
                self.store.update(sticker_id, status=STEP_STATUS[step])

            reprocess(
                clip.video,
                self.store.input_path(sticker_id),
                clip.model_input,
                out_dir=self.store.data_dir,
                name=f"{STICKERS_DIR}/{sticker_id}",
                targets=self.targets,
                emote_focus=focus,
                decor=sticker.decor or decor_layers.DecorMode.AUTO,
                framing=clip.framing,
                framing_subject=clip.framing_subject,
                meta=clip.meta,
                progress=progress,
            )
            if self.store.get(sticker_id) is None:
                return
            self.store.update(
                sticker_id, status=Status.DONE, error="", job="", job_raw="", raw_id=clip.id
            )
        except Exception as exc:
            log.error("reprocess %s failed: %s", sticker_id, self.redact(repr(exc)))
            if self.store.get(sticker_id) is not None:
                message = (
                    f"Не вдалося використати ролик: {exc}"
                    if isinstance(exc, FileNotFoundError)
                    else self.explain(exc)
                )
                self.store.update(sticker_id, status=Status.ERROR, error=message, job="")


WorkerFactory = Callable[[Store, Settings], Worker]
