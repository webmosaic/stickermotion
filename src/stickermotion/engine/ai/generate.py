"""Wan 2.2 i2v fast generation and the raw-video cache.

ADR-011: ``wan-video/wan-2.2-i2v-fast``, 480p, the model input frame is both the first and
the last image. ADR-015: 81 frames played at 30 fps (2.7 s) by default. 0.05$ per video
(flat, checked 2026-10-08).

Clip length (model schema, checked 2026-10-08): ``num_frames`` 81..121 (81 recommended)
and ``frames_per_second`` 5..30. "Pricing is based on the video duration at 16 fps", so
the price follows ``num_frames`` only: 81 frames = 5.06 s at 16 fps = 0.05$, whatever the
playback fps. Both are configurable (``wan_params``); the defaults follow ADR-015.

Resolution: ``480p`` (default) or ``720p``. Published price per video (checked
2026-10-09 on the model page): 0.05$ at 480p, 0.11$ at 720p (base variant, no
interpolation).

The raw video is cached under ``<cache>/<sha256>.mp4``, keyed by the model input frame
(the PNG with its framing), the prompt, the model and its parameters. A repeated run with
the same inputs reuses it without a paid call unless regeneration is forced.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from stickermotion.engine.ai.costs import Price
from stickermotion.engine.ai.provider import ReplicateClient, output_url, png_bytes

WAN_MODEL = "wan-video/wan-2.2-i2v-fast"
WAN_PRICE = Price(estimate_usd=0.05, flat_usd=0.05)
WAN_RESOLUTION_PRICES = {"480p": 0.05, "720p": 0.11}
WAN_PARAMS: dict[str, Any] = {
    "resolution": "480p",
    "num_frames": 81,
    "frames_per_second": 30,
    "go_fast": True,
    "interpolate_output": False,
}
WAN_FRAMES_RANGE = (81, 121)
WAN_FPS_RANGE = (5, 30)
# The price is quoted for the default clip: 81 frames at 16 fps.
WAN_PRICED_FRAMES = 81
MAX_SEED = 2**31 - 1


def random_seed() -> int:
    """A fresh seed for one clip; kept with the clip so it can be repeated."""
    import secrets

    return secrets.randbelow(MAX_SEED + 1)


def wan_params(
    frames: int | None = None,
    fps: int | None = None,
    resolution: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Wan parameters with ``num_frames`` / ``frames_per_second`` / ``resolution`` checked.

    ``seed`` is added only when given, so earlier cache keys stay valid.
    """
    params = dict(WAN_PARAMS)
    if seed is not None:
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed must be in 0..{MAX_SEED}, got {seed}")
        params["seed"] = int(seed)
    if resolution is not None:
        if resolution not in WAN_RESOLUTION_PRICES:
            raise ValueError(
                f"resolution must be one of {', '.join(WAN_RESOLUTION_PRICES)}, got {resolution!r}"
            )
        params["resolution"] = resolution
    if frames is not None:
        low, high = WAN_FRAMES_RANGE
        if not low <= frames <= high:
            raise ValueError(f"frames must be in {low}..{high} (Wan 2.2 schema), got {frames}")
        params["num_frames"] = frames
    if fps is not None:
        low, high = WAN_FPS_RANGE
        if not low <= fps <= high:
            raise ValueError(f"fps must be in {low}..{high} (Wan 2.2 schema), got {fps}")
        params["frames_per_second"] = fps
    return params


def wan_price(params: dict[str, Any]) -> Price:
    """Price per video at the resolution, scaled with the clip duration at 16 fps
    (``num_frames``)."""
    base = WAN_RESOLUTION_PRICES[params.get("resolution", "480p")]
    usd = round(base * params["num_frames"] / WAN_PRICED_FRAMES, 4)
    return Price(estimate_usd=usd, flat_usd=usd)


def cache_key(frame_png: bytes, prompt: str, model: str, params: dict[str, Any]) -> str:
    """SHA-256 over the input frame bytes, prompt, model and parameters."""
    digest = hashlib.sha256()
    digest.update(frame_png)
    meta = {"prompt": prompt, "model": model, "params": params}
    digest.update(json.dumps(meta, sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class Generation:
    video: Path
    cache_key: str
    cached: bool
    cost_usd: float
    prediction_id: str | None = None


def wan_input(image_ref: str, prompt: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "image": image_ref,
        "last_image": image_ref,
        **(params if params is not None else WAN_PARAMS),
    }


def generate(
    client: ReplicateClient,
    frame: Image.Image,
    prompt: str,
    cache_dir: Path,
    regenerate: bool = False,
    params: dict[str, Any] | None = None,
) -> Generation:
    """Raw video for ``frame`` + ``prompt``: from the cache, or one paid Wan call.

    ``params`` (see ``wan_params``) default to ``WAN_PARAMS``; the default keeps the
    cache keys of earlier runs valid.
    """
    params = dict(params if params is not None else WAN_PARAMS)
    payload = png_bytes(frame.convert("RGB"))
    key = cache_key(payload, prompt, WAN_MODEL, params)
    path = cache_dir / f"{key}.mp4"
    if path.is_file() and path.stat().st_size > 0 and not regenerate:
        client.ledger.record(
            step="generate", model=WAN_MODEL, params=params, status="cached", cache_key=key
        )
        return Generation(path, key, cached=True, cost_usd=0.0)
    ref = client.image_ref(payload, "frame.png")
    spent_before = client.ledger.spent_usd
    prediction = client.predict(
        WAN_MODEL, wan_input(ref, prompt, params), wan_price(params), "generate"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(client.download(output_url(prediction.get("output"))))
    tmp.replace(path)
    cost = round(client.ledger.spent_usd - spent_before, 6)
    return Generation(path, key, cached=False, cost_usd=cost, prediction_id=prediction.get("id"))
