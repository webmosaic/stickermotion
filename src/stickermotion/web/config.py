"""Web service settings from environment variables (secrets never live in code)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from stickermotion.engine.ai.costs import DEFAULT_BUDGET_USD
from stickermotion.engine.ai.provider import read_token
from stickermotion.publish.config import PublishConfig

DEFAULT_DAILY_BUDGET_USD = 1.0
DEFAULT_MAX_UPLOAD_MB = 10.0
SECRET_FILE = ".secret_key"


def _float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """``APP_PASSWORD``: the only login. ``DATA_DIR``: files, SQLite, cache, cost journal.

    ``SECRET_KEY`` signs the session cookie; when unset a random key is kept in
    ``DATA_DIR/.secret_key`` (mode 600). ``DAILY_BUDGET_USD`` caps the spend per calendar
    day, ``RUN_BUDGET_USD`` one generation. ``COOKIE_SECURE=1`` behind HTTPS.
    """

    password: str
    data_dir: Path
    token: str | None = field(default=None, repr=False)
    secret_key: str = field(default="", repr=False)
    daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD
    run_budget_usd: float = DEFAULT_BUDGET_USD
    max_upload_bytes: int = int(DEFAULT_MAX_UPLOAD_MB * 1024 * 1024)
    cookie_secure: bool = False
    publish: PublishConfig = field(default_factory=PublishConfig, repr=False)
    """Telegram and Discord bots; tokens stay inside, never rendered."""

    def __post_init__(self) -> None:
        if not self.password:
            raise ValueError("APP_PASSWORD is not set")
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.secret_key:
            self.secret_key = load_secret(self.data_dir / SECRET_FILE)

    @classmethod
    def from_env(cls, dotenv: Path | None = Path(".env")) -> Settings:
        _load_dotenv(dotenv)
        return cls(
            password=os.environ.get("APP_PASSWORD", ""),
            data_dir=Path(os.environ.get("DATA_DIR", "data")),
            token=read_token(dotenv),
            secret_key=os.environ.get("SECRET_KEY", ""),
            daily_budget_usd=_float("DAILY_BUDGET_USD", DEFAULT_DAILY_BUDGET_USD),
            run_budget_usd=_float("RUN_BUDGET_USD", DEFAULT_BUDGET_USD),
            max_upload_bytes=int(_float("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB) * 1024 * 1024),
            cookie_secure=_flag("COOKIE_SECURE"),
            publish=PublishConfig.from_env(),
        )


def _load_dotenv(dotenv: Path | None) -> None:
    """Fill unset variables from a ``KEY=value`` file (environment wins)."""
    if dotenv is None or not dotenv.is_file():
        return
    for line in dotenv.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        key = key.strip()
        if sep and key and not key.startswith("#") and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def load_secret(path: Path) -> str:
    """Random signing key kept next to the data, created on first start."""
    if path.is_file():
        value = path.read_text().strip()
        if value:
            return value
    value = secrets.token_urlsafe(48)
    path.write_text(value + "\n")
    path.chmod(0o600)
    return value
