"""Login, CSRF and request-size guards for the single-user web UI."""

from __future__ import annotations

import hmac
import secrets
from typing import Any

from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

SESSION_USER = "auth"
SESSION_CSRF = "csrf"
CSRF_FIELD = "csrf_token"
CSRF_HEADER = "x-csrf-token"
# Multipart framing on top of the file itself.
UPLOAD_OVERHEAD = 64 * 1024


def password_ok(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.encode(), expected.encode())


def csrf_token(request: Request) -> str:
    """Per-session token, created on first use."""
    token = request.session.get(SESSION_CSRF)
    if not token:
        token = secrets.token_urlsafe(32)
        request.session[SESSION_CSRF] = token
    return str(token)


def csrf_ok(request: Request, form: Any | None) -> bool:
    expected = request.session.get(SESSION_CSRF)
    if not expected:
        return False
    given = request.headers.get(CSRF_HEADER) or (form.get(CSRF_FIELD) if form else None) or ""
    return hmac.compare_digest(str(given).encode(), str(expected).encode())


def logged_in(request: Request) -> bool:
    return bool(request.session.get(SESSION_USER))


class BodyLimitMiddleware:
    """Rejects request bodies above ``limit`` bytes with 413 before they are parsed."""

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        length = headers.get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.limit:
            await _too_large(scope, receive, send)
            return
        received = 0
        limit = self.limit

        async def guarded() -> Any:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _TooLargeError
            return message

        try:
            await self.app(scope, guarded, send)
        except _TooLargeError:
            await _too_large(scope, receive, send)


class _TooLargeError(Exception):
    pass


async def _too_large(scope: Scope, receive: Receive, send: Send) -> None:
    response = PlainTextResponse("Файл завеликий.", status_code=413)
    await response(scope, receive, send)
