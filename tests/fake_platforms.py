"""In-memory fakes of the Telegram Bot API and the Discord REST API (tests).

Answers follow the documented response shapes (``{"ok": true, "result": ...}`` /
``{"ok": false, "error_code": ..., "description": ...}`` for Telegram, JSON objects and
``{"code": ..., "message": ...}`` errors for Discord). No request leaves the process.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from stickermotion.publish.http import Response

TG_TOKEN = "123456789:AAH-TESTSECRETtelegramTOKEN_abcdefg"
DC_TOKEN = "MTIzNDU2Nzg5.TESTSECRET.discordTOKEN_abcdefghijk"
GUILD = "987654321012345678"


def parse_multipart(headers: dict[str, str], body: bytes | None) -> dict[str, Any]:
    kind = headers.get("Content-Type", "")
    match = re.search(r"boundary=(\S+)", kind)
    if not match or body is None:
        return {}
    boundary = match.group(1).encode()
    out: dict[str, Any] = {}
    for chunk in body.split(b"--" + boundary):
        chunk = chunk.strip(b"\r\n")
        if not chunk or chunk == b"--":
            continue
        head, _, value = chunk.partition(b"\r\n\r\n")
        name = re.search(rb'name="([^"]+)"', head)
        if not name:
            continue
        key = name.group(1).decode()
        if b"filename=" in head:
            out[key] = {
                "filename": re.search(rb'filename="([^"]*)"', head).group(1).decode(),  # type: ignore[union-attr]
                "data": value,
            }
        else:
            out[key] = value.decode()
    return out


def _ok(result: Any) -> Response:
    return Response(200, json.dumps({"ok": True, "result": result}).encode())


def _err(code: int, description: str, **extra: Any) -> Response:
    body = {"ok": False, "error_code": code, "description": description, **extra}
    return Response(code, json.dumps(body).encode())


@dataclass
class FakeTelegram:
    username: str = "ExampleBot"
    owner_id: int = 4242
    owner_started: bool = True
    token: str = TG_TOKEN
    sets: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    fail: dict[str, Response] = field(default_factory=dict)
    """Method name -> forced error response (consumed once)."""
    _counter: int = 0

    def methods(self) -> list[str]:
        return [m for m, _ in self.calls]

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> Response:
        path = urlparse(url).path
        match = re.fullmatch(r"/bot([^/]+)/(\w+)", path)
        assert match, url
        token, name = match.groups()
        fields = parse_multipart(headers, body)
        self.calls.append((name, fields))
        if token != self.token:
            return _err(401, "Unauthorized")
        if name in self.fail:
            return self.fail.pop(name)
        handler = getattr(self, f"m_{name}", None)
        assert handler, name
        return handler(fields)

    # Methods

    def m_getMe(self, _f: dict[str, Any]) -> Response:  # noqa: N802
        return _ok({"id": 1, "is_bot": True, "first_name": "Example", "username": self.username})

    def m_getStickerSet(self, f: dict[str, Any]) -> Response:  # noqa: N802
        found = self.sets.get(f["name"])
        if found is None:
            return _err(400, "Bad Request: STICKERSET_INVALID")
        return _ok(found)

    def _sticker(self, f: dict[str, Any], spec: dict[str, Any], kind: str) -> dict[str, Any]:
        attach = spec["sticker"].removeprefix("attach://")
        upload = f[attach]
        assert spec["format"] == "video" and upload["filename"].endswith(".webm")
        self._counter += 1
        unique = "u" + hashlib.sha1(upload["data"]).hexdigest()[:10]
        item = {
            "file_id": f"CAACAg{self._counter:04d}",
            "file_unique_id": unique,
            "type": kind,
            "width": 512,
            "height": 512,
            "is_animated": False,
            "is_video": True,
            "emoji": spec["emoji_list"][0],
        }
        if kind == "custom_emoji":
            item["custom_emoji_id"] = f"5{self._counter:018d}"
        return item

    def _owner_ok(self, f: dict[str, Any]) -> Response | None:
        if int(f["user_id"]) != self.owner_id:
            return _err(400, "Bad Request: USER_ID_INVALID")
        if not self.owner_started:
            return _err(400, "Bad Request: PEER_ID_INVALID")
        return None

    def m_createNewStickerSet(self, f: dict[str, Any]) -> Response:  # noqa: N802
        problem = self._owner_ok(f)
        if problem:
            return problem
        name = f["name"]
        if not name.lower().endswith(f"_by_{self.username.lower()}"):
            return _err(400, "Bad Request: invalid sticker set name is specified")
        if name in self.sets:
            return _err(400, "Bad Request: sticker set name is already occupied")
        kind = f.get("sticker_type", "regular")
        stickers = [self._sticker(f, s, kind) for s in json.loads(f["stickers"])]
        self.sets[name] = {
            "name": name,
            "title": f["title"],
            "sticker_type": kind,
            "stickers": stickers,
        }
        return _ok(True)

    def m_addStickerToSet(self, f: dict[str, Any]) -> Response:  # noqa: N802
        problem = self._owner_ok(f)
        if problem:
            return problem
        found = self.sets.get(f["name"])
        if found is None:
            return _err(400, "Bad Request: STICKERSET_INVALID")
        item = self._sticker(f, json.loads(f["sticker"]), found["sticker_type"])
        # "If exactly the same sticker had already been added to the set, then the set
        # isn't changed."
        if all(s["file_unique_id"] != item["file_unique_id"] for s in found["stickers"]):
            found["stickers"].append(item)
        return _ok(True)

    def m_deleteStickerFromSet(self, f: dict[str, Any]) -> Response:  # noqa: N802
        for found in self.sets.values():
            for item in found["stickers"]:
                if item["file_id"] == f["sticker"]:
                    found["stickers"].remove(item)
                    return _ok(True)
        return _err(400, "Bad Request: STICKER_INVALID")


@dataclass
class FakeDiscord:
    token: str = DC_TOKEN
    guild_id: str = GUILD
    premium_tier: int = 0
    emojis: list[dict[str, Any]] = field(default_factory=list)
    stickers: list[dict[str, Any]] = field(default_factory=list)
    accept_webp: bool = True
    calls: list[tuple[str, str, Any]] = field(default_factory=list)
    fail: dict[str, Response] = field(default_factory=dict)
    """"METHOD /path-suffix" -> forced error response (consumed once)."""
    _counter: int = 100

    def requests(self, method: str = "", suffix: str = "") -> list[tuple[str, str, Any]]:
        return [c for c in self.calls if c[0].startswith(method) and c[1].endswith(suffix)]

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> Response:
        path = urlparse(url).path.removeprefix("/api/v10")
        if headers.get("Content-Type", "").startswith("multipart"):
            payload: Any = parse_multipart(headers, body)
        else:
            payload = json.loads(body) if body else None
        self.calls.append((method, path, payload))
        if headers.get("Authorization") != f"Bot {self.token}":
            return self._error(401, 0, "401: Unauthorized")
        key = f"{method} {path.rsplit('/', 1)[-1]}"
        if key in self.fail:
            return self.fail.pop(key)
        prefix = f"/guilds/{self.guild_id}"
        if not path.startswith(prefix):
            return self._error(404, 10004, "Unknown Guild")
        rest = path.removeprefix(prefix)
        if method == "GET" and rest == "":
            return self._json(
                {
                    "id": self.guild_id,
                    "name": "My Server",
                    "premium_tier": self.premium_tier,
                    "emojis": self.emojis,
                    "stickers": self.stickers,
                }
            )
        if method == "GET" and rest == "/stickers":
            return self._json(self.stickers)
        if method == "POST" and rest == "/emojis":
            return self._create_emoji(payload)
        if method == "POST" and rest == "/stickers":
            return self._create_sticker(payload)
        match = re.fullmatch(r"/(emojis|stickers)/(\d+)", rest)
        if match:
            kind, item_id = match.groups()
            items = self.emojis if kind == "emojis" else self.stickers
            item = next((i for i in items if i["id"] == item_id), None)
            if item is None:
                return self._error(404, 10014 if kind == "emojis" else 10060, "Unknown")
            if method == "GET":
                return self._json(item)
            if method == "DELETE":
                items.remove(item)
                return Response(204, b"")
        raise AssertionError(f"unexpected {method} {path}")

    @staticmethod
    def _json(data: Any, status: int = 200) -> Response:
        return Response(status, json.dumps(data).encode())

    def _error(self, status: int, code: int, message: str, **extra: Any) -> Response:
        return self._json({"code": code, "message": message, **extra}, status)

    def _next_id(self) -> str:
        self._counter += 1
        return f"1200000000000000{self._counter}"

    def _create_emoji(self, payload: dict[str, Any]) -> Response:
        name = payload["name"]
        if not re.fullmatch(r"[A-Za-z0-9_]{2,32}", name):
            return self._error(400, 50035, "Invalid Form Body")
        image = payload["image"]
        kind = image.split(";", 1)[0].removeprefix("data:")
        if kind == "image/webp" and not self.accept_webp:
            return self._error(
                400,
                50035,
                "Invalid Form Body",
                errors={
                    "image": {
                        "_errors": [
                            {"code": "BINARY_TYPE_MAX_SIZE", "message": "Invalid image data"}
                        ]
                    }
                },
            )
        tier_slots = {0: 50, 1: 100, 2: 150, 3: 250}[self.premium_tier]
        if sum(1 for e in self.emojis if e.get("animated")) >= tier_slots:
            return self._error(400, 30008, "Maximum number of emojis reached (50)")
        item = {"id": self._next_id(), "name": name, "animated": True, "kind": kind}
        self.emojis.append(item)
        return self._json(item, 201)

    def _create_sticker(self, payload: dict[str, Any]) -> Response:
        name = payload["name"]
        if not 2 <= len(name) <= 30:
            return self._error(400, 50035, "Invalid Form Body")
        assert payload["file"]["filename"].endswith(".png")
        description = payload.get("description", "")
        assert description == "" or 2 <= len(description) <= 100
        item = {
            "id": self._next_id(),
            "name": name,
            "tags": payload["tags"],
            "description": description,
            "format_type": 2,
        }
        self.stickers.append(item)
        return self._json(item)


@dataclass
class Platforms:
    """Routes by host to the Telegram or Discord fake (the web app has one transport)."""

    telegram: FakeTelegram = field(default_factory=FakeTelegram)
    discord: FakeDiscord = field(default_factory=FakeDiscord)

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> Response:
        host = urlparse(url).hostname
        if host == "api.telegram.org":
            return self.telegram(method, url, headers, body)
        if host == "discord.com":
            return self.discord(method, url, headers, body)
        raise AssertionError(f"unexpected host {host}")
