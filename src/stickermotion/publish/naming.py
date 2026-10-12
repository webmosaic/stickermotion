"""Names and emoji for publication: Discord names, Telegram set names, emoji choice.

The emoji associated with a sticker are picked by a simple keyword rule from the action
description (free, no model call); the user can change them on the result page.
"""

from __future__ import annotations

import re
import unicodedata

# Discord: emoji names 2-32 characters, sticker names 2-30; we keep both to [A-Za-z0-9_].
DISCORD_EMOJI_NAME = (2, 32)
DISCORD_STICKER_NAME = (2, 30)
# Telegram set name: letters, digits, underscores, starts with a letter, no "__",
# ends with "_by_<bot>", 1-64 characters.
TELEGRAM_SET_NAME_MAX = 64
TELEGRAM_EMOJI_MAX = 20

DEFAULT_EMOJI = "🙂"

_UK_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "h",
    "ґ": "g",
    "д": "d",
    "е": "e",
    "є": "ie",
    "ж": "zh",
    "з": "z",
    "и": "y",
    "і": "i",
    "ї": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ь": "",
    "ю": "iu",
    "я": "ia",
    "ъ": "",
    "ы": "y",
    "э": "e",
    "ё": "io",
    "'": "",
    "’": "",
    "ʼ": "",
}


def transliterate(text: str) -> str:
    out = []
    for char in text:
        lower = char.lower()
        if lower in _UK_LATIN:
            latin = _UK_LATIN[lower]
            out.append(latin.capitalize() if char != lower else latin)
        else:
            out.append(char)
    decomposed = unicodedata.normalize("NFKD", "".join(out))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def discord_name(source: str, limits: tuple[int, int] = DISCORD_EMOJI_NAME) -> str:
    """Valid Discord name from any text (file name, title): latin, digits, ``_``."""
    low, high = limits
    name = re.sub(r"[^A-Za-z0-9_]+", "_", transliterate(source)).strip("_")
    name = re.sub(r"_+", "_", name)[:high].strip("_")
    if len(name) < low:
        name = (name + "_sticker").strip("_")[:high] if name else "sticker"
    return name


def discord_name_problem(name: str, limits: tuple[int, int] = DISCORD_EMOJI_NAME) -> str | None:
    """User-facing reason why ``name`` is not a valid Discord name, else ``None``."""
    low, high = limits
    if not re.fullmatch(r"[A-Za-z0-9_]+", name or ""):
        return "Ім'я може містити лише латинські літери, цифри і _."
    if not low <= len(name) <= high:
        return f"Ім'я має бути від {low} до {high} символів, зараз {len(name)}."
    return None


def unique_name(name: str, taken: set[str], high: int) -> str:
    """``name`` or ``name_2``, ``name_3``... that is not in ``taken`` (case-insensitive)."""
    lowered = {t.lower() for t in taken}
    if name.lower() not in lowered:
        return name
    for index in range(2, 1000):
        suffix = f"_{index}"
        candidate = name[: high - len(suffix)] + suffix
        if candidate.lower() not in lowered:
            return candidate
    raise ValueError("no free name")  # pragma: no cover


def telegram_set_name(slug: str, bot_username: str, suffix: str = "") -> str:
    """``<slug>[_<suffix>]_by_<bot>`` by the Bot API rules; raises ``ValueError``."""
    base = re.sub(r"[^A-Za-z0-9_]+", "_", transliterate(slug)).strip("_")
    base = re.sub(r"_+", "_", base)
    if suffix:
        base = f"{base}_{suffix}"
    if not base or not base[0].isalpha():
        base = "s" + base
    tail = f"_by_{bot_username}"
    name = base[: TELEGRAM_SET_NAME_MAX - len(tail)].rstrip("_") + tail
    problem = telegram_set_name_problem(name, bot_username)
    if problem:
        raise ValueError(problem)
    return name


def telegram_set_name_problem(name: str, bot_username: str) -> str | None:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        return "Ім'я пака: лише латинські літери, цифри і _, починається з літери."
    if "__" in name:
        return "Ім'я пака не може містити два _ поспіль."
    if not name.lower().endswith(f"_by_{bot_username.lower()}"):
        return f"Ім'я пака має закінчуватися на _by_{bot_username}."
    if len(name) > TELEGRAM_SET_NAME_MAX:
        return f"Ім'я пака довше за {TELEGRAM_SET_NAME_MAX} символів."
    return None


# Emoji

_JOINERS = {"‍", "️", "︎", "⃣"}


def _is_modifier(char: str) -> bool:
    return char in _JOINERS or 0x1F3FB <= ord(char) <= 0x1F3FF or 0xE0020 <= ord(char) <= 0xE007F


def split_emoji(text: str) -> tuple[str, ...]:
    """Emoji found in free text, in order, without duplicates (ZWJ sequences kept)."""
    found: list[str] = []
    current = ""
    joined = False
    for char in text:
        if current and (_is_modifier(char) or joined):
            current += char
            joined = char == "‍"
            continue
        if current:
            found.append(current)
            current = ""
        category = unicodedata.category(char)
        if category == "So" or (0x1F1E6 <= ord(char) <= 0x1F1FF):
            current = char
            joined = False
    if current:
        found.append(current)
    # Regional indicator pairs form one flag.
    merged: list[str] = []
    for item in found:
        if (
            merged
            and len(merged[-1]) == 1
            and len(item) == 1
            and all(0x1F1E6 <= ord(c) <= 0x1F1FF for c in merged[-1] + item)
        ):
            merged[-1] += item
        else:
            merged.append(item)
    unique = list(dict.fromkeys(merged))
    return tuple(unique[:TELEGRAM_EMOJI_MAX])


_RULES: tuple[tuple[str, str], ...] = (
    (r"\b(yawn|sleep|sleepy|asleep|nap|snore)", "😴"),
    (r"\b(wave|waves|waving|greet|hello|hi\b)", "👋"),
    (r"\b(laugh|laughs|giggle|lol)", "😂"),
    (r"\b(cry|cries|crying|tear|sob)", "😭"),
    (r"\b(heart|love|kiss|blow)", "❤️"),
    (r"\b(dance|dances|dancing)", "💃"),
    (r"\b(angry|anger|rage|stomp|frown)", "😠"),
    (r"\b(clap|claps|applaud)", "👏"),
    (r"\b(thumb|thumbs|nod|nods|approve)", "👍"),
    (r"\b(think|thinks|ponder|chin)", "🤔"),
    (r"\b(shrug)", "🤷"),
    (r"\b(wink|winks)", "😉"),
    (r"\b(surprise|surprised|gasp|shock)", "😮"),
    (r"\b(eat|eats|chew|bite|snack)", "😋"),
    (r"\b(drink|sip|coffee|tea|cup)", "☕"),
    (r"\b(sing|sings|singing|music)", "🎤"),
    (r"\b(jump|jumps|hop|hops|bounce|celebrat|cheer)", "🎉"),
    (r"\b(smile|smiles|grin|happy)", "😊"),
    (r"\b(blush|shy)", "😊"),
    (r"\b(sad|sigh)", "😔"),
)


def suggest_emoji(*texts: str, limit: int = 2) -> tuple[str, ...]:
    """Emoji for a sticker from its action description (English or Ukrainian words)."""
    text = " ".join(t for t in texts if t).lower()
    picked: list[str] = []
    for pattern, emoji in _RULES:
        if emoji not in picked and re.search(pattern, text):
            picked.append(emoji)
        if len(picked) >= limit:
            break
    return tuple(picked) or (DEFAULT_EMOJI,)
