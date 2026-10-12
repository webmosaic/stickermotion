"""Action text for the web UI: the automatic description in Ukrainian and the way back.

The vision model (``describe.describe``) writes the action in English. The user reads and
edits it in Ukrainian, so the same cheap LLM (``openai/gpt-5-nano``) translates it to
Ukrainian, and the user's text back to English before generation. Text without Cyrillic
is taken as English and sent as is; an unchanged automatic text reuses its English
original, so neither costs a call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image

from stickermotion.engine.ai.describe import DESCRIBE_MODEL, DESCRIBE_PRICE, describe
from stickermotion.engine.ai.framing import frame_for_model
from stickermotion.engine.ai.provider import ReplicateClient, output_text

TRANSLATE_PRICE = DESCRIBE_PRICE
LANGUAGES = {"uk": "Ukrainian", "en": "English"}
MAX_TEXT = 600

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def has_cyrillic(text: str) -> bool:
    return bool(_CYRILLIC.search(text))


def clean(text: str) -> str:
    """Single-spaced text without wrapping quotes, cut to ``MAX_TEXT`` characters."""
    text = " ".join(text.split()).strip().strip('"').strip("«»").strip()
    return text[:MAX_TEXT]


def translate_input(text: str, target: str) -> dict[str, object]:
    language = LANGUAGES[target]
    return {
        "system_prompt": (
            "You translate short descriptions of a character's motion for an animation "
            "tool. Reply with the translation only, no quotes, no comments."
        ),
        "prompt": (
            f"Translate into {language}. Keep the meaning, body parts and directions exact; "
            f"plain present tense.\n\n{text}"
        ),
        "reasoning_effort": "minimal",
        "verbosity": "low",
        "max_completion_tokens": 1000,
    }


def translate(client: ReplicateClient, text: str, target: str) -> str:
    """One paid call to the cheap LLM (about 0.0001$)."""
    prediction = client.predict(
        DESCRIBE_MODEL, translate_input(text, target), TRANSLATE_PRICE, f"translate_{target}"
    )
    result = clean(output_text(prediction.get("output")))
    if not result:
        raise ValueError("the translation came back empty")
    return result


@dataclass(frozen=True)
class AutoText:
    english: str
    ukrainian: str


def auto_description(client: ReplicateClient, image: Image.Image) -> AutoText:
    """Vision model description (action plus decorations) and its Ukrainian translation.

    The green model frame is tried first (as in the CLI). Replicate's moderation sometimes
    flags it (E005; such a failure is not billed), so the
    character on a neutral grey background is tried once more before giving up.
    """
    frame, _ = frame_for_model(image)
    found = describe(client, frame, image)
    english = clean(" ".join(p for p in (found.action, found.decorations) if p.strip()))
    return AutoText(english, translate(client, english, "uk"))


def to_english(client: ReplicateClient, text: str, auto_uk: str, auto_en: str) -> str:
    """English action for the video model from what the user left in the text field."""
    text = clean(text)
    if not text:
        raise ValueError("empty description")
    if auto_en and text == clean(auto_uk):
        return auto_en
    if not has_cyrillic(text):
        return text
    return translate(client, text, "en")
