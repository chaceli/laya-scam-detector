"""Unicode-script detection and language routing for Laya checkpoints."""
from __future__ import annotations

import unicodedata


_NON_LATIN_SCRIPT_PREFIXES = (
    "cjk unified ideograph",
    "hangul syllable",
    "devanagari",
    "arabic",
    "hebrew",
    "cyrillic",
    "greek",
    "tamil",
    "thai",
    "bengali",
    "gurmukhi",
    "telugu",
    "tibetan",
    "myanmar",
    "khmer",
    "lao",
    "georgian",
    "armenian",
)

_PREFIX_TO_TAG = {
    "cjk unified ideograph": "cjk_han",
    "hangul syllable": "hangul",
    "devanagari": "devanagari",
    "arabic": "arabic",
    "hebrew": "hebrew",
    "cyrillic": "cyrillic",
    "greek": "greek",
    "tamil": "tamil",
    "thai": "thai",
    "bengali": "bengali",
    "gurmukhi": "gurmukhi",
    "telugu": "telugu",
    "tibetan": "tibetan",
    "myanmar": "myanmar",
    "khmer": "khmer",
    "lao": "lao",
    "georgian": "georgian",
    "armenian": "armenian",
}

LATIN_SCRIPT = "latin"


def detect_script(text: str) -> str:
    """Detect dominant non-Latin script, falling back to 'latin'."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            block = unicodedata.name(ch, "")
        except ValueError:
            block = ""
        matched = False
        for prefix in _NON_LATIN_SCRIPT_PREFIXES:
            if block.lower().startswith(prefix):
                tag = _PREFIX_TO_TAG[prefix]
                counts[tag] = counts.get(tag, 0) + 1
                matched = True
                break
        if not matched:
            counts[LATIN_SCRIPT] = counts.get(LATIN_SCRIPT, 0) + 1
    if not counts:
        return LATIN_SCRIPT
    return max(counts.items(), key=lambda kv: kv[1])[0]


class ScriptRouter:
    """Route text to English or Multilingual checkpoint based on script."""

    def route(self, text: str) -> str:
        script = detect_script(text)
        if script == LATIN_SCRIPT:
            return "english"
        return "multilingual"

    def route_with_reason(self, text: str) -> tuple[str, str]:
        script = detect_script(text)
        if script == LATIN_SCRIPT:
            return "english", f"script={script}"
        return "multilingual", f"script={script} (non-Latin)"