"""Unicode-script detection and language routing for Laya checkpoints.

Supports dual-checkpoint deployment:
- english (ModernBERT-large, 421M) — Latin script
- multilingual (mmBERT-base, 322M) — CJK/Hebrew/Arabic/Devanagari etc.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import Optional

from src.laya_onnx import OnnxLayaClient


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


class Router:
    """High-level router combining English + Multilingual clients.

    Routing policy:
    - Latin script (English, Spanish, French, etc.) → English checkpoint
    - Non-Latin script (CJK, Hebrew, Arabic, etc.) → Multilingual checkpoint
    - If multilingual checkpoint is missing, falls back to English (with warning)

    Per the research:
    - Laya English checkpoint: zero-shot 0.342 typed-decisions; ModernBERT
      backbone shreds CJK characters (Khmer 0.000 acc @ 0.952 conf).
    - Laya multilingual checkpoint (mmBERT-base, 256k vocab): native
      multilingual support, ~2x faster than English checkpoint.
    """

    def __init__(
        self,
        english_dir: Path | str | None = None,
        multilingual_dir: Path | str | None = None,
        providers: Optional[list[str]] = None,
    ):
        if not english_dir and not multilingual_dir:
            raise ValueError("at least one of english_dir / multilingual_dir is required")
        self.english = None
        if english_dir:
            self.english = OnnxLayaClient(english_dir, providers=providers)
        self.multilingual = None
        if multilingual_dir and Path(multilingual_dir).exists():
            try:
                self.multilingual = OnnxLayaClient(multilingual_dir, providers=providers)
            except Exception as e:
                print(f"⚠ Multilingual checkpoint failed to load: {e}", file=sys.stderr)
        self.scripts = ScriptRouter()

    def _pick(self, text: str) -> tuple[OnnxLayaClient, str, str]:
        model_tag = self.scripts.route(text)
        reason = self.scripts.route_with_reason(text)[1]
        if model_tag == "multilingual" and self.multilingual is not None:
            return self.multilingual, "multilingual", reason
        if model_tag == "english" and self.english is not None:
            return self.english, "english", reason
        if self.multilingual is not None:
            if model_tag == "english":
                reason += " — English checkpoint unavailable, using multilingual"
            return self.multilingual, "multilingual", reason
        reason += " — multilingual checkpoint unavailable, using English (known limitation)"
        return self.english, "english", reason

    def predict(
        self,
        state: str,
        questions: dict,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        model: Optional[str] = None,
    ) -> dict:
        if isinstance(state, dict):
            import json
            state = json.dumps(state, ensure_ascii=False)
        if model is None:
            client, model_tag, reason = self._pick(state)
        else:
            reason = "explicit override"
            if model == "multilingual" and self.multilingual is not None:
                client, model_tag = self.multilingual, "multilingual"
            elif model == "english" and self.english is not None:
                client, model_tag = self.english, "english"
            elif self.multilingual is not None:
                client, model_tag = self.multilingual, "multilingual"
                reason += " — requested checkpoint unavailable, using multilingual"
            elif self.english is not None:
                client, model_tag = self.english, "english"
                reason += " — requested checkpoint unavailable, using English"
            else:
                raise RuntimeError("no checkpoint available to serve this request")
        result = client.predict(state, questions, max_len=max_len, head_max_len=head_max_len)
        result["routing"] = {"model": model_tag, "reason": reason}
        return result

    def route(self, text: str) -> str:
        return self.scripts.route(text)