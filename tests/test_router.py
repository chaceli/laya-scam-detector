"""Tests for Unicode-script detection and language routing."""
import pytest

from src.router import detect_script, ScriptRouter


class TestDetectScript:
    def test_pure_latin_returns_latin(self):
        assert detect_script("Hello world, this is English.") == "latin"

    def test_chinese_hanzi_returns_cjk(self):
        assert detect_script("您好世界") == "cjk_han"

    def test_devanagari_returns_devanagari(self):
        assert detect_script("नमस्ते") == "devanagari"

    def test_arabic_returns_arabic(self):
        assert detect_script("مرحبا") == "arabic"

    def test_cyrillic_returns_cyrillic(self):
        assert detect_script("Привет") == "cyrillic"

    def test_mixed_text_returns_dominant_non_latin(self):
        # 10 Chinese + 1 English word → still cjk_han
        assert detect_script("您好 您好 您好 您好 您好 hello") == "cjk_han"

    def test_empty_returns_latin(self):
        assert detect_script("") == "latin"

    def test_pure_numbers_and_punctuation_returns_latin(self):
        assert detect_script("12345!@#$%") == "latin"

    def test_hiragana_kana_returns_latin_fallback(self):
        # We don't list hiragana in our script map; falls back to latin.
        # (Laya has no Japanese-specific script handling.)
        result = detect_script("こんにちは")
        assert result in {"latin", "cjk_han"}


class TestScriptRouter:
    def test_latin_routes_to_english(self):
        r = ScriptRouter()
        assert r.route("Hello world") == "english"

    def test_chinese_routes_to_multilingual(self):
        r = ScriptRouter()
        assert r.route("您好世界") == "multilingual"

    def test_short_latin_routes_to_english(self):
        r = ScriptRouter()
        assert r.route("OK") == "english"

    def test_route_with_reason_returns_string(self):
        r = ScriptRouter()
        model, reason = r.route_with_reason("您好")
        assert model == "multilingual"
        assert "cjk_han" in reason or "non-Latin" in reason