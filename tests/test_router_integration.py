"""Tests for top-level Router (combines English + Multilingual clients)."""
import json
import os

import pytest

from src.router import Router


@pytest.fixture(scope="module")
def router():
    return Router(
        english_dir="models/laya-onnx-en",
        multilingual_dir="models/laya-onnx-multilingual",
    )


@pytest.fixture(scope="module")
def schema():
    return json.loads(open("schemas/scam.json").read())


@pytest.mark.skipif(
    not os.path.exists("models/laya-onnx-en/model.onnx"),
    reason="model not downloaded",
)
class TestRouterIntegration:
    def test_router_loads_english(self, router):
        assert router.english is not None

    def test_router_multilingual_is_none_when_missing(self, router):
        # laya-multilingual-onnx does not exist, so fallback to English
        assert router.multilingual is None

    def test_router_predict_english_text(self, router, schema):
        result = router.predict(
            "Hello world, just checking my account.",
            schema,
        )
        assert result["routing"]["model"] == "english"
        assert "is_scam" in result["answers"]

    def test_router_predict_chinese_text_falls_back_to_english(self, router, schema):
        """When multilingual checkpoint is unavailable, non-Latin text
        still routes to English (documented limitation)."""
        result = router.predict(
            "您好世界",
            schema,
        )
        # Routes to 'english' because multilingual checkpoint doesn't exist
        assert result["routing"]["model"] == "english"
        assert "fallback" in result["routing"]["reason"] or "English" in result["routing"]["reason"]

    def test_explicit_model_override(self, router, schema):
        result = router.predict(
            "Hello world",
            schema,
            model="english",
        )
        assert result["routing"]["model"] == "english"
        assert "explicit" in result["routing"]["reason"]

    def test_explicit_multilingual_override_falls_back_when_missing(self, router, schema):
        result = router.predict(
            "Hello world",
            schema,
            model="multilingual",
        )
        # multilingual not loaded; falls back to english
        assert result["routing"]["model"] == "english"