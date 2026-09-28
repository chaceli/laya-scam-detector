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


def _multilingual_dir() -> str:
    """Prefer the fine-tuned bundle; fall back to the base multilingual dir."""
    finetuned = "models/laya-onnx-multilingual-finetuned"
    if os.path.exists(f"{finetuned}/model.onnx"):
        return finetuned
    return "models/laya-onnx-multilingual"


@pytest.fixture(scope="module")
def router_with_ml():
    """Router with both English and multilingual checkpoints loaded (when available)."""
    return Router(
        english_dir="models/laya-onnx-en",
        multilingual_dir=_multilingual_dir(),
    )


@pytest.mark.skipif(
    not os.path.exists(f"{_multilingual_dir()}/model.onnx"),
    reason="Multilingual ONNX checkpoint required",
)
class TestRouterMultilingualOnly:
    """Deployment mode used by the Hugging Face Space: only the fine-tuned
    multilingual checkpoint is shipped, so English must be optional."""

    def test_router_constructs_without_english(self):
        r = Router(english_dir=None, multilingual_dir=_multilingual_dir())
        assert r.english is None
        assert r.multilingual is not None

    def test_requires_at_least_one_checkpoint(self):
        with pytest.raises(ValueError):
            Router(english_dir=None, multilingual_dir=None)

    def test_latin_text_served_by_multilingual(self, schema):
        r = Router(english_dir=None, multilingual_dir=_multilingual_dir())
        out = r.predict("Hello world", schema)
        assert out["routing"]["model"] == "multilingual"
        assert "English checkpoint unavailable" in out["routing"]["reason"]

    def test_chinese_text_served_by_multilingual(self, schema):
        r = Router(english_dir=None, multilingual_dir=_multilingual_dir())
        out = r.predict("您好世界", schema)
        assert out["routing"]["model"] == "multilingual"

    def test_explicit_english_request_falls_back(self, schema):
        r = Router(english_dir=None, multilingual_dir=_multilingual_dir())
        out = r.predict("Hello", schema, model="english")
        assert out["routing"]["model"] == "multilingual"
        assert "requested checkpoint unavailable" in out["routing"]["reason"]


@pytest.mark.skipif(
    not (os.path.exists("models/laya-onnx-en/model.onnx")
         and os.path.exists(f"{_multilingual_dir()}/model.onnx")),
    reason="Both English and multilingual ONNX checkpoints required",
)
class TestRouterWithMultilingual:
    def test_both_clients_loaded(self, router_with_ml):
        assert router_with_ml.english is not None
        assert router_with_ml.multilingual is not None

    def test_chinese_routes_to_multilingual(self, router_with_ml, schema):
        result = router_with_ml.predict("您好世界", schema)
        assert result["routing"]["model"] == "multilingual"
        assert "non-Latin" in result["routing"]["reason"]

    def test_hebrew_routes_to_multilingual(self, router_with_ml, schema):
        result = router_with_ml.predict("שלום", schema)
        assert result["routing"]["model"] == "multilingual"

    def test_english_routes_to_english(self, router_with_ml, schema):
        result = router_with_ml.predict("Hello world", schema)
        assert result["routing"]["model"] == "english"

    def test_explicit_multilingual_overrides_script(self, router_with_ml, schema):
        result = router_with_ml.predict("Hello", schema, model="multilingual")
        assert result["routing"]["model"] == "multilingual"