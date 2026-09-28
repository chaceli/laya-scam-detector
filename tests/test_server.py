"""API tests for the Laya web playground."""
import pytest
from fastapi.testclient import TestClient

from server.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_ok_with_models(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["models"]["english"] is True
        assert body["models"]["multilingual"] is True
        assert "multilingual_dir" in body


class TestSamples:
    def test_samples_returns_presets(self, client):
        r = client.get("/api/samples")
        assert r.status_code == 200
        samples = r.json()["samples"]
        assert len(samples) >= 4
        langs = {s["lang"] for s in samples}
        assert {"zh", "en"}.issubset(langs)
        for s in samples:
            assert s["id"] and s["label"] and s["text"]


class TestPredict:
    def test_all_primitives_returns_three_answers(self, client):
        r = client.post("/api/predict", json={
            "text": "您好，我是XX快递客服，您有一个包裹丢失需要理赔。",
            "primitives": ["noul", "score", "choice"],
            "model": "auto",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body["answers"].keys()) == {"is_scam", "risk_level", "scam_category"}
        assert "routing" in body and "latency_ms" in body
        assert 0.0 <= body["answers"]["is_scam"]["noul"] <= 1.0

    def test_subset_primitives_only_returns_selected(self, client):
        r = client.post("/api/predict", json={
            "text": "Hello world",
            "primitives": ["noul"],
        })
        assert r.status_code == 200, r.text
        assert set(r.json()["answers"].keys()) == {"is_scam"}

    def test_auto_routes_chinese_to_multilingual(self, client):
        r = client.post("/api/predict", json={
            "text": "您好世界",
            "primitives": ["noul"],
            "model": "auto",
        })
        assert r.json()["routing"]["model"] == "multilingual"

    def test_force_english_overrides_routing(self, client):
        r = client.post("/api/predict", json={
            "text": "您好世界",
            "primitives": ["noul"],
            "model": "english",
        })
        assert r.json()["routing"]["model"] == "english"

    def test_custom_questions_override_schema(self, client):
        r = client.post("/api/predict", json={
            "text": "This is definitely a scam message about money.",
            "primitives": ["noul"],
            "questions": {
                "is_scam": {"type": "noul", "instructions": "Is this malicious?"}
            },
        })
        assert r.status_code == 200, r.text
        assert "is_scam" in r.json()["answers"]