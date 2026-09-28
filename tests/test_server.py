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