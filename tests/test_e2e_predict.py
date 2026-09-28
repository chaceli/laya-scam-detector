"""End-to-end inference test using the real ONNX model and full scam schema."""
import json
import os

import pytest

from src.laya_onnx import OnnxLayaClient


CHECKPOINT_DIR = "models/laya-onnx-en"


@pytest.fixture(scope="module")
def client():
    return OnnxLayaClient(CHECKPOINT_DIR)


@pytest.fixture(scope="module")
def schema():
    return json.loads(open("schemas/scam.json").read())


@pytest.mark.skipif(
    not os.path.exists("models/laya-onnx-en/model.onnx"),
    reason="model not downloaded",
)
class TestEndToEnd:
    def test_english_phishing_email_gets_high_risk(self, client, schema):
        result = client.predict(
            "URGENT: Your PayPal account has been limited. Click http://paypa1-secure.tk/login to verify.",
            schema,
        )
        is_scam = result["answers"]["is_scam"]["noul"]
        assert is_scam > 0.5, f"phishing email scored too low: {is_scam}"
        assert result["answers"]["scam_category"]["choice"] != "benign"

    def test_english_benign_email_gets_low_risk(self, client, schema):
        result = client.predict(
            "Hi, just confirming our meeting at 3pm tomorrow. Looking forward to it.",
            schema,
        )
        is_scam = result["answers"]["is_scam"]["noul"]
        assert is_scam < 0.5, f"benign email scored too high: {is_scam}"

    def test_latency_under_3_seconds_on_cpu(self, client, schema):
        # All 3 primitives in one call should still finish in <3s on M4 Pro CPU
        result = client.predict("Test message", schema)
        assert result["latency_ms"] < 3000, f"too slow: {result['latency_ms']}ms"

    def test_returns_all_three_primitives(self, client, schema):
        result = client.predict("Some random message", schema)
        assert "is_scam" in result["answers"]
        assert "risk_level" in result["answers"]
        assert "scam_category" in result["answers"]
        # Each primitive has its core field
        assert "noul" in result["answers"]["is_scam"]
        assert "score" in result["answers"]["risk_level"]
        assert "choice" in result["answers"]["scam_category"]

    def test_repeated_calls_are_consistent(self, client, schema):
        """Same input → same output (no autoregressive sampling)."""
        text = "Verify your bank account at https://example-secure.tk"
        r1 = client.predict(text, schema)
        r2 = client.predict(text, schema)
        for q in ["is_scam", "risk_level", "scam_category"]:
            assert r1["answers"][q] == r2["answers"][q], (
                f"inconsistent for {q}: {r1['answers'][q]} != {r2['answers'][q]}"
            )