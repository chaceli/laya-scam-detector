"""Tests for OnnxLayaClient: predict, tokenize, softmax."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.laya_onnx import OnnxLayaClient, _tokenize_question, _softmax_with_temperature


CHECKPOINT_DIR = Path("models/laya-onnx-en")
SCHEDULED = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.onnx").exists(),
    reason="ONNX checkpoint not downloaded",
)


@pytest.fixture(scope="module")
def client():
    if not (CHECKPOINT_DIR / "model.onnx").exists():
        return None
    return OnnxLayaClient(CHECKPOINT_DIR)


class TestSoftmax:
    def test_uniform_logit_returns_uniform_prob(self):
        logits = np.array([1.0, 1.0, 1.0])
        probs = _softmax_with_temperature(logits, temperature=1.0)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3], atol=1e-6)

    def test_high_temp_makes_distribution_softer(self):
        logits = np.array([3.0, 0.0, 0.0])
        probs = _softmax_with_temperature(logits, temperature=10.0)
        assert probs[0] < 0.5
        assert probs[1] > 0.2

    def test_low_temp_makes_distribution_sharper(self):
        logits = np.array([3.0, 0.0, 0.0])
        probs = _softmax_with_temperature(logits, temperature=0.1)
        assert probs[0] > 0.99

    def test_probabilities_sum_to_one(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        probs = _softmax_with_temperature(logits, temperature=1.0)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            _softmax_with_temperature(np.array([1.0]), temperature=0)


class TestTokenizeQuestion:
    @SCHEDULED
    def test_choice_question_returns_one_position_per_option(self, client):
        tokens, positions = _tokenize_question(
            tokenizer=client.tokenizer,
            question="Which department?",
            options={"billing": "invoices and refunds", "tech": "bugs"},
            head_max_len=192,
            cls_id=client.cls_id,
            sep_id=client.sep_id,
            mask_id=client.mask_id,
        )
        assert len(tokens) > 0
        assert len(positions) == 2
        assert all(0 <= p < len(tokens) for p in positions)
        # Each marker position must be a [MASK] token
        for p in positions:
            assert tokens[p] == client.mask_id

    @SCHEDULED
    def test_three_options_yields_three_positions(self, client):
        tokens, positions = _tokenize_question(
            tokenizer=client.tokenizer,
            question="Pick",
            options={"a": "one", "b": "two", "c": "three"},
            head_max_len=192,
            cls_id=client.cls_id,
            sep_id=client.sep_id,
            mask_id=client.mask_id,
        )
        assert len(positions) == 3


class TestPredict:
    @SCHEDULED
    def test_predict_noul_returns_probability(self, client):
        result = client.predict(
            state="This is a phishing attempt.",
            questions={"is_fraud": {"type": "noul", "instructions": "Is this a scam?"}},
        )
        assert "answers" in result
        assert "is_fraud" in result["answers"]
        assert "noul" in result["answers"]["is_fraud"]
        prob = result["answers"]["is_fraud"]["noul"]
        assert 0.0 <= prob <= 1.0

    @SCHEDULED
    def test_predict_choice_returns_distribution(self, client):
        result = client.predict(
            state="Hi, billing question about invoice 12345.",
            questions={
                "dept": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "payments", "tech": "bugs"},
                }
            },
        )
        ans = result["answers"]["dept"]
        assert "choice" in ans
        assert "probabilities" in ans
        assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-4

    @SCHEDULED
    def test_predict_score_returns_expected_value(self, client):
        result = client.predict(
            state="This is urgent!",
            questions={
                "urgency": {
                    "type": "score",
                    "instructions": "Rate urgency",
                    "criteria": ["low", "medium", "high"],
                }
            },
        )
        ans = result["answers"]["urgency"]
        assert "score" in ans
        assert 0.0 <= ans["score"] <= 2.0

    @SCHEDULED
    def test_predict_empty_state_raises(self, client):
        with pytest.raises(ValueError):
            client.predict("", {"q": {"type": "noul", "instructions": "X"}})

    @SCHEDULED
    def test_predict_unknown_qtype_raises(self, client):
        with pytest.raises(ValueError):
            client.predict(
                "test",
                {"q": {"type": "unknown", "instructions": "X"}},
            )

    @SCHEDULED
    def test_predict_latency_recorded(self, client):
        result = client.predict(
            "test",
            {"q": {"type": "noul", "instructions": "Is this a scam?"}},
        )
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0


class TestOnnxLayaClientInit:
    @SCHEDULED
    def test_special_tokens_cached(self, client):
        assert client.cls_id is not None
        assert client.sep_id is not None
        assert client.mask_id is not None

    @SCHEDULED
    def test_config_max_len_loaded(self, client):
        # rl_agent_config.json has max_len=512
        assert client.default_max_len == 512

    @SCHEDULED
    def test_config_head_max_len_loaded(self, client):
        # rl_agent_config.json has head_max_len=192
        assert client.default_head_max_len == 192