"""Tests for report module: summary stats, category matrix, rendering."""
import json
from pathlib import Path

import pytest

from src.report import _summary, _category_table, render_report


@pytest.fixture
def fake_results():
    return [
        {
            "id": "a", "language": "zh", "expected_risk": 5,
            "expected_category": "phishing", "state_preview": "phishing msg",
            "routing": {"model": "english"},
            "latency_ms": 50,
            "predicted": {"is_scam_noul": 0.9, "risk_level": 4.0,
                          "scam_category": "phishing", "category_probs": {}},
        },
        {
            "id": "b", "language": "zh", "expected_risk": 1,
            "expected_category": "benign", "state_preview": "hi msg",
            "routing": {"model": "english"},
            "latency_ms": 60,
            "predicted": {"is_scam_noul": 0.1, "risk_level": 1.0,
                          "scam_category": "benign", "category_probs": {}},
        },
        {
            "id": "c", "language": "en", "expected_risk": 4,
            "expected_category": "investment_scam", "state_preview": "crypto",
            "routing": {"model": "english"},
            "latency_ms": 70,
            "predicted": {"is_scam_noul": 0.7, "risk_level": 3.5,
                          "scam_category": "investment_scam", "category_probs": {}},
        },
    ]


class TestSummary:
    def test_summary_returns_dict(self, fake_results):
        s = _summary(fake_results)
        assert "is_scam_accuracy" in s
        assert "p50_latency_ms" in s

    def test_perfect_classification_high_risk(self, fake_results):
        s = _summary(fake_results)
        # All 3 correctly classified at threshold 0.5
        # a: 5 -> scam (1) pred 0.9 -> scam (1) TP
        # b: 1 -> benign (0) pred 0.1 -> benign (0) TN
        # c: 4 -> scam (1) pred 0.7 -> scam (1) TP
        assert s["is_scam_accuracy"] == 1.0
        assert s["is_scm_tp"] == 2
        assert s["is_scm_tn"] == 1

    def test_risk_level_mae(self, fake_results):
        s = _summary(fake_results)
        # a: 4.0 - 5 = -1, b: 1.0 - 1 = 0, c: 3.5 - 4 = -0.5
        # MAE = (1 + 0 + 0.5) / 3 = 0.5
        assert abs(s["risk_level_mae"] - 0.5) < 1e-9

    def test_category_accuracy(self, fake_results):
        s = _summary(fake_results)
        assert s["scam_category_accuracy"] == 1.0

    def test_by_language_breakdown(self, fake_results):
        s = _summary(fake_results)
        assert "by_language" in s
        assert "zh" in s["by_language"]
        assert "en" in s["by_language"]
        assert s["by_language"]["zh"]["n"] == 2
        assert s["by_language"]["en"]["n"] == 1

    def test_empty_results_returns_empty_dict(self):
        assert _summary([]) == {}


class TestCategoryTable:
    def test_returns_markdown_table(self, fake_results):
        md = _category_table(fake_results)
        assert "| expected" in md
        assert "phishing" in md
        assert "benign" in md

    def test_empty_returns_placeholder(self):
        md = _category_table([])
        assert "No category data" in md


class TestRenderReport:
    def test_render_creates_files(self, fake_results, tmp_path):
        schema = json.loads(Path("schemas/scam.json").read_text())
        md_path, jsonl_path = render_report(fake_results, schema, tmp_path)
        assert md_path.exists()
        assert jsonl_path.exists()
        content = md_path.read_text()
        assert "Laya Scam-Phrase Evaluation Report" in content
        assert "Headline Metrics" in content
        assert "Per-Language" in content
        assert "Failure analysis" in content

    def test_render_includes_per_sample_table(self, fake_results, tmp_path):
        schema = json.loads(Path("schemas/scam.json").read_text())
        md_path, _ = render_report(fake_results, schema, tmp_path)
        content = md_path.read_text()
        # Each sample ID should appear in the All samples table
        for sample in fake_results:
            assert sample["id"] in content