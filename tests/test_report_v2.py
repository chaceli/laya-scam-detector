"""report.py v2 扩展：分类别召回 / 分体裁 FPR / 阈值曲线。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.report import (
    _hardneg_fpr, _per_category_recall, _threshold_table, render_report,
)


def _res(noul, exp_risk, cat_pred=None, cat_exp=None, source=None):
    return {
        "id": "x", "type": "single", "language": "zh",
        "expected_risk": exp_risk, "expected_category": cat_exp,
        "source": source, "state_preview": "",
        "predicted": {
            "is_scam_noul": noul, "is_scam_label": int(noul >= 0.5),
            "risk_level": 1, "scam_category": cat_pred,
            "category_probs": None,
        },
        "routing": {}, "latency_ms": 10.0, "error": None,
    }


class TestPerCategoryRecall:
    def test_math(self):
        rs = [
            _res(0.9, 4, "rebate_scam", "rebate_scam", "ccl2023"),
            _res(0.9, 4, "phishing", "rebate_scam", "ccl2023"),
            _res(0.1, 1, "benign", "benign", "fgrc_scd_sms"),
        ]
        out = dict((c, (n, k, r)) for c, n, k, r in _per_category_recall(rs))
        assert out["rebate_scam"] == (2, 1, 0.5)
        assert out["benign"][2] == 1.0


class TestHardnegFpr:
    def test_overall_and_genres(self):
        rs = [
            _res(0.6, 1, source="hn_financial_notice"),
            _res(0.3, 1, source="hn_financial_notice"),
            _res(0.9, 1, source="hn_promotion"),
        ]
        genres, overall = _hardneg_fpr(rs)
        assert overall == (3, 2, 2 / 3)
        assert dict((s, fpr) for s, _, _, fpr in genres) == {
            "hn_financial_notice": 0.5, "hn_promotion": 1.0}

    def test_none_when_no_hardneg(self):
        assert _hardneg_fpr([_res(0.9, 4, source="ccl2023")]) is None


class TestThresholdTable:
    def test_has_mid_threshold_row(self):
        lines = _threshold_table([
            _res(0.9, 5), _res(0.2, 1), _res(0.8, 4), _res(0.4, 2),
        ])
        assert any(l.startswith("| 0.50 |") for l in lines)


class TestRenderReport:
    def test_sections_written(self, tmp_path):
        rs = [
            _res(0.6, 1, "benign", "benign", "hn_financial_notice"),
            _res(0.9, 4, "rebate_scam", "rebate_scam", "ccl2023"),
        ]
        md_path, _ = render_report(rs, {}, tmp_path)
        text = md_path.read_text()
        assert "## Per-Category Recall" in text
        assert "rebate_scam" in text
        assert "## Hard-Negative FPR by Genre" in text
        assert "| **overall** |" in text
        assert "## Threshold-Recall Curve" in text