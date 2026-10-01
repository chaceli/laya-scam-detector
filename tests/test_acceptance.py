"""check_acceptance v2 解析函数测试（合成报告，无模型）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from check_acceptance import parse_category_recall, parse_overall_fpr  # noqa: E402

FPR_MD = """## Hard-Negative FPR by Genre

| Source | n | FP | FPR |
|---|---|---|---|
| hn_financial_notice | 75 | 1 | 0.013 |
| **overall** | 600 | 9 | 0.015 |
"""

RECALL_MD = """## Per-Category Recall

| Category | n | Correct | Recall |
|---|---|---|---|
| rebate_scam | 120 | 108 | 0.900 |
| benign | 80 | 72 | 0.900 |
"""


def test_parse_overall_fpr():
    assert parse_overall_fpr(FPR_MD) == 0.015


def test_parse_overall_fpr_missing():
    assert parse_overall_fpr("no table here") is None


def test_parse_category_recall():
    assert parse_category_recall(RECALL_MD, "rebate_scam") == 0.900


def test_parse_category_recall_missing():
    assert parse_category_recall(RECALL_MD, "crypto_scam") is None