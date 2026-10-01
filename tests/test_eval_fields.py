"""_expected_category 口径统一测试（design §4.2）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval import _expected_category


def test_category_field_wins():
    assert _expected_category(
        {"category": "benign", "expected_category": "spam"}) == "benign"


def test_expected_category_fallback():
    assert _expected_category({"expected_category": "spam"}) == "spam_general"


def test_none_when_absent():
    assert _expected_category({"state": "x"}) is None


def test_normalize_applied():
    assert _expected_category({"category": "ham"}) == "benign"