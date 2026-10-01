"""compose_train ratio invariants (synthetic rows, no real data)."""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from dataset_mix import compose_train, HARDNEG_SOURCES, stable_id  # noqa: E402


def mk(text, is_scam, source, category=None):
    return {
        "id": f"{source}-{abs(hash(text)) % 10**8}",
        "text": text, "is_scam": is_scam,
        "risk": 4 if is_scam else 1,
        "category": category or ("benign" if not is_scam else "spam_general"),
        "language": "zh", "source": source,
    }


def test_caps_and_rebate_floor():
    rows = []
    rows += [mk(f"ccl-r{i}", 1, "ccl2023", "rebate_scam") for i in range(4000)]
    rows += [mk(f"ccl-nr{i}", 1, "ccl2023", "phishing") for i in range(500)]
    rows += [mk(f"inv{i}", 1, "fgrc_scd_sms", "investment_scam") for i in range(4000)]
    rows += [mk(f"ph{i}", 1, "fbs_sms", "phishing") for i in range(200)]
    rows += [mk(f"tele{i}", 1, "teleantifraud", "spam_general") for i in range(3000)]
    rows += [mk(f"hn{i}", 0, "hn_financial_notice") for i in range(600)]
    rows += [mk(f"nb{i}", 0, "nb_natural_benign") for i in range(200)]
    rows += [mk(f"tip{i}", 0, "fgrc_scd_sms") for i in range(3000)]
    rows += [mk(f"nat{i}", 0, "ealvaradob") for i in range(3000)]

    train, stats = compose_train(rows, eval_ids=set(), train_target=2000, seed=42)

    pos = [r for r in train if r["is_scam"] == 1]
    neg = [r for r in train if r["is_scam"] == 0]
    assert len(train) == 2000
    assert stats["n_pos"] == 900 and stats["n_neg"] == 1100

    # rebate_scam ≥25% 正样本
    assert sum(1 for r in pos if r["category"] == "rebate_scam") >= 0.25 * 900 - 1
    # CCL 非刷单返利 ≤20% 正样本（rebate 豁免——spec 勘误）
    ccl_nr = [r for r in pos if r["source"] == "ccl2023"
              and r["category"] != "rebate_scam"]
    assert len(ccl_nr) <= 0.20 * 900 + 1
    # Tele ≤10% 正样本
    assert sum(1 for r in pos if r["source"] == "teleantifraud") <= 0.10 * 900 + 1
    # 单类别 ≤30% 正样本
    for c, k in Counter(r["category"] for r in pos).items():
        assert k <= 0.30 * 900 + 1, f"{c} over cap"
    # FGRC 提示类 10–15% 负样本
    tips = [r for r in neg if r["source"] in ("fgrc_scd_sms", "fgrc_scd_dialog")]
    assert 0.10 * 1100 - 1 <= len(tips) <= 0.15 * 1100 + 1
    # 难负全保留（不超过负样本预算时）
    assert sum(1 for r in neg if r["source"] in HARDNEG_SOURCES) == 600


def test_rebate_floor_error_when_insufficient():
    rows = [mk(f"p{i}", 1, "fbs_sms", "phishing") for i in range(100)]
    rows += [mk(f"n{i}", 0, "ealvaradob") for i in range(100)]
    with pytest.raises(RuntimeError, match="rebate_scam"):
        compose_train(rows, eval_ids=set(), train_target=1000, seed=42)


def test_eval_ids_excluded():
    rows = [mk(f"hn{i}", 0, "hn_financial_notice") for i in range(50)]
    rows += [mk(f"p{i}", 1, "ccl2023", "rebate_scam") for i in range(100)]
    leak = rows[0]["id"]
    train, _ = compose_train(rows, eval_ids={leak}, train_target=100, seed=42)
    assert all(r["id"] != leak for r in train)


def test_generated_cap_enforced():
    rows = [mk(f"hn{i}", 0, "hn_financial_notice") for i in range(2000)]
    rows += [mk(f"p{i}", 1, "ccl2023", "rebate_scam") for i in range(1000)]
    with pytest.raises(RuntimeError, match="生成样本"):
        compose_train(rows, eval_ids=set(), train_target=2000, seed=42)


def test_stable_id_matches_build_dataset():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bd", Path(__file__).resolve().parent.parent / "scripts" / "build_dataset.py")
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    assert stable_id("样本", "hn_x") == bd.stable_id("样本", "hn_x")
