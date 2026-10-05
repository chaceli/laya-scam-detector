"""CCL 同源留出切分的不变量测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_ccl_holdout import build_holdout  # noqa: E402


def _mk(case_id, label):
    return {"案件编号": case_id, "案情描述": f"案件{case_id}的描述文本", "案件类别": label}


def test_counts_and_disjoint():
    rows = [_mk(i, "刷单返利类") for i in range(3000)]
    rows += [_mk(10000 + i, "虚假网络投资理财类") for i in range(200)]
    hold, rest = build_holdout(rows, seed=42, n_rebate=1000, n_other=50)
    reb = [r for r in hold if r["category"] == "rebate_scam"]
    inv = [r for r in hold if r["category"] == "investment_scam"]
    assert len(reb) == 1000
    assert len(inv) == 50
    # 留出与剩余按 案件编号 不交叉（hold 行 id = ccl_holdout_<案件编号>；rest 行保留 案件编号）
    h_cases = {r["id"].removeprefix("ccl_holdout_") for r in hold}
    r_cases = {str(r["案件编号"]) for r in rest}
    assert not (h_cases & r_cases)


def test_deterministic_by_seed():
    rows = [_mk(i, "刷单返利类") for i in range(3000)]
    h1, _ = build_holdout(rows, seed=42)
    h2, _ = build_holdout(rows, seed=42)
    assert [r["id"] for r in h1] == [r["id"] for r in h2]
    h3, _ = build_holdout(rows, seed=7)
    assert [r["id"] for r in h1] != [r["id"] for r in h3]


def test_holdout_row_shape():
    rows = [_mk(i, "刷单返利类") for i in range(2000)]
    hold, _ = build_holdout(rows, seed=1, n_rebate=100)
    r = hold[0]
    assert set(r) >= {"id", "type", "state", "language", "expected_risk", "category", "source"}
    assert r["type"] == "single" and r["state"]
    assert r["expected_risk"] == 4 and r["language"] == "zh"
