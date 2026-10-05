"""切 CCL2023 同源留出集，供 v3 的 Gate4 评测（主指标）。

同源留出：1000 条刷单返利 + 其余各类各 50 条。按 案件编号 切分（同一案件不跨训练/留出
两侧），seed 固定可复现。输出评测格式，category 用 CCL2023_LABEL_MAP 的真值。

用法: python scripts/build_ccl_holdout.py [--seed 42]
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import CCL2023_LABEL_MAP  # noqa: E402

SRC = Path("datasets/raw/ccl2023/train.json")
OUT = Path("datasets/ccl_rebate_eval.jsonl")
REBATE_LABEL = "刷单返利类"


def build_holdout(rows, seed=42, n_rebate=1000, n_other=50):
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("案件类别") in CCL2023_LABEL_MAP:
            by_label[r["案件类别"]].append(r)

    hold_src: list[dict] = []
    for label, items in by_label.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        k = n_rebate if label == REBATE_LABEL else n_other
        hold_src.extend(shuffled[: min(k, len(shuffled))])

    hold = [{
        "id": f"ccl_holdout_{r['案件编号']}",
        "type": "single",
        "state": r["案情描述"],
        "language": "zh",
        "expected_risk": 4,
        "category": CCL2023_LABEL_MAP[r["案件类别"]],
        "source": "ccl2023_holdout",
    } for r in hold_src]

    held_ids = {r["案件编号"] for r in hold_src}
    rest = [r for r in rows if r.get("案件编号") not in held_ids]
    return hold, rest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = json.load(open(SRC))
    hold, _ = build_holdout(rows, seed=args.seed)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in hold))
    print(f"✓ {OUT}: {len(hold)} 条")
    print("  category:", dict(Counter(r["category"] for r in hold).most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
