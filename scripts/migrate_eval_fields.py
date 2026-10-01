"""统一评测字段口径：为 datasets/*.jsonl 补 category 字段（design §4.2）。"""
import json
from pathlib import Path

FILES = [Path("datasets/eval.jsonl"), Path("datasets/multi.jsonl"),
         Path("datasets/single.jsonl"), Path("datasets/public.jsonl")]

for path in FILES:
    if not path.exists():
        print(f"! 缺失 {path}")
        continue
    rows = [json.loads(l) for l in path.open() if l.strip()]
    changed = 0
    for rec in rows:
        if "category" not in rec:
            derived = rec.get("expected_category")
            if derived is None:
                risk = rec.get("expected_risk")
                derived = ("spam_general"
                           if (risk is not None and risk >= 4) else "benign")
            rec["category"] = derived
            changed += 1
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"✓ {path}: {changed}/{len(rows)} 补 category")