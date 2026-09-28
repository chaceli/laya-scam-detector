"""Create a Kaggle-ready subsample of the training data.

LoRA fine-tuning needs 10-50k samples, not 346k. A smaller upload:
1. Uploads much faster (30MB vs 461MB)
2. Trains faster (4-5h → 1-2h)
3. Still enough signal for LoRA (r=8, 3.5M trainable params)

Stratifies by (source, language, is_scam) to preserve distribution.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

SRC = Path("datasets/training")
DST = Path("kaggle/upload")
DST.mkdir(parents=True, exist_ok=True)

TRAIN_SIZE = 30_000
VAL_SIZE = 3_000
SEED = 42


def subsample(path: Path, target_size: int, stratify: bool = True) -> list[dict]:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not stratify:
        random.seed(SEED)
        random.shuffle(rows)
        return rows[:target_size]

    # Stratify by (source, language, is_scam)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("source", "?"), r.get("language", "?"), r.get("is_scam", -1))
        groups[key].append(r)

    random.seed(SEED)
    for items in groups.values():
        random.shuffle(items)

    # Proportional allocation
    total = len(rows)
    result = []
    for key, items in groups.items():
        share = max(1, round(target_size * len(items) / total))
        result.extend(items[:share])
    random.shuffle(result)
    return result[:target_size]


def main() -> int:
    print("Subsampling for Kaggle upload...")
    train = subsample(SRC / "train.jsonl", TRAIN_SIZE)
    val = subsample(SRC / "val.jsonl", VAL_SIZE)

    with (DST / "train.jsonl").open("w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (DST / "val.jsonl").open("w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  ✓ train.jsonl: {len(train):,} rows")
    print(f"  ✓ val.jsonl: {len(val):,} rows")

    # Distribution summary
    from collections import Counter
    print("\nTrain language:", dict(Counter(r.get("language") for r in train).most_common()))
    print("Train is_scam:", dict(Counter(r.get("is_scam") for r in train)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())