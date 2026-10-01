"""切分难负评测集（永不进训练）与训练池。design §7.2。

评测集：8 体裁 + 自然 benign 各 75 条（不足则全取并警告）。
输出：datasets/hardneg_eval.jsonl（评测格式）
     + datasets/hard_negatives/train_pool.jsonl（训练格式）。
contrastive_pairs.jsonl 不参与切分（train-only）。
用法：python scripts/build_hardneg_evalset.py
"""
import json
import random
import sys
from pathlib import Path

DATA = Path("datasets/hard_negatives")
OUT_EVAL = Path("datasets/hardneg_eval.jsonl")
OUT_TRAIN = DATA / "train_pool.jsonl"
PER_GENRE = 75
SEED = 42


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def main() -> int:
    rng = random.Random(SEED)
    eval_rows: list[dict] = []
    train_rows: list[dict] = []
    genre_files = sorted(DATA.glob("hn_*.jsonl"))
    nb = DATA / "nb_natural_benign.jsonl"
    if nb.exists():
        genre_files.append(nb)
    for p in genre_files:
        if p.name == "train_pool.jsonl":
            continue
        rows = load_jsonl(p)
        if not rows:
            print(f"  ! {p.name} 为空，跳过")
            continue
        rng.shuffle(rows)
        k = min(PER_GENRE, len(rows))
        if len(rows) < PER_GENRE + 40:
            print(f"  ! {p.stem} 仅 {len(rows)} 条（建议 ≥115：75 eval + 40 train）")
        source = rows[0]["source"]
        for r in rows[:k]:
            eval_rows.append({
                "id": r["id"], "type": "single", "state": r["text"],
                "language": r.get("language", "zh"), "expected_risk": 1,
                "category": "benign", "source": source,
            })
        train_rows.extend(rows[k:])
        print(f"  ✓ {p.stem}: eval {k} / train {len(rows) - k}")

    eval_ids = {r["id"] for r in eval_rows}
    train_ids = {r["id"] for r in train_rows}
    overlap = eval_ids & train_ids
    assert not overlap, f"评测/训练 id 相交 {len(overlap)} 条 —— 切分 bug"

    with OUT_EVAL.open("w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_TRAIN.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {OUT_EVAL}: {len(eval_rows)} 条")
    print(f"✓ {OUT_TRAIN}: {len(train_rows)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())