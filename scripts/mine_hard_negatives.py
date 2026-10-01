"""滚动挖掘最难负样本（design §6.4；FPR>2% 时启动，硬上限 2 轮）。

mine  : v2 模型给候选池打分 → Top-K 复核清单（verdict 留空待人工）
merge : 人工填 verdict=keep/drop 后，把 keep 行追加进 train_pool.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.laya_onnx import OnnxLayaClient

DATA = Path("datasets/hard_negatives")
SCHEMA = json.loads(Path("schemas/scam.json").read_text())


def mode_mine(args) -> int:
    if not Path(args.model_dir, "model.onnx").exists():
        print(f"✗ 模型缺失: {args.model_dir}")
        return 2
    cand = DATA / "mining_candidates.jsonl"
    if not cand.exists():
        print(f"✗ {cand} 缺失 —— 先跑 generate_negatives.py --out 生成候选池")
        return 2
    rows = [json.loads(l) for l in cand.open() if l.strip()]
    print(f"候选池 {len(rows)} 条，打分中…")
    client = OnnxLayaClient(args.model_dir)
    scored = []
    for i, r in enumerate(rows):
        pred = client.predict(r["text"], SCHEMA)
        scored.append((pred["answers"]["is_scam"]["noul"], r))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}")
    scored.sort(key=lambda x: -x[0])
    out = DATA / f"mining_round{args.round}_review.jsonl"
    with out.open("w") as f:
        for noul, r in scored[: args.top]:
            rec = dict(r)
            rec["noul"] = noul
            rec["verdict"] = ""
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✓ {out}: Top {min(args.top, len(scored))} 复核清单")
    print("  人工操作：把每行 verdict 填 keep（确认真 benign）或 drop（实为诈骗/假负例）")
    return 0


def mode_merge(args) -> int:
    review = DATA / f"mining_round{args.round}_review.jsonl"
    pool = DATA / "train_pool.jsonl"
    keep = [json.loads(l) for l in review.open() if l.strip()
            and json.loads(l).get("verdict") == "keep"]
    with pool.open("a") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {pool} += {len(keep)} keep 行（drop 行不入池）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["mine", "merge"])
    ap.add_argument("--model-dir",
                    default="models/laya-onnx-multilingual-finetuned-v2")
    ap.add_argument("--round", type=int, required=True, choices=[1, 2])
    ap.add_argument("--top", type=int, default=1000)
    args = ap.parse_args()
    return mode_mine(args) if args.mode == "mine" else mode_merge(args)


if __name__ == "__main__":
    sys.exit(main())