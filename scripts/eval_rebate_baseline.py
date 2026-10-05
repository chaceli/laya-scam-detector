"""v1 基线：rebate_scam 类在真实评测集上的 is_scam 召回。

用 taxonomy 无关的 noul（是否诈骗）做主指标 —— v1 模型训练于 13 类体系，
choice 头没有 rebate_scam 选项，拿它衡量 v2 目标类是无效的。

必须同时跑良性对照（hardneg_eval 675 条难负样本）：只报召回没有意义，
一个把所有输入都判为诈骗的模型召回是 100%，但什么也没学到。

用法:
  python scripts/eval_rebate_baseline.py \
      --model models/laya-onnx-multilingual-finetuned-fp16
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.laya_onnx import OnnxLayaClient  # noqa: E402

SCHEMA = json.loads(Path("schemas/scam.json").read_text())
NOUL_Q = {"is_scam": SCHEMA["is_scam"]}


def load(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.open() if l.strip()]
    # hardneg_eval.jsonl stores the message under "state"; the training-side
    # files use "text". Normalise so both eval inputs work.
    for r in rows:
        if "text" not in r and "state" in r:
            r["text"] = r["state"]
    return [r for r in rows if r.get("text")]


def score_rows(client: OnnxLayaClient, rows: list[dict],
               label: str) -> list[dict]:
    out = []
    t0 = time.perf_counter()
    for i, r in enumerate(rows):
        p = client.predict(r["text"], NOUL_Q)
        out.append({"text": r["text"], "p_true": p["answers"]["is_scam"]["noul"],
                    "latency_ms": p["latency_ms"]})
        if (i + 1) % 50 == 0:
            print(f"  {label}: {i+1}/{len(rows)}")
    print(f"  {label}: {len(rows)} 条 / {time.perf_counter()-t0:.1f}s")
    return out


def report(name: str, scored: list[dict], positives: bool, threshold: float) -> dict:
    flagged = [s for s in scored if s["p_true"] >= threshold]
    rate = len(flagged) / len(scored) if scored else 0.0
    kind = "召回(>=t 判为诈骗)" if positives else "误报率(>=t 误判 benign 为诈骗)"
    print(f"\n{name}")
    print(f"  {kind}: {rate:.3f}  ({len(flagged)}/{len(scored)})")
    ps = sorted(s["p_true"] for s in scored)
    if ps:
        print(f"  p_true 分位: p10={ps[len(ps)//10]:.3f} "
              f"p50={ps[len(ps)//2]:.3f} p90={ps[9*len(ps)//10]:.3f}")
    return {"name": name, "n": len(scored), "rate": rate, "flagged": len(flagged)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/laya-onnx-multilingual-finetuned-fp16")
    ap.add_argument("--eval-set", default="datasets/rebate_eval_real.jsonl")
    ap.add_argument("--control", default="datasets/hardneg_eval.jsonl")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="reports/rebate_baseline_v1.md")
    args = ap.parse_args()

    if not Path(args.model, "model.onnx").exists():
        print(f"✗ 模型缺失: {args.model}")
        return 2

    client = OnnxLayaClient(args.model)
    print(f"模型: {args.model}")

    results = []
    lines = ["# v1 基线：rebate_scam 真实评测集召回", "",
             f"模型 `{args.model}` · 阈值 {args.threshold} · "
             f"指标 noul(is_scam)，taxonomy 无关（v1 无 rebate_scam 类）", ""]

    for path, positives, name in (
        (Path(args.eval_set), True, "rebate 真实评测集（诈骗正样本）"),
        (Path(args.control), False, "hardneg 难负样本（benign 对照）"),
    ):
        if not path.exists():
            print(f"! 跳过缺失: {path}")
            continue
        rows = load(path)
        scored = score_rows(client, rows, name)
        r = report(name, scored, positives, args.threshold)
        results.append(r)
        lines += [f"## {name}", "",
                  f"- n = {r['n']}",
                  f"- 判定为诈骗: {r['flagged']} ({r['rate']:.3f})", ""]
        if not positives:
            worst = sorted(scored, key=lambda s: -s["p_true"])[:5]
            lines += ["误报最高的 5 条：", ""]
            lines += [f"- `{w['text'][:50]}` p={w['p_true']:.3f}" for w in worst]
            lines.append("")

    if len(results) == 2:
        pos, neg = results[0], results[1]
        print(f"\n判别力 gap（召回 - 误报）: {pos['rate'] - neg['rate']:.3f}")
        lines += [f"**判别力 gap（召回 − 误报）= {pos['rate'] - neg['rate']:.3f}**", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
