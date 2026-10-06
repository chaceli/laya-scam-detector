"""生成 v2 vs v3 训练前后对比报告（含真实文本）。

取样：CCL 同源留出 rebate（含被 v2 漏判的）、42 条跨源警方/媒体案件、难负样本 benign。
对每条跑 v2 与 v3，逐条展示 文本 + 两版预测，突出 Gate4 的修复与 FPR 的保持。

用法: python scripts/gen_v3_comparison_report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.laya_onnx import OnnxLayaClient  # noqa: E402

SCHEMA = json.loads(Path("schemas/scam.json").read_text())
V2 = "models/laya-onnx-multilingual-finetuned-v2"
V3 = "models/laya-onnx-multilingual-finetuned-v3"
OUT = Path("reports/v2-vs-v3-comparison.md")


def load(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def predict(client, text):
    a = client.predict(text, SCHEMA)["answers"]
    return a["is_scam"]["noul"], a["scam_category"]["choice"]


def fmt(t, n=110):
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n] + "…"


def main() -> int:
    c2, c3 = OnnxLayaClient(V2), OnnxLayaClient(V3)

    ccl = [r for r in load("datasets/ccl_rebate_eval.jsonl")
           if r["category"] == "rebate_scam"]
    cross = load("datasets/rebate_eval_real.jsonl")
    benign = [r for r in load("datasets/hardneg_eval.jsonl")]

    import random
    rng = random.Random(7)
    groups = [
        ("CCL 同源留出 · 刷单返利（真实案件）", ccl, "state", rng.sample(ccl, 8)),
        ("跨源 · 警方/媒体案件（不同语域）", cross, "text", cross[:6]),
        ("难负样本 · 合法文本（FPR 保持）", benign, "state", rng.sample(benign, 6)
         if len(benign) >= 6 else benign),
    ]

    lines = [
        "# v2 → v3 训练前后对比（真实文本）",
        "",
        "v2 = 合成 rebate 训练；v3 = 真实 CCL2023 训练。展示两版在同一批**真实文本**上的逐条预测差异。",
        "",
        "格式：`is_scam 概率 / 预测类别`。",
        "",
    ]
    for title, rows, key, sample in groups:
        lines += [f"## {title}", "",
                  "| # | 文本 | v2 | v3 |", "|---|---|---|---|"]
        for i, r in enumerate(sample, 1):
            t = r[key]
            n2, k2 = predict(c2, t)
            n3, k3 = predict(c3, t)
            mark = " ✅" if k3 == r.get("category") else ""
            lines.append(
                f"| {i} | {fmt(t)} | `{n2:.2f} / {k2}` | `{n3:.2f} / {k3}`{mark} |")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
