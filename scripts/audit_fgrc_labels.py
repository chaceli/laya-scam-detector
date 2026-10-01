"""用 v1 ONNX 模型审计 FGRC 标签噪声（design §4.2：抽样发现错配，二元用途可控）。

用法：PYTHONPATH=. python scripts/audit_fgrc_labels.py [--sample 2000]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.laya_onnx import OnnxLayaClient

MODEL_DIR = "models/laya-onnx-multilingual-finetuned"  # v1
TRAIN = "datasets/training/train.jsonl"
OUT = "reports/fgrc_audit.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    args = ap.parse_args()

    if not Path(args.model_dir, "model.onnx").exists():
        print(f"✗ 模型缺失: {args.model_dir}/model.onnx")
        return 2

    rows = []
    with open(TRAIN) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["source"].startswith("fgrc"):
                    rows.append(r)
    rng = random.Random(42)
    rng.shuffle(rows)
    rows = rows[: args.sample]
    if not rows:
        print(f"✗ {TRAIN} 无 fgrc 行（先跑 Task 7）")
        return 2

    client = OnnxLayaClient(args.model_dir)
    schema = json.loads(Path("schemas/scam.json").read_text())

    noise_flag, hard_fp = [], []  # scam→benign 分歧 / benign→scam 分歧
    for i, r in enumerate(rows):
        pred = client.predict(r["text"], schema)
        noul = pred["answers"]["is_scam"]["noul"]
        if r["is_scam"] == 1 and noul < 0.5:
            noise_flag.append((r, noul))
        elif r["is_scam"] == 0 and noul >= 0.5:
            hard_fp.append((r, noul))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}")

    lines = [
        "# FGRC 标签噪声审计（v1 模型 vs 训练标签）",
        "",
        f"- 样本: {len(rows)} 条 fgrc 行（seed=42）",
        f"- label=scam & pred<0.5（疑似标签噪声）: {len(noise_flag)}",
        f"- label=benign & pred≥0.5（疑似难负/误报源）: {len(hard_fp)}",
        "",
        "## 疑似标签噪声 Top 100（模型判 benign，标签 scam）",
        "",
        "| noul | label category | text |",
        "|---|---|---|",
    ]
    for r, noul in sorted(noise_flag, key=lambda x: x[1])[:100]:
        preview = r["text"][:80].replace("|", "\\|")
        lines.append(f"| {noul:.2f} | {r['category']} | {preview} |")
    lines += ["", "## 疑似难负 Top 100（模型判 scam，标签 benign）", "",
              "| noul | text |", "|---|---|"]
    for r, noul in sorted(hard_fp, key=lambda x: -x[1])[:100]:
        preview = r["text"][:80].replace("|", "\\|")
        lines.append(f"| {noul:.2f} | {preview} |")
    Path(OUT).write_text("\n".join(lines) + "\n")
    print(f"✓ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())