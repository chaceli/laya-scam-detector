"""跑 v2 四套评测并调用四门槛验收（design §1、§7.2）。

用法：python scripts/run_full_eval.py [--model-dir models/laya-onnx-multilingual-finetuned-v2]
"""
import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

V2_DEFAULT = "models/laya-onnx-multilingual-finetuned-v2"
HOLDOUT = Path("datasets/eval_holdout_v2.jsonl")
TEST = Path("datasets/training/test.jsonl")
EVAL_INPUTS = [
    ("hardneg", "datasets/hardneg_eval.jsonl"),
    ("holdout", str(HOLDOUT)),
    ("handwritten", "datasets/eval.jsonl"),
    # 第 4 套：英文侧不回退证据（design §7.2）——无门槛消费，仅出报告
    ("public", "datasets/public.jsonl"),
]


def ensure_holdout() -> None:
    """600 条中文 holdout（test.jsonl 采样，seed=42）+ 两个 rebate 评测集。

    rebate_scam 必须显式并入：test.jsonl 来自 split_dataset(real)，而合成源
    只进 train，因此 test.jsonl 里 rebate_scam 为 0 条。不并入的话 Gate4 的
    分类别召回表查不到该类，parse_category_recall 返回 None，gate 必然失败。
    并入两个互补口径：
    - datasets/ccl_rebate_eval.jsonl —— CCL 同源留出（主指标，1000 条，n 大）
    - datasets/rebate_eval_real.jsonl —— 42 条警方/媒体跨源集（不同语域，泛化检查）
    两者均与训练集零重叠（见 build_dataset 的归一化文本排除）。
    """
    if HOLDOUT.exists():
        return
    rows = [json.loads(l) for l in TEST.open() if l.strip()]
    zh = [r for r in rows if r.get("language") == "zh"]
    rng = random.Random(42)
    rng.shuffle(zh)
    picked = zh[:600]
    assert len(picked) == 600, f"test.jsonl 中文行不足 600: {len(zh)}"
    out = [{
        "id": r["id"], "type": "single", "state": r["text"],
        "language": "zh", "expected_risk": r["risk"],
        "category": r["category"], "source": r["source"],
    } for r in picked]

    n_rebate = 0
    for eval_file in ("datasets/ccl_rebate_eval.jsonl",
                      "datasets/rebate_eval_real.jsonl"):
        p = Path(eval_file)
        if not p.exists():
            continue
        for l in p.open():
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            r.setdefault("id", f"rebate_eval_{n_rebate}")
            r.setdefault("type", "single")
            r.setdefault("language", "zh")
            if "state" not in r and "text" in r:
                r["state"] = r["text"]
            out.append(r)
            n_rebate += 1
    assert n_rebate, "两个 rebate 评测集都为空 —— Gate4 将无 rebate_scam 可评"

    with HOLDOUT.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {HOLDOUT}: {len(picked)} 中文 + {n_rebate} rebate = {len(out)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=V2_DEFAULT)
    args = ap.parse_args()

    if not Path(args.model_dir, "model.onnx").exists():
        print(f"✗ 模型缺失: {args.model_dir}")
        return 2

    ensure_holdout()
    paths: dict[str, Path] = {}
    for name, inp in EVAL_INPUTS:
        cmd = [sys.executable, "main.py", "--eval", "--input", inp,
               "--output", "reports/", "--multilingual-dir", args.model_dir]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            print(f"✗ {name} eval 失败:\n{r.stderr[-500:]}")
            return 2
        m = re.search(r"✓ Report: (.+)", r.stderr)
        if not m:
            print(f"✗ {name} eval 未输出报告路径")
            return 2
        paths[name] = Path(m.group(1).strip())
        print(f"✓ {name}: {paths[name]}")

    cmd = [sys.executable, "scripts/check_acceptance.py",
           "--hardneg", str(paths["hardneg"]),
           "--holdout", str(paths["holdout"]),
           "--handwritten", str(paths["handwritten"])]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())