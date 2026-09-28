"""Convert the fine-tuned ONNX model from fp32 to fp16.

Halves the checkpoint (1.29GB -> ~645MB) so it can ship as a GitHub Release
asset and load faster in a Hugging Face Space. Inputs/outputs are kept in
their original dtypes (keep_io_types=True) so the existing OnnxLayaClient
contract is unchanged.

Usage:
  .venv/bin/python scripts/quantize_onnx_fp16.py \
      --src models/laya-onnx-multilingual-finetuned \
      --dst models/laya-onnx-multilingual-finetuned-fp16
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/laya-onnx-multilingual-finetuned")
    ap.add_argument("--dst", default="models/laya-onnx-multilingual-finetuned-fp16")
    ap.add_argument("--atol", type=float, default=0.05,
                    help="max abs logit difference tolerated for fp16")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not (src / "model.onnx").exists():
        print(f"✗ {src}/model.onnx not found")
        return 1

    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    print(f"Loading {src/'model.onnx'} (external data)...")
    model = onnx.load(str(src / "model.onnx"))
    print(f"  graph loaded; initializers: {len(model.graph.initializer)}")

    print("Converting to fp16 (keep_io_types=True)...")
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print(f"Saving fp16 model → {dst}/model.onnx ...")
    onnx.save(
        model_fp16,
        str(dst / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=1024,
    )

    # Copy non-weight assets unchanged
    for sub in ["tokenizer", "categories.json", "rl_agent_config.json", "config.json"]:
        s = src / sub
        if s.exists():
            d = dst / sub
            if s.is_dir():
                shutil.copytree(s, d)
            else:
                shutil.copy(s, d)
            print(f"  ✓ copied {sub}")

    fp32_mb = (src / "model.onnx.data").stat().st_size / 1e6
    fp16_mb = (dst / "model.onnx.data").stat().st_size / 1e6
    print(f"\nSize: {fp32_mb:.0f} MB → {fp16_mb:.0f} MB ({fp16_mb/fp32_mb*100:.0f}%)")

    # Verify numeric agreement against fp32 on a real input
    print("\nVerifying fp32 vs fp16 logits...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.laya_onnx import OnnxLayaClient

    c32 = OnnxLayaClient(src)
    c16 = OnnxLayaClient(dst)
    probes = [
        "您好，我是XX快递客服，您有一个包裹在运输途中丢失，请点击链接理赔。",
        "妈，我今晚回家吃饭，大概6点到家。",
    ]
    questions = {"is_scam": {"type": "noul", "instructions": "Is this a scam?"}}
    max_diff = 0.0
    for text in probes:
        p32 = c32.predict(text, questions)["answers"]["is_scam"]["noul"]
        p16 = c16.predict(text, questions)["answers"]["is_scam"]["noul"]
        d = abs(p32 - p16)
        max_diff = max(max_diff, d)
        print(f"  '{text[:20]}…' fp32={p32:.5f} fp16={p16:.5f} diff={d:.5f}")

    print(f"\nmax probability diff: {max_diff:.5f} (tolerance {args.atol})")
    if max_diff > args.atol:
        print("✗ fp16 deviates too much")
        return 1
    print("✓ fp16 conversion verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())