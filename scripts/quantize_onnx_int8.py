"""Quantize the fine-tuned ONNX model to int8 for in-browser inference.

Dynamic per-channel int8 shrinks the checkpoint to ~28% of fp32 (370 MB
including the tokenizer), which is what the Hugging Face Static Space
downloads. Verified: 10/10 decision agreement vs fp16, max probability
delta 0.0072.

Quantizing must start from the fp32 export — the quantizer emits fp32
scales, and an fp16 source produces fp16 scales that DequantizeLinear
rejects with an InvalidGraph error.

Usage:
  .venv/bin/python scripts/quantize_onnx_int8.py \
      --src models/laya-onnx-multilingual-finetuned \
      --dst models/laya-onnx-multilingual-finetuned-int8
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

PROBES = [
    ("您好，我是XX快递客服，您有一个包裹在运输途中丢失，请点击链接填写个人信息理赔。", "scam", "delivery_fraud"),
    ("【最高人民检察院】您涉嫌洗钱，请将资金转入安全账户配合调查。", "scam", "impersonation"),
    ("宝贝儿，我这边投资平台稳赚不赔，先投5000试试，保证一周翻倍。", "scam", "investment_scam"),
    ("恭喜您被抽中二等奖，奖金10万元，请尽快联系客服领取。", "scam", "phishing"),
    ("妈，我今晚回家吃饭，大概6点到家。", "benign", None),
    ("老板，明天上午9点的会议改到下午3点，请准时参加。", "benign", None),
    ("您的快递已签收，感谢您选择京东。", "benign", None),
    ("URGENT: Verify your PayPal account at http://fake.tk/login", "scam", "phishing"),
    ("Hi, just confirming our meeting at 3pm tomorrow.", "benign", None),
    ("Your Amazon order #12345 has shipped, arriving Friday.", "benign", None),
]

COPY_EXTRA = ["tokenizer", "categories.json", "rl_agent_config.json", "config.json"]


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="models/laya-onnx-multilingual-finetuned")
    ap.add_argument("--dst", default="models/laya-onnx-multilingual-finetuned-int8")
    ap.add_argument("--per-channel", action="store_true", default=True)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not (src / "model.onnx").exists():
        print(f"✗ {src}/model.onnx not found (quantize from the fp32 export)")
        return 1

    from onnxruntime.quantization import QuantType, quantize_dynamic

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print(f"Quantizing {src}/model.onnx (dynamic QInt8, per_channel={args.per_channel})...")
    quantize_dynamic(
        model_input=str(src / "model.onnx"),
        model_output=str(dst / "model.onnx"),
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
        use_external_data_format=True,
    )
    for sub in COPY_EXTRA:
        s = src / sub
        if not s.exists():
            continue
        (shutil.copytree if s.is_dir() else shutil.copy)(s, dst / sub)
        print(f"  ✓ {sub}")

    fp32_mb, int8_mb = dir_size(src) / 1e6, dir_size(dst) / 1e6
    print(f"\nSize: {fp32_mb:.0f} MB → {int8_mb:.0f} MB ({int8_mb/fp32_mb*100:.0f}%)")

    print("\nVerifying against the fp32 model...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.laya_onnx import OnnxLayaClient

    schema = json.loads(Path("schemas/scam.json").read_text())
    ref = OnnxLayaClient(src)
    q = OnnxLayaClient(dst)
    ok_scam = ok_cat = 0
    max_diff = 0.0
    for text, exp_scam, exp_cat in PROBES:
        a_ref = ref.predict(text, schema)["answers"]
        a_q = q.predict(text, schema)["answers"]
        p_ref, p_q = a_ref["is_scam"]["noul"], a_q["is_scam"]["noul"]
        max_diff = max(max_diff, abs(p_ref - p_q))
        s_ok = (p_ref >= 0.5) == (p_q >= 0.5)
        c_ok = a_ref["scam_category"]["choice"] == a_q["scam_category"]["choice"]
        ok_scam += s_ok
        ok_cat += c_ok
        print(f"  {text[:24]:26s} scam {p_ref:.3f}/{p_q:.3f} {'✓' if s_ok else '✗'}"
              f"  cat {'✓' if c_ok else '✗'}")

    n = len(PROBES)
    print(f"\nis_scam agreement: {ok_scam}/{n}")
    print(f"category agreement: {ok_cat}/{n}")
    print(f"max probability diff: {max_diff:.4f}")
    if ok_scam != n or ok_cat != n:
        print("✗ int8 changed a decision")
        return 1
    print("✓ int8 quantization verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())