"""Verbose demo: print every step of Laya inference for a sample message.

Shows:
- Original text + question schema
- Tokenized sequence with [CLS]/[SEP]/[MASK] positions highlighted
- Each ONNX input tensor (shape, dtype, first/last values)
- Raw logits from ONNX
- Temperature-scaled logits
- Softmax probabilities
- Final typed answer

Run with:
    .venv/bin/python test_verbose.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from src.laya_onnx import (
    OnnxLayaClient,
    _tokenize_question,
    _softmax_with_temperature,
    QTYPE_CHOICE,
    QTYPE_NOUL,
    QTYPE_SCORE,
)


SAMPLES = [
    ("zh", "中文公检法诈骗",
     "【最高人民检察院】您涉嫌洗钱，已立案侦查，请将资金转入安全账户配合调查，联系警官00861-396-xxx。"),
    ("en", "PayPal phishing",
     "URGENT: Your PayPal account has been limited. Click here to verify: http://paypa1-secure.tk/login"),
]


def format_token_ids(token_ids: list[int], tokenizer: Tokenizer,
                     option_positions: list[int], max_show: int = 60) -> str:
    """Show tokens with [MASK] positions highlighted."""
    pieces = []
    for i, tid in enumerate(token_ids[:max_show]):
        tok_str = tokenizer.decode([tid])
        marker = " <<<MASK" if i in option_positions else ""
        pieces.append(f"  [{i:3}] {tid:6} = {tok_str!r}{marker}")
    if len(token_ids) > max_show:
        pieces.append(f"  ... ({len(token_ids) - max_show} more tokens)")
    return "\n".join(pieces)


def run_one_verbose(client: OnnxLayaClient, label: str, text: str, schema: dict) -> None:
    print("=" * 78)
    print(f"📨 Sample: {label}")
    print("=" * 78)
    print(f"\n─── INPUT ───")
    print(f"text ({len(text)} chars): {text}")
    print(f"\nschema: {json.dumps(schema, ensure_ascii=False, indent=2)}")

    answers: dict = {}

    print("\n" + "─" * 78)
    print("─── PER-QUESTION INFERENCE ───")
    print("─" * 78)

    for qname, q in schema.items():
        qtype_str = q["type"]
        qtype = {"choice": QTYPE_CHOICE, "score": QTYPE_SCORE, "noul": QTYPE_NOUL}[qtype_str]
        instructions = q.get("instructions", "")
        criteria = q.get("criteria", [])

        if qtype == QTYPE_CHOICE:
            options = {k: str(v) for k, v in criteria.items()}
        elif qtype == QTYPE_NOUL:
            options = {"false": "no", "true": "yes"}
        elif qtype == QTYPE_SCORE:
            options = {f"level_{i}": str(c) for i, c in enumerate(criteria)}

        print(f"\n>>> QUESTION: {qname!r} (type={qtype_str})")
        print(f"    instructions: {instructions}")
        print(f"    options: {options}")

        # Step 1: Tokenize
        head_tokens, head_positions = _tokenize_question(
            tokenizer=client.tokenizer,
            question=instructions,
            options=options,
            head_max_len=client.default_head_max_len,
            cls_id=client.cls_id,
            sep_id=client.sep_id,
            mask_id=client.mask_id,
        )
        remaining = client.default_max_len - len(head_tokens) - 1
        state_ids = client.tokenizer.encode(text, add_special_tokens=False).ids
        state_ids = state_ids[: max(0, remaining)]
        token_ids = head_tokens + state_ids + [client.sep_id]
        marker_positions = [p for p in head_positions if p < len(token_ids)]

        print(f"\n─── STEP 1: TOKENIZE ────")
        print(f"  head_max_len: {client.default_head_max_len}  (option token budget)")
        print(f"  max_len:      {client.default_max_len}  (head + state)")
        print(f"  question prefix tokens: {len(head_tokens)}")
        print(f"  state tokens: {len(state_ids)}")
        print(f"  trailing [SEP]: 1")
        print(f"  TOTAL token sequence length: {len(token_ids)}")
        print(f"\n  Token sequence (first 60 of {len(token_ids)}):")
        print(format_token_ids(token_ids, client.tokenizer, marker_positions, max_show=60))

        print(f"\n  [MASK] positions: {marker_positions}  ({len(marker_positions)} options)")
        for i, p in enumerate(marker_positions):
            label_i = list(options.keys())[i]
            print(f"    pos {p}: {label_i}")

        # Step 2: Build ONNX inputs
        n_opts = len(marker_positions)
        padded_pos = np.zeros((1, 16), dtype=np.int64)
        padded_pos[0, :n_opts] = marker_positions
        padded_mask = np.zeros((1, 16), dtype=bool)
        padded_mask[0, :n_opts] = True

        inputs = {
            "input_ids":      np.array([token_ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(token_ids)), dtype=np.int64),
            "marker_pos":     padded_pos,
            "marker_mask":    padded_mask,
            "qtype":          np.array([qtype], dtype=np.int64),
        }

        print(f"\n─── STEP 2: ONNX INPUTS ────")
        for k, v in inputs.items():
            print(f"  {k}:")
            print(f"    shape: {v.shape}, dtype: {v.dtype}")
            flat = v.flatten()
            preview = flat[:6].tolist()
            ellipsis = "..." if len(flat) > 6 else ""
            print(f"    values (first 6): {preview}{ellipsis}")
            if k == "qtype":
                type_name = {0: "CHOICE", 1: "SCORE", 2: "NOUL"}[flat[0]]
                print(f"    meaning: qtype={int(flat[0])} = {type_name}")
            elif k == "marker_pos":
                valid = [int(x) for x in flat[:n_opts]]
                print(f"    valid [MASK] positions: {valid}")

        # Step 3: Run ONNX
        outputs = client.session.run(None, inputs)
        logits = outputs[0][0, :n_opts]
        raw_logits = logits.copy()

        print(f"\n─── STEP 3: ONNX OUTPUT (raw logits) ────")
        print(f"  output[0] shape: {outputs[0].shape}")
        print(f"  sliced logits ({n_opts} options): {logits.tolist()}")
        for i, (label_i, logit) in enumerate(zip(options.keys(), logits)):
            print(f"    [{i}] {label_i}: raw_logit = {float(logit):.4f}")

        # Step 4: Temperature
        temperature = client._temperature_for(qtype, n_opts)
        print(f"\n─── STEP 4: TEMPERATURE ────")
        print(f"  fitted T for qtype={qtype_str}, n_opts={n_opts}: {temperature}")

        # Step 5: Softmax
        probs = _softmax_with_temperature(logits, temperature)
        print(f"\n─── STEP 5: SOFTMAX (with temperature) ────")
        print(f"  probs: {probs.tolist()}")
        for i, (label_i, prob) in enumerate(zip(options.keys(), probs)):
            print(f"    [{i}] {label_i}: prob = {float(prob):.4f}")

        # Step 6: Final answer
        if qtype == QTYPE_NOUL:
            p_true = float(probs[1]) if n_opts > 1 else float(probs[0])
            answer = {
                "noul": p_true,
                "probabilities": {"false": float(probs[0]), "true": p_true},
                "confidence": max(float(probs[0]), p_true),
            }
        elif qtype == QTYPE_CHOICE:
            argmax = int(np.argmax(probs))
            labels = list(criteria.keys())
            answer = {
                "choice": labels[argmax],
                "probabilities": {labels[i]: float(probs[i]) for i in range(n_opts)},
                "confidence": float(probs[argmax]),
            }
        elif qtype == QTYPE_SCORE:
            n = n_opts
            expected = sum(float(probs[i]) * i for i in range(n))
            levels = criteria if isinstance(criteria, list) else list(criteria.values())
            answer = {
                "score": expected,
                "max_score": n - 1,
                "distribution": {levels[i]: float(probs[i]) for i in range(n)},
                "confidence": float(probs.max()),
            }

        print(f"\n─── STEP 6: FINAL ANSWER ────")
        print(f"  {json.dumps(answer, ensure_ascii=False, indent=2)}")

        answers[qname] = answer

    print(f"\n─── AGGREGATED RESULT ───")
    print(json.dumps({"answers": answers}, ensure_ascii=False, indent=2))
    print()


def main() -> int:
    print("=" * 78)
    print("Laya 详细推理 Demo — 显示模型每一步的输入输出")
    print("=" * 78)
    print("\n加载模型...")
    client = OnnxLayaClient("models/laya-onnx-en")
    schema = json.load(open("schemas/scam.json"))
    print(f"✓ model.onnx loaded, max_len={client.default_max_len}, "
          f"head_max_len={client.default_head_max_len}\n")

    for lang, label, text in SAMPLES:
        run_one_verbose(client, label, text, schema)

    return 0


if __name__ == "__main__":
    sys.exit(main())