---
library_name: onnxruntime
license: apache-2.0
base_model: convaiinnovations/laya
tags:
  - onnx
  - scam-detection
  - phishing-detection
  - fraud-detection
  - text-classification
  - multilingual
  - chinese
  - laya
  - system-one
  - calibrated-decisions
language:
  - zh
  - en
  - multilingual
pipeline_tag: text-classification
---

# Laya Multilingual — Scam Detection (fp16 ONNX)

A LoRA fine-tune of [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya)
(mmBERT-base, 322M parameters, 100+ languages) for **Chinese/English scam-phrase
risk detection**, exported to **ONNX Runtime** (no PyTorch required at inference).

Trained entirely on-device on an Apple M4 Pro (MPS backend) — see the
[GitHub project](https://github.com/chaceli/laya-scam-detector) for the full
pipeline (dataset construction, training, evaluation).

## What it does

Give it a piece of text plus typed questions; it returns typed decisions with
calibrated probabilities in a **single forward pass** (≈125 ms on CPU for all
three questions). It never generates text, so there is nothing to parse and
nothing to hallucinate.

Three question primitives:

| Primitive | Question | Output |
|---|---|---|
| `noul` | Is this a scam? | P(true) ∈ [0,1] |
| `score` | How high is the risk? | expected level 0-4 + distribution |
| `choice` | What category of scam? | one of 13 labels + full distribution |

## 13-class taxonomy

`benign`, `phishing`, `crypto_scam`, `investment_scam`, `lottery_scam`,
`job_scam`, `loan_scam`, `impersonation`, `romance_scam`, `delivery_fraud`,
`marketing`, `adult_content`, `spam_general`

## Results

600-sample Chinese holdout (never seen during training):

| Metric | Value |
|---|---|
| is_scam accuracy | **0.967** |
| is_scam precision | 0.989 |
| is_scam recall | 0.957 |
| is_scam F1 | 0.972 |
| 13-class accuracy | **0.810** |
| risk_level MAE | 1.52 |
| p50 latency (CPU) | 125 ms |
| p95 latency (CPU) | 228 ms |

Handwritten mixed set (38 samples) — before → after fine-tuning:

| Metric | English base | This model |
|---|---|---|
| is_scam accuracy | 0.868 | **0.921** |
| is_scam F1 | 0.909 | **0.943** |
| Chinese accuracy | 0.840 | **0.920** |
| 13-class accuracy | 0.500 | **0.632** |
| false positives | 5 | **3** |
| p50 latency | 652 ms | **131 ms** |

The latency win is structural: mmBERT's 256k vocabulary tokenizes Chinese at
~1.5 chars/token, where an English-only ModernBERT vocabulary shredded it.

## Training

| Item | Value |
|---|---|
| Base | `convaiinnovations/laya-multilingual` (mmBERT-base, 322M) |
| Method | LoRA r=8, alpha=16, dropout=0.05 on `Wqkv`+`Wo` |
| Trainable params | 1,148,928 (0.36% of base) |
| Objective | cross-entropy (strictly proper log score) on `is_scam` + `category` |
| Data | 15,000 balanced samples from 5 public datasets (zh 84.5% / en 15.5%) |
| Hardware | Apple M4 Pro, MPS (peak 1.3 GB RAM) |
| Time | 73 minutes (3 epochs) |

Data sources: FGRC-SCD (Chinese telecom fraud), ealvaradob (English phishing),
FBS_SMS (Chinese fake-base-station), ScamShield, UCI SMS Spam, plus a small
synthetic seed for the two rarest classes.

## Files

| File | Size | Notes |
|---|---|---|
| `model.onnx` | 2.8 MB | computation graph |
| `model.onnx.data` | 643 MB | fp16 weights |
| `tokenizer/` | 33 MB | mmBERT 256k-vocab tokenizer |
| `categories.json` | — | the 13-class taxonomy |

fp32 → fp16 max probability difference measured at **0.00001**.

## Usage

This is a *decision* model: it does not chat. Feed it the exact input contract
below (matching the ONNX graph).

```python
import json
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
tok = Tokenizer.from_file("tokenizer/tokenizer.json")

# mmBERT special tokens: <bos>=2 (CLS), <eos>=1 (SEP), <mask>=4, <pad>=0
CLS, SEP, MASK, PAD = 2, 1, 4, 0

def tokenize_question(question: str, options: list[str], head_max_len: int = 192):
    ids = [CLS] + tok.encode(f"choice question: {question}",
                             add_special_tokens=False).ids + [SEP]
    positions = []
    per_option = max(8, (head_max_len - 16) // len(options))
    for opt in options:
        positions.append(len(ids))
        ids.append(MASK)
        ids += tok.encode(" " + opt, add_special_tokens=False).ids[:per_option]
    ids.append(SEP)
    return ids, positions

def predict(text: str, instructions: str, options: list[str], qtype: int):
    head_ids, positions = tokenize_question(instructions, options)
    room = 512 - len(head_ids) - 1
    state_ids = tok.encode(text, add_special_tokens=False).ids[:room]
    ids = head_ids + state_ids + [SEP]

    out = session.run(None, {
        "input_ids":      np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        "marker_pos":     np.array([positions], dtype=np.int64),
        "marker_mask":    np.ones((1, len(positions)), dtype=bool),
        "qtype":          np.array([qtype], dtype=np.int64),  # 0=choice 1=score 2=noul
    })[0][0]

    import numpy as np
    p = np.exp(out - out.max()); p = p / p.sum()
    return dict(zip(options, p.round(4).tolist()))

print(predict(
    "您好，我是XX快递客服，您有一个包裹在运输途中丢失，请点击链接理赔。",
    "Is this a scam?",
    ["false", "true"],
    qtype=2,
))
# {'false': 0.0006, 'true': 0.9994}
```

A full reference client (routing, temperature scaling, batch mode, HTTP server)
is in the GitHub repo: https://github.com/chaceli/laya-scam-detector

## Limitations

- **Not a chatbot.** It answers only the typed questions you supply.
- **Confidence is not ground truth.** A 0.95 probability does not mean 95%
  correctness on your data — calibrate thresholds on your own traffic.
- **Category label frequency is imbalanced.** `crypto_scam` and `marketing`
  had only a handful of training examples; expect low recall for those.
- **English was not the fine-tuning target.** English text is still served
  well, but if English accuracy is the goal, fine-tune on English data.
- **State can contain injection.** Malicious content inside the input can sway
  the judgement; never treat this as your only line of defence.
- **Type safety ≠ correctness.** The output space is locked to your options,
  but the model can still choose the wrong one.

## Intended use

Screening, triage, routing, and guardrails for scam/spam detection in
Chinese and English text. Not for autonomous, irreversible actions
(payments, account deletion, compliance approvals) without human review.

## License

Apache-2.0, inheriting from the Convai Innovations base model.
