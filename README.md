# Laya Scam-Phrase Detector

Local ONNX Runtime deployment of the Laya decision model for scam-phrase risk evaluation. No PyTorch dependency.

See `research/laya-jev-research-report.md` for background on Laya vs Jev.

## Results summary (zero-shot, no fine-tuning)

### Handwritten dataset (38 samples, 20 ZH + 10 EN + 8 multi-turn)

| Metric | Value | Target | Status |
|---|---|---|---|
| is_scam accuracy | 0.868 | >=0.75 | pass |
| is_scam precision | 0.833 | -- | -- |
| is_scam **recall** | **1.000** | -- | **no missed scams** |
| is_scam F1 | 0.909 | >=0.65 | pass |
| risk_level MAE | 1.78 | <=1.5 | slightly over |
| scam_category accuracy | 0.605 | >=0.30 | exceeded 2x |
| p50 latency | 575ms | -- | -- |
| 0 errors | | | -- |

Per-language: en 0.923 / zh 0.840 accuracy.

### Public SMS Spam Collection (400 samples, third-party)

| Metric | Value |
|---|---|
| is_scam accuracy | 0.905 |
| is_scam precision | 0.872 |
| is_scam recall | 0.950 |
| is_scam F1 | 0.909 |
| p50 latency | 507ms |

**Zero-shot baseline is robust** — recall of 1.0 on handwritten and 0.95 on public data means Laya catches essentially all scams without any fine-tuning.

## Post-evaluation decision

Per design doc section 8.4 thresholds:
- is_scam F1 0.909 >= 0.7 — zero-shot is **above threshold** for production use
- Recall 1.000 / 0.950 — **critical security metric** (catching scams) is excellent
- Risk MAE 1.78 slightly above target — fine for binary gating, less precise for ordinal
- Category accuracy 0.605 — exceeds target by 2x

**Conclusion:** Zero-shot baseline is **production-ready for binary scam detection** (is_scam primitive). Micro-tuning could improve ordinal scoring (risk_level) but is **not required** for the security-critical classification task.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/download_models.sh
```

## Usage

```bash
# Single prediction (CLI)
python main.py --predict "您好，我是XX快递客服..." --questions schemas/scam.json

# Route check (no model load)
python main.py --route "中文字符串"

# Batch evaluation
python main.py --eval --input datasets/eval.jsonl --output reports/

# With public SMS Spam dataset (third-party)
python main.py --eval --input datasets/public.jsonl --output reports/
```

### Web Playground

Interactive UI over the fine-tuned model — paste text, pick which of the
three Laya primitives to run, and inspect verdicts, probability
distributions, latency, and routing.

```bash
pip install -e ".[serve]"
PYTHONPATH=. python -m uvicorn server.app:app --port 8000
# open http://127.0.0.1:8000
```

Features:
- Input text (Chinese/English) with 4 one-click preset samples
- Pick any subset of the three primitives: `noul` (scam yes/no),
  `score` (1-5 risk), `choice` (13 scam categories)
- Switch model: auto-route / English base / fine-tuned multilingual
- "Advanced" panel to edit instructions and criteria inline
- Results per primitive: verdict, full probability distribution bars,
  latency, and the resolved checkpoint

API: `GET /api/health`, `GET /api/samples`, `GET /api/defaults`,
`POST /api/predict`.

## Phase 2: Fine-tuned multilingual model (complete)

The zero-shot baseline struggled with Chinese (accuracy 0.840 vs 0.923 English)
and the original 8-class schema. We fine-tuned `convaiinnovations/laya-multilingual`
(mmBERT-base, 322M, 100+ languages) with LoRA on multi-source scam data —
**entirely on-device on an Apple M4 Pro** (no cloud GPU).

### Training setup

| Item | Value |
|---|---|
| Base model | `convaiinnovations/laya-multilingual` (mmBERT-base, 322M) |
| Method | LoRA r=8 on `Wqkv`+`Wo` (1.15M trainable / 323M = 0.36%) |
| Datasets | FGRC-SCD (62k ZH), scamshield (37k EN), ealvaradob (78k EN), FBS_SMS (14k ZH), UCI SMS Spam (400), synthetic seed (35) |
| Training samples | 15,000 balanced (stratified) |
| Hardware | **Apple M4 Pro, MPS backend** (peak 1.3 GB of 24 GB) |
| Runtime | **73 minutes** (3 epochs) |

### 13-class schema

Extended from 8 to 13 categories: `benign`, `phishing`, `crypto_scam`,
`investment_scam`, `lottery_scam`, `job_scam`, `loan_scam`, `impersonation`,
`romance_scam`, `delivery_fraud`, `marketing`, `adult_content`, `spam_general`.

### Results — all acceptance targets met

600-sample Chinese holdout (from the training data's held-out test split):

| Metric | Target | Achieved |
|---|---|---|
| Chinese is_scam accuracy | >= 0.90 | **0.967** |
| is_scam recall | >= 0.95 | **0.957** |
| 13-class accuracy | >= 0.70 | **0.810** |
| p50 latency (M4 Pro CPU) | — | **125 ms** |
| p95 latency | — | 228 ms |

Handwritten set (38 samples) — before → after:

| Metric | English baseline | Fine-tuned multilingual |
|---|---|---|
| is_scam accuracy | 0.868 | **0.921** |
| is_scam precision | 0.833 | **0.893** |
| is_scam F1 | 0.909 | **0.943** |
| 13-class accuracy | 0.500 | **0.632** |
| false positives | 5 | **3** |
| p50 latency | 652 ms | **131 ms** |

Per-language accuracy on the handwritten set: en 0.923, zh 0.840 → **0.920**.

The latency win is structural: mmBERT's 256k vocabulary tokenizes Chinese at
~1.5 chars/token, where the English ModernBERT vocabulary shredded it.

### Reproduce

```bash
# Phase 1: data prep
python scripts/fetch_datasets.py       # download 5 public datasets (~800MB)
python scripts/build_dataset.py         # build datasets/training/*.jsonl (13-class)
python scripts/build_kaggle_subset.py  # 30k stratified subset

# Phase 2: train locally on Apple Silicon (MPS)
HF_HUB_DISABLE_XET=1 python scripts/train_local_lora.py \
  --train datasets/training/subset30k_train.jsonl \
  --val datasets/training/subset3k_val.jsonl \
  --max-samples 15000 --epochs 3 --batch-size 8 --grad-accum 2

# Phase 3: export ONNX
HF_HUB_DISABLE_XET=1 python scripts/export_local_onnx.py

# Phase 4: evaluate
python main.py --eval --input datasets/eval.jsonl --output reports/
python scripts/check_acceptance.py reports/<latest>.md
```

An optional Kaggle path (`kaggle/`) is kept as a cloud fallback.

## Status

All phases complete.

- Phase 1 (data prep + 13-class schema): complete
- Phase 2 (local MPS LoRA training): complete — 73 min, targets met
- Phase 3 (ONNX export): complete — PyTorch vs ONNX diff 3.3e-06
- Phase 4 (eval + integration): complete — acceptance 3/3

Tests: 98 passing, 2 skipped (network-gated tokenization probe).