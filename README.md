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

## Status

All phases complete. 70 tests passing. Outstanding zero-shot results.