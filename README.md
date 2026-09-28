# Laya Scam-Phrase Detector

Local ONNX Runtime deployment of the Laya decision model for scam-phrase risk evaluation. No PyTorch dependency.

See `research/laya-jev-research-report.md` for background on Laya vs Jev.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/download_models.sh
python main.py --predict "您好，我是XX快递客服..." --questions schemas/scam.json
```

## Status

Phase 1: project skeleton (this commit).