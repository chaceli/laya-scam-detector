#!/usr/bin/env bash
# One-shot v2 pipeline resume: Task 13 build -> invariant check -> Task 15 train (background).
# Usage: bash scripts/run_v2_pipeline.sh
# Prereq: datasets/raw/ccl2023/{train.json,test.json} in place (CodaLab or AI Studio mirror).
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_JSON="datasets/raw/ccl2023/train.json"
TEST_JSON="datasets/raw/ccl2023/test.json"

[ -f "$TRAIN_JSON" ] || { echo "MISSING: $TRAIN_JSON (download from CodaLab or AI Studio dataset 215947)"; exit 2; }
[ -f "$TEST_JSON" ] || { echo "MISSING: $TEST_JSON"; exit 2; }
source .venv/bin/activate

echo "=== Task 13: v2 dataset build (~5 min) ==="
python scripts/build_dataset.py 2>&1 | tee reports/build_v2.log | tail -30

echo ""
echo "=== Invariant verification ==="
python - <<'PYEOF'
import json
from collections import Counter
train = [json.loads(l) for l in open('datasets/training/train.jsonl')]
eval_ids = {json.loads(l)['id'] for l in open('datasets/hardneg_eval.jsonl')}
pos = [r for r in train if r['is_scam'] == 1]
c = Counter(r['category'] for r in pos)
gen = sum(1 for r in train if r.get('source', '').startswith(('hn_', 'nb_', 'contrastive')))
leak = [r['id'] for r in train if r['id'] in eval_ids]
print('n', len(train), '| pos', len(pos), '| rebate', round(c['rebate_scam']/len(pos), 3), '| gen', round(gen/len(train), 3), '| leak', len(leak))
assert len(train) == 45000 and not leak
assert c['rebate_scam']/len(pos) >= 0.25 - 0.001
assert gen/len(train) <= 0.30 + 0.001
print('INVARIANTS OK')
PYEOF

echo ""
echo "=== Task 15: launching v2 training (background, est. 4-5h on MPS) ==="
nohup env HF_HUB_DISABLE_XET=1 python -u scripts/train_local_lora.py \
  --train datasets/training/train.jsonl --val datasets/training/val.jsonl \
  --max-samples 45000 --epochs 4 --batch-size 8 --grad-accum 2 \
  --output models/laya-lora-finetuned-v2 > reports/train_v2.log 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"
echo "Log: tail -f reports/train_v2.log"
echo ""
echo "After training completes (log shows 'Training done'), run:"
echo "  HF_HUB_DISABLE_XET=1 python scripts/export_local_onnx.py --state models/laya-lora-finetuned-v2/model_state.pt --output models/laya-onnx-multilingual-finetuned-v2"
echo "  python scripts/run_full_eval.py   # Task 16 four-gate acceptance"