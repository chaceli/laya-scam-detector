#!/usr/bin/env bash
# One-shot v2 pipeline resume: Task 13 build -> invariant check -> Task 15 train (background).
# Usage: bash scripts/run_v2_pipeline.sh
# Prereq: datasets/synthetic/rebate_scam.jsonl (run gen_rebate_synthetic.py).
# CCL2023 is no longer required: rebate_scam comes from the template generator.
set -euo pipefail
cd "$(dirname "$0")/.."

REBATE_SYNTH="datasets/synthetic/rebate_scam.jsonl"

[ -f "$REBATE_SYNTH" ] || { echo "MISSING: $REBATE_SYNTH (run: python scripts/gen_rebate_synthetic.py)"; exit 2; }
source .venv/bin/activate

echo "=== Task 13: v2 dataset build (~5 min) ==="
python scripts/build_dataset.py 2>&1 | tee reports/build_v2.log | tail -30

echo ""
echo "=== Invariant verification ==="
python - <<'PYEOF'
import json, sys
sys.path.insert(0, 'scripts')
from dataset_mix import _norm_text
from collections import Counter

train = [json.loads(l) for l in open('datasets/training/train.jsonl')]
pos = [r for r in train if r['is_scam'] == 1]
c = Counter(r['category'] for r in pos)
gen = sum(1 for r in train if r.get('source', '').startswith(('hn_', 'nb_', 'contrastive')))

# Leak check is by normalised text, not id: stable_id hashes the source in, so
# eval rows (source=rebate_eval_real) never share an id with training rows
# (source=syn_rebate) and an id-only check would silently pass.
hardneg_eval = {json.loads(l)['state'] for l in open('datasets/hardneg_eval.jsonl')}
rebate_eval = {json.loads(l)['text'] for l in open('datasets/rebate_eval_real.jsonl')}
train_norm = {_norm_text(r['text']) for r in train}
leak = ({_norm_text(t) for t in hardneg_eval | rebate_eval} & train_norm)

print('n', len(train), '| pos', len(pos),
      '| rebate', round(c['rebate_scam']/len(pos), 3),
      '| gen', round(gen/len(train), 3), '| leak', len(leak))
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