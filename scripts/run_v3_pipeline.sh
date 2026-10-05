#!/usr/bin/env bash
# v3: 真实 CCL rebate 从头重训 -> 导出 ONNX。前置: CCL 数据就位 + 留出集已生成。
# 用法: bash scripts/run_v3_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

[ -f datasets/raw/ccl2023/train.json ] || { echo "MISSING datasets/raw/ccl2023/train.json"; exit 2; }
[ -f datasets/ccl_rebate_eval.jsonl ] || { echo "MISSING datasets/ccl_rebate_eval.jsonl (run build_ccl_holdout.py)"; exit 2; }

echo "=== build v3 dataset ==="
python scripts/build_dataset.py 2>&1 | tail -20

echo ""
echo "=== invariant check ==="
python - <<'PYEOF'
import json, sys
sys.path.insert(0, 'scripts')
from dataset_mix import _norm_text
from collections import Counter

train = [json.loads(l) for l in open('datasets/training/train.jsonl')]
pos = [r for r in train if r['is_scam'] == 1]
c = Counter(r['category'] for r in pos)
ev = set()
for f in ('datasets/ccl_rebate_eval.jsonl', 'datasets/rebate_eval_real.jsonl'):
    for l in open(f):
        r = json.loads(l)
        ev.add(_norm_text(r.get('text') or r.get('state')))
tn = {_norm_text(r['text']) for r in train}
print('n', len(train), '| pos', len(pos),
      '| rebate', round(c['rebate_scam']/len(pos), 3), '| leak', len(ev & tn))
assert len(train) == 45000 and not (ev & tn)
assert c['rebate_scam']/len(pos) >= 0.25 - 0.001
print('INVARIANTS OK')
PYEOF

echo ""
echo "=== train v3 (background, ~5h MPS) ==="
nohup env HF_HUB_DISABLE_XET=1 python -u scripts/train_local_lora.py \
  --train datasets/training/train.jsonl --val datasets/training/val.jsonl \
  --max-samples 45000 --epochs 4 --batch-size 8 --grad-accum 2 \
  --output models/laya-lora-finetuned-v3 > reports/train_v3.log 2>&1 &
echo "PID: $!  log: reports/train_v3.log"
echo ""
echo "After training: export + gates"
echo "  HF_HUB_DISABLE_XET=1 python scripts/export_local_onnx.py --state models/laya-lora-finetuned-v3/model_state.pt --output models/laya-onnx-multilingual-finetuned-v3"
echo "  python scripts/run_full_eval.py --model-dir models/laya-onnx-multilingual-finetuned-v3"
