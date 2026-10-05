# v3 实施计划：用真实 CCL2023 修 Gate4，保住 FPR

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用已到位的真实 CCL2023 数据从头重训 v3，把 `rebate_scam` 类别召回从 0 拉到 ≥0.80，同时保持 Gate1 FPR ≤0.02、Gate2/Gate3 不回退。

**Architecture:** 三处改动：(1) CCL 数据落位到 loader 期望路径；(2) 新增同源留出切分脚本产出 `datasets/ccl_rebate_eval.jsonl`；(3) `build_dataset` 移除合成 rebate 接线、把 CCL 留出纳入文本级排除。随后跑既有训练/导出/验收管线，产物落 v3 目录，与 v2 逐门对比。

**Tech Stack:** Python 3.14, ONNX Runtime, Laya SDK（mmBERT-base），LoRA (peft)，pytest。数据侧纯标准库。

## Global Constraints

- `TRAIN_TARGET=45000`、`POS_FRACTION=0.45` → `n_pos=20250`
- `REBATE_FLOOR=0.25`（rebate ≥ 5,062 正样本）、`CCL_NONREBATE_CAP=0.20`、`PER_CATEGORY_CAP=0.30`、`GEN_CAP=0.30`、`TELE_POS_CAP=0.10`
- 训练超参不变：LoRA r=8、4 epoch、batch 8 × grad-accum 2、lr 2e-4、max_len 256、MPS
- 评测集排除一律按**归一化文本**（`dataset_mix._norm_text`），不得用 id（`stable_id` 含 source）
- `datasets/raw/`、`datasets/training/`、`models/` 已 gitignore；`datasets/ccl2023/` 需新增 gitignore
- git 提交用 `/Library/Developer/CommandLineTools/usr/bin/git`（系统 `/usr/bin/git` 被 Xcode 许可拦截），conventional commits，每条加 Sisyphus 署名
- 验收门槛：Gate4 同源 ≥0.80 且 Gate1 ≤0.02 且 Gate2/Gate3 相对 v2 抖动 ≤0.01

---

## File Structure

| 文件 | 责任 | 动作 |
|---|---|---|
| `datasets/raw/ccl2023/{train,test}.json` | CCL 原始数据（loader 数据源） | 放置（从 `datasets/ccl2023/` 复制） |
| `.gitignore` | 忽略用户放置目录 | 修改（加 `datasets/ccl2023/`） |
| `scripts/build_ccl_holdout.py` | 切同源留出 → `datasets/ccl_rebate_eval.jsonl` | 新建 |
| `tests/test_ccl_holdout.py` | 切分不变量测试 | 新建 |
| `scripts/build_dataset.py` | 移除合成 rebate、纳入 CCL 留出排除 | 修改 |
| `datasets/synthetic/rebate_scam.jsonl` | 旧合成数据 | 删除 |
| `scripts/run_v3_pipeline.sh` | v3 一键：build → 不变量 → 训练 | 新建（基于 run_v2_pipeline.sh） |
| `reports/gate_v3_result.md` | v3 四门 + v2 对比 | 新建 |
| `AGENTS.md` | 更新 Gate4 状态、CCL 就位 | 修改 |

---

### Task 1: CCL 数据落位 + gitignore

**Files:**
- Create: `datasets/raw/ccl2023/train.json`, `datasets/raw/ccl2023/test.json`（复制）
- Modify: `.gitignore`

**Interfaces:**
- Produces: `datasets/raw/ccl2023/{train,test}.json` 就位，供 `load_ccl2023` 读取

- [ ] **Step 1: 复制数据到 loader 期望路径**

```bash
mkdir -p datasets/raw/ccl2023
cp datasets/ccl2023/train.json datasets/raw/ccl2023/train.json
cp datasets/ccl2023/test.json  datasets/raw/ccl2023/test.json
ls -lh datasets/raw/ccl2023/
```
Expected: 两个文件（~85M / ~10M）

- [ ] **Step 2: gitignore 用户放置目录**

在 `.gitignore` 末尾追加（保留已有行）：
```
# CCL2023 原始数据（用户在 datasets/ccl2023 放置；副本进 datasets/raw/ 供 loader）
datasets/ccl2023/
```

- [ ] **Step 3: 验证 loader 能加载且无 unmapped 标签**

```bash
source .venv/bin/activate
python - <<'PY'
import sys; sys.path.insert(0, 'scripts')
from dataset_loaders import load_ccl2023
from collections import Counter
rows = load_ccl2023()
print("总行:", len(rows))
print("source:", {r['source'] for r in rows})
print("category:", dict(Counter(r['category'] for r in rows).most_common()))
assert len(rows) > 80000, f"预期 >80k, 实得 {len(rows)}"
assert rows[0]['source'] == 'ccl2023'
print("OK")
PY
```
Expected: `总行: 82210`，rebate_scam 28,367，无 RuntimeError

- [ ] **Step 4: 确认 datasets/ 下无新增未跟踪大文件**

```bash
GIT=/Library/Developer/CommandLineTools/usr/bin/git
$GIT status --short | grep ccl2023 || echo "(datasets/ccl2023 已被忽略, 符合预期)"
```

- [ ] **Step 5: Commit**

```bash
GIT=/Library/Developer/CommandLineTools/usr/bin/git
$GIT add .gitignore
$GIT commit -m "chore(data): ignore user-placed datasets/ccl2023 raw dir" -m "CCL2023 is now available; the working copy lives in datasets/ccl2023 (gitignored) and is copied to datasets/raw/ccl2023 for load_ccl2023. Raw data stays out of git." -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 2: CCL 同源留出切分脚本（TDD）

**Files:**
- Create: `scripts/build_ccl_holdout.py`
- Create: `tests/test_ccl_holdout.py`
- Produces: `datasets/ccl_rebate_eval.jsonl`

**Interfaces:**
- Consumes: `datasets/raw/ccl2023/train.json`；`schemas.scam_categories.CCL2023_LABEL_MAP`
- Produces: `build_holdout(rows, seed=42, n_rebate=1000, n_other=50) -> tuple[list[dict], list[dict]]`
  返回 (holdout_rows, remaining_rows)；holdout 行的形态为评测格式
  `{"id","type","state","language","expected_risk","category","source"}`

- [ ] **Step 1: 写失败测试**

`tests/test_ccl_holdout.py`：
```python
"""CCL 同源留出切分的不变量测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_ccl_holdout import build_holdout  # noqa: E402


def _mk(case_id, label):
    return {"案件编号": case_id, "案情描述": f"案件{case_id}的描述文本", "案件类别": label}


def test_counts_and_disjoint():
    rows = [_mk(i, "刷单返利类") for i in range(3000)]
    rows += [_mk(10000 + i, "虚假网络投资理财类") for i in range(200)]
    hold, rest = build_holdout(rows, seed=42, n_rebate=1000, n_other=50)
    reb = [r for r in hold if r["category"] == "rebate_scam"]
    inv = [r for r in hold if r["category"] == "investment_scam"]
    assert len(reb) == 1000
    assert len(inv) == 50
    # 留出与剩余按 案件编号 不交叉
    h_ids = {r["id"] for r in hold}
    r_ids = {r["id"] for r in rest}
    assert not (h_ids & r_ids)


def test_deterministic_by_seed():
    rows = [_mk(i, "刷单返利类") for i in range(3000)]
    h1, _ = build_holdout(rows, seed=42)
    h2, _ = build_holdout(rows, seed=42)
    assert [r["id"] for r in h1] == [r["id"] for r in h2]
    h3, _ = build_holdout(rows, seed=7)
    assert [r["id"] for r in h1] != [r["id"] for r in h3]


def test_holdout_row_shape():
    rows = [_mk(i, "刷单返利类") for i in range(2000)]
    hold, _ = build_holdout(rows, seed=1, n_rebate=100)
    r = hold[0]
    assert set(r) >= {"id", "type", "state", "language", "expected_risk", "category", "source"}
    assert r["type"] == "single" and r["state"]
    assert r["expected_risk"] == 4 and r["language"] == "zh"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_ccl_holdout.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'build_ccl_holdout'`）

- [ ] **Step 3: 实现脚本**

`scripts/build_ccl_holdout.py`：
```python
"""切 CCL2023 同源留出集，供 v3 的 Gate4 评测（主指标）。

同源留出：1000 条刷单返利 + 其余 11 类各 50 条。按 案件编号 切分（不跨训练/留出两侧），
seed 固定可复现。输出评测格式，category 用 CCL2023_LABEL_MAP 的真值。

用法: python scripts/build_ccl_holdout.py [--seed 42]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import CCL2023_LABEL_MAP  # noqa: E402

SRC = Path("datasets/raw/ccl2023/train.json")
OUT = Path("datasets/ccl_rebate_eval.jsonl")
REBATE_LABEL = "刷单返利类"


def build_holdout(rows, seed=42, n_rebate=1000, n_other=50):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in rows:
        if r.get("案件类别") in CCL2023_LABEL_MAP:
            by_label[r["案件类别"]].append(r)

    hold_src, rest_ids = [], set()
    # 每类洗牌后取前 k（rebate 取 n_rebate，其它取 n_other）
    for label, items in by_label.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        k = n_rebate if label == REBATE_LABEL else n_other
        k = min(k, len(shuffled))
        hold_src.extend(shuffled[:k])
        for r in shuffled[k:]:
            rest_ids.add(r["案件编号"])

    hold = [{
        "id": f"ccl_holdout_{r['案件编号']}",
        "type": "single",
        "state": r["案情描述"],
        "language": "zh",
        "expected_risk": 4,
        "category": CCL2023_LABEL_MAP[r["案件类别"]],
        "source": "ccl2023_holdout",
    } for r in hold_src]

    held_ids = {r["案件编号"] for r in hold_src}
    rest = [r for r in rows if r.get("案件编号") not in held_ids]
    return hold, rest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = json.load(open(SRC))
    hold, rest = build_holdout(rows, seed=args.seed)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in hold))
    from collections import Counter
    print(f"✓ {OUT}: {len(hold)} 条")
    print("  category:", dict(Counter(r["category"] for r in hold).most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_ccl_holdout.py -v`
Expected: 3 passed

- [ ] **Step 5: 实跑生成留出集**

```bash
python scripts/build_ccl_holdout.py
```
Expected: `✓ datasets/ccl_rebate_eval.jsonl: 1550 条`，rebate_scam 1000

- [ ] **Step 6: Commit**

```bash
GIT=/Library/Developer/CommandLineTools/usr/bin/git
$GIT add scripts/build_ccl_holdout.py tests/test_ccl_holdout.py datasets/ccl_rebate_eval.jsonl
$GIT commit -m "feat(eval): carve same-source CCL holdout for Gate4" -m "Splits CCL2023 train by case id into 1000 rebate + 50-per-class holdout (1550 rows), seed-fixed and disjoint from the training remainder. Same-source view for the Gate4 primary metric; the 42 cross-register cases remain as a generalisation check." -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 3: build_dataset 移除合成 rebate + 纳入 CCL 留出排除（TDD）

**Files:**
- Modify: `scripts/build_dataset.py:341-390`
- Delete: `datasets/synthetic/rebate_scam.jsonl`
- Test: `tests/test_mix_invariants.py`（追加）或 `tests/test_build_dataset_rebate.py`（新建）

**Interfaces:**
- Consumes: `datasets/ccl_rebate_eval.jsonl`（Task 2）、`load_ccl2023`（Task 1）
- Produces: `main()` 构建的 `datasets/training/train.jsonl` 用真实 CCL rebate，无合成源

- [ ] **Step 1: 写失败测试**

`tests/test_build_dataset_rebate.py`：
```python
"""build_dataset 的 rebate 接线：真实 CCL 取代合成，CCL 留出必排除。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_mix import _norm_text  # noqa: E402
import json


def test_synthetic_rebate_file_removed():
    assert not Path("datasets/synthetic/rebate_scam.jsonl").exists(), \
        "合成 rebate 应已删除"


def test_ccl_holdout_excluded_from_train():
    """若构建产物已存在，CCL 留出文本不得出现在训练集。"""
    from build_ccl_holdout import build_holdout
    hold, _ = build_holdout(json.load(open("datasets/raw/ccl2023/train.json")))
    hold_norm = {_norm_text(r["state"]) for r in hold}
    train_path = Path("datasets/training/train.jsonl")
    if not train_path.exists():
        return  # 构建未跑则跳过（Task 4 会实跑）
    train_norm = {_norm_text(json.loads(l)["text"]) for l in train_path.open()}
    assert not (hold_norm & train_norm), f"CCL 留出泄漏 {len(hold_norm & train_norm)} 条"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_build_dataset_rebate.py -v`
Expected: FAIL（`test_synthetic_rebate_file_removed` —— 文件仍存在）

- [ ] **Step 3: 删除合成数据文件**

```bash
GIT=/Library/Developer/CommandLineTools/usr/bin/git
$GIT rm datasets/synthetic/rebate_scam.jsonl
```

- [ ] **Step 4: 修改 build_dataset.py**

把 341-390 段的合成 rebate 读取替换为仅"CCL 留出排除"（不再读合成文件）：
```python
    # —— v2/v3: rebate_scam 现由真实 CCL2023 提供（合成已移除）——
    # 排除集：42 条跨源案件 + CCL 同源留出。一律按归一化文本排除（id 含 source，
    # 跨源永不匹配）。见 docs/plans/2026-10-05-v3-real-ccl-rebate-design.md §3。
    rebate_eval_texts: set[str] = set()
    for _eval_file in ("datasets/rebate_eval_real.jsonl",
                       "datasets/ccl_rebate_eval.jsonl"):
        _p = Path(_eval_file)
        if _p.exists():
            with _p.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        rebate_eval_texts.add(r.get("text") or r.get("state"))
            print(f"  ✓ {_eval_file}: {len(rebate_eval_texts):,} 排除文本累计")
```
并把 `compose_train` 调用改为（去掉 `rebate_rows`）：
```python
    from dataset_mix import compose_train
    train, mix_stats = compose_train(
        real + synth + hardneg_rows,
        eval_ids=hardneg_eval_ids,
        eval_texts=rebate_eval_texts,
    )
```
同时删除 `rebate_rows` 的定义与 `for r in rebate_rows:` 的 id 填充循环（383 行附近）。

- [ ] **Step 5: 跑测试确认通过（合成已删）**

Run: `python -m pytest tests/test_build_dataset_rebate.py -v`
Expected: `test_synthetic_rebate_file_removed` PASS；留出测试因训练集未重建而 skip/return

- [ ] **Step 6: 跑既有不动量测试确认无回归**

Run: `python -m pytest tests/test_mix_invariants.py tests/test_dataset_mix.py -q`
Expected: 全通过

- [ ] **Step 7: Commit**

```bash
GIT=/Library/Developer/CommandLineTools/usr/bin/git
$GIT add scripts/build_dataset.py tests/test_build_dataset_rebate.py
$GIT commit -m "feat(data): source rebate from real CCL, drop synthetic + exclude CCL holdout" -m "build_dataset no longer reads the synthetic rebate file; it now excludes both the 42 cross-register cases and the new CCL same-source holdout by normalised text. Removes rebate_rows wiring so compose_train takes real CCL rebate via load_ccl2023." -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 4: 构建数据集 + 不变量校验

**Files:**
- Produces: `datasets/training/{train,val,test}.jsonl`

**Interfaces:**
- Consumes: Task 1-3 的 loader/排除改动
- Produces: 45k 训练集，供 Task 5 训练

- [ ] **Step 1: 实跑 build_dataset**

```bash
source .venv/bin/activate
python scripts/build_dataset.py 2>&1 | tail -40
```
Expected: `Split: train=45,000`；`rebate_scam` 出现在 Train category distribution；无 CCL unmapped 报错

- [ ] **Step 2: 校验不变量（rebate 占比 / 泄漏 / 生成占比）**

```bash
python - <<'PY'
import json, sys
sys.path.insert(0,'scripts')
from dataset_mix import _norm_text
from collections import Counter
train=[json.loads(l) for l in open('datasets/training/train.jsonl')]
pos=[r for r in train if r['is_scam']==1]
c=Counter(r['category'] for r in pos)
print('rebate share:', round(c['rebate_scam']/len(pos),3), '(need >=0.25)')
assert c['rebate_scam']/len(pos) >= 0.25-0.001
# 泄漏检查：CCL 留出 + 42 集
ev=set()
for f in ('datasets/ccl_rebate_eval.jsonl','datasets/rebate_eval_real.jsonl'):
    for l in open(f):
        r=json.loads(l); ev.add(_norm_text(r.get('text') or r.get('state')))
tn={_norm_text(r['text']) for r in train}
print('leak:', len(ev & tn), '(need 0)')
assert not (ev & tn)
# CCL 是否真进训练
print('ccl2023 正样本:', sum(1 for r in pos if r['source']=='ccl2023'))
print('INVARIANTS OK')
PY
```
Expected: `INVARIANTS OK`，rebate share ≈0.25，leak 0，ccl2023 正样本 ≈9,112

- [ ] **Step 3: 跑完整测试套件**

Run: `python -m pytest -q 2>&1 | tail -3`
Expected: 按 `N passed` 判定（可能 exit 134 的已知原生析构抖动，忽略退出码）

- [ ] **Step 4: Commit**

```bash
GIT=/Library/Developer/CommandLineTools/usr/bin/git
$GIT add datasets/ccl_rebate_eval.jsonl
$GIT commit -m "chore(data): v3 dataset built from real CCL, invariants green" -m "Records the built v3 training set state: rebate floor met from real CCL, zero normalised-text leak from either eval set. datasets/training/ is gitignored, so this commit only tracks the holdout eval artifact." -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 5: 训练 v3 + 导出 ONNX

**Files:**
- Create: `scripts/run_v3_pipeline.sh`
- Produces: `models/laya-lora-finetuned-v3/model_state.pt` → `models/laya-onnx-multilingual-finetuned-v3/`

**Interfaces:**
- Consumes: Task 4 的 45k 训练集
- Produces: v3 ONNX bundle（`model.onnx`、`model.onnx.data`、`tokenizer/`、`categories.json`）

- [ ] **Step 1: 新建 v3 管线脚本**

`scripts/run_v3_pipeline.sh`（基于 run_v2_pipeline.sh，去掉合成前置检查，输出 v3 目录）：
```bash
#!/usr/bin/env bash
# v3: 真实 CCL rebate 从头重训 -> 导出 ONNX。前置: Task 1-4 已完成。
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

[ -f datasets/raw/ccl2023/train.json ] || { echo "MISSING CCL train.json"; exit 2; }
[ -f datasets/ccl_rebate_eval.jsonl ] || { echo "MISSING ccl_rebate_eval.jsonl"; exit 2; }

echo "=== build v3 dataset ==="
python scripts/build_dataset.py 2>&1 | tail -20

echo "=== train v3 (background, ~5h MPS) ==="
nohup env HF_HUB_DISABLE_XET=1 python -u scripts/train_local_lora.py \
  --train datasets/training/train.jsonl --val datasets/training/val.jsonl \
  --max-samples 45000 --epochs 4 --batch-size 8 --grad-accum 2 \
  --output models/laya-lora-finetuned-v3 > reports/train_v3.log 2>&1 &
echo "PID: $!  log: reports/train_v3.log"
```

- [ ] **Step 2: 启动训练**

```bash
bash scripts/run_v3_pipeline.sh
```
Expected: 打印 PID；`reports/train_v3.log` 开始输出 `epoch 1 step ...`

- [ ] **Step 3: 训练完成后导出 ONNX**

```bash
HF_HUB_DISABLE_XET=1 python scripts/export_local_onnx.py \
  --state models/laya-lora-finetuned-v3/model_state.pt \
  --output models/laya-onnx-multilingual-finetuned-v3 2>&1 | tail -10
```
Expected: `✓ ONNX matches PyTorch`，max diff <1e-4，`categories.json (14 classes)`

- [ ] **Step 4: 端到端验证 v3 可加载推理**

```bash
python - <<'PY'
import json, sys; sys.path.insert(0,'.')
from src.laya_onnx import OnnxLayaClient
c=OnnxLayaClient('models/laya-onnx-multilingual-finetuned-v3')
S=json.loads(open('schemas/scam.json').read())
p=c.predict("我被人拉去做刷单返利任务，垫付了三千多", S)
print("is_scam:", round(p['answers']['is_scam']['noul'],3),
      "| cat:", p['answers']['scam_category']['choice'])
PY
```
Expected: `cat: rebate_scam`（v2 此处是 investment_scam）

- [ ] **Step 5: Commit（仅脚本；模型目录 gitignore）**

```bash
GIT=/Library/Developer/CommandLineTools/usr/bin/git
$GIT add scripts/run_v3_pipeline.sh
$GIT commit -m "feat(scripts): v3 pipeline - real-CCL retrain + export" -m "Adds the v3 one-shot pipeline mirroring v2 but sourcing rebate from real CCL and writing to models/*-v3. models/ is gitignored." -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 6: v3 四门验收 + v2 对比

**Files:**
- Modify: `scripts/run_full_eval.py`（把 CCL 留出纳入 holdout）
- Create: `reports/gate_v3_result.md`

**Interfaces:**
- Consumes: v3 ONNX（Task 5）
- Produces: 四门结果 + v2/v3 逐门对比表

- [ ] **Step 1: 把 CCL 留出纳入 eval holdout**

修改 `scripts/run_full_eval.py::ensure_holdout()`：在追加 42 条 `rebate_eval_real.jsonl` 之后，同样追加 `datasets/ccl_rebate_eval.jsonl` 的行（字段 `state`/`category` 已是评测格式）。确保 `parse_category_recall(holdout, "rebate_scam")` 能读到 1,000+ 条。

- [ ] **Step 2: 跑 v3 四门**

```bash
source .venv/bin/activate
python scripts/run_full_eval.py --model-dir models/laya-onnx-multilingual-finetuned-v3 2>&1 | tail -12
```
Expected: 打印四门；`Gate4 rebate_scam 召回` 应 ≥0.80

- [ ] **Step 3: 单独测 42 条跨源口径**

```bash
python - <<'PY'
import json, sys; sys.path.insert(0,'.')
from src.laya_onnx import OnnxLayaClient
S=json.loads(open('schemas/scam.json').read())
c=OnnxLayaClient('models/laya-onnx-multilingual-finetuned-v3')
rows=[json.loads(l) for l in open('datasets/rebate_eval_real.jsonl')]
hit=sum(1 for r in rows if c.predict(r['text'],S)['answers']['scam_category']['choice']=='rebate_scam')
print(f"42 条跨源 rebate 召回: {hit}/{len(rows)} = {hit/len(rows):.3f}")
PY
```

- [ ] **Step 4: 写对比报告 `reports/gate_v3_result.md`**

含：四门 v2 vs v3 表、CCL 同源 rebate 召回、42 条跨源召回、通过判定（Gate4≥0.80 且 Gate1≤0.02 且 Gate2/3 抖动≤0.01）、若 FPR 反弹的诊断说明。

- [ ] **Step 5: Commit**

```bash
GIT=/Library/Developer/CommandLineTools/usr/bin/git
$GIT add scripts/run_full_eval.py reports/gate_v3_result.md reports/eval-*.md reports/results-*.jsonl
$GIT commit -m "test(eval): v3 four-gate result + v2 comparison" -m "Adds the CCL same-source holdout to the eval holdout and reports the v3 gates against v2. Success = rebate recall >=0.80 on the same-source holdout with FPR still <=0.02 and no regression on holdout/handwritten." -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

### Task 7: 文档更新

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: 更新 AGENTS.md**

- Gotchas 里"CCL 未就位"改为"CCL 已就位（`datasets/raw/ccl2023/`），Gate4 由 v3 处理"；
- 记录 v3 结果与 Gate4 状态；
- 保留 flaky-pytest 与归一化文本排除两条既有 gotcha。

- [ ] **Step 2: Commit**

```bash
GIT=/Library/Developer/CommandLineTools/usr/bin/git
$GIT add AGENTS.md
$GIT commit -m "docs(agents): CCL2023 landed, record v3 Gate4 outcome" -m "Updates the rebate section: CCL is now present, synthetic rebate removed, Gate4 measured on the CCL same-source holdout plus the 42 cross-register cases." -m "Ultraworked with [Sisyphus](https://github.com/code-yeongyu/oh-my-openagent)" -m "Co-authored-by: Sisyphus <clio-agent@sisyphuslabs.ai>"
```

---

## Self-Review

**Spec coverage:**
- §2.1 落位 → Task 1 ✓
- §2.2 删合成 → Task 3 ✓
- §2.3 loader → Task 1 Step 3 验证 ✓
- §3.1 CCL 留出 → Task 2 ✓
- §3.2 42 条跨源 → Task 6 Step 3 ✓
- §3.3 Gate4 判定 → Task 6 ✓
- §4 训练 → Task 5 ✓
- §5 验证 → Task 6 ✓
- §6 风险 → Task 6 Step 4（FPR 反弹诊断）✓
- §7 验收 7 项 → 覆盖（CCL 就位 T1 / 删合成+不变量 T3-4 / 零泄漏 T4 / 训练导出 T5 / 报告 T6 / AGENTS T7 / 测试 T4 Step3）✓

**Placeholder scan:** 无 TBD/TODO；代码步骤均含实际代码。Task 5 Step 1 给出完整脚本。

**Type consistency:** `build_holdout(rows, seed, n_rebate, n_other)` 在测试与实现签名一致；留出行字段 `{id,type,state,language,expected_risk,category,source}` 在 Task 2 产出、Task 3/6 消费一致；`_norm_text` 全程复用。

**已知实现注意点（给执行者）：**
- Task 3 Step 4 的具体行号会因插入而偏移，以"替换合成 rebate 读取段 + compose_train 调用"为准。
- Task 6 Step 1 的 `ensure_holdout` 现只追加 42 条；需并行追加 CCL 留出（两文件字段名不同：42 集用 `text`，CCL 留出用 `state`）。
- `datasets/ccl_rebate_eval.jsonl` 需入库（评测证据），Task 2/Task 4 提交它。
