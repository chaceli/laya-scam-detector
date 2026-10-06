# Laya 反诈模型 v3 增训复现指南

> 本文记录 v3 增训（用真实 CCL2023 数据修复 `rebate_scam` 类别）的完整过程、原理、脚本、产物、
> 对比数据与部署方法，供他人复现与借鉴。
>
> 关联文档：设计 `docs/plans/2026-10-05-v3-real-ccl-rebate-design.md`、实施计划
> `docs/plans/2026-10-05-v3-real-ccl-rebate-impl.md`；验收 `reports/gate_v3_result.md`；
> 逐条对比 `reports/v2-vs-v3-comparison.md`。

---

## 0. TL;DR

| 项 | 值 |
|---|---|
| 目标 | 修复 14 类体系里 `rebate_scam`（刷单返利）类别召回 = 0 的问题 |
| 手段 | 用真实 CCL2023 案件数据取代合成数据，**从头重训 v3**（配方不变） |
| 结果 | 四门验收 **4/4**；Gate4 rebate 召回 **0.000 → 0.950** |
| 训练成本 | Apple M4 Pro MPS，4 epoch / 45k 样本，**约 5 小时** |
| 产物 | LoRA state → ONNX(fp32) → ONNX(fp16 643MB) → HF 模型仓 → HF Static Space |

---

## 1. 原理：为什么要增训，以及为什么必须用真实数据

### 1.1 问题现象
v2 模型在**真实**刷单返利案件上，`choice`（14 类分类）头**从不输出** `rebate_scam`：
42 条真实案件里 0 条判对，52% 被误判为 `investment_scam`、27% 判为 `benign`。
但同一模型对**合成**的招募广告文本（"招聘刷单员，日结佣金，先垫付后返利"）却能正确输出
`rebate_scam`（0.94）。

### 1.2 根因：训练分布 ≠ 真实分布（register mismatch）
- **训练数据**是模板生成的"**招募广告**"体（recruiter-ad register）：招聘口吻、短句、话术模板
- **真实案件**是"**受害人/警方笔录**"体（case-narrative register）：叙述口吻、长文本、案情复盘

模型学到的是"广告体 → rebate"的表面关联，而非"刷单返利 = 诈骗"的**语义**，因此在真实笔录上
完全失效。用一句话概括：**合成数据能训练类别，但训练不出跨语域的泛化。**

### 1.3 为什么"改输出/调阈值"救不了
诊断证据：对 42 条真实案件，`choice` 头给 `rebate_scam` 的**平均概率仅 0.0013**，而
`investment_scam` 是 0.651。这不是"被微弱压过"，而是该类**几乎不激活**。因此：
- 类别先验重加权 ✗（只能重排已有概率质量，无法凭空造出）
- 阈值微调 / argmax 重排 ✗（概率质量≈0）
- **唯一有效路径：用真实数据重训**（本方案）；或改分类体系（需产品决策，本方案未采纳）

### 1.4 一个关键区分：这是"分类"失败，不是"检测"失败
即便 v2，`noul`（是否诈骗）头在真实案件上召回已达 0.881——**模型知道这是诈骗，只是归类错**。
v3 修的是"归类"，不是"发现"。

---

## 2. 方法

### 2.1 整体流程

```
CCL2023 原始数据
      │  build_dataset.py（dataset_loaders.load_ccl2023）
      ▼
45k 训练集（compose_train 按配方组装：rebate floor + 三类 caps）
      │  train_local_lora.py（LoRA 微调 mmBERT）
      ▼
LoRA state (model_state.pt)
      │  export_local_onnx.py
      ▼
ONNX fp32 ──quantize_onnx_fp16.py──▶ ONNX fp16 (643MB)
      │                                    │
      │  run_full_eval.py + check_acceptance.py（四门）    │ push_hf_model.py
      ▼                                    ▼
reports/gate_v3_result.md            HF 模型仓
                                          │ push_hf_static.py
                                          ▼
                                      HF Static Space（浏览器内推理）
```

### 2.2 数据配比（**配方不变**，与 v2 一致）

`scripts/dataset_mix.py::compose_train` 的硬约束：

| 常量 | 值 | 含义 |
|---|---|---|
| `TRAIN_TARGET` | 45,000 | 训练集总规模 |
| `POS_FRACTION` | 0.45 | 正样本占比 → n_pos = 20,250 |
| `REBATE_FLOOR` | 0.25 | **rebate_scam ≥ 25% 正样本**（5,062 条）|
| `CCL_NONREBATE_CAP` | 0.20 | CCL 非返利 ≤ 20% 正样本 |
| `PER_CATEGORY_CAP` | 0.30 | 任意单类别 ≤ 30% 正样本 |
| `GEN_CAP` | 0.30 | 生成样本 ≤ 30% 训练总量 |
| `TELE_POS_CAP` | 0.10 | TeleAntiFraud ≤ 10% 正样本 |

关键改动：**rebate_scam 的来源**从"合成模板"换成"真实 CCL2023"，其余配方一字未改
（这正是"保住 v2 FPR 战果"的保证——FPR 由配方而非权重决定）。

实际 CCL 用量：由于 caps 约束，82k CCL 里约 **9,401 条**进入训练（rebate 5,351 + 非返利封顶 4,050）。

### 2.3 训练设置（与 v2 一致）

| 项 | 值 |
|---|---|
| 基座 | `convaiinnovations/laya` multilingual（mmBERT-base, 322M）|
| 方法 | LoRA r=8 / alpha=16，target 为 `Wqkv`+`Wo`（可训练 1.15M / 0.36%）|
| 优化 | 4 epoch，batch 8 × grad-accum 2，lr 2e-4，max_len 256 |
| 硬件 | Apple M4 Pro，MPS 后端 |
| 耗时 | ~5.0 小时（17,875s）|

---

## 3. 数据

### 3.1 CCL2023（真实数据，本方案的核心输入）

- 来源：CCL2023-FCC 评测（CodaLab competition 12558 / 百度 AI Studio dataset 215947）
- 布局：`datasets/ccl2023/{train,test}.json`（用户放置，gitignored），复制到 `datasets/raw/ccl2023/`（loader 读取处）
- `train.json`：**82,210 条**，字段 `案件编号` / `案情描述` / `案件类别`，12 类
- `test.json`：10,276 条，**无标注**（赛事测试集）→ 不可用于评测
- 12 类分布（train）：

| 类别 | 数量 | 映射到 |
|---|---|---|
| 刷单返利类 | **28,367** | rebate_scam |
| 冒充电商物流客服类 | 11,018 | delivery_fraud |
| 虚假网络投资理财类 | 9,469 | investment_scam |
| 贷款、代办信用卡类 | 8,883 | loan_scam |
| 虚假征信类 | 6,771 | loan_scam |
| 虚假购物、服务类 | 5,647 | phishing |
| 冒充公检法及政府机关类 | 3,651 | impersonation |
| 冒充领导、熟人类 | 3,525 | impersonation |
| 网络游戏产品虚假交易类 | 1,723 | spam_general |
| 网络婚恋、交友类 | 1,324 | romance_scam |
| 网黑案件 | 958 | spam_general |
| 冒充军警购物类诈骗 | 874 | impersonation |

映射表见 `schemas/scam_categories.py::CCL2023_LABEL_MAP`（**strict**，未覆盖即报错，不走兜底）。

### 3.2 训练集最终构成（v3，45,000 条）

正样本 20,250 / 负样本 24,750。正样本类别：

```
rebate_scam    5351  (26.4%，满足 floor)
loan_scam      2569
delivery_fraud 2336
investment_scam 2232
impersonation  2156
phishing       2046
spam_general   1883
romance_scam    993
lottery_scam    456
job_scam        195
adult_content    29
crypto_scam       4
```

正样本来源（Top）：CCL2023 真实 9,401、FGRC-SCD 7,879、FBS 1,837、ealvaradob 767…
负样本来源（Top）：ealvaradob 6,485、phishing_email 5,761、生成难负 4,546、ChiFraud 2,475…

### 3.3 评测集（两层口径）

| 集 | 文件 | 规模 | 作用 |
|---|---|---|---|
| CCL 同源留出 | `datasets/ccl_rebate_eval.jsonl` | 1,550（1,000 rebate + 每标签 50）| **Gate4 主指标**（大样本，n=1000 → ±3%）|
| 跨源真实案件 | `datasets/rebate_eval_real.jsonl` | 42（警方/媒体通报，带 `origin`）| 泛化检查（不同语域）|
| 难负样本 | `datasets/hardneg_eval.jsonl` | 675 | Gate1 FPR |
| 手写集 | `datasets/eval.jsonl` | 38 | Gate3 |

**隔离**：三类评测集都按**归一化文本**排除出训练（见 §7.1 踩坑 1），同一 `案件编号` 不跨
训练/留出两侧。

---

## 4. 脚本清单

| 脚本 | 作用 |
|---|---|
| `scripts/dataset_loaders.py` | `load_ccl2023()` 等数据源加载器；读 `datasets/raw/ccl2023/`，按 `CCL2023_LABEL_MAP` 严格映射 |
| `scripts/dataset_mix.py` | `compose_train()` 配比组装：rebate floor + 三类 caps；`_norm_text()` 归一化排除 |
| `scripts/build_dataset.py` | 汇总所有源 → 去重 → `compose_train` → 写 `datasets/training/{train,val,test}.jsonl` |
| `scripts/build_ccl_holdout.py` | 切 CCL 同源留出（1000 rebate + 50/标签）→ `datasets/ccl_rebate_eval.jsonl` |
| `scripts/run_v3_pipeline.sh` | 一键：build → 不变量校验 → 后台训练 |
| `scripts/train_local_lora.py` | LoRA 微调（mmBERT + MPS），产出 `model_state.pt` + `train_meta.json` |
| `scripts/export_local_onnx.py` | 导出 ONNX（fp32），复制 tokenizer + 写 14 类 `categories.json` |
| `scripts/quantize_onnx_fp16.py` | fp32 → fp16（1287MB → 643MB），保持 IO dtype，验证 logits 差异 |
| `scripts/run_full_eval.py` | 跑 4 套评测 → 调 `check_acceptance.py`；`ensure_holdout()` 组装 holdout |
| `scripts/check_acceptance.py` | 四门槛判定（FPR / holdout / 手写 / rebate 召回）|
| `scripts/gen_v3_comparison_report.py` | 生成 v2-vs-v3 逐条真实文本对比报告 |
| `deploy/push_hf_model.py` | 上传 fp16 bundle 到 HF 模型仓（`--model-dir` 可选版本）|
| `deploy/push_hf_static.py` | 部署 HF Static Space（含 tokenizer gz bundle）|

---

## 5. 复现步骤

### 前置
- Python ≥3.10，`pip install -e ".[dev]"`（pytest）与 `[serve]`（可选）
- Apple Silicon（MPS）或等价环境；`HF_HUB_DISABLE_XET=1`（HF 下载必需）
- 环境有 SOCKS 代理时：`pip install "httpx[socks]"`（huggingface_hub 需要）
- CCL2023 数据放到 `datasets/raw/ccl2023/{train,test}.json`

### 步骤

```bash
cd <repo>
source .venv/bin/activate
export HF_HUB_DISABLE_XET=1

# 1) 切 CCL 同源留出（可复现，seed=42）
python scripts/build_ccl_holdout.py
#    → datasets/ccl_rebate_eval.jsonl (1550 条: 1000 rebate + 50/标签)

# 2) 构建 45k 训练集（真实 CCL rebate 取代合成）
python scripts/build_dataset.py
#    → datasets/training/{train,val,test}.jsonl

# 3) 不变量校验（rebate ≥25% / 零泄漏 / 生成占比 ≤30%）
python - <<'PY'
import json, sys; sys.path.insert(0,'scripts')
from dataset_mix import _norm_text
from collections import Counter
tr=[json.loads(l) for l in open('datasets/training/train.jsonl')]
pos=[r for r in tr if r['is_scam']==1]
c=Counter(r['category'] for r in pos)
ev=set()
for f in ('datasets/ccl_rebate_eval.jsonl','datasets/rebate_eval_real.jsonl'):
    for l in open(f):
        r=json.loads(l); ev.add(_norm_text(r.get('text') or r.get('state')))
tn={_norm_text(r['text']) for r in tr}
print('n',len(tr),'rebate',round(c['rebate_scam']/len(pos),3),'leak',len(ev&tn))
assert len(tr)==45000 and not (ev&tn) and c['rebate_scam']/len(pos)>=0.249
print('INVARIANTS OK')
PY

# 4) 训练（~5h MPS）
HF_HUB_DISABLE_XET=1 python -u scripts/train_local_lora.py \
  --train datasets/training/train.jsonl --val datasets/training/val.jsonl \
  --max-samples 45000 --epochs 4 --batch-size 8 --grad-accum 2 \
  --output models/laya-lora-finetuned-v3

# 5) 导出 ONNX
HF_HUB_DISABLE_XET=1 python scripts/export_local_onnx.py \
  --state models/laya-lora-finetuned-v3/model_state.pt \
  --output models/laya-onnx-multilingual-finetuned-v3

# 6) fp16 量化（可选，用于部署）
python scripts/quantize_onnx_fp16.py \
  --src models/laya-onnx-multilingual-finetuned-v3 \
  --dst models/laya-onnx-multilingual-finetuned-v3-fp16

# 7) 四门验收
python scripts/run_full_eval.py --model-dir models/laya-onnx-multilingual-finetuned-v3
#    → 打印四门 + reports/gate_v3_result.md 素材

# 8) 逐条对比报告
python scripts/gen_v3_comparison_report.py
#    → reports/v2-vs-v3-comparison.md
```

---

## 6. 产物清单

| 产物 | 路径 | 说明 |
|---|---|---|
| LoRA state | `models/laya-lora-finetuned-v3/model_state.pt` | 321.9M 参数合并权重 + `train_meta.json` |
| ONNX fp32 | `models/laya-onnx-multilingual-finetuned-v3/` | `model.onnx` + `model.onnx.data`(1.29GB) + tokenizer + 14 类 categories.json |
| ONNX fp16 | `models/laya-onnx-multilingual-finetuned-v3-fp16/` | 部署用，`model.onnx.data` **643MB** |
| HF 模型仓 | `LiChace/laya-scam-detector-onnx-v3` | fp16 bundle；权重 sha256 `4854eae9…` |
| HF Space | `LiChace/laya-scam-detector` | https://lichace-laya-scam-detector.static.hf.space |
| 验收报告 | `reports/gate_v3_result.md` | 四门 + v2/v3 对比 |
| 对比报告 | `reports/v2-vs-v3-comparison.md` | 真实文本逐条 |
| v1 基线 | `reports/gate_baseline_v1.md` | 训练前对照 |

`models/`、`datasets/training/`、`datasets/raw/` 均 gitignored（大文件不入库）；`reports/` 入库作证据。

---

## 7. 对比测试数据

### 7.1 四门验收（v1 → v2 → v3）

| Gate | v1 (零样本) | v2 (合成 rebate) | **v3 (真实 CCL)** | 目标 |
|---|---|---|---|---|
| Gate1 难负 FPR | 0.593 | 0.004 | **0.006** | ≤0.02 |
| Gate2 holdout zh 准确 | 0.795 | 0.964 | **0.989** | ≥0.90 |
| Gate2 holdout zh 召回 | 0.925 | 0.964 | **0.998** | ≥0.95 |
| Gate3 手写类别准确 | 0.658 | 0.763 | 0.711 | ≥0.70 |
| **Gate4 rebate_scam 类别召回** | **0.000** | **0.000** | **0.950** | ≥0.80 |
| 通过门数 | 0/4 | 3/4 | **4/4** | 4/4 |

> 说明：Gate3 从 v2 的 0.763 微降到 0.711（仍达阈）。这是 14 类边界在真实数据下重新校准的
> 副作用；手写集仅 38 条，±0.05 波动属正常。

### 7.2 Gate4 的两层口径（v3）

| 口径 | n | v2 | v3 |
|---|---|---|---|
| CCL 同源留出（主）| 1,550（1,000 rebate）| — | **0.950** |
| 跨源真实案件 | 42 | 0.000 | **0.952**（40/42）|

两个独立口径都 ≈0.95，跨源集尤其说明 v3 学到的是**语义**而非 CCL 模板指纹。

### 7.3 逐条真实文本对比（节选自 `reports/v2-vs-v3-comparison.md`）

格式：`is_scam 概率 / 预测类别`

**CCL 同源留出（真实案件）**

| 文本（截断）| v2 | v3 |
|---|---|---|
| 2022年12月7日…添加微信好友…录入微信刷单群…下载"乐橙"APP…做任务刷单 | `0.00 / benign` | `1.00 / rebate_scam` ✅ |
| 2022年10月06日…被对方以刷单返利为由通过网银转账骗取现金 | `1.00 / benign` | `1.00 / rebate_scam` ✅ |
| 2023年1月18日…在快手找兼职…下载"桃隐社区"APP进行刷单 | `0.99 / phishing` | `1.00 / rebate_scam` ✅ |

**跨源真实案件（警方/媒体）**

| 文本（截断）| v2 | v3 |
|---|---|---|
| 市民王某被拉进兼职微信群…点关注、刷好评…升级代理高返现…账户风控冻结 | `1.00 / investment_scam` | `1.00 / rebate_scam` ✅ |
| 合肥居民小Z…需完成充值刷单任务才能交友…操作失误需转账修复 | `1.00 / romance_scam` | `1.00 / rebate_scam` ✅ |

**合法难负样本（FPR 保持）**

| 文本 | v2 | v3 |
|---|---|---|
| 【唯品会】通知：您尾号3620的快件已到达，请至学校快递柜领取。| `0.00 / benign` | `0.00 / benign` ✅ |
| 【曹操出行】鄞州区写字楼招跑腿骑手，时薪35元… | `0.00 / benign` | `0.00 / benign` ✅ |

### 7.4 线上实测（部署后，浏览器内推理）

```
2022年10月29日…报警称其被诈骗…              → rebate_scam 99.7%
2022年12月17日…来所报警…                   → rebate_scam 99.8%
市民王某被拉进兼职微信群…刷好评…            → rebate_scam 98.8%
```

---

## 8. 部署方法

### 8.1 上传模型到 HF

```bash
export HF_TOKEN=hf_xxx
export HF_HUB_DISABLE_XET=1

# 打包并上传 fp16 bundle（v3）
python deploy/push_hf_model.py \
  --repo-id LiChace/laya-scam-detector-onnx-v3 \
  --model-dir models/laya-onnx-multilingual-finetuned-v3-fp16
```

上传内容：`model.onnx` + `model.onnx.data` + `tokenizer/` + `categories.json`（+ 自动附 model card）。
上传后用 LFS `oid`(sha256) 与本地 `shasum -a 256` 比对确认一致。

### 8.2 部署 Static Space

```bash
python deploy/push_hf_static.py --repo-id LiChace/laya-scam-detector
```

Static Space 免费；它打包 `index.html/app.js/style.css` + **tokenizer.json.gz(5MB)**，
模型权重运行时从 `MODEL_BASE`（HF 模型仓）拉取。

### 8.3 切换 Space 指向新模型

编辑 `web-static/app.js` 的 `MODEL_BASE`：
```js
const MODEL_BASE = window.__MODEL_BASE
  || "https://huggingface.co/LiChace/laya-scam-detector-onnx-v3/resolve/main";
```
**同时必须同步 `web-static/app.js` 里的 `SCHEMA`**（见 §9 踩坑 2）——类别数与模型头一致。
然后重跑 `push_hf_static.py`。

### 8.4 线上验证要点

- 正确 host 是 `https://lichace-laya-scam-detector.static.hf.space`（**带 `.static.`**；裸 `*.hf.space` 会 404）
- 首访需下载 ~681MB fp16 模型（浏览器缓存后秒开）
- 模型权重经 HF 的 Xet 302 重定向到 CDN；本环境需 `HF_HUB_DISABLE_XET=1` 仅影响上传/下载工具，
  浏览器端 fetch 跟随重定向正常

---

## 9. 踩坑与教训（复现时务必注意）

### 9.1 评测集隔离不能按 id
`stable_id(text, source)` 把 **source 一起哈希**。评测行（source=`real_case`）与训练行
（source=`syn_rebate`）同文本也会得到**不同 id**，所以 `rid in eval_ids` 的排除**永不触发**——
评测集会静默混入训练，使 Gate4 变成自测。**必须按归一化文本排除**（`dataset_mix._norm_text`），
并加回归测试锁死。

### 9.2 评测集必须是真实数据（反例警示）
最初用**另一个模板生成器**造了 182 条"评测集"，重复短语 43 次、30% 与训练共享 4-gram——
它**什么都没测到**。换成 42 条真实警方/媒体案件后才暴露真问题。**任何 rebate 评测集必须
是真实案件叙述**。

### 9.3 部署的 schema 必须与模型头一致（本次最隐蔽的 bug）
v3 模型是 14 类，但 `web-static/app.js` 里**硬编码的 SCHEMA 是 13 类**（缺 `rebate_scam`）。
后果：浏览器只给模型 13 个选项，14 类的头永远选不到 rebate，全部落到 `investment_scam`。
这在 v1/v2 时代被"反正标签是错的"掩盖，v3 修对了答案反而暴露。
**排查方法**：先证伪权重（比对 HF LFS sha256 vs 本地）→ 再证伪 tokenization（抓浏览器
token ids 逐位比对）→ 排除后即定位到 schema。**切换类别数时，模型、tokenizer、SCHEMA、
`categories.json` 四者必须同步。**

### 9.4 Gate4 需要留出集里真的含有该类
holdout 从 `test.jsonl` 采样，而 `test.jsonl` 来自 `split_dataset(real)`，**不含合成源**——
若不显式并入 rebate 评测集，`parse_category_recall` 返回 `None`，Gate 必判失败。已修复为
`ensure_holdout()` 显式并入 CCL 留出 + 42 条跨源集。

### 9.5 `pytest` 退出码 134 是已知抖动
全量套件约 1/3 概率在打印 summary 后以 `libc++abi: recursive_mutex lock failed` 中止
（onnxruntime + torch + Laya tokenizers 同进程析构竞争）。**按 `N passed` 判定，不看退出码**
（当前 177 passed / 2 skipped）。

### 9.6 环境小坑
- `HF_HUB_DISABLE_XET=1` 必需（xet 后端在本环境失败）
- SOCKS 代理下 `huggingface_hub` 需 `pip install "httpx[socks]"`
- 系统 `/usr/bin/git` 可能被 Xcode 许可拦截，用 `/Library/Developer/CommandLineTools/usr/bin/git`
- `scripts/gen_rebate_synthetic.py` 与 `datasets/synthetic/rebate_scam.jsonl` 是 v2 遗留，
  v3 已退役（保留脚本供历史参考）

---

## 10. 结论

- **Gate1（FPR 0.593→0.006）** 与 **Gate3（0.658→0.711）** 由 v2 配方达成，v3 保持
- **Gate4 由真实 CCL 数据达成 0→0.950**，这是 v3 的核心增量
- 复现的关键三点：① 真实数据（CCL2023）② 配方不变（保 FPR）③ 四者同步（模型/schema/tokenizer/categories）
- 已知边界：Gate3 微降属重校准；rebate/investment 的真实语义边界仍是易混淆点（v3 跨源仅 40/42）
