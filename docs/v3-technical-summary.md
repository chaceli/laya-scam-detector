# Laya 反诈模型 v3 增训：技术总结

> 面向有技术背景的读者（工程师 / 算法 / 技术 PM）。介于"大白话版"
> （`v3-plain-language-summary.md`）与"完整复现指南"（`v3-incremental-training-guide.md`）之间。
> 重点讲清：**问题怎么定位的、方案怎么设计的、训练怎么跑的、效果差多少、Laya 底座值不值得用。**

---

## 1. 问题定位

### 1.1 现象
v2 模型是 14 类体系，其中 `rebate_scam`（刷单返利，中国发案量最高的电诈类型）应有独立分类。
但在 **42 条真实公安/媒体案件**上，`choice`（14 类）头对 `rebate_scam` 的召回 = **0**：
52% 判成 `investment_scam`、27% 判成 `benign`。

### 1.2 定位过程（三层证据）
1. **对比检测头 vs 分类头**：同一批案件，`noul`（is_scam）头召回 0.881——**模型知道这是诈骗，
   只是归类错**。所以问题是"分类"而非"检测"。
2. **看概率质量分布**：`choice` 头给 `rebate_scam` 的**平均概率仅 0.0013**，`investment_scam` 是 0.651。
   即该类**几乎不激活**，不是"被微弱压过"。
3. **归因到数据分布**：同一模型对**合成**招募广告体文本能正确输出 `rebate_scam`（0.94），对**真实**
   案件笔录体却不行。→ **训练分布 ≠ 推理分布**（register / distribution shift）。

### 1.3 为什么后处理救不了
| 手段 | 结论 | 原因 |
|---|---|---|
| 类别先验重加权 | ✗ | softmax 重加权只能重排已有质量，无法造出 0.0013 之外的质量 |
| 阈值微调 / argmax 重排 | ✗ | 类别概率质量 ≈ 0，无阈值可调 |
| **换真实数据重训** | ✓ | 唯一能改变模型内部表征的路径 |

代价对比：改分类体系（合并 rebate→investment）也可行，但只到 0.667，且需产品决策——本方案未采纳。

---

## 2. 技术方案

### 2.1 数据源
**CCL2023-FCC**（公安反诈大数据平台真实案件）：
- `train.json` **82,210 条**，字段 `案件编号`/`案情描述`/`案件类别`，**12 类**
- `test.json` 10,276 条**无标注**（赛事测试集）→ 不可用于评测
- 12 类经 `schemas/scam_categories.py::CCL2023_LABEL_MAP` **严格映射**到我们的 14 类（未覆盖即报错，
  不走 `spam_general` 兜底，避免标签污染）
- 关键：**28,367 条 `刷单返利类`** 正是缺失的真实信号

### 2.2 数据配比（配方**不变**）
沿用 v2 的 `compose_train` 硬约束，只换 rebate 的**来源**（合成 → 真实）：

```
TRAIN_TARGET=45000   POS_FRACTION=0.45    → n_pos=20,250
REBATE_FLOOR=0.25    → rebate_scam ≥ 5,062 条
CCL_NONREBATE_CAP=0.20   PER_CATEGORY_CAP=0.30   GEN_CAP=0.30   TELE_POS_CAP=0.10
```

**为什么"配方不变"是关键**：v2 的 FPR 从 59.3% 压到 0.4%，靠的是**配方**（难负样本加权 + 跨桶
上限），不是某组权重。保持配方 → 换数据不应当破坏 FPR。这用一次实验就验证了（见 §4）。

实际 CCL 用量：受 caps 约束，82k 中约 **9,401 条**进训练（rebate 5,351 + 非返利封顶 4,050）。

### 2.3 微调方案
| 项 | 值 | 说明 |
|---|---|---|
| 基座 | `convaiinnovations/laya` multilingual | mmBERT-base，322M 参数，100+ 语言 |
| 方法 | **LoRA r=8 / α=16** | 仅注入 `Wqkv`+`Wo`，可训练 **1.15M / 323M = 0.36%** |
| 超参 | 4 epoch，batch 8 × grad-accum 2，lr 2e-4，max_len 256 | 与 v2 一致 |
| 硬件 | Apple M4 Pro（MPS） | 无云 GPU |
| 耗时 | **17,875s ≈ 5.0 小时** | |

LoRA 的价值：**极低显存 + 参数高效**，且合并后与全量微调产出的部署形态一致。

---

## 3. 训练过程

### 3.1 流水线
```
CCL2023 原始数据
   │  build_ccl_holdout.py      切 1,000 rebate 同源留出（按 案件编号，seed=42）
   │  dataset_loaders.load_ccl2023()   82k → 14 类映射
   │  compose_train()           按配方组装（floor + caps）
   ▼
45,000 训练集（rebate 5,351 / 26.4%）
   │  train_local_lora.py       LoRA 微调，MPS
   ▼
model_state.pt（合并权重 321.9M）
   │  export_local_onnx.py      ONNX fp32 + tokenizer + 14 类 categories.json
   ▼
onnx fp32 (1.29GB) ──quantize_onnx_fp16.py──▶ onnx fp16 (643MB)
   │  run_full_eval.py + check_acceptance.py   四门
   ▼
reports/gate_v3_result.md
   │  push_hf_model.py / push_hf_static.py
   ▼
HF 模型仓 → HF Static Space（浏览器内推理）
```

### 3.2 训练曲线（epoch 级验证）
| epoch | train loss | val zh is_scam acc | val 14-class acc |
|---|---|---|---|
| 1 | 0.815 | 0.943 | 0.840 |
| 2 | 0.423 | 0.959 | 0.882 |
| 3 | 0.312 | 0.964 | 0.876 |
| 4 | 0.255 | **0.966** | 0.872 |

对比 v2 的 epoch 4（0.966/0.872）——**v3 的验证指标与 v2 基本持平**（略低），因为真实 CCL 数据
比合成数据更难、更杂。**验证指标持平但 Gate4 从 0 → 0.95**，正是"换成真实数据"的价值：
真实数据没有让验证 loss 更好看，但让模型学到了**真实的类别边界**。

### 3.3 关键工程约束（复现必踩）
- **评测集排除按归一化文本，不能按 id**：`stable_id` 把 source 一起哈希，评测行与训练行 source
  不同 → id 永不相等 → 排除逻辑静默失效，评测集混入训练变自测。已改 `_norm_text` + 回归测试。
- **Gate4 留出集必须真含该类**：否则 `parse_category_recall` 返回 `None`，Gate 必判死。
- **部署 schema 必须与模型头同步**：见 §6.2。

---

## 4. 效果对比

### 4.1 四门验收（v1 → v2 → v3）
| Gate | 指标 | v1（零样本）| v2（合成 rebate）| **v3（真实 CCL）** | 阈值 |
|---|---|---|---|---|---|
| 1 | 难负集 FPR | 0.593 | 0.004 | **0.006** | ≤0.02 |
| 2 | holdout zh acc | 0.795 | 0.964 | **0.989** | ≥0.90 |
| 2 | holdout zh recall | 0.925 | 0.964 | **0.998** | ≥0.95 |
| 3 | 手写集 14 类 acc | 0.658 | 0.763 | 0.711 | ≥0.70 |
| **4** | **rebate_scam 召回** | **0.000** | **0.000** | **0.950** | ≥0.80 |
| | **通过** | 0/4 | 3/4 | **4/4** | |

### 4.2 Gate4 双口径（防自欺）
| 口径 | 来源 | n | v2 | v3 |
|---|---|---|---|---|
| 同源留出（主）| CCL train 切分 | 1,550（1,000 rebate）| — | **0.950** |
| 跨源检查 | 警方/媒体通报 | 42 | 0.000 | **0.952**（40/42）|

**两个独立口径同时 ≈0.95** 是强证据：同源留出测"学会 CCL 的模式"，跨源集测"迁移到其它语域"。
跨源也达标 → **模型学的是语义（刷单返利的诈骗特征），不是 CCL 的模板指纹**。

### 4.3 逐条真实文本（节选，`reports/v2-vs-v3-comparison.md`）
格式 `is_scam 概率 / 类别`：

| 真实案件（节选）| v2 | v3 |
|---|---|---|
| 2022年12月…录入微信刷单群…下载"乐橙"APP…做任务刷单 | `0.00 / benign` | `1.00 / rebate_scam` ✅ |
| 2023年1月…快手找兼职…下载"桃隐社区"APP进行刷单 | `0.99 / phishing` | `1.00 / rebate_scam` ✅ |
| 市民王某…点关注、刷好评…升级代理高返现 | `1.00 / investment_scam` | `1.00 / rebate_scam` ✅ |
| 【唯品会】您尾号3620的快件已到达…（合法）| `0.00 / benign` | `0.00 / benign`（FPR 保持）|

### 4.4 副作用评估
- **FPR 0.004 → 0.006**：仍严于阈值 3 倍，属真实正样本分布变化带来的正常扰动
- **手写集 0.763 → 0.711**：仍是"合格"，是 14 类边界在真实数据下重新校准的副作用（n=38，±0.05）

### 4.5 ONNX 导出保真
fp32 vs fp16 logits 最大概率差 **0.00000**（容差 0.05）；PyTorch vs ONNX 最大差 **3.81e-06**。
部署无损。

---

## 5. Laya 可行性评估

**结论：Laya 底座可行。本次失败是数据问题，不是模型容量问题。** 三条技术证据：

1. **同底座换数据即达标**：模型架构、LoRA 配置、配方全未变，仅换 rebate 数据来源 → 4/4 通过。
   说明底座表征能力足够，瓶颈在数据分布。
2. **零样本已有信号**：v1 从未见过 `rebate_scam` 类，`noul` 头在真实案件上已召回 0.881——底座的
   通用诈骗语义理解已到位，缺的只是细粒度分类监督。
3. **极低的部署成本**：
   - LoRA 仅 0.36% 参数可训练 → 单机 MPS 5h
   - 导出 ONNX 后**无 PyTorch 依赖**，浏览器内（onnxruntime-web WASM）即可推理 → 已上线
     https://lichace-laya-scam-detector.static.hf.space

**边界**：`rebate_scam` 与 `investment_scam` 的语义边界天然模糊（"充值返利"两类都像），v3 跨源
仍错判 2/42；`job_scam` 等长尾类样本少，仍依赖数据配比。这些是**数据层面**的持续工作，不是底座限制。

---

## 6. 经验与坑

### 6.1 评测集不能用合成冒充真实
最初用模板生成 182 条"评测集"，重复短语 43 次、30% 与训练共享 4-gram → **什么都没测到**。
换 42 条真实案件才暴露真问题。**评测集必须真实**。

### 6.2 部署 schema 必须与模型头四者同步
`web-static/app.js` 硬编码的 `SCHEMA` 仍是 13 类（缺 `rebate_scam`），导致 14 类模型永远选不到
该类，全部落到 `investment_scam`。排查：① 比对 HF LFS sha256 vs 本地（证伪权重）→ ② 抓浏览器
token ids vs Python（证伪分词）→ ③ 定位到 schema。
**切类别数时，模型 / tokenizer / SCHEMA / categories.json 必须同步。**

### 6.3 其他
- `pytest` 约 1/3 概率以 exit 134 中止（原生析构竞争），按 `N passed` 判定
- HF 下载需 `HF_HUB_DISABLE_XET=1`；SOCKS 代理需 `httpx[socks]`

---

## 7. 结论

- **Gate4 从 0 → 0.950**，靠的是把 rebate 数据从"合成招募广告体"换成"真实案件笔录体"
- **配方不变守住了 FPR 战果**（0.006，远低于 0.02）
- **双口径 + 真实文本对比**证明模型学到的是语义而非模板
- **Laya 底座 + LoRA + 真实数据**在单机低成本下达到生产可用，且可浏览器内 offline 运行

复现步骤见 `docs/v3-incremental-training-guide.md`；验收数据见 `reports/gate_v3_result.md`；
逐条对比见 `reports/v2-vs-v3-comparison.md`。
