# v3 设计：用真实 CCL2023 修 Gate4，保住 FPR 战果

日期：2026-10-05
状态：待用户复核
前置：`docs/plans/2026-09-30-laya-fp-reduction-{design,impl}.md`（v2）

## 1. 背景与目标

### 现状（v2，2026-10-05 验收）
| Gate | v2 | 目标 | 状态 |
|---|---|---|---|
| Gate1 难负 FPR | 0.004 | ≤0.02 | ✅ |
| Gate2 holdout zh 准确/召回 | 0.964/0.964 | ≥0.90/≥0.95 | ✅ |
| Gate3 手写类别准确 | 0.763 | ≥0.70 | ✅ |
| Gate4 rebate_scam 类别召回 | **0/42** | ≥0.80 | ❌ |

Gate4 失败根因（已定位）：v2 在真实刷单返利文本上从不输出 `rebate_scam`（choice 头平均概率
0.0013，vs investment_scam 0.651）。合成 rebate 训练数据是"招募广告"语域，真实案件是"受害
人/警方叙述"语域，分布不匹配。**不是类别不可学**——模型对合成语域能正确输出 rebate（0.94）。

### 触发
用户已下载 CCL2023-FCC 至 `datasets/ccl2023/{train,test}.json`：
- train.json：82,210 条，12 类，**28,367 条刷单返利**（真实标注，字段 `案情描述`/`案件类别`）
- test.json：10,276 条，**无标注**（赛事测试集）

### 目标与约束
- **主目标**：`rebate_scam` 类别召回 ≥0.80（真实数据）
- **硬约束**：Gate1 FPR 不得从 0.004 反弹超过 0.02；Gate2/Gate3 不回退
- **手段**：从头重训 v3，**配方不变**（难负样本 + 三类 caps + rebate floor）

## 2. 数据接入

### 2.1 落位
- `load_ccl2023` 读 `datasets/raw/ccl2023/`（`RAW = Path("datasets/raw")`），字段键 `案情描述`/
  `案件类别` 已匹配。把用户的 `datasets/ccl2023/{train,test}.json` 放到 `datasets/raw/ccl2023/`。
- `.gitignore` 增 `datasets/ccl2023/`（原始大数据，与 `datasets/raw/` 惯例一致，不入库）。

### 2.2 替换合成 rebate
- **删除** `datasets/synthetic/rebate_scam.jsonl` 并移除 `build_dataset.py` 中对该文件的读取接线
  （git 历史保留，可回退）。真实 CCL rebate 取代合成（用户裁定"用 CCL 全 12 类"，未选共存）。
- `scripts/gen_rebate_synthetic.py` 保留但不再被管线引用（留作历史/回退参考）。
- `datasets/rebate_eval_real.jsonl`（42 条跨源集）保留，作为跨源泛化检查（见 §3.2）。

### 2.3 loader
- `load_ccl2023` 已在 `build_dataset.py` 加载列表中，数据到位后自动拉全 12 类。
- `CCL2023_LABEL_MAP` 已覆盖 12 个类字符串（strict，未映射即报错）——数据就位后先跑一次验证不报错。

## 3. 评测设计（方案 A：双口径）

### 3.1 CCL 同源留出（主指标）
- 新增切分脚本：从 CCL train 按行切出留出集（每行是一条案件，`案件编号` 为其 id），**1,000 条
  rebate**（rebate 召回用，n=1000 → 95% 置信区间约 ±3%）+ **其它 11 类各留出 50 条**（共 550 条，
  宏观复核）。切分 seed 固定。
- 留出集写为独立文件 `datasets/ccl_rebate_eval.jsonl`（评测格式，含真 `category`）。
- 训练用的 rebate 池 = 28,367 − 1,000 = 27,367 条，仍远超 floor（5,062）。
- 留出集**不得进训练**：在 `build_dataset` 把该文件的**归一化文本**加入 `eval_texts` 排除集
  （与 42 条集同一机制，复用 `dataset_mix._norm_text` + 既有回归测试）。
- ⚠️ 关键隔离：CCL 其余类也在训练集里，必须保证 rebate 留出样本不以**其它类别**形式混入训练
  （同一 `案件编号` 只归留出或训练一侧，不跨侧；排除按文本，天然覆盖）。

### 3.2 跨源检查（保留 42 条）
- 保留 `datasets/rebate_eval_real.jsonl`（42 条警方/媒体案件），作不同语域的泛化检查。
- 该集已在排除机制内（`eval_texts`）。

### 3.3 Gate4 判定
- 主指标：CCL 同源留出 rebate 召回。
- 补充：42 条跨源召回。
- 两者都作为 Gate4 报告字段；**通过要求同源 ≥0.80**，跨源作为泛化证据一并记录（不单独卡阈值，
  因其 n 小）。

### 3.4 已知局限
- 同源留出与训练同分布，会高估真实世界表现 → 用 42 条跨源集兜底观察。
- CCL test.json 无标注，不可用于评测。

## 4. 训练

- 适配 `scripts/run_v2_pipeline.sh`（或新建 v3 版）：build_dataset → 不变量校验 → 从头训 v3。
- 配方不变：`compose_train` 的 caps/floor 保持 v2 值（`REBATE_FLOOR=0.25`、`CCL_NONREBATE_CAP=0.20`、
  `PER_CATEGORY_CAP=0.30`、`GEN_CAP=0.30`、`TRAIN_TARGET=45000`、`POS_FRACTION=0.45`）。
- 超参不变：LoRA r=8、4 epoch、batch 8 × grad-accum 2、lr 2e-4、max_len 256、MPS。
- 产物：`models/laya-lora-finetuned-v3` → ONNX `models/laya-onnx-multilingual-finetuned-v3`。

## 5. 验证

- v3 跑四门，逐门与 v2 对比（`reports/gate_v3_result.md`）。
- **通过标准**：Gate4 同源 rebate 召回 ≥0.80 **且** Gate1 FPR ≤0.02 **且** Gate2/Gate3 不回退
  （相对 v2 各自允许 ≤0.01 的抖动）。

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| FPR 反弹（真实正样本分布扰动） | 配方不变为主防线；若反弹先诊断，不硬推上线 |
| 同源留出高估 | 42 条跨源集兜底 |
| CCL 与既有源文本重叠 | 已核验与 42 条集、旧 train 整句重叠均为 0；build_dataset 入口按文本去重 |
| 长文本截断（p90=617 字） | max_chars=1024 截断，与现状一致 |

## 7. 验收清单（交付）
- [ ] CCL 数据就位 + loader 无 unmapped 报错
- [ ] 合成 rebate 已移除，build_dataset 用真实 CCL 构建 45k，不变量全绿（含 rebate ≥25%）
- [ ] CCL 留出集与训练集零泄漏（归一化文本级）
- [ ] v3 训练完成 + ONNX 导出（PyTorch/ONNX 差异 <1e-4）
- [ ] 四门报告 + v2/v3 对比
- [ ] `AGENTS.md` 更新（CCL 已就位，Gate4 状态）
- [ ] 测试通过（`pytest`，按 N passed 判定）
