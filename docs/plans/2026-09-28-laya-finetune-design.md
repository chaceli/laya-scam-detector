# Laya 多语增量训练（微调）以支持中文 — 设计文档

**日期**：2026-09-28
**作者**：Sisyphus
**状态**：已批准

---

## 1. 目标

基于 `laya-multilingual` (mmBERT-base, 322M) base，用多源公开数据集（中文 + 英文）做增量微调，导出 ONNX 多语版，**显著提升中文诈骗话术检测能力**，并扩展类别到 13 类。

**核心问题**：
- 现有零样本英文 checkpoint 在中文上 is_scam accuracy 仅 0.840
- 8 类 schema 不足以表达中文诈骗细分类型
- 中文 tokenization 退化（ModernBERT-large 拆字过细）

**验收标准**（硬性）：
- 中文 is_scam accuracy ≥ 0.90（基线 0.840）
- 中文 is_scam recall ≥ 0.95
- 13 类 accuracy ≥ 0.70（基线 0.605 over 8 类）
- 校准 ECE < 0.05（基线 0.081）
- ONNX vs PyTorch logit 差异 < 1e-5

---

## 2. 用户决策摘要（来自 brainstorming）

| 维度 | 选择 |
|---|---|
| 微调起点 | `laya-multilingual` (mmBERT-base, 100+ 语言含中文) |
| 微调策略 | LoRA (4-8GB 显存) |
| 数据集 | 多源混合 (~50k 样本) |
| 类别标签 | 扩展到 12+ 类 |
| 部署形态 | ONNX 多语版导出 |
| 执行方案 | **方案 C**：Kaggle 训练 + 本地评估 + ONNX 多语版 |

---

## 3. 端到端流程

> **修订（2026-09-28）**：训练从 Kaggle 2×T4 改为**本地 Mac M4 Pro (MPS)**。
> 技术验证确认本地可行：模型 1.31GB 内存、LoRA 1.15M 可训参数、前向+反向正常。
> Kaggle 路径（`kaggle/`）保留作备选。详见 §13。

```
[本地] Phase A: 数据准备
  scripts/fetch_datasets.py    → 下载 5 个公开数据集
  scripts/build_dataset.py     → 统一 schema + 13 类映射 + 切分
  scripts/build_kaggle_subset.py → 30k 分层子集

[本地 MPS] Phase B: LoRA 微调 (~45min-2h)
  scripts/train_local_lora.py
  - 加载 laya-multilingual (mmBERT-base) via laya SDK
  - 注入 LoRA r=8 (target_modules=["Wqkv","Wo"])
  - 预编码数据集（tokenization 移出训练循环）
  - 交叉熵（log score，严格 proper scoring rule）训练
  - 每 epoch 验证 is_scam + category 准确率
  - 保存合并后 model_state.pt

[本地] Phase C: ONNX 导出
  scripts/export_local_onnx.py
  - 加载 base + model_state.pt
  - 导出 ONNX（匹配 OnnxLayaClient 契约）
  - 验证 logit diff < 1e-3 vs PyTorch
  - 输出 models/laya-onnx-multilingual-finetuned/

[本地] Phase D: 评估与部署
  - 改造 src/router.py 双 checkpoint 路由
  - 更新 schemas/scam.json 到 13 类
  - main.py --eval 在手写集 + 公开集 + 训练 holdout 上跑
  - scripts/check_acceptance.py 验收
  - 对比 reports/eval-*.md 改进表格
  - README 更新最终评估数字
```

---

## 4. 数据来源 + 类别映射

### 4.1 数据集组合（~30k 训练样本）

| 数据集 | 规模 | 语言 | 许可证 | 用途 |
|---|---|---|---|---|
| FGRC-SCD (Abooooo) | <1k | 中文 | MIT | 中文测试集补充 |
| Scam_Message_9_Language (vichetkao) | 12.8k | 9 国含中文 | MIT | 多语基础 |
| SpamShield-Datasets (M-Arjun) | 149k | 23 国含中文 1.2k | CC-BY-4.0 | **主要训练源** |
| FBS_SMS_Dataset (fl-wxiao) | 14k | 中文 | 学术 | 中文真实诈骗 |
| ealvaradob/phishing-dataset | 73k | 英文 | 研究 | 英文钓鱼 |
| UC Irvine SMS Spam | 5.6k | 英文 | 学术 | 英文基线 |

**过滤**：
- SpamShield 排除 Dutch/Italian（不平衡）
- SpamShield 保留合成 20%（增广多样性）
- 训练集 1:1 scam:benign 平衡

### 4.2 13 类 schema（替换原 8 类）

```json
{
  "is_scam": {"type": "noul"},
  "risk_level": {"type": "score", "criteria": ["1","2","3","4","5"]},
  "scam_category": {
    "type": "choice",
    "criteria": {
      "benign": "正常消息",
      "phishing": "钓鱼链接/账号盗取",
      "crypto_scam": "加密货币诈骗",
      "investment_scam": "虚假投资",
      "lottery_scam": "虚假中奖/抽奖",
      "job_scam": "虚假招聘",
      "loan_scam": "虚假贷款",
      "impersonation": "冒充公检法/客服",
      "romance_scam": "杀猪盘/情感诱导",
      "delivery_fraud": "快递理赔",
      "marketing": "营销/促销",
      "adult_content": "成人/色情",
      "spam_general": "其他垃圾"
    }
  }
}
```

### 4.3 类别映射规则

```python
# schemas/scam_categories.py
CATEGORY_MAP = {
    "spam": "spam_general",
    "phishing": "phishing",
    "crypto": "crypto_scam",
    "marketing": "marketing",
    "job_scam": "job_scam",
    "giveaway": "lottery_scam",
    "adult": "adult_content",
    "promo": "marketing",
    "AD:Loan": "loan_scam",
    "AD:Network_service": "spam_general",
    "AD:Other": "spam_general",
    "FR:Financial": "investment_scam",
    "FR:Phishing(Bank)": "phishing",
    "FR:Phishing(Other)": "phishing",
    "FR:Other": "spam_general",
    "IL:Escort_service": "adult_content",
    "IL:Fake_ID_and_invoice": "impersonation",
    "IL:Gambling": "lottery_scam",
    "IL:Political_propaganda": "spam_general",
    "Other": "spam_general",
    "low_risk_sms": "benign",
    "high_risk_sms": "spam_general",
    "fraud_call": "impersonation",
    "phishing_link": "phishing",
}
```

---

## 5. Kaggle 笔记本设计

### 5.1 笔记本结构（`kaggle/laya_finetune_multilingual.ipynb`）

```
[Cell 1] Setup: pip install laya peft trl datasets transformers accelerate
[Cell 2] 加载 base: laya.load("convaiinnovations/laya", subfolder="multilingual")
[Cell 3] 注入 LoRA adapter (r=8, target_modules=attn)
[Cell 4] 加载数据集: /kaggle/input/scam-detection/*train*.jsonl
[Cell 5] RLCDTrainer (Brier + spherical + RPS reward, GRPO policy gradient)
[Cell 6] 训练循环: 4 epochs, batch=32 effective, lr=2e-4
[Cell 7] 合并 adapter: peft_model.merge_and_unload()
[Cell 8] 拟合每类型温度: fit_all_temperatures(eval_set)
[Cell 9] 保存到 /kaggle/working/
[Cell 10] 推送到 HF Hub: <USER>/laya-multilingual-finetuned-scam
```

### 5.2 训练超参数

```python
BATCH_SIZE = 8        # per T4
GRAD_ACCUM = 4        # effective 32
LR = 2e-4             # LoRA standard
EPOCHS = 4
WARMUP_RATIO = 0.06
MAX_SEQ = 512
GRPO_K = 8            # samples per group
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
```

### 5.3 验证策略

每 epoch eval：
- is_scam accuracy / F1 / recall
- risk_level MAE
- per-type 温度拟合
- 校准 ECE

提前停止：F1 不再提升 2 epoch。

---

## 6. ONNX 导出 + Router 改造

### 6.1 合并 + 导出（`scripts/merge_and_export_onnx.py`）

```python
# 1. 下载 base + adapter
# 2. 合并 LoRA
peft_model = PeftModel.from_pretrained(base_model, adapter_path)
merged = peft_model.merge_and_unload()
# 3. 导出 ONNX (使用社区 export/export_onnx.py 参考实现)
export_laya_to_onnx(merged, tokenizer, output_dir)
# 4. 验证 logit diff
max_diff = validate_onnx_vs_pytorch(onnx_path, merged, test_inputs)
assert max_diff < 1e-5
# 5. 推送 HF Hub: laya-onnx-multilingual-finetuned
```

输出结构：
```
models/laya-onnx-multilingual-finetuned/
├── model.onnx               (~3MB)
├── model.onnx.data          (~644MB FP16)
├── tokenizer/tokenizer.json
├── rl_agent_config.json     (with fitted temperatures)
└── categories.json          (13 类映射)
```

### 6.2 Router 改造（`src/router.py`）

```python
class Router:
    def __init__(self, english_dir, multilingual_dir, providers=None):
        self.english = OnnxLayaClient(english_dir, providers=providers)        # 421M
        self.multilingual = OnnxLayaClient(multilingual_dir, providers=providers)  # 322M

    def _pick(self, text):
        if detect_script(text) == LATIN_SCRIPT:
            return self.english
        return self.multilingual
```

CLI：`--model multilingual` 显式覆盖；自动路由：CJK/Hebrew/Arabic/Devanagari → multilingual。

---

## 7. 评估方案

### 7.1 测试集

| 测试集 | 来源 | 数量 | 语言 |
|---|---|---|---|
| 现有手写 | datasets/eval.jsonl | 38 | 中英 |
| 现有公开 SMS Spam | datasets/public.jsonl | 400 | 英 |
| 新增 SpamShield holdout | test split | ~3k | 23 国 |
| 新增 FGRC-SCD | test split | ~100 | 中文 |
| 新增合成 edge cases | 脚本生成 | ~30 | 混合 |

### 7.2 目标指标

| 指标 | 基线 | 微调目标 |
|---|---|---|
| 中文 is_scam accuracy | 0.840 | **≥ 0.90** |
| 中文 is_scam recall | (FN 数据) | **≥ 0.95** |
| 13 类 accuracy | 0.605 (8 类) | **≥ 0.70** |
| 校准 ECE | 0.081 | **< 0.05** |
| 中文误报率 | (0.16) | **≤ 0.10** |
| 延迟 (M4 Pro CPU) | 575ms | **< 700ms** |

### 7.3 ONNX 验证

```python
# max_logit_diff vs PyTorch < 1e-5
test_inputs = [("Test text 1", {"q": {"type": "noul", "instructions": "X"}})]
pytorch_logits = run_pytorch(merged, test_inputs)
onnx_logits = run_onnx("models/laya-onnx-multilingual-finetuned/", test_inputs)
assert max_diff(pytorch_logits, onnx_logits) < 1e-5
```

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| Kaggle OOM | gradient checkpointing + max_seq=512 + 增量保存 |
| 类别映射偏差 | 在原始测试集上验证不退化 |
| 中文 tokenization 不佳 | 验证 mmBERT 在中文上 chars/token ≥ 1.5 |
| ONNX 精度漂移 | 强制 max_logit_diff < 1e-5 验证 |
| 训练数据偏见（合成 20%） | train/val/test 按来源 stratify；test 仅用原始 |
| 模型变大（322M → 647MB ONNX）| FP16 量化；按需 int8 |
| Kaggle 30h/周不足 | 单次 5h，留 25h 给多轮调参 |
| 多语 ONNX 兼容性未验证 | 导出后跑小批量 ONNX vs PyTorch diff 验证 |

---

## 9. 验收清单

### 必需
- [ ] 中文 is_scam accuracy ≥ 0.90
- [ ] 中文 is_scam recall ≥ 0.95
- [ ] 13 类 accuracy ≥ 0.70
- [ ] 校准 ECE < 0.05
- [ ] 中文误报率 ≤ 0.10
- [ ] ONNX vs PyTorch logit diff < 1e-5
- [ ] 70+ 现有测试全过
- [ ] 单条推理延迟 < 700ms (M4 Pro CPU)

### 目标（nice-to-have）
- [ ] 中文 noul F1@0.5 ≥ 0.85
- [ ] 类别 accuracy ≥ 0.75
- [ ] ECE < 0.03

---

## 10. 文件结构

```
laya/
├── docs/plans/
│   ├── 2026-09-28-laya-scam-detection-design.md     (既有)
│   ├── 2026-09-28-laya-scam-detection-impl.md       (既有)
│   ├── 2026-09-28-laya-finetune-design.md          # 本文档
│   └── 2026-09-28-laya-finetune-impl.md            # 实现计划（由 writing-plans 生成）
├── kaggle/
│   └── laya_finetune_multilingual.ipynb             # 新建（本地）
├── scripts/
│   ├── fetch_datasets.py                            # 新建
│   ├── build_dataset.py                             # 新建
│   └── merge_and_export_onnx.py                    # 新建
├── schemas/
│   ├── scam.json                                    # 更新到 13 类
│   └── scam_categories.py                           # 新建（映射规则）
├── src/
│   ├── router.py                                    # 改造（双 checkpoint）
│   └── ...
├── datasets/
│   ├── training/                                    # 新建
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── ...
├── models/
│   ├── laya-onnx-en/                               # 既有
│   └── laya-onnx-multilingual/                      # 新建（待微调后替换）
└── reports/
    ├── eval-20260928-*.md                          # 既有（基线）
    ├── finetune-progress-*.md                      # 新建（微调进展）
    └── eval-finetuned-*.md                         # 新建（最终对比）
```

---

## 11. 依赖新增

```toml
# pyproject.toml 新增
[project.optional-dependencies]
train = [
    "peft>=0.10",           # LoRA
    "trl>=0.8",              # RLCD 训练器
    "accelerate>=0.27",      # 多 GPU 调度
    "bitsandbytes>=0.41",   # 4-bit 量化（可选）
]
```

无需新增核心运行时依赖（ONNX / tokenizers / numpy / huggingface_hub 已就位）。

---

## 12. 不在范围内

- 自定义架构（替换 mmBERT）
- 多轮对话微调（先聚焦单条）
- 完整工作流（intake/routing/observation）
- 生产部署（Docker / K8s / 监控系统）
- 自动 retraining pipeline