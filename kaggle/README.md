# Kaggle 微调操作指南（Task 21）

本目录包含用于 Kaggle 微调的全部材料。**这一步必须手动完成** — Kaggle 需要交互式网页操作。

---

## 📦 材料清单

| 文件 | 大小 | 用途 |
|---|---|---|
| `laya_finetune_multilingual.ipynb`（上一层目录）| — | 训练笔记本 |
| `upload/scam-detection-training.zip` | 13MB | 训练数据集（30k train + 3k val）|
| `upload/train.jsonl` | 40MB | 原始训练数据（备用）|
| `upload/val.jsonl` | 4.9MB | 原始验证数据（备用）|

---

## 🚀 操作步骤

### 步骤 1：上传数据集到 Kaggle

1. 打开 https://www.kaggle.com/datasets → **New Dataset**
2. 上传 `kaggle/upload/scam-detection-training.zip`
3. 数据集名称设为 **`scam-detection-training`**（笔记本硬编码此路径）
4. 可见性：**Private**（保护训练数据）
5. 创建

**验证**：数据集创建后，Kaggle 路径应为 `/kaggle/input/scam-detection-training/`，包含 `train.jsonl` 和 `val.jsonl`。

### 步骤 2：创建 Kaggle Notebook

1. 打开 https://www.kaggle.com/code → **New Notebook**
2. 菜单 **File** → **Import Notebook** → 上传 `kaggle/laya_finetune_multilingual.ipynb`
3. 右侧设置：
   - **Accelerator**: `GPU T4 x2`（免费 2×T4）
   - **Persistence**: `Files only`
   - **Internet**: `On`（需要下载 Laya 权重）
4. 左侧 **Add Data** → 搜索并附加 `scam-detection-training` 数据集

### 步骤 3：修改占位符

在**第 8 个 cell**（Push to HuggingFace Hub）中找到：

```python
HF_USER = "YOUR_USERNAME"
```

替换为你的 HuggingFace 用户名。**前置条件**：
- HF 账号已创建
- 在 Kaggle 的 **Add-ons → Secrets** 中添加 `HF_TOKEN` 密钥（HF 写权限 token）
  - token 从 https://huggingface.co/settings/tokens 创建（类型：Write）

### 步骤 4：执行

点击 **Run All**。预计运行 1-2 小时（30k 样本、LoRA、4 epochs）。

**可能的问题**：
| 症状 | 解决 |
|---|---|
| `CUDA out of memory` | 减少 `per_device_train_batch_size` 到 4，`gradient_accumulation_steps` 到 8 |
| `HF_ENDPOINT` 访问慢 | 已有 `hf-mirror.com` 回退；若仍慢，改用 Kaggle 的 HF 缓存 |
| 训练 loss 不降 | 检查 LoRA target_modules 是否匹配 mmBERT 层名 |
| **`model` 变量未定义** | 说明 Laya SDK 的 Router API 与我们预期不同 — 参考下方"API 适配"章节 |

### 步骤 5：保存产出

训练成功后，笔记本会自动推送两个 repo 到你的 HF 账号：
- `<HF_USER>/laya-multilingual-finetuned-scam`（合并后模型）
- `<HF_USER>/laya-multilingual-scam-adapter`（LoRA adapter）

**记录以下信息到 `reports/finetune-progress-v1.md`**：
- Kaggle run URL
- 总运行时间
- 最终训练/验证 loss
- 每 epoch 的评估指标
- HF Hub repo URL
- 任何报错/调整

### 步骤 6：告知我

Kaggle 跑完后，把 `HF_USER` 和 run URL 发我，我继续执行：
- **Task 25**: `scripts/merge_and_export_onnx.py`（本地合并 + ONNX 导出）
- **Task 29-31**: 最终评估 + 集成

---

## ⚠️ API 适配说明（重要）

笔记本 Cell 2 假设 Laya SDK 的 Router 有 `.multilingual.model` 属性。但实测 Laya 0.3.21 的 Router API 是：

```python
router = Router(device="cuda", preload=True)
router.models  # {'english': ('convaiinnovations/laya', None),
               #  'multilingual': ('convaiinnovations/laya', 'multilingual'),
               #  'typed-decisions': ('convaiinnovations/laya', 'typed-decisions')}
```

**没有** `router.english` / `router.multilingual` 直接属性。

**在 Kaggle 上首次运行 Cell 2 后，根据实际 API 调整**：

```python
# 可能需要改成：
from laya import Router
router = Router(device="cuda", preload=True)
router.preload(["multilingual"])
agent = router.load("multilingual")   # 返回 Agent 实例
model = agent.model                    # 拿到 HF 模型
```

如果 `router.load()` 返回的不是带 `.model` 的对象，尝试：
```python
print(type(agent))
print([a for a in dir(agent) if not a.startswith("_")])
# 找到 encoder / model / backbone 之类的属性
```

**这也是为什么 Phase 4 无法完全自动化** — Laya SDK 的 PyTorch 内部结构需要在 Kaggle 实机确认。

---

## 📋 Checklist

- [ ] 数据集上传到 Kaggle（`scam-detection-training`）
- [ ] Notebook 导入 Kaggle
- [ ] 加速器设 `GPU T4 x2`
- [ ] 数据集已附加
- [ ] `HF_TOKEN` secret 已添加
- [ ] `HF_USER` 占位符已替换
- [ ] Cell 2 API 已适配（如需）
- [ ] Run All 成功
- [ ] 两个 HF repo 已推送
- [ ] `reports/finetune-progress-v1.md` 已记录
