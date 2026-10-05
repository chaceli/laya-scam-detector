# v1 Gate 基线（训练前对照）

模型 `models/laya-onnx-multilingual-finetuned-fp16` · 2026-10-05

v2 训练启动前的完整四门基线，用于训练后逐门对比。

```
✗ Gate1 难负评测集 FPR: 0.593 (need ≤ 0.02)
✗ Gate2 holdout zh acc/recall: 0.795 / 0.925 (need ≥0.90 / ≥0.95)
✗ Gate3 手写集 scam_category acc: 0.658 (need ≥ 0.70)
✗ Gate4 rebate_scam 召回: 0.06 (need ≥ 0.80)
0/4 criteria met
```

## Gate4 的 0.06 是预期值，不是缺陷

v1 训练于 13 类体系，choice 头根本没有 `rebate_scam` 这个输出，
所以它几乎不会把样本判成该类。此处记录 0.06 是为了说明 **Gate4 已被修复为可测量**：
在此之前 holdout 采样自 test.jsonl（源自 split_dataset(real)，不含合成源），
`rebate_scam` 计数为 0，`parse_category_recall` 只会返回 None，
Gate4 无论模型训练得多好都会失败。已改为并入 `datasets/rebate_eval_real.jsonl`。

注意区分两个不同的召回口径：

- **0.06**（本表）= 14 类 *分类* 召回，v1 无此类，v2 的真实目标
- **0.923**（`reports/rebate_baseline_v1.md`）= taxonomy 无关的 is_scam 检测召回，v1 已达标
