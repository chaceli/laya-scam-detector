# 本地部署 Laya 并完成诈骗话术风险测试 — 设计文档

**日期**：2026-09-28
**作者**：Sisyphus (Orchestrator)
**状态**：已批准

---

## 1. 目标

基于上一轮 `research/laya-jev-research-report.md` 的洞察，在本地（macOS Apple M4 Pro, 24GB, 无 NVIDIA GPU）部署开源决策模型 Laya，使用 ONNX Runtime（无需 PyTorch），并在零样本条件下对**诈骗话术**进行风险评估测试。产出 CLI 工具、JSONL 原始结果、Markdown 评估报告。

**不在本期范围**：微调、生产化打包、HTTP API 服务（作为后续阶段）。

---

## 2. 用户决策摘要（来自 brainstorming）

| 维度 | 选择 |
|---|---|
| 输入类型 | 混合（单条 + 多轮对话）|
| 微调策略 | 渐进式：先零样本评估，根据结果决定是否微调 |
| 部署方式 | **ONNX Runtime**（免 PyTorch、更轻、更快）|
| 测试集 | 公开数据集 + 手写补充 |
| 输出形式 | CLI 工具 + JSONL + Markdown 报告 |

---

## 3. 架构总览

```
CLI 入口 (cli.py)
  │
  ▼
OnnxLayaClient (laya_onnx.py · 核心)
  ├─ Router (语言检测 <1ms)
  │   ├─ English/Latin → laya-onnx (512 tokens)
  │   └─ Devanagari/CJK/非 Latin → laya-multilingual-onnx (1024)
  ├─ predict(state, questions) → {answers, routing, latency_ms}
  │   ├─ Tokenize [CLS] q [SEP] [MASK] opt [SEP] state [SEP]
  │   ├─ ONNX Runtime 前向 → logits
  │   ├─ Softmax(logits / T_qtype) over [MASK] positions
  │   └─ 应用 fitted temperature（校准概率）
  └─ predict_batch(requests)
  │
  ▼
评估与报告 (eval.py · report.py)
  ├─ datasets/single.jsonl    手写 30 条单条样本
  ├─ datasets/multi.jsonl     手写 10 条多轮对话
  ├─ datasets/public.jsonl    公开数据集（SMS Spam 等）
  └─ 输出 reports/eval-<ts>.md + results-<ts>.jsonl
```

---

## 4. 核心组件

### 4.1 `OnnxLayaClient`（`src/laya_onnx.py`）

**构造参数**：`checkpoint_dir: Path, device: str = "cpu"`

**加载**：
- `tokenizer.json`（Hugging Face `tokenizers` 库）
- `model.onnx`（`ort.InferenceSession`）
- `rl_agent_config.json`（max_len、head_max_len、fitted temperatures）

**方法**：
- `predict(state, questions, max_len=None, head_max_len=None) → dict`
- `predict_batch(requests) → list[dict]`
- `_tokenize_question(q, options, head_max_len) → (token_ids, option_positions)`
- `_softmax_with_temperature(logits, qtype) → np.ndarray`

### 4.2 `Router`（`src/router.py`）

- `LATIN_SCRIPT = "latin"`
- `NON_LATIN_SCRIPTS = ["devanagari", "cjk_han", "cyrillic", "arabic", "hebrew", "tamil", "thai", ...]`
- `detect_script(text) → str`：基于 Unicode 脚本块检测
- `route(text) → Literal["english", "multilingual"]`
- `predict(state, questions) → dict`：自动选 checkpoint
- `predict_batch(requests) → list[dict]`：按 checkpoint 分组批处理

### 4.3 CLI 入口（`src/cli.py` / `main.py`）

```bash
# 单条预测
python main.py --predict "您好，我是XX快递客服..." --questions schemas/scam.json

# 批量评估
python main.py --eval --input datasets/eval.jsonl --output reports/

# 固定 checkpoint
python main.py --predict "..." --model english
python main.py --predict "..." --model multilingual

# 调试：只看路由（不下载模型）
python main.py --route "中文字符串"
```

### 4.4 测试集结构（`datasets/`）

```jsonl
{"id": "single-001", "type": "single", "language": "zh", "expected_risk": 5,
 "expected_category": "delivery_fraud",
 "state": "您好，我是XX快递客服，您有一个丢失的包裹需要理赔...",
 "source": "handwritten"}
{"id": "multi-001", "type": "multi_turn", "language": "zh", "expected_risk": 5,
 "turns": [
   {"role": "scammer", "text": "..."},
   {"role": "victim",  "text": "..."}],
 "source": "handwritten"}
{"id": "public-001", "type": "single", "language": "en", "expected_label": "spam",
 "state": "FREE entry in our2.5k comp...", "source": "SMS Spam Collection"}
```

### 4.5 Schema（`schemas/scam.json`）

```json
{
  "is_scam": {
    "type": "noul",
    "instructions": "Is this message a scam or phishing attempt?"
  },
  "risk_level": {
    "type": "score",
    "instructions": "How high is the risk this is a scam?",
    "criteria": [
      "1 - benign",
      "2 - suspicious",
      "3 - likely scam",
      "4 - high confidence scam",
      "5 - definitive fraud"
    ]
  },
  "scam_category": {
    "type": "choice",
    "instructions": "What type of scam does this most resemble?",
    "criteria": {
      "delivery_fraud": "fake package, refund scams",
      "phishing": "fake links, credential theft",
      "romance_scam": "pig butchering, emotional manipulation",
      "investment_scam": "fake returns, crypto fraud",
      "impersonation": "fake officials, fake customer service",
      "lottery_scam": "fake prizes, lucky draws",
      "benign": "legitimate message"
    }
  }
}
```

---

## 5. 数据流

### 5.1 单条推理

```
1. predict(state, questions)
2. Router.detect_script(state) → 选 checkpoint
4. _tokenize_question(q, options, head_max_len) → token_ids, marker_pos
   拼接 → total_token_seq_len = CLS q + MASK opt1..optN + SEP state SEP
   截断至 max_len
5. ONNX session.run({input_ids, attention_mask, marker_pos, marker_mask, qtype}) → logits
6. softmax(logits / T_qtype) over [MASK] positions → probabilities
7. 构造 answers：
   - noul → P(true)
   - score → 加权期望等级
   - choice → argmax + 完整分布
8. 返回 {answers, routing: {model, repo, reason, script}, latency_ms}
```

### 5.2 多轮对话

```
1. 接收 turns: [{scammer, victim}, ...]
2. 拼接（默认）：
   "[S1] {scammer_turn_1}\n[V1] {victim_turn_1}\n[S2] {scammer_turn_2}\n..."
   长度 > max_len 时：保留头部 + 尾部 + 中间省略
3. 用 scam_schema 评估拼接后的 state
4. 备选：滑动窗口 — 把对话切成 N 个 chunk，每个 chunk 独立评估，
   最后聚合（max risk / majority vote category）
```

### 5.3 批量评估

```
1. 加载 datasets/eval.jsonl
2. 每条样本 → 构造 state → router.predict → results.jsonl
3. 聚合统计：accuracy@is_scam, MAE@risk_level, confusion_matrix, ECE, latency
4. 渲染 reports/eval-<ts>.md
```

---

## 6. ONNX 输入/输出契约

```python
# 5 个输入
"input_ids":      int64[batch, seq_len]
"attention_mask": int64[batch, seq_len]
"marker_pos":     int64[batch, max_options]    # 每个 [MASK] 在序列中的位置
"marker_mask":    bool[batch, max_options]     # 有效选项掩码
"qtype":          int64[batch]                 # 0=choice, 1=score, 2=noul

# 1 个输出
"logits":         float32[batch, max_options]
```

**重要约束**：每次 ONNX 调用只评估一个 question 类型。多个 questions 需要多次调用。

---

## 7. 错误处理

| 失败场景 | 处理策略 |
|---|---|
| 模型未下载 | CLI 启动前 fail-fast + 打印 huggingface-cli 下载命令 |
| ONNX Runtime 加载失败 | 错误透传，提示检查 onnxruntime 版本（≥1.18）与架构 |
| state 超出 max_len | 头部+尾部保留，中间省略，warn 一次 |
| 选项超过 head_max_len | 截断过长描述；不足 2 选项则报错 |
| 同名 choice label / 空 criteria | 拒绝请求，错误消息点明字段 |
| state 为空 | 拒绝请求 |
| 推理 OOM（M4 Pro 24GB 罕见）| 捕获 ort 异常，回退更短 max_len 重试 |
| 网络抖动下载 | 重试 3 次，指数退避 |
| 所有 question 都失败 | exit code 2，stderr 输出问题列表 |

**日志策略**：warn/error → stderr；正常 JSON → stdout（便于管道）。

---

## 8. 测试与成功标准

### 8.1 烟雾测试（必须通过）

```python
# tests/test_smoke.py
def test_router_detects_latin_and_chinese()
def test_predict_returns_structured_answers()
def test_batch_consistent_with_single()
def test_cli_help_exits_zero()
```

### 8.2 评估指标

| 指标 | 目标 | 备注 |
|---|---|---|
| is_scam AUC | ≥ 0.75 | 公开数据集零样本基线 |
| is_scam F1 @ 0.5 | ≥ 0.65 | 同上 |
| risk_level MAE | ≤ 1.5 | 5 级量表 |
| scam_category accuracy | ≥ 0.30 | 6+1 类别，零样本期望较低 |
| ECE | 报告数值 | 后续可拟合温度优化 |
| p50 latency | 报告数值 | 评估报告必含 |
| 通过率 | smoke 100% | 推理 + 评估跑通 |

### 8.3 评估方法

- **手写**：30-60 条样本，覆盖 7 类诈骗（快递/刷单/冒充公检法/杀猪盘/钓鱼/中奖/贷款）+ 正常消息，混入中英文
- **公开**：SMS Spam Collection（英文 ~5.5k 条）；Hugging Face 中文诈骗话术数据集（如 `AcmcUkDistributedAi/ScamDialogue`）
- **结果矩阵**：每类诈骗的 precision/recall
- **失败案例分析**：哪些被漏判？哪些被误判？

### 8.4 微调触发条件（下一期决策）

- AUC < 0.7 或 F1 < 0.6 → 强烈建议微调
- AUC 0.7-0.85 → 报告局限性，建议业务增加规则引擎
- AUC > 0.85 → 零样本基线已够用

微调路径：准备 1k+ 标注 → Kaggle 2×T4 跑 laya 微调 notebook → 导出 ONNX → 替换 `models/laya-onnx-en/`。

---

## 9. 文件结构

```
laya/
├── README.md                            # 项目说明 + 快速开始
├── docs/
│   └── plans/
│       ├── 2026-09-28-laya-scam-detection-design.md   # 本文档
│       └── 2026-09-28-laya-scam-detection-impl.md     # 由 writing-plans 生成
├── research/
│   └── laya-jev-research-report.md      # 已有
├── src/
│   ├── __init__.py
│   ├── laya_onnx.py                     # OnnxLayaClient
│   ├── router.py                        # 语言路由
│   ├── tokenizer.py                     # token 化助手
│   ├── cli.py                           # CLI 入口
│   ├── eval.py                          # 批量评估
│   └── report.py                        # Markdown 报告渲染
├── datasets/
│   ├── single.jsonl                     # 30 条手写单条
│   ├── multi.jsonl                      # 10 条手写多轮
│   ├── public.jsonl                     # 公开数据集
│   └── eval.jsonl                       # 合并（生成）
├── schemas/
│   └── scam.json                        # 三个原语 schema
├── models/                              # gitignore
│   ├── laya-onnx-en/                    # huggingface-cli download
│   └── laya-onnx-multilingual/
├── reports/                             # gitignore
│   ├── eval-<timestamp>.md
│   └── results-<timestamp>.jsonl
├── tests/
│   ├── __init__.py
│   ├── test_smoke.py
│   └── fixtures/
├── scripts/
│   ├── download_models.sh               # 下载两个 checkpoint
│   └── build_eval_set.py                 # 把三个 sources 合并为 eval.jsonl
├── pyproject.toml                       # 依赖：onnxruntime, tokenizers, numpy, huggingface_hub
└── .gitignore                           # models/, reports/, __pycache__, .venv/
```

---

## 10. 依赖

```toml
# pyproject.toml
[project]
name = "laya-scam-detector"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "onnxruntime>=1.18",
    "tokenizers>=0.15",
    "numpy>=1.24",
    "huggingface_hub>=0.24",
]
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov", "ruff"]
```

**不引入**：torch、transformers、laya SDK（保持 ONNX 路径纯净）。

---

## 11. 风险与缓解

| 风险 | 缓解 |
|---|---|
| `inferenceprince/laya-onnx` 没有多语版 | 退化为英文 checkpoint 在中文上 zero-shot（已知会"自信地错"，但能跑通） |
| huggingface-cli 拉取失败 | 备选：手动下载 tarball |
| ONNX Runtime 不支持 M4 Pro | onnxruntime>=1.18 已支持 arm64；若失败回退 to `pip install onnxruntime-cpu` |
| 0.3.x 接口变更 | 锁定 `inferenceprince/laya-onnx` 版本，pin commit |
| 测试样本构造主观 | 标注双盲：构造 → 标注 → 复核 |
| 公开数据集下载慢/失败 | 可全部用手写 30 条 + 多语 10 条完成最低评估 |