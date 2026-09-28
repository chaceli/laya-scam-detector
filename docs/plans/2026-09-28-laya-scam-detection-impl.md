# Laya 本地部署 + 诈骗话术风险测试 — 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 macOS Apple M4 Pro 上通过 ONNX Runtime 部署开源决策模型 Laya（无 PyTorch），并在零样本条件下评估其对中文/英文诈骗话术的风险识别能力。

**Architecture:** 自写 ONNX 推理客户端包装 `inferenceprince/laya-onnx`；Router 按 Unicode 脚本自动选 checkpoint；CLI + JSONL + Markdown 报告三层产出；TDD 驱动，按烟雾测试 → 评估脚本 渐进推进。

**Tech Stack:** Python 3.10+, onnxruntime ≥1.18, tokenizers, numpy, huggingface_hub; pytest; bash; git

**设计文档:** `docs/plans/2026-09-28-laya-scam-detection-design.md`

---

## 工作分解总览

```
Phase 1: 项目骨架              (Task 1-4)
Phase 2: 模型下载与验证        (Task 5-7)
Phase 3: 数据集与 schema       (Task 8-11)
Phase 4: 核心组件（TDD）      (Task 12-19)
Phase 5: CLI 与评估            (Task 20-25)
Phase 6: 烟雾测试与首次评估    (Task 26-29)
Phase 7: 报告渲染              (Task 30-31)
```

并行机会：Task 8/9（手写数据集 A/B 可分别写）、Task 12-14（tokenizer/router 可同步进行）

---

## Phase 1: 项目骨架

### Task 1: 初始化 pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`

**Step 1:** 写 pyproject.toml

```toml
[project]
name = "laya-scam-detector"
version = "0.1.0"
description = "Local Laya deployment for scam-phrase risk evaluation via ONNX Runtime"
requires-python = ">=3.10"
dependencies = [
    "onnxruntime>=1.18",
    "tokenizers>=0.15",
    "numpy>=1.24",
    "huggingface_hub>=0.24",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov", "ruff>=0.6"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --strict-markers"
```

**Step 2:** 写 .gitignore

```
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
.venv/
venv/
.env

models/
reports/
*.onnx
*.safetensors

.DS_Store
*.swp
```

**Step 3:** 提交

```bash
git add pyproject.toml .gitignore
git commit -m "chore: bootstrap project with pyproject.toml and .gitignore"
```

---

### Task 2: 创建目录骨架

**Files:**
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/.gitkeep`
- Create: `datasets/.gitkeep`
- Create: `schemas/.gitkeep`
- Create: `models/.gitkeep`
- Create: `reports/.gitkeep`
- Create: `scripts/.gitkeep`

**Step 1:** 创建所有空目录（用 mkdir -p）

```bash
mkdir -p src tests tests/fixtures datasets schemas models reports scripts
touch src/__init__.py tests/__init__.py tests/fixtures/.gitkeep \
      datasets/.gitkeep schemas/.gitkeep models/.gitkeep \
      reports/.gitkeep scripts/.gitkeep
```

**Step 2:** 提交

```bash
git add src tests datasets schemas models reports scripts
git commit -m "chore: create project directory skeleton"
```

---

### Task 3: 设置 Python 环境

**Step 1:** 创建 conda/venv 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Step 2:** 安装基础依赖

```bash
pip install -e ".[dev]"
```

Expected: 安装 onnxruntime, tokenizers, numpy, huggingface_hub, pytest, ruff 全成功。

**Step 3:** 验证 Python 可导入

```bash
python -c "import onnxruntime as ort; print('onnxruntime', ort.__version__)"
python -c "import tokenizers; print('tokenizers', tokenizers.__version__)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
```

Expected: 四个 print 都输出对应版本号。

---

### Task 4: 创建 README

**Files:**
- Create: `README.md`

**Step 1:** 写 README.md

```markdown
# Laya Scam-Phrase Detector

Local ONNX Runtime deployment of the Laya decision model ([luaya.convaiinnovations.com](https://laya.convaiinnovations.com/)) for scam-phrase risk evaluation. No PyTorch dependency.

See `research/laya-jev-research-report.md` for background.

## Quick start

```bash
pip install -e ".[dev]"
bash scripts/download_models.sh
python main.py --predict "您好，我是XX快递客服..." --questions schemas/scam.json
```

## Status

Phase 1: project skeleton (this commit).
```

**Step 2:** 提交

```bash
git add README.md
git commit -m "docs: add README with quick-start placeholder"
```

---

## Phase 2: 模型下载与验证

### Task 5: 写下载脚本

**Files:**
- Create: `scripts/download_models.sh`

**Step 1:** 写下载脚本

```bash
#!/usr/bin/env bash
# Download Laya ONNX checkpoints (English + Multilingual) from Hugging Face.
# Idempotent: skips already-downloaded files.

set -euo pipefail

REPO_EN="inferenceprince/laya-onnx"
REPO_ML="inferenceprince/laya-multilingual-onnx"  # may not exist yet
TARGET_EN="models/laya-onnx-en"
TARGET_ML="models/laya-onnx-multilingual"

download() {
    local repo="$1"
    local target="$2"
    if [ -d "$target" ] && [ -f "$target/model.onnx" ]; then
        echo "✓ $target already exists, skipping"
        return 0
    fi
    echo "↓ Downloading $repo → $target"
    mkdir -p "$target"
    if ! huggingface-cli download "$repo" --local-dir "$target"; then
        echo "✗ Failed to download $repo"
        return 1
    fi
}

download "$REPO_EN" "$TARGET_EN"

if huggingface-cli download "$REPO_ML" --local-dir "$TARGET_ML" 2>/dev/null; then
    echo "✓ Multilingual checkpoint downloaded"
else
    echo "⚠ Multilingual checkpoint unavailable; fallback to English for non-Latin"
fi
```

**Step 2:** 给予执行权限

```bash
chmod +x scripts/download_models.sh
```

**Step 3:** 提交

```bash
git add scripts/download_models.sh
git commit -m "feat(scripts): add model download script for ONNX checkpoints"
```

---

### Task 6: 下载英文 checkpoint

**Step 1:** 运行下载脚本

```bash
bash scripts/download_models.sh
```

Expected: 下载 `inferenceprince/laya-onnx` 到 `models/laya-onnx-en/`。

**Step 2:** 验证文件存在

```bash
ls -lh models/laya-onnx-en/
test -f models/laya-onnx-en/model.onnx
test -f models/laya-onnx-en/tokenizer/tokenizer.json
test -f models/laya-onnx-en/rl_agent_config.json
```

Expected: 看到 `model.onnx` (~840MB), `tokenizer/tokenizer.json` (~4MB), `rl_agent_config.json` (几 KB)。

**Step 3:** 提交下载配置（不是模型本身，模型不入 git）

```bash
git status  # 确认 models/ 在 .gitignore 中，没有被加入
git log --oneline -3
```

---

### Task 7: 编写 ONNX 加载烟雾测试

**Files:**
- Create: `tests/test_model_load.py`

**Step 1:** 写测试

```python
"""Smoke test for ONNX model loading."""
import json
from pathlib import Path

import onnxruntime as ort
import pytest
from tokenizers import Tokenizer


CHECKPOINT_DIR = Path("models/laya-onnx-en")


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.onnx").exists(),
    reason="ONNX checkpoint not downloaded; run scripts/download_models.sh",
)
def test_model_onnx_loads():
    session = ort.InferenceSession(
        CHECKPOINT_DIR / "model.onnx",
        providers=["CPUExecutionProvider"],
    )
    assert session is not None
    inputs = [inp.name for inp in session.get_inputs()]
    assert "input_ids" in inputs
    assert "marker_pos" in inputs


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "tokenizer/tokenizer.json").exists(),
    reason="tokenizer not downloaded",
)
def test_tokenizer_loads():
    tok = Tokenizer.from_file(str(CHECKPOINT_DIR / "tokenizer/tokenizer.json"))
    encoded = tok.encode("Hello world")
    assert len(encoded.ids) == 2


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "rl_agent_config.json").exists(),
    reason="rl_agent_config not downloaded",
)
def test_config_loads():
    cfg = json.loads((CHECKPOINT_DIR / "rl_agent_config.json").read_text())
    assert "max_len" in cfg or "head_max_len" in cfg
```

**Step 2:** 运行测试

```bash
pytest tests/test_model_load.py -v
```

Expected: 全部 PASS 或 SKIP（如果模型未下载则 SKIP）。

**Step 3:** 提交

```bash
git add tests/test_model_load.py
git commit -m "test: add smoke tests for ONNX model, tokenizer, config loading"
```

---

## Phase 3: 数据集与 schema

### Task 8: 写 scam schema

**Files:**
- Create: `schemas/scam.json`

**Step 1:** 写 schema

```json
{
  "is_scam": {
    "type": "noul",
    "instructions": "Is this message a scam, fraud, phishing, or social engineering attempt?"
  },
  "risk_level": {
    "type": "score",
    "instructions": "How high is the risk that this is malicious?",
    "criteria": [
      "1 - clearly benign (normal message)",
      "2 - mildly suspicious (some red flags)",
      "3 - likely scam (multiple fraud signals)",
      "4 - high confidence scam (typical fraud pattern)",
      "5 - definitive fraud (obvious scam)"
    ]
  },
  "scam_category": {
    "type": "choice",
    "instructions": "What category of scam does this most resemble?",
    "criteria": {
      "delivery_fraud": "fake courier, refund, or lost package",
      "phishing": "fake links, credential theft, account verification",
      "romance_scam": "pig butchering, emotional manipulation, long game",
      "investment_scam": "fake returns, crypto, stock tips",
      "impersonation": "fake police, government, bank, customer service",
      "lottery_scam": "fake prize, lucky draw, congratulations",
      "loan_scam": "fake loan offers, predatory lending",
      "benign": "legitimate message"
    }
  }
}
```

**Step 2:** 提交

```bash
git add schemas/scam.json
git commit -m "feat(schemas): add scam detection schema with three primitives"
```

---

### Task 9: 手写单条中文测试集

**Files:**
- Create: `datasets/single.jsonl`

**Step 1:** 写 30 条单条测试样本

每条 JSONL 一行。覆盖 7 类诈骗 + 正常消息，混入中英文。

```jsonl
{"id": "single-zh-001", "type": "single", "language": "zh", "expected_risk": 5, "expected_category": "delivery_fraud", "state": "您好，我是XX快递客服，您有一个包裹在运输途中丢失，现需要您配合操作进行理赔，请点击链接填写个人信息：http://kd-gs.xyz/claim。", "source": "handwritten"}
{"id": "single-zh-002", "type": "single", "language": "zh", "expected_risk": 5, "expected_category": "lottery_scam", "state": "恭喜您！您的号码被抽中二等奖，奖金10万元，请尽快联系客服领取，扣费后请联系。", "source": "handwritten"}
{"id": "single-zh-003", "type": "single", "language": "zh", "expected_risk": 4, "expected_category": "romance_scam", "state": "亲爱的，我在柬埔寨遇到了麻烦，需要您帮我汇一笔钱过去，等我回去就还您，并且我会娶您。", "source": "handwritten"}
{"id": "single-zh-004", "type": "single", "language": "zh", "expected_risk": 5, "expected_category": "impersonation", "state": "【最高人民检察院】您涉嫌洗钱，已立案侦查，请将资金转入安全账户配合调查，联系警官00861-396-xxx。", "source": "handwritten"}
{"id": "single-zh-005", "type": "single", "language": "zh", "expected_risk": 5, "expected_category": "investment_scam", "state": "内部消息！这只股票明日必涨30%，老师带您精准建仓，仅剩3个名额，添加老师微信领牛股。", "source": "handwritten"}
{"id": "single-zh-006", "type": "single", "language": "zh", "expected_risk": 4, "expected_category": "loan_scam", "state": "无需征信，秒到账10万！低息无抵押贷款，下载App即可申请，额度高放款快。", "source": "handwritten"}
{"id": "single-zh-007", "type": "single", "language": "zh", "expected_risk": 5, "expected_category": "phishing", "state": "您的银行账户将被停用，请点击链接 https://icbc-verify.tk 重新认证身份，24小时内未验证将冻结账户。", "source": "handwritten"}
{"id": "single-zh-008", "type": "single", "language": "zh", "expected_risk": 1, "expected_category": "benign", "state": "您好，明天下午3点我们在咖啡厅见面，麻烦带一下那份文件，谢谢。", "source": "handwritten"}
{"id": "single-zh-009", "type": "single", "language": "zh", "expected_risk": 1, "expected_category": "benign", "state": "妈，我今晚回家吃饭，大概6点到家。", "source": "handwritten"}
{"id": "single-zh-010", "type": "single", "language": "zh", "expected_risk": 4, "expected_category": "delivery_fraud", "state": "【京东白条】尊敬的用户，您的白条账户存在异常，请立即联系客服400-xxx-xxxx 进行账户核实，否则将影响您的征信。", "source": "handwritten"}
{"id": "single-zh-011", "type": "single", "language": "zh", "expected_risk": 2, "expected_category": "benign", "state": "您的快递已到达菜鸟驿站，取件码123456，请凭码取件。", "source": "handwritten"}
{"id": "single-zh-012", "type": "single", "language": "zh", "expected_risk": 4, "expected_category": "impersonation", "state": "您好，我是XX市公安局的，您涉嫌一起跨境诈骗案，需要您配合调查并将名下资产转移到安全账户。", "source": "handwritten"}
{"id": "single-zh-013", "type": "single", "language": "zh", "expected_risk": 5, "expected_category": "romance_scam", "state": "宝贝儿，我在这边投资了一个平台，稳赚不赔，你要不要和我一起？先投5000试试，我保证你一周翻倍。", "source": "handwritten"}
{"id": "single-zh-014", "type": "single", "language": "zh", "expected_risk": 5, "expected_category": "phishing", "state": "您的Apple ID在异地登录，已被锁定，如非本人操作请点此链接解锁：https://apple-id-unlock.buzz", "source": "handwritten"}
{"id": "single-zh-015", "type": "single", "language": "zh", "expected_risk": 5, "expected_category": "investment_scam", "state": "XX数字货币交易所，充值返利20%，注册即送100USDT，提现秒到，老师一对一指导，带您财富自由。", "source": "handwritten"}
{"id": "single-zh-016", "type": "single", "language": "zh", "expected_risk": 1, "expected_category": "benign", "state": "您的快递已签收，感谢您选择京东，期待再次为您服务。", "source": "handwritten"}
{"id": "single-zh-017", "type": "single", "language": "zh", "expected_risk": 1, "expected_category": "benign", "state": "您好，请帮我查一下这个月的账单，谢谢。", "source": "handwritten"}
{"id": "single-zh-018", "type": "single", "language": "zh", "expected_risk": 3, "expected_category": "lottery_scam", "state": "恭喜您被抽中成为本周幸运用户，奖品为iPhone 15一部，请支付99元运费即可领取。", "source": "handwritten"}
{"id": "single-zh-019", "type": "single", "language": "zh", "expected_risk": 4, "expected_category": "loan_scam", "state": "您的信用卡额度已提升至20万，无需任何审核，请立即登录 http://credit-apply.buzz 激活使用。", "source": "handwritten"}
{"id": "single-zh-020", "type": "single", "language": "zh", "expected_risk": 1, "expected_category": "benign", "state": "老板，明天上午9点的会议改到下午3点，麻烦您准时参加。", "source": "handwritten"}
{"id": "single-en-001", "type": "single", "language": "en", "expected_risk": 5, "expected_category": "phishing", "state": "URGENT: Your PayPal account has been limited. Click here to verify: http://paypa1-secure.tk/login", "source": "handwritten"}
{"id": "single-en-002", "type": "single", "language": "en", "expected_risk": 5, "expected_category": "investment_scam", "state": "Make $5000/day with our secret crypto trading algorithm. 100% guaranteed returns. Join now: https://moon-cash.io", "source": "handwritten"}
{"id": "single-en-003", "type": "single", "language": "en", "expected_risk": 4, "expected_category": "romance_scam", "state": "My darling, I'm stuck in Manila and lost my wallet. Please wire $2000 to this Western Union and I'll pay you back when I return.", "source": "handwritten"}
{"id": "single-en-004", "type": "single", "language": "en", "expected_risk": 5, "expected_category": "impersonation", "state": "This is Officer Johnson from the IRS. You owe back taxes. If you don't pay immediately, a warrant will be issued for your arrest.", "source": "handwritten"}
{"id": "single-en-005", "type": "single", "language": "en", "expected_risk": 1, "expected_category": "benign", "state": "Hi, just confirming our meeting at 3pm tomorrow. Looking forward to it.", "source": "handwritten"}
{"id": "single-en-006", "type": "single", "language": "en", "expected_risk": 1, "expected_category": "benign", "state": "Your Amazon order #12345 has shipped and will be delivered on Friday.", "source": "handwritten"}
{"id": "single-en-007", "type": "single", "language": "en", "expected_risk": 4, "expected_category": "lottery_scam", "state": "CONGRATULATIONS! You've won $1,000,000 in our sweepstakes! Send $500 processing fee to claim your prize!", "source": "handwritten"}
{"id": "single-en-008", "type": "single", "language": "en", "expected_risk": 3, "expected_category": "loan_scam", "state": "Pre-approved personal loan up to $50,000 with no credit check. Apply today and get funds in 24 hours!", "source": "handwritten"}
{"id": "single-en-009", "type": "single", "language": "en", "expected_risk": 5, "expected_category": "delivery_fraud", "state": "FedEx: Package on hold. Verify address at https://fedex-tracking.buzz to avoid return.", "source": "handwritten"}
{"id": "single-en-010", "type": "single", "language": "en", "expected_risk": 1, "expected_category": "benign", "state": "Mom is fine. Dad's blood pressure was a bit high but the doctor says it's manageable. Talk tomorrow.", "source": "handwritten"}
```

**Step 2:** 验证 JSONL 格式

```bash
python -c "import json; lines = [json.loads(l) for l in open('datasets/single.jsonl')]; print(f'{len(lines)} samples loaded')"
```

Expected: `30 samples loaded`。

**Step 3:** 提交

```bash
git add datasets/single.jsonl
git commit -m "feat(datasets): add 30 handwritten single-message scam test samples (ZH+EN)"
```

---

### Task 10: 手写多轮对话测试集

**Files:**
- Create: `datasets/multi.jsonl`

**Step 1:** 写 8 条多轮对话样本（典型诈骗剧本）

```jsonl
{"id": "multi-zh-001", "type": "multi_turn", "language": "zh", "expected_risk": 5, "expected_category": "impersonation", "source": "handwritten", "turns": [{"role": "scammer", "text": "您好，我是XX检察院王检察官，工号00291298。"}, {"role": "victim", "text": "什么事？"}, {"role": "scammer", "text": "您涉嫌一起跨境洗钱案，涉案金额280万，已被通缉。"}, {"role": "victim", "text": "我没有啊！"}, {"role": "scammer", "text": "需要您配合调查，将名下存款转入安全账户接受核查，否则将强制执行。"}, {"role": "victim", "text": "怎么转？"}, {"role": "scammer", "text": "请下载指定App登录，账户是6217********，先转50万保证金。"}]}
{"id": "multi-zh-002", "type": "multi_turn", "language": "zh", "expected_risk": 5, "expected_category": "romance_scam", "source": "handwritten", "turns": [{"role": "scammer", "text": "在吗？"}, {"role": "victim", "text": "在的"}, {"role": "scammer", "text": "我最近在做一个投资平台，回报特别高，要不要一起？"}, {"role": "victim", "text": "什么平台？"}, {"role": "scammer", "text": "XX数字资产交易所，先投5000试试水，我保证你一周翻倍。"}, {"role": "victim", "text": "能信吗？"}, {"role": "scammer", "text": "放心，我已经赚了几十万了，我把账户截图给你看，绝对稳赚不赔。"}]}
{"id": "multi-zh-003", "type": "multi_turn", "language": "zh", "expected_risk": 5, "expected_category": "delivery_fraud", "source": "handwritten", "turns": [{"role": "scammer", "text": "【京东客服】您购买的商品出现质量问题，我们将三倍赔偿。"}, {"role": "victim", "text": "真的吗？"}, {"role": "scammer", "text": "请加QQ群123456，我们客服会指导您操作。"}, {"role": "victim", "text": "好的"}, {"role": "scammer", "text": "请先在群里领取50元体验金，然后点击链接开通理赔通道。"}, {"role": "victim", "text": "链接是什么？"}, {"role": "scammer", "text": "https://jd-claim.buzz，请填写身份证号、银行卡号、验证码。"}]}
{"id": "multi-en-001", "type": "multi_turn", "language": "en", "expected_risk": 5, "expected_category": "tech_support_scam", "source": "handwritten", "turns": [{"role": "scammer", "text": "Hello, this is Microsoft Support. Your computer has been infected with a virus."}, {"role": "victim", "text": "Oh no, what should I do?"}, {"role": "scammer", "text": "You need to install TeamViewer so I can fix it remotely."}, {"role": "victim", "text": "Ok, how?"}, {"role": "scammer", "text": "Go to teamviewer-msfix.tech and enter the code 987654. I'll take control."}, {"role": "scammer", "text": "I see you have a banking trojan. We need to secure your accounts immediately."}, {"role": "victim", "text": "How much will it cost?"}, {"role": "scammer", "text": "$499 for lifetime subscription. Please pay via gift cards from CVS."}]}
{"id": "multi-en-002", "type": "multi_turn", "language": "en", "expected_risk": 4, "expected_category": "investment_scam", "source": "handwritten", "turns": [{"role": "scammer", "text": "Hi, I'm a financial advisor. Want to learn about a great opportunity?"}, {"role": "victim", "text": "Sure, what is it?"}, {"role": "scammer", "text": "Bitcoin arbitrage. Our AI bot guarantees 5% daily returns with no risk."}, {"role": "victim", "text": "That sounds too good"}, {"role": "scammer", "text": "Many clients doubled their money in a month. Start with just $1000."}, {"role": "scammer", "text": "I can show you our verified returns. Click this link to see audited results."}]}
{"id": "multi-zh-004", "type": "multi_turn", "language": "zh", "expected_risk": 1, "expected_category": "benign", "source": "handwritten", "turns": [{"role": "user", "text": "在吗？"}, {"role": "friend", "text": "在的，怎么了"}, {"role": "user", "text": "周末一起去爬山吗？"}, {"role": "friend", "text": "可以啊，几点出发？"}, {"role": "user", "text": "早上7点，老地方见"}, {"role": "friend", "text": "好嘞，记得带水和防晒霜"}]}
{"id": "multi-en-003", "type": "multi_turn", "language": "en", "expected_risk": 1, "expected_category": "benign", "source": "handwritten", "turns": [{"role": "user", "text": "Are you free for lunch tomorrow?"}, {"role": "colleague", "text": "Yes, around noon works."}, {"role": "user", "text": "Great, let's meet at the new Italian place."}, {"role": "colleague", "text": "Perfect. I'll book a table for 12:30."}]}
{"id": "multi-zh-005", "type": "multi_turn", "language": "zh", "expected_risk": 4, "expected_category": "phishing", "source": "handwritten", "turns": [{"role": "scammer", "text": "您的微信号存在违规行为，已被限制登录。"}, {"role": "victim", "text": "怎么会？"}, {"role": "scammer", "text": "请访问 http://wechat-unlock.tk 进行实名认证。"}, {"role": "victim", "text": "要填什么信息？"}, {"role": "scammer", "text": "您的手机号、身份证号、银行卡号以及收到的验证码。"}, {"role": "victim", "text": "验证码也可以给别人吗？"}, {"role": "scammer", "text": "是的，这是腾讯官方的安全验证流程。"}]}
```

**Step 2:** 验证 JSONL

```bash
python -c "import json; lines = [json.loads(l) for l in open('datasets/multi.jsonl')]; print(f'{len(lines)} multi-turn samples loaded'); print(f'total turns: {sum(len(l[\"turns\"]) for l in lines)}')"
```

Expected: `8 multi-turn samples loaded` + `total turns: 56`（按实际数）

**Step 3:** 提交

```bash
git add datasets/multi.jsonl
git commit -m "feat(datasets): add 8 handwritten multi-turn dialogue scam samples"
```

---

### Task 11: 合并脚本与最终数据集

**Files:**
- Create: `scripts/build_eval_set.py`
- Create: `datasets/eval.jsonl`（由脚本生成）

**Step 1:** 写合并脚本

```python
"""Combine datasets/{single,multi}.jsonl into datasets/eval.jsonl."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    ROOT / "datasets" / "single.jsonl",
    ROOT / "datasets" / "multi.jsonl",
]
TARGET = ROOT / "datasets" / "eval.jsonl"


def main() -> int:
    seen_ids: set[str] = set()
    out_lines = []
    for src in SOURCES:
        if not src.exists():
            print(f"⚠ missing source: {src}", file=sys.stderr)
            continue
        with src.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj["id"] in seen_ids:
                    raise ValueError(f"duplicate id: {obj['id']}")
                seen_ids.add(obj["id"])
                out_lines.append(line)
    TARGET.write_text("\n".join(out_lines) + "\n")
    print(f"✓ Wrote {len(out_lines)} samples to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2:** 运行合并脚本

```bash
python scripts/build_eval_set.py
```

Expected: `✓ Wrote 38 samples to .../datasets/eval.jsonl`

**Step 3:** 提交

```bash
git add scripts/build_eval_set.py datasets/eval.jsonl
git commit -m "feat(datasets): add merge script and combined eval.jsonl (38 samples)"
```

---

## Phase 4: 核心组件（TDD）

### Task 12: 写 router 脚本检测

**Files:**
- Create: `tests/test_router.py`
- Create: `src/router.py`

**Step 1:** 先写失败测试

```python
"""Tests for Unicode-script detection and language routing."""
import pytest

from src.router import detect_script, ScriptRouter


class TestDetectScript:
    def test_pure_latin_returns_latin(self):
        assert detect_script("Hello world, this is English.") == "latin"

    def test_chinese_hanzi_returns_cjk(self):
        assert detect_script("您好世界") == "cjk_han"

    def test_devanagari_returns_devanagari(self):
        assert detect_script("नमस्ते") == "devanagari"

    def test_arabic_returns_arabic(self):
        assert detect_script("مرحبا") == "arabic"

    def test_cyrillic_returns_cyrillic(self):
        assert detect_script("Привет") == "cyrillic"

    def test_mixed_text_returns_dominant_non_latin(self):
        # 10 Chinese + 1 English word → still cjk_han
        assert detect_script("您好 您好 您好 您好 您好 hello") == "cjk_han"

    def test_empty_returns_latin(self):
        assert detect_script("") == "latin"


class TestScriptRouter:
    def test_latin_routes_to_english(self):
        r = ScriptRouter()
        assert r.route("Hello world") == "english"

    def test_chinese_routes_to_multilingual(self):
        r = ScriptRouter()
        assert r.route("您好世界") == "multilingual"

    def test_short_latin_routes_to_english(self):
        r = ScriptRouter()
        assert r.route("OK") == "english"
```

**Step 2:** 运行测试确认失败

```bash
pytest tests/test_router.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.router'`

**Step 3:** 实现 router

```python
"""Unicode-script detection and language routing for Laya checkpoints."""
from __future__ import annotations

import unicodedata


# Map: first letter Unicode block → script tag
_NON_LATIN_SCRIPT_NAMES = {
    "CJK Unified Ideographs": "cjk_han",
    "Hangul Syllables": "hangul",
    "Devanagari": "devanagari",
    "Arabic": "arabic",
    "Hebrew": "hebrew",
    "Cyrillic": "cyrillic",
    "Greek": "greek",
    "Tamil": "tamil",
    "Thai": "thai",
    "Bengali": "bengali",
    "Gurmukhi": "gurmukhi",
    "Telugu": "telugu",
    "Tibetan": "tibetan",
    "Myanmar": "myanmar",
    "Khmer": "khmer",
    "Lao": "lao",
    "Georgian": "georgian",
    "Armenian": "armenian",
}

LATIN_SCRIPT = "latin"


def detect_script(text: str) -> str:
    """Detect dominant non-Latin script, falling back to 'latin'."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        # Get Unicode block name via unicodedata
        try:
            block = unicodedata.name(ch, "")
        except ValueError:
            block = ""
        # Block names start with the script name
        matched = False
        for prefix, tag in _NON_LATIN_SCRIPT_NAMES.items():
            if block.startswith(prefix):
                counts[tag] = counts.get(tag, 0) + 1
                matched = True
                break
        if not matched:
            counts[LATIN_SCRIPT] = counts.get(LATIN_SCRIPT, 0) + 1
    if not counts:
        return LATIN_SCRIPT
    return max(counts.items(), key=lambda kv: kv[1])[0]


class ScriptRouter:
    """Route text to English or Multilingual checkpoint based on script."""

    def route(self, text: str) -> str:
        script = detect_script(text)
        if script == LATIN_SCRIPT:
            return "english"
        return "multilingual"

    def route_with_reason(self, text: str) -> tuple[str, str]:
        """Return (model_name, reason_string) for debugging."""
        script = detect_script(text)
        if script == LATIN_SCRIPT:
            return "english", f"script={script}"
        return "multilingual", f"script={script} (non-Latin)"
```

**Step 4:** 运行测试确认通过

```bash
pytest tests/test_router.py -v
```

Expected: 全部 PASS（10 tests）。

**Step 5:** 提交

```bash
git add tests/test_router.py src/router.py
git commit -m "feat(src): add Unicode-script detector and ScriptRouter"
```

---

### Task 13: 写 ONNX 客户端 TDD 测试

**Files:**
- Create: `tests/test_laya_onnx.py`

**Step 1:** 写失败测试

```python
"""Tests for OnnxLayaClient: predict, tokenize, softmax."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.laya_onnx import OnnxLayaClient, _tokenize_question, _softmax_with_temperature


CHECKPOINT_DIR = Path("models/laya-onnx-en")
SCHEDULED = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.onnx").exists(),
    reason="ONNX checkpoint not downloaded",
)


@pytest.fixture(scope="module")
def client():
    if not (CHECKPOINT_DIR / "model.onnx").exists():
        return None
    return OnnxLayaClient(CHECKPOINT_DIR)


class TestTokenizeQuestion:
    @SCHEDULED
    def test_tokenize_choice_question_returns_token_ids_and_positions(self, client):
        tokens, positions = _tokenize_question(
            tokenizer=client.tokenizer,
            question="Which department?",
            options={"billing": "invoices and refunds", "tech": "bugs"},
            head_max_len=192,
            cls_id=client.cls_id,
            sep_id=client.sep_id,
            mask_id=client.mask_id,
        )
        assert len(tokens) > 0
        assert len(positions) == 2  # one per option
        assert all(0 <= p < len(tokens) for p in positions)


class TestSoftmax:
    def test_softmax_uniform_logit_returns_uniform_prob(self):
        logits = np.array([1.0, 1.0, 1.0])
        probs = _softmax_with_temperature(logits, temperature=1.0)
        np.testing.assert_allclose(probs, [1/3, 1/3, 1/3], atol=1e-6)

    def test_softmax_high_temp_makes_distribution_softer(self):
        logits = np.array([3.0, 0.0, 0.0])
        probs = _softmax_with_temperature(logits, temperature=10.0)
        # At T=10, the distribution is very soft
        assert probs[0] < 0.5
        assert probs[1] > 0.2

    def test_softmax_low_temp_makes_distribution_sharper(self):
        logits = np.array([3.0, 0.0, 0.0])
        probs = _softmax_with_temperature(logits, temperature=0.1)
        assert probs[0] > 0.99

    def test_softmax_probabilities_sum_to_one(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        probs = _softmax_with_temperature(logits, temperature=1.0)
        assert abs(probs.sum() - 1.0) < 1e-6


class TestPredict:
    @SCHEDULED
    def test_predict_noul_returns_probability(self, client):
        result = client.predict(
            state="This is a phishing attempt.",
            questions={"is_fraud": {"type": "noul", "instructions": "Is this a scam?"}},
        )
        assert "answers" in result
        assert "is_fraud" in result["answers"]
        assert "noul" in result["answers"]["is_fraud"]
        prob = result["answers"]["is_fraud"]["noul"]
        assert 0.0 <= prob <= 1.0

    @SCHEDULED
    def test_predict_choice_returns_distribution(self, client):
        result = client.predict(
            state="Hi, billing question about invoice 12345.",
            questions={
                "dept": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "payments", "tech": "bugs"},
                }
            },
        )
        ans = result["answers"]["dept"]
        assert "choice" in ans
        assert "probabilities" in ans
        assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-4
```

**Step 2:** 运行测试确认失败

```bash
pytest tests/test_laya_onnx.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.laya_onnx'`

---

### Task 14: 实现 _tokenize_question 与 _softmax

**Files:**
- Create: `src/laya_onnx.py`（部分）

**Step 1:** 写最小化的 _softmax 和 _tokenize_question

```python
"""ONNX Runtime wrapper for Laya decision model.

Public API:
    OnnxLayaClient(checkpoint_dir) → predict(state, questions) → dict
    _tokenize_question(...) → (token_ids, option_positions)
    _softmax_with_temperature(logits, temperature) → np.ndarray
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _softmax_with_temperature(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert logits to probabilities with temperature scaling."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = logits.astype(np.float64) / temperature
    scaled = scaled - scaled.max()  # numerical stability
    exp = np.exp(scaled)
    return (exp / exp.sum()).astype(np.float32)


def _tokenize_question(
    *,
    tokenizer,
    question: str,
    options: dict[str, str],
    head_max_len: int,
    cls_id: int,
    sep_id: int,
    mask_id: int,
) -> tuple[list[int], list[int]]:
    """Tokenize a single question into the Laya ONNX input format.

    Returns (token_ids, option_positions_in_sequence).
    """
    # Encode question text
    q_ids = tokenizer.encode("choice question: " + question, add_special_tokens=False).ids

    token_ids: list[int] = [cls_id] + q_ids + [sep_id]
    option_positions: list[int] = []

    for label, description in options.items():
        option_positions.append(len(token_ids))
        token_ids.append(mask_id)
        opt_text = f" {label}: {description}"
        opt_ids = tokenizer.encode(opt_text, add_special_tokens=False).ids
        # Truncate each option to leave room for others (rough budget)
        per_option_budget = max(8, (head_max_len - 16) // max(len(options), 1))
        token_ids.extend(opt_ids[:per_option_budget])
    token_ids.append(sep_id)

    return token_ids, option_positions
```

**Step 2:** 运行测试，验证 _softmax 与 _tokenize_question 通过

```bash
pytest tests/test_laya_onnx.py::TestSoftmax tests/test_laya_onnx.py::TestTokenizeQuestion -v
```

Expected: 6 tests pass（soft 4 + tokenize 1，已运行的就这些）+ 3 skip（predict 还没实现）

**Step 3:** 提交

```bash
git add tests/test_laya_onnx.py src/laya_onnx.py
git commit -m "feat(src): add softmax and tokenize helpers with passing unit tests"
```

---

### Task 15: 实现 OnnxLayaClient 主体

**Files:**
- Modify: `src/laya_onnx.py`

**Step 1:** 在文件末尾追加 OnnxLayaClient 类

```python
import onnxruntime as ort
from tokenizers import Tokenizer

QTYPE_CHOICE = 0
QTYPE_SCORE = 1
QTYPE_NOUL = 2


class OnnxLayaClient:
    """Laya decision model client using ONNX Runtime (no PyTorch)."""

    def __init__(self, checkpoint_dir: Path | str, providers: list[str] | None = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.session = ort.InferenceSession(
            str(self.checkpoint_dir / "model.onnx"),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(
            str(self.checkpoint_dir / "tokenizer" / "tokenizer.json")
        )
        cfg_path = self.checkpoint_dir / "rl_agent_config.json"
        self.config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        # Special token IDs (cached for hot path)
        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        self.mask_id = self.tokenizer.token_to_id("[MASK]")
        if any(t is None for t in (self.cls_id, self.sep_id, self.mask_id)):
            raise ValueError(
                "Tokenizer missing required special tokens [CLS]/[SEP]/[MASK]"
            )
        # Defaults from config
        self.default_max_len = int(self.config.get("max_len", 512))
        self.default_head_max_len = int(self.config.get("head_max_len", 192))

    def _tokenize_for_question(
        self, state: str, question_text: str, options: dict[str, str],
        max_len: int, head_max_len: int,
    ) -> tuple[list[int], list[int]]:
        """Build (token_ids, option_positions) for state + one question."""
        head_tokens, head_positions = _tokenize_question(
            tokenizer=self.tokenizer,
            question=question_text,
            options=options,
            head_max_len=head_max_len,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            mask_id=self.mask_id,
        )
        if len(head_tokens) >= max_len:
            return head_tokens[:max_len], head_positions

        remaining = max_len - len(head_tokens) - 1  # -1 for trailing [SEP]
        state_ids = self.tokenizer.encode(state, add_special_tokens=False).ids
        state_ids = state_ids[: max(0, remaining)]
        token_ids = head_tokens + state_ids + [self.sep_id]
        # Clamp positions if state truncation removed markers
        option_positions = [p for p in head_positions if p < len(token_ids)]
        return token_ids, option_positions

    def _run_one(
        self, token_ids: list[int], option_positions: list[int], qtype: int,
    ) -> np.ndarray:
        """Run ONNX forward; return raw logits [num_options]."""
        if not option_positions:
            raise ValueError("no options to score")
        seq_len = len(token_ids)
        inputs = {
            "input_ids": np.array([token_ids], dtype=np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "marker_pos": np.array([option_positions], dtype=np.int64),
            "marker_mask": np.ones((1, len(option_positions)), dtype=bool),
            "qtype": np.array([qtype], dtype=np.int64),
        }
        outputs = self.session.run(None, inputs)
        return outputs[0][0]  # [num_options]

    def _answer_for_qtype(
        self, qtype: int, logits: np.ndarray, criteria: list | dict,
        temperature: float,
    ) -> dict:
        """Convert raw logits to typed answer based on qtype."""
        probs = _softmax_with_temperature(logits, temperature)
        if qtype == QTYPE_NOUL:
            # noul uses one [MASK] (false:) and one (true:)
            # Layout depends on caller; we trust caller mapped correctly
            p_true = float(probs[1]) if len(probs) > 1 else float(probs[0])
            return {"noul": p_true, "probabilities": {"false": float(probs[0]), "true": p_true}}
        if qtype == QTYPE_CHOICE:
            labels = list(criteria.keys())
            dist = {labels[i]: float(probs[i]) for i in range(len(labels))}
            argmax_idx = int(np.argmax(probs))
            return {"choice": labels[argmax_idx], "probabilities": dist, "confidence": float(probs[argmax_idx])}
        if qtype == QTYPE_SCORE:
            # ordinal: levels 0..N-1, expected value = sum(p_i * i)
            n = len(probs)
            expected = sum(float(probs[i]) * i for i in range(n))
            levels = criteria if isinstance(criteria, list) else list(criteria.values())
            dist = {levels[i]: float(probs[i]) for i in range(n)}
            return {"score": expected, "max_score": n - 1, "distribution": dist, "confidence": float(probs.max())}
        raise ValueError(f"unknown qtype: {qtype}")

    def _temperature_for(self, qtype: int, option_count: int) -> float:
        """Look up fitted temperature; default 1.0."""
        temps = self.config.get("fitted_temperatures", {})
        # Try exact key (qtype, option_count); else (qtype); else 1.0
        if isinstance(temps, dict):
            key1 = (qtype, option_count)
            key2 = qtype
            if key1 in temps:
                return float(temps[key1])
            if key2 in temps:
                return float(temps[key2])
        return 1.0

    def predict(
        self,
        state: str,
        questions: dict,
        max_len: int | None = None,
        head_max_len: int | None = None,
    ) -> dict:
        """Run a single predict. Returns {answers, routing, latency_ms}."""
        import time
        max_len = max_len or self.default_max_len
        head_max_len = head_max_len or self.default_head_max_len

        if not isinstance(state, str) or not state:
            raise ValueError("state must be a non-empty string")

        answers: dict = {}
        started = time.perf_counter()
        for name, q in questions.items():
            qtype_str = q.get("type", "noul")
            qtype_map = {"choice": QTYPE_CHOICE, "score": QTYPE_SCORE, "noul": QTYPE_NOUL}
            qtype = qtype_map[qtype_str]
            instructions = q.get("instructions", "")
            criteria = q.get("criteria", [])
            if qtype == QTYPE_CHOICE:
                if not isinstance(criteria, dict) or len(criteria) < 2:
                    raise ValueError(f"choice question '{name}' needs criteria dict with ≥2 keys")
                options = {k: str(v) for k, v in criteria.items()}
            elif qtype == QTYPE_NOUL:
                options = {"false": "no", "true": "yes"}
            elif qtype == QTYPE_SCORE:
                if not isinstance(criteria, list) or len(criteria) < 2:
                    raise ValueError(f"score question '{name}' needs criteria list with ≥2 levels")
                options = {f"level_{i}": str(c) for i, c in enumerate(criteria)}
            else:
                raise ValueError(f"unknown question type: {qtype_str}")

            tokens, positions = self._tokenize_for_question(
                state=state,
                question_text=instructions,
                options=options,
                max_len=max_len,
                head_max_len=head_max_len,
            )
            logits = self._run_one(tokens, positions, qtype)
            temperature = self._temperature_for(qtype, len(options))
            answers[name] = self._answer_for_qtype(qtype, logits, criteria, temperature)
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "answers": answers,
            "routing": {"model": "unknown", "reason": "OnnxLayaClient direct call"},
            "latency_ms": latency_ms,
        }

    def predict_batch(self, requests: list[dict]) -> list[dict]:
        """Predict for a list of {state, questions} dicts."""
        return [self.predict(**r) for r in requests]
```

**Step 2:** 运行测试

```bash
pytest tests/test_laya_onnx.py -v
```

Expected: 全部 PASS（如果 checkpoint 已下载）或 SKIP（未下载）。

**Step 3:** 提交

```bash
git add src/laya_onnx.py
git commit -m "feat(src): implement OnnxLayaClient with predict and predict_batch"
```

---

### Task 16: 端到端冒烟测试（推理）

**Files:**
- Create: `tests/test_e2e_predict.py`

**Step 1:** 写端到端测试

```python
"""End-to-end inference test using the real ONNX model."""
import pytest

from src.laya_onnx import OnnxLayaClient


CHECKPOINT_DIR = "models/laya-onnx-en"


@pytest.fixture(scope="module")
def client():
    return OnnxLayaClient(CHECKPOINT_DIR)


@pytest.fixture(scope="module")
def schema():
    import json
    return json.loads(open("schemas/scam.json").read())


@pytest.mark.skipif(
    not __import__("os").path.exists("models/laya-onnx-en/model.onnx"),
    reason="model not downloaded",
)
class TestEndToEnd:
    def test_phishing_email_gets_high_risk(self, client, schema):
        result = client.predict(
            "URGENT: Your PayPal account has been limited. Click http://paypa1-secure.tk/login to verify.",
            schema,
        )
        is_scam = result["answers"]["is_scam"]["noul"]
        assert is_scam > 0.3, f"phishing email scored too low: {is_scam}"

    def test_benign_email_gets_low_risk(self, client, schema):
        result = client.predict(
            "Hi, just confirming our meeting at 3pm tomorrow. Looking forward to it.",
            schema,
        )
        is_scam = result["answers"]["is_scam"]["noul"]
        assert is_scam < 0.7, f"benign email scored too high: {is_scam}"

    def test_latency_under_2_seconds_on_cpu(self, client, schema):
        result = client.predict(
            "Test message", schema
        )
        assert result["latency_ms"] < 2000, f"too slow: {result['latency_ms']}ms"
```

**Step 2:** 运行

```bash
pytest tests/test_e2e_predict.py -v
```

Expected: PASS 或 SKIP（取决于是否下载）。

**Step 4:** 提交

```bash
git add tests/test_e2e_predict.py
git commit -m "test: add end-to-end inference tests for phishing/benign classification"
```

---

## Phase 17: Router + 客户端集成

### Task 17: 实现 Router.predict（顶层入口）

**Files:**
- Modify: `src/router.py`

**Step 1:** 在 `src/router.py` 末尾添加集成层

```python
"""High-level Router: combines two OnnxLayaClients (English + Multilingual)."""
from pathlib import Path

from src.laya_onnx import OnnxLayaClient


class Router:
    """Auto-routes requests to English or Multilingual checkpoint based on script."""

    def __init__(
        self,
        english_dir: Path | str,
        multilingual_dir: Path | str | None = None,
        providers: list[str] | None = None,
        preload: bool = True,
    ):
        self.english = OnnxLayaClient(english_dir, providers=providers)
        self.multilingual = None
        if multilingual_dir and Path(multilingual_dir).exists():
            try:
                self.multilingual = OnnxLayaClient(multilingual_dir, providers=providers)
            except Exception as e:
                import sys
                print(f"⚠ Multilingual checkpoint failed to load: {e}", file=sys.stderr)
        self.scripts = ScriptRouter()

    def _pick(self, text: str) -> tuple[OnnxLayaClient, str, str]:
        model_tag = self.scripts.route(text)
        if model_tag == "multilingual" and self.multilingual is not None:
            return self.multilingual, model_tag, self.scripts.route_with_reason(text)[1]
        return self.english, "english", self.scripts.route_with_reason(text)[1]

    def predict(
        self,
        state: str,
        questions: dict,
        max_len: int | None = None,
        head_max_len: int | None = None,
        model: str | None = None,
    ) -> dict:
        # Coerce state if dict
        if isinstance(state, dict):
            import json
            state = json.dumps(state, ensure_ascii=False)
        if model is None:
            client, model_tag, reason = self._pick(state)
        else:
            client = self.multilingual if model == "multilingual" and self.multilingual else self.english
            model_tag = model
            reason = "explicit override"
        result = client.predict(state, questions, max_len=max_len, head_max_len=head_max_len)
        result["routing"] = {"model": model_tag, "reason": reason}
        return result

    def route(self, state: str) -> str:
        """Just return the route decision (no inference)."""
        return self.scripts.route(state)
```

**Step 2:** 写 router 测试

```python
# tests/test_router_integration.py
import pytest

from src.router import Router


@pytest.fixture(scope="module")
def router():
    return Router(english_dir="models/laya-onnx-en", multilingual_dir="models/laya-onnx-multilingual")


@pytest.mark.skipif(
    not __import__("os").path.exists("models/laya-onnx-en/model.onnx"),
    reason="model not downloaded",
)
class TestRouterIntegration:
    def test_router_loads_english(self, router):
        assert router.english is not None

    def test_router_predict_chinese_text(self, router):
        import json
        result = router.predict(
            "您好，我是XX快递客服，您有一个包裹丢失需要理赔",
            json.loads(open("schemas/scam.json").read()),
        )
        assert result["routing"]["model"] in {"english", "multilingual"}
        assert "is_scam" in result["answers"]

    def test_explicit_model_override(self, router):
        import json
        result = router.predict(
            "Hello world",
            json.loads(open("schemas/scam.json").read()),
            model="english",
        )
        assert result["routing"]["model"] == "english"
```

**Step 3:** 运行测试

```bash
pytest tests/test_router_integration.py -v
```

Expected: PASS 或 SKIP。

**Step 4:** 提交

```bash
git add src/router.py tests/test_router_integration.py
git commit -m "feat(src): add top-level Router with auto script-based checkpoint selection"
```

---

## Phase 5: CLI 与评估

### Task 18: 实现 main.py CLI

**Files:**
- Create: `main.py`

**Step 1:** 写 main.py

```python
"""CLI for local Laya scam-phrase evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.router import Router
from src.eval import run_evaluation, load_schema
from src.report import render_report


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="laya-scam-detector",
        description="Local Laya decision model — scam-phrase risk evaluation",
    )
    parser.add_argument("--predict", help="Single state string to evaluate")
    parser.add_argument("--questions", help="JSON file with question schema")
    parser.add_argument("--model", choices=["english", "multilingual"], help="Force checkpoint")
    parser.add_argument("--route", help="Just detect script and route (no inference)")
    parser.add_argument("--eval", action="store_true", help="Run batch evaluation")
    parser.add_argument("--input", help="Input JSONL file for --eval")
    parser.add_argument("--output", default="reports/", help="Output directory for reports")
    parser.add_argument("--english-dir", default="models/laya-onnx-en")
    parser.add_argument("--multilingual-dir", default="models/laya-onnx-multilingual")

    args = parser.parse_args()

    # Mode: --route only (no model load)
    if args.route:
        from src.router import ScriptRouter
        sr = ScriptRouter()
        model = sr.route(args.route)
        print(json.dumps({"input": args.route, "routed_to": model, "reason": sr.route_with_reason(args.route)[1]}, ensure_ascii=False))
        return 0

    # Load Router
    try:
        router = Router(
            english_dir=args.english_dir,
            multilingual_dir=args.multilingual_dir,
        )
    except Exception as e:
        print(f"✗ Failed to load model: {e}", file=sys.stderr)
        print(f"  Run: bash scripts/download_models.sh", file=sys.stderr)
        return 2

    # Mode: --eval
    if args.eval:
        if not args.input:
            print("✗ --eval requires --input <file.jsonl>", file=sys.stderr)
            return 2
        schema = load_schema()
        results = run_evaluation(router, args.input, schema)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path, jsonl_path = render_report(results, schema, out_dir)
        print(f"✓ Report: {report_path}", file=sys.stderr)
        print(f"✓ Results: {jsonl_path}", file=sys.stderr)
        return 0

    # Mode: --predict
    if args.predict:
        if not args.questions:
            print("✗ --predict requires --questions <file.json>", file=sys.stderr)
            return 2
        schema = json.loads(Path(args.questions).read_text())
        try:
            result = router.predict(args.predict, schema, model=args.model)
        except Exception as e:
            print(f"✗ Inference failed: {e}", file=sys.stderr)
            return 3
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2:** 测试 help

```bash
python main.py --help
```

Expected: 显示帮助文本，exit code 0。

**Step 3:** 测试 --route

```bash
python main.py --route "您好世界"
python main.py --route "Hello world"
```

Expected（stdout）：
- `{"input": "您好世界", "routed_to": "multilingual", ...}`
- `{"input": "Hello world", "routed_to": "english", ...}`

**Step 4:** 提交

```bash
git add main.py
git commit -m "feat(cli): add main.py with --predict --route --eval modes"
```

---

### Task 19: 实现 eval.py（批量评估）

**Files:**
- Create: `src/eval.py`

**Step 1:** 写 eval.py

```python
"""Batch evaluation over JSONL test sets."""
from __future__ import annotations

import json
from pathlib import Path

from src.router import Router


def load_schema(path: str = "schemas/scam.json") -> dict:
    return json.loads(Path(path).read_text())


def _state_from_record(rec: dict) -> str:
    """Normalize a test record into a state string."""
    if rec.get("type") == "multi_turn" and "turns" in rec:
        # Join turns as "S1 ... V1 ... S2 ..."
        parts = []
        for i, turn in enumerate(rec["turns"]):
            role_short = turn.get("role", "?")[0].upper()
            parts.append(f"[{role_short}{i+1}] {turn.get('text', '')}")
        return "\n".join(parts)
    state = rec.get("state", "")
    return state


def _thresholded_is_scam(p_noul: float, threshold: float = 0.5) -> int:
    return 1 if p_noul >= threshold else 0


def run_evaluation(router: Router, input_jsonl: str, schema: dict) -> list[dict]:
    """Run all samples through router.predict; return enriched records."""
    results = []
    with open(input_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = _state_from_record(rec)
            try:
                pred = router.predict(state, schema)
            except Exception as e:
                pred = {"error": str(e), "answers": {}}
            enriched = {
                "id": rec.get("id"),
                "type": rec.get("type"),
                "language": rec.get("language"),
                "expected_risk": rec.get("expected_risk"),
                "expected_category": rec.get("expected_category"),
                "expected_label": rec.get("expected_label"),
                "source": rec.get("source"),
                "state_preview": state[:100],
                "predicted": {
                    "is_scam_noul": pred.get("answers", {}).get("is_scam", {}).get("noul"),
                    "risk_level": pred.get("answers", {}).get("risk_level", {}).get("score"),
                    "scam_category": pred.get("answers", {}).get("scam_category", {}).get("choice"),
                    "category_probs": pred.get("answers", {}).get("scam_category", {}).get("probabilities"),
                },
                "routing": pred.get("routing"),
                "latency_ms": pred.get("latency_ms"),
                "error": pred.get("error"),
            }
            results.append(enriched)
    return results
```

**Step 2:** 写 eval 测试

```python
# tests/test_eval.py
import json
from pathlib import Path

import pytest

from src.eval import _state_from_record, _thresholded_is_scam


class TestStateFromRecord:
    def test_single_message_returns_state(self):
        rec = {"type": "single", "state": "Hello world"}
        assert _state_from_record(rec) == "Hello world"

    def test_multi_turn_formats_with_speaker_tags(self):
        rec = {
            "type": "multi_turn",
            "turns": [
                {"role": "scammer", "text": "Hi"},
                {"role": "victim", "text": "Hello"},
            ],
        }
        out = _state_from_record(rec)
        assert "[S1] Hi" in out
        assert "[V2] Hello" in out


class TestThresholdedIsScam:
    def test_above_threshold_returns_1(self):
        assert _thresholded_is_scam(0.6) == 1

    def test_below_threshold_returns_0(self):
        assert _thresholded_is_scam(0.4) == 0

    def test_at_threshold_returns_1(self):
        assert _thresholded_is_scam(0.5) == 1
```

**Step 3:** 运行测试

```bash
pytest tests/test_eval.py -v
```

Expected: 5 tests pass.

**Step 4:** 提交

```bash
git add src/eval.py tests/test_eval.py
git commit -m "feat(src): add eval module with state normalization and thresholded prediction"
```

---

### Task 20: 实现 report.py（评估报告渲染）

**Files:**
- Create: `src/report.py`

**Step 1:** 写 report.py

```python
"""Render Markdown evaluation report from evaluation results."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median


def _safe_get(pred: dict, *path, default=None):
    cur = pred
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


def _summary(results: list[dict]) -> dict:
    """Compute aggregate metrics."""
    is_scam_pairs = [
        (r["predicted"]["is_scam_noul"], int(r["expected_risk"] >= 4) if r.get("expected_risk") is not None else None)
        for r in results if r["predicted"]["is_scam_noul"] is not None
    ]
    is_scam_pairs = [(p, y) for p, y in is_scam_pairs if y is not None]

    if not is_scam_pairs:
        return {}

    y_true = [y for _, y in is_scam_pairs]
    y_pred = [1 if p >= 0.5 else 0 for p, _ in is_scam_pairs]
    y_prob = [p for p, _ in is_scam_pairs]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    # Risk level MAE
    risk_pairs = [
        (r["predicted"]["risk_level"], r["expected_risk"])
        for r in results
        if r["predicted"]["risk_level"] is not None and r.get("expected_risk") is not None
    ]
    risk_mae = mean(abs(p - e) for p, e in risk_pairs) if risk_pairs else None

    # Category accuracy
    cat_pairs = [
        (r["predicted"]["scam_category"], r["expected_category"])
        for r in results
        if r["predicted"]["scam_category"] and r.get("expected_category")
    ]
    cat_acc = sum(1 for p, e in cat_pairs if p == e) / max(len(cat_pairs), 1) if cat_pairs else None

    # Latency
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    p50_lat = median(latencies) if latencies else None
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else None

    return {
        "n": len(is_scam_pairs),
        "is_scam_accuracy": accuracy,
        "is_scam_precision": precision,
        "is_scam_recall": recall,
        "is_scam_f1": f1,
        "is_scm_tp": tp, "is_scm_fp": fp, "is_scm_fn": fn, "is_scm_tn": tn,
        "risk_level_mae": risk_mae,
        "scam_category_accuracy": cat_acc,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "n_errors": sum(1 for r in results if r.get("error")),
    }


def _failure_table(results: list[dict], n: int = 10) -> str:
    """Return markdown table of worst false negatives and false positives."""
    # False negatives: high expected risk, low predicted noul
    fn = []
    fp = []
    for r in results:
        p = r["predicted"]["is_scam_noul"]
        e = r["expected_risk"]
        if p is None or e is None:
            continue
        if e >= 4 and p < 0.5:
            fn.append((e - p, r))
        if e <= 2 and p >= 0.5:
            fp.append((p - e, r))

    fn.sort(key=lambda x: -x[0])
    fp.sort(key=lambda x: -x[0])

    lines = ["## Failure analysis", ""]
    lines.append(f"### False Negatives (missed scams, n={len(fn)})")
    lines.append("")
    if not fn:
        lines.append("_None_")
    else:
        lines.append("| ID | Lang | Expected | Pred noul | Preview |")
        lines.append("|---|---|---|---|---|")
        for _, r in fn[:n]:
            preview = (r.get("state_preview") or "").replace("|", "\\|")[:60]
            lines.append(f"| {r['id']} | {r.get('language')} | {r['expected_risk']} | {r['predicted']['is_scam_noul']:.2f} | {preview} |")
    lines.append("")
    lines.append(f"### False Positives (benign flagged as scam, n={len(fp)})")
    lines.append("")
    if not fp:
        lines.append("_None_")
    else:
        lines.append("| ID | Lang | Expected | Pred noul | Preview |")
        lines.append("|---|---|---|---|---|")
        for _, r in fp[:n]:
            preview = (r.get("state_preview") or "").replace("|", "\\|")[:60]
            lines.append(f"| {r['id']} | {r.get('language')} | {r['expected_risk']} | {r['predicted']['is_scam_noul']:.2f} | {preview} |")
    return "\n".join(lines)


def _category_table(results: list[dict]) -> str:
    """Confusion-matrix-like table for scam_category."""
    counter = Counter()
    for r in results:
        p = r["predicted"]["scam_category"]
        e = r["expected_category"]
        if p and e:
            counter[(e, p)] += 1
    if not counter:
        return "_No category data._"
    expected_cats = sorted({e for e, _ in counter.keys()})
    predicted_cats = sorted({p for _, p in counter.keys()})
    lines = ["| expected ↓ / predicted → | " + " | ".join(predicted_cats) + " |",
             "|" + "---|" * (len(predicted_cats) + 1)]
    for e in expected_cats:
        row = [f"| **{e}**"]
        for p in predicted_cats:
            row.append(str(counter.get((e, p), 0)))
        lines.append(" | ".join(row) + " |")
    return "\n".join(lines)


def render_report(results: list[dict], schema: dict, out_dir: Path) -> tuple[Path, Path]:
    """Render Markdown + JSONL to out_dir; return (md_path, jsonl_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"eval-{ts}.md"
    jsonl_path = out_dir / f"results-{ts}.jsonl"

    # Write JSONL
    jsonl_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")

    # Write Markdown
    s = _summary(results)
    md = []
    md.append(f"# Laya Scam-Phrase Evaluation Report")
    md.append(f"_Generated: {datetime.now().isoformat()}_")
    md.append("")
    md.append(f"**Total samples:** {len(results)}")
    md.append("")
    md.append("## Headline Metrics")
    md.append("")
    if s:
        md.append("| Metric | Value |")
        md.append("|---|---|")
        md.append(f"| is_scam accuracy (threshold 0.5) | {s['is_scam_accuracy']:.3f} |")
        md.append(f"| is_scam precision | {s['is_scam_precision']:.3f} |")
        md.append(f"| is_scam recall | {s['is_scam_recall']:.3f} |")
        md.append(f"| is_scam F1 | {s['is_scam_f1']:.3f} |")
        md.append(f"| risk_level MAE | {s['risk_level_mae']:.2f} |" if s['risk_level_mae'] is not None else "| risk_level MAE | n/a |")
        md.append(f"| scam_category accuracy | {s['scam_category_accuracy']:.3f} |" if s['scam_category_accuracy'] is not None else "| scam_category accuracy | n/a |")
        md.append(f"| p50 latency | {s['p50_latency_ms']:.0f} ms |" if s['p50_latency_ms'] is not None else "| p50 latency | n/a |")
        md.append(f"| p95 latency | {s['p95_latency_ms']:.0f} ms |" if s['p95_latency_ms'] is not None else "| p95 latency | n/a |")
        md.append(f"| errors | {s['n_errors']} |")
    md.append("")
    md.append("## Confusion Matrix (is_scam, threshold 0.5)")
    md.append("")
    if s:
        md.append("|  | pred=scam | pred=benign |")
        md.append("|---|---|---|")
        md.append(f"| actual=scam | {s['is_scm_tp']} (TP) | {s['is_scm_fn']} (FN) |")
        md.append(f"| actual=benign | {s['is_scm_fp']} (FP) | {s['is_scm_tn']} (TN) |")
    md.append("")
    md.append("## Scam Category Confusion")
    md.append("")
    md.append(_category_table(results))
    md.append("")
    md.append(_failure_table(results))
    md.append("")
    md.append("## All samples")
    md.append("")
    md.append("| ID | Lang | Exp | Pred | Exp cat | Pred cat | Latency |")
    md.append("|---|---|---|---|---|---|---|")
    for r in results:
        pred = r["predicted"]
        md.append(
            f"| {r['id']} | {r.get('language')} | {r.get('expected_risk')} | "
            f"{pred['is_scam_noul']:.2f} | {r.get('expected_category') or ''} | "
            f"{pred['scam_category'] or ''} | {r.get('latency_ms', 0):.0f}ms |"
        )
    md.append("")

    md_path.write_text("\n".join(md))
    return md_path, jsonl_path
```

**Step 2:** 写 report 测试

```python
# tests/test_report.py
import json
from pathlib import Path

import pytest

from src.report import _summary, _category_table, render_report


@pytest.fixture
def fake_results():
    return [
        {"id": "a", "language": "zh", "expected_risk": 5, "expected_category": "phishing",
         "state_preview": "phishing msg", "routing": {"model": "english"},
         "latency_ms": 50,
         "predicted": {"is_scam_noul": 0.9, "risk_level": 4.0, "scam_category": "phishing",
                       "category_probs": {}}},
        {"id": "b", "language": "zh", "expected_risk": 1, "expected_category": "benign",
         "state_preview": "hi msg", "routing": {"model": "english"},
         "latency_ms": 60,
         "predicted": {"is_scam_noul": 0.1, "risk_level": 1.0, "scam_category": "benign",
                       "category_probs": {}}},
    ]


class TestSummary:
    def test_summary_returns_dict(self, fake_results):
        s = _summary(fake_results)
        assert "is_scam_accuracy" in s
        assert "p50_latency_ms" in s

    def test_summary_perfect_classification(self, fake_results):
        s = _summary(fake_results)
        assert s["is_scam_accuracy"] == 1.0
        assert s["is_scm_tp"] == 1
        assert s["is_scm_tn"] == 1


class TestRenderReport:
    def test_render_creates_files(self, fake_results, tmp_path):
        schema = json.loads(Path("schemas/scam.json").read_text())
        md_path, jsonl_path = render_report(fake_results, schema, tmp_path)
        assert md_path.exists()
        assert jsonl_path.exists()
        assert "Laya Scam-Phrase" in md_path.read_text()
```

**Step 3:** 运行测试

```bash
pytest tests/test_report.py -v
```

Expected: 3 tests pass.

**Step 4:** 提交

```bash
git add src/report.py tests/test_report.py
git commit -m "feat(src): add Markdown report renderer with metrics, confusion matrix, failure analysis"
```

---

## Phase 6: 烟雾测试与首次评估

### Task 21: CLI 烟雾测试

**Files:**
- Create: `tests/test_cli.py`

**Step 1:** 写测试

```python
"""CLI smoke tests using subprocess."""
import json
import os
import subprocess
import sys


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args, check=True, env=None):
    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = REPO
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "main.py", *args],
        capture_output=True, text=True,
        cwd=REPO,
        env=full_env,
        check=check,
    )


class TestCLISmoke:
    def test_help_exits_zero(self):
        result = _run("--help")
        assert result.returncode == 0
        assert "laya-scam-detector" in result.stdout

    def test_route_chinese_routes_to_multilingual(self):
        result = _run("--route", "您好世界")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["routed_to"] == "multilingual"

    def test_route_english_routes_to_english(self):
        result = _run("--route", "Hello world")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["routed_to"] == "english"
```

**Step 2:** 运行

```bash
pytest tests/test_cli.py -v
```

Expected: PASS。

**Step 3:** 提交

```bash
git add tests/test_cli.py
git commit -m "test: add CLI smoke tests for --help and --route"
```

---

### Task 22: 首次评估运行

**Step 1:** 在 eval 集上跑批量评估

```bash
python main.py --eval --input datasets/eval.jsonl --output reports/
```

Expected: 退出码 0；打印 `✓ Report: reports/eval-<ts>.md` 与 `✓ Results: reports/results-<ts>.jsonl`。

**Step 2:** 检查输出文件

```bash
ls -lh reports/
head -50 reports/eval-*.md | head -50
```

**Step 3:** 检查关键指标（应有数值）

在报告中确认：
- is_scam accuracy / F1 有数值
- p50 latency 有数值
- 失败分析表存在

**Step 4:** 提交（只提交报告，不提交模型）

```bash
git status  # 确认 reports/ 在 .gitignore 中，或选择性 commit
```

如果选择 commit 报告：

```bash
git add reports/eval-*.md reports/results-*.jsonl
git commit -m "docs: add first evaluation report on 38 handwritten samples"
```

---

### Task 23: 公开数据集集成（条件性）

**Files:**
- Create: `scripts/fetch_sms_spam.py`

**Step 1:** 写下载脚本（仅下载，不强制使用）

```python
"""Optionally download SMS Spam Collection from Hugging Face datasets.

The dataset is large (~5.5k rows). We sample 200 ham + 200 spam for evaluation.
"""
import json
import random
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("⚠ 'datasets' not installed; skip. Run: pip install datasets", file=sys.stderr)
    sys.exit(0)


def main() -> int:
    out = Path("datasets/public.jsonl")
    try:
        ds = load_dataset("ucirvine/sms_spam", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}", file=sys.stderr)
        return 1

    records = list(ds)
    print(f"Loaded {len(records)} records from SMS Spam Collection")

    spam = [r for r in records if r["label"] == 1]
    ham = [r for r in records if r["label"] == 0]
    random.seed(42)
    sample = random.sample(spam, min(200, len(spam))) + random.sample(ham, min(200, len(ham)))
    random.shuffle(sample)

    with out.open("w") as f:
        for i, r in enumerate(sample):
            obj = {
                "id": f"public-sms-{i:04d}",
                "type": "single",
                "language": "en",
                "expected_label": "spam" if r["label"] == 1 else "benign",
                "expected_risk": 5 if r["label"] == 1 else 1,
                "expected_category": "phishing" if r["label"] == 1 else "benign",
                "state": r["sms"],
                "source": "SMS Spam Collection",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {len(sample)} samples to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2:** 安装并运行

```bash
pip install datasets
python scripts/fetch_sms_spam.py
```

Expected: `✓ Wrote 400 samples to datasets/public.jsonl`

**Step 3:** 重新跑评估（合并所有数据集）

```bash
# Update build_eval_set.py to include public.jsonl if present
# (or just run eval on it directly)
python main.py --eval --input datasets/public.jsonl --output reports/
```

**Step 4:** 提交

```bash
git add scripts/fetch_sms_spam.py datasets/public.jsonl
git commit -m "feat(datasets): add SMS Spam Collection fetcher and 400 sampled records"
```

---

## Phase 7: 收尾

### Task 24: 全套烟雾测试套件

**Step 1:** 运行所有测试

```bash
pytest -v
```

Expected: 全部 PASS 或 SKIP（没有网络/模型的情况）。

**Step 2:** 修复任何失败

如果失败，根据 traceback 修复，重新跑。

**Step 3:** 提交（如果有修复）

```bash
git add -A
git commit -m "test: fix failing tests" || echo "nothing to commit"
```

---

### Task 25: 评估后决策点

**Step 1:** 查看最终报告

```bash
cat reports/eval-*.md | head -80
```

**Step 2:** 基于设计文档 §8.4 的微调触发条件做记录

- 如果 AUC < 0.7 → 在文档末尾追加 "微调建议" 章节，列出需要的标注数据量、训练步骤
- 如果 AUC ≥ 0.85 → 在文档末尾追加 "零样本基线可用" 章节，说明后续可优化方向

**Step 3:** 在 README.md 加评估结果摘要

```bash
# 手动添加：
echo -e "\n## Latest evaluation\n\nSee reports/eval-<timestamp>.md for headline metrics." >> README.md
```

**Step 4:** 提交

```bash
git add README.md
git commit -m "docs: add evaluation results to README"
```

---

## 总提交清单（预期）

```
chore: bootstrap project with pyproject.toml and .gitignore
chore: create project directory skeleton
docs: add README with quick-start placeholder
feat(scripts): add model download script for ONNX checkpoints
test: add smoke tests for ONNX model, tokenizer, config loading
feat(schemas): add scam detection schema with three primitives
feat(datasets): add 30 handwritten single-message scam test samples (ZH+EN)
feat(datasets): add 8 handwritten multi-turn dialogue scam samples
feat(datasets): add merge script and combined eval.jsonl (38 samples)
feat(src): add Unicode-script detector and ScriptRouter
feat(src): add softmax and tokenize helpers with passing unit tests
feat(src): implement OnnxLayaClient with predict and predict_batch
test: add end-to-end inference tests for phishing/benign classification
feat(src): add top-level Router with auto script-based checkpoint selection
feat(cli): add main.py with --predict --route --eval modes
feat(src): add eval module with state normalization and thresholded prediction
feat(src): add Markdown report renderer with metrics, confusion matrix, failure analysis
test: add CLI smoke tests for --help and --route
docs: add first evaluation report on 38 handwritten samples
feat(datasets): add SMS Spam Collection fetcher and 400 sampled records
docs: add evaluation results to README
```

---

## 执行方式

按照 writing-plans skill 的执行交接流程，下一步：

**Plan complete and saved to `docs/plans/2026-09-28-laya-scam-detection-impl.md`.**

**两个执行选项：**

1. **Subagent-Driven (this session)** — 每个任务派遣一个新的 subagent 执行，任务间人工 review，快速迭代
2. **Parallel Session (separate)** — 打开新会话，使用 executing-plans skill，批量执行 + 检查点

请告诉我哪种执行方式。