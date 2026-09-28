# Laya 与 Jev 模型深度调研报告

*调研日期：2026-09-28 | 信息源：15+ 篇英文/中文技术博客、官方仓库、Hugging Face 模型卡、独立评测 | 置信度：高（核心事实多源交叉验证），中（基准对比数据）*

---

## 执行摘要

**Laya** 与 **Jev** 是 2026 年 9 月前后诞生的"System One 决策模型"，两者采用 **非自回归编码器架构**，专门用于结构化决策（分类、评分、布尔判断），而不是生成文本。它们填补了"用 GPT 类大模型做选择题"这一过度场景的空白——把 500ms–2s 的 token 生成延迟压缩到 30–300ms，并附带数学可校准的置信度。

**核心对比**：
- **Jev**（TypeSafe AI，闭源 API）：2026-09-15 发布，由前 OpenAI 研究员、InstructGPT 共同作者 Diogo Almeida 创立，融资 4000 万美元（DCVC 领投）。定位"开箱即用的通用决策云服务"。
- **Laya**（ConvAI Innovations，开源）：2026-09-18 由印度独立研究者 Nandakishor Mukkunnoth 发布，Apache 2.0 协议。定位"可微调的本地专家模型"，发布后 48 小时登顶 Hugging Face 趋势榜，GitHub 星标 26.7k+。

**关键启示**：这不是"开源击败闭源"的简单叙事——Laya 在零样本精度、上下文长度（512–1024 vs 64k）、选项上限（~20 vs 255）方面弱于 Jev；但在 **延迟（33ms vs 236ms）**、**校准误差 ECE（0.081 vs 0.246，温度校准后）**、**数据主权（本地 vs 云端）**、**成本（$0 vs $0.042/M tokens）** 上显著领先。

---

## 1. Jev 模型：TypeSafe AI 的闭源决策 API

### 1.1 背景与定位

- **发布**：2026-09-15，由旧金山初创公司 TypeSafe AI 出 stealth 推出 ([XY Space](http://xyspace.dev/blog/what-is-jev))
- **创始人**：Diogo Almeida（前 OpenAI 研究员，InstructGPT 共同作者、GPT-4 技术报告作者之一），联合创始人 Erik Gafni、Sasha Sheng ([Blockchain Council](https://www.blockchain-council.org/ai/jev-architecture-explained))
- **融资**：4000 万美元种子轮，DCVC 领投 ([XY Space](http://xyspace.dev/blog/what-is-jev))
- **命名由来**：以 19 世纪英国经济学家 William Stanley Jevons（杰文斯悖论）命名，呼应"决策成本下降将带来调用量爆炸"的商业信念 ([magrop.net](https://blog.margrop.net/en/post/jev-system-one-non-generative-decision-model/))

### 1.2 架构与技术特征

TypeSafe **未公开架构、参数量、训练数据**——这是 Jev 与 Laya 最大的信息不对称 ([wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya))。社区根据反向工程推测其接近 BERT 风格的编码器，但无官方确认。

三项核心技术支柱 ([Blockchain Council](https://www.blockchain-council.org/ai/jev-architecture-explained))：
1. **Parallel Sampling（并行采样）**：放弃 token-by-token 自回归循环，对 state + typed schema 一次前向并行评估
2. **Typed Invariants（类型不变性）**：输出被 logit 层限定为开发者预设的枚举、布尔或数值范围，从结构上消除格式幻觉
3. **RLCD（Reinforcement Learning for Calibrated Decisions）**：以严格正确的评分规则作为信号，训练模型输出"认知诚实"的概率

### 1.3 三种决策原语（Primitives）

| 原语 | 输入 | 输出 | 典型用例 |
|---|---|---|---|
| `choice` | 状态 + 候选选项列表 | 选中项 + 每项概率 + 置信度 | 路由、意图、工具选择 |
| `score` | 状态 + 有序评分标准 | 加权期望分 + 等级分布 + 置信度 | 紧急度、风险、情感强度 |
| `noul` | 状态 + 是/否问题 | P(true) ∈ [0,1] | 钓鱼检测、注入检测、退款意图 |

([attentionheads.blog](https://www.attentionheads.blog/p/jev-a-system-one-ai-primitive))

### 1.4 关键规格

| 维度 | 规格 |
|---|---|
| 价格 | **$0.042 / 百万输入 token**；输出免费 |
| 延迟 | 70–500ms（官方）/ 236–276ms p50（第三方 [AbdelStark](https://github.com/AbdelStark/jev-benchmarks)、[nibzard](https://github.com/nibzard/decision-model-benchmark) 独立测量）|
| 上下文 | 64k tokens / 请求（state + 最长单问题 ≤ 32k）|
| 选项上限 | `choice` 最多 255 个选项 |
| 速率限制 | 250,000 tokens/s + 1,200 requests/min（动态调整）|
| 输入模态 | 仅文本（字符串、JSON 对象、文本数组）；不支持图像/音频/视频 |
| 主要语言 | 英语优先，中文/CJK 可用但精度可能下降 |
| 定制能力 | **不支持微调/LoRA**；所有客户共享同一套权重 |

([wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya))

### 1.5 使用场景

Jev 在以下场景中展现价值 ([attentionheads.blog](https://www.attentionheads.blog/p/jev-a-system-one-ai-primitive), [note.com/retail_shirokuma](https://note.com/retail_shirokuma/n/n88ec35695e5a?hl=en))：

1. **模型路由器（Model Router）**：在用户请求进来时判断调用哪个 LLM（便宜模型 vs 昂贵模型）
2. **意图识别与工单分流**：自动路由客服请求至对应团队
3. **提示注入 / 越狱检测**：作为 LLM 的前置防线（但官方承认对对抗性内容仍有脆弱性）
4. **结构化提取与重排序**：知识图谱构建、RAG 段落相关性评分
5. **Agent 工具选择**：决定下一步调用哪个工具、判断工具调用是否成功
6. **实时决策密集场景**：游戏 NPC 行为决策、机器人 FSD 仿真（开发者 Justin Schroeder 一小时内用 Jev 重建了特斯拉 FSD 仿真 [gist.ly](https://gist.ly/youtube-summarizer/jev-ai-the-worlds-fastest-decision-making-model)）

### 1.6 生态集成

- **LangChain**：官方集成指南已发布
- **Vercel AI Gateway**：`typesafe/jev` 路由
- **Cloudflare Workers AI**：作为 `typesafe/jev` 上线（2026-09-18 起，零留存模式可选）
- **Pydantic AI**：`TypeSafeModel` 类型化字段自动交由 Jev 处理

([XY Space](http://xyspace.dev/blog/what-is-jev))

### 1.7 已记录的局限

TypeSafe 官方文档坦诚列出 Jev 1.13 的"jagged edges"（[wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya)）：
- 字面理解问题、计数/算术不可靠
- 日期比较、间接推理能力弱
- 当 state 中充斥无关内容时精度下降
- 易受 prompt injection 影响（state 中的对抗性内容可左右判断）
- 同一问题与其否定的两个 noul 概率之和可能为 1.19（违反概率公理）

### 1.8 开源替代

- **Tev1-4B-experimental**（Together AI，2026-09-23）：开放权重的 Jev-like 模型，训练成本约 $17；400 项任务准确率 90.0% vs Jev 97.2%，但更便宜 ([XY Space](http://xyspace.dev/blog/what-is-jev))
- **OpenJev**（社区 AlexWortega）：用 Qwen3.5 4B 作 backbone，从最后 token 的 logits 直接抽取分类概率
- **SemIf / Kev**：从开源 LLM 读 token 概率模拟 Jev 输出，Jev 一致率约 85%

---

## 2. Laya 模型：ConvAI Innovations 的开源决策引擎

### 2.1 背景与发布

- **作者**：Nandakishor Mukkunnoth，独立研究者，ConvAI Innovations CEO，来自印度喀拉拉邦 ([theleftshift.com](https://www.theleftshift.com/an-indian-ai-model-that-doesnt-chat-claims-top-spot-on-hugging-face))
- **首次模型**："从训练到推送 HF、GitHub、撰写博客只用 15 小时"——作者本人语 ([theleftshift.com](https://www.theleftshift.com/an-indian-ai-model-that-doesnt-chat-claims-top-spot-on-hugging-face))
- **发布**：2026-09-18
- **关键论文**：arXiv:2503.23303（SalesRLAgent，2025-03）、arXiv:2510.01237（置信度感知路由，2025-09）
- **爆红**：发布后 48 小时登顶 Hugging Face Trending #1，被 CEO Clément Delangue 公开推荐；截至最新统计 GitHub 26.7k stars、2.3k forks ([github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya))
- **许可证**：Apache 2.0

### 2.2 架构

Laya 采用"**双向编码器 + 决策头**"架构，与传统 BERT 类似但专门为决策任务重新训练 ([wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya))：

```
State (text or JSON) + typed questions and options
   │ packed as: [CLS] question [SEP] [MASK] option1 [MASK] option2 ... [SEP] state [SEP]
   ▼
ModernBERT-large encoder (395M, 28 bidirectional attention layers)
   ▼
Decision-head Transformer (~25M, 2 layers)
   ├─ Option scorer: hidden state at each [MASK] → logit → temperature scaling → softmax
   └─ Act / Escalate head: [CLS] + distribution stats → "act automatically" or "escalate"
```

每个选项前有独立的 `[MASK]` 标记；多个问题批处理后 **单次前向传播** 完成 ([mer.vin](https://mer.vin/news/laya-the-33ms-open-source-decision-model-beating-jev))。

### 2.3 三个 Checkpoint + Router

| Checkpoint | 编码器 | 参数 | 默认上下文 | 用途 |
|---|---|---|---|---|
| `laya` | ModernBERT-large | 421M | 512 tokens | 英语 |
| `laya-multilingual` | mmBERT-base | 322M | 1024 tokens（可扩至 8k）| 100+ 语言，约快 2x |
| `laya-typed-decisions` | ModernBERT-large | 421M | 1024 tokens | 客服、发票、安全告警、agent 可观测性 |

([huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya))

**Router** 是个纯 Python 的语言路由器，在前向传播前 <1ms 检测 Unicode 脚本（22 种字母），决定调用哪个 checkpoint。存在原因：**英文 checkpoint 在非拉丁脚本上"自信地错误"——高棉语上 0.952 confidence 但 0.000 accuracy** ([huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya))。

### 2.4 RLCD 训练细节（已公开）

- **不使用监督交叉熵损失**
- 奖励 = 严格正确的评分规则组合（log + spherical，对有序问题额外加 ranked probability score）
- 更新 = REINFORCE + GRPO-style group-mean baseline（组均值基线）
- **Act/Escalate 头**：用代价矩阵训练——答对 +1，答错 -3，转人工 -0.5；模型学到只有当正确率 > 62.5% 时才自动执行 ([wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya))
- 多轮对话使用 TD(λ=1.0)

### 2.5 校准（Calibration）

Laya 的核心卖点是 **数学可校准的概率**——它训练目标就是让 0.95 的概率在实际中真的对应 95% 准确率。但出厂时是过度自信的：

| Checkpoint | 拟合前 ECE | 拟合后 ECE（每问题类型+选项数一个温度） |
|---|---|---|
| `laya` | 0.466 | **0.081** |
| `laya-multilingual` | 0.314 | **0.106**（多语版出厂不带拟合温度）|

([huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya))

### 2.6 使用场景

官方列出的九大生产级工作流 ([mer.vin](https://mer.vin/news/laya-the-33ms-open-source-decision-model-beating-jev))：

| 任务 | 准确率 | 备注 |
|---|---|---|
| Enron 邮件垃圾过滤 | **0.993** | F1 0.993, ECE 0.013 |
| 钓鱼邮件检测 | **0.980** | F1 0.979, ECE 0.012 |
| LLM 守卫/越狱检测（held-out ToxicChat） | 0.755–0.762 | 50% selective coverage 时达 0.931 |
| RAG 段落相关性筛选 | 0.657 | 单次前向 |
| 工单路由（10-way）| 0.522 | |

作者本人的语义自动化方案（在 DEV.to 上展示）：路由部门分类 + 紧急度评分 + 流失风险预测 + 钓鱼检测全部用一次 `router.predict()` 完成 ([dev.to/nandakishor_m](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me))。

---

## 3. Laya 部署方法（重点分析）

### 3.1 安装与快速启动

最简方式：

```bash
pip install laya
```

Python 3.10+ 要求（依赖 `huggingface_hub` 1.x、`transformers` 5.x、`torch` 2.14）。

**可选项（extras）** ([github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya))：
- `laya[serve]` — HTTP 服务器
- `laya[mcp]` — MCP 服务器（Claude Code、Cursor、OpenClaw 集成）
- `laya[langchain]` — LangChain/LangGraph 集成
- `laya[llamaindex]` — LlamaIndex selectors
- `laya[crewai]` — CrewAI 路由
- `laya[onnx]` — ONNX Runtime
- `laya[fast]` — TileLang GPU fast path

### 3.2 五种部署路径

#### A. Python SDK（最常用）

```python
from laya import Router

router = Router(preload=True)  # 预加载所有 checkpoint，避免语言切换时的 7-10s 重载

result = router.predict(state, questions)
# 自动路由：根据语言/脚本选 checkpoint
# 单次前向返回所有问题的答案
```

([huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya))

#### B. HTTP Server（兼容 Jev 协议）

```bash
pip install "laya[serve]"
LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve  # 0.0.0.0:8000
```

**关键特性**：暴露 `POST /v1/systemone` 接口——**与 TypeSafe Jev 的 API 完全相同**。现有 Jev 客户端只需改 base URL 即可使用 Laya ([huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya))。

- 默认 `0.0.0.0` 监听，无认证
- 设置 `LAYA_API_KEY` 后需 `Authorization: Bearer <key>`
- 畸形问题返回 422 并指明错误

#### C. ONNX Runtime（CPU/GPU/浏览器）

`inferenceprince/laya-onnx` 是社区导出的 ONNX 版本（fp16），**无需 PyTorch** ([huggingface.co/inferenceprince/laya-onnx](https://huggingface.co/inferenceprince/laya-onnx))：
- 3.3 MB 计算图 + 842.6 MB fp16 权重
- 支持 `CPUExecutionProvider`、`CUDAExecutionProvider`、浏览器（wasm）
- 已被 TypeScript/Node.js 项目广泛使用

#### D. Apple Silicon / Core ML / MLX

- **laya-coreml**（`aac6fef/laya-coreml`）：macOS 15+/iOS 18+ 导出，支持 ANE（Apple Neural Engine），内存仅 1GB，ANE 上短决策 FP16 推理极快 ([github.com/mizorewww/laya-coreml](https://github.com/mizorewww/laya-coreml/blob/main/docs/USAGE.md))
- **laya-mlx**（社区）：MLX 移植，Apple Silicon 原生，M3 Max 上 60 decisions/sec 玩贪吃蛇

#### E. TypeScript / Node.js

`laya-ts` 包（`npm install laya-ts`）从同一仓库的 `laya-ts-v*` 标签发布，支持浏览器和 Node.js ([github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya))。

### 3.3 Docker Compose 部署

仓库自带 Docker Compose 配置（`compose.yaml` / `compose.cuda.yaml` / `compose.http.yaml` / `compose.spark.yaml`），支持 CPU/GPU 多环境 ([github.com/NandhaKishorM/laya](https://github.com/NandhaKishorM/laya))。

### 3.4 端侧 / 嵌入式

- **Apple Silicon**：Core ML 导出后约 1GB 内存即可运行
- **AX8850**（爱芯元智）：知乎 2026-09-22 报告在端侧 NPU 上跑通开源平替，**33ms 链路可嵌入每次请求** ([腾讯云转载](https://cloud.tencent.com.cn/developer/article/2750506))
- **ESP32 / 资源受限设备**：理论上 421M 模型需 ~2GB 内存，社区在尝试量化到更低配置

### 3.5 模型微调（Fine-Tuning）

仓库自带完整 Jupyter 笔记本 `laya_finetune_typed_decisions_2xT4_kaggle.ipynb`：
- 硬件：Kaggle 免费 2× T4 GPUs
- 数据：约 30,000 问题
- 训练：4 epochs，约 4–5 小时
- 流程：造数据 → RLCD 训练 → 温度拟合 → 评测 → 推送到 Hugging Face Hub

([mer.vin](https://mer.vin/news/laya-the-33ms-open-source-decision-model-beating-jev))

### 3.6 生产部署关键参数

| 配置 | 单请求延迟 | 模型重载 |
|---|---|---|
| `Router()`（默认懒加载，`max_loaded=2`）| 检测 <1ms（首次某语言后）| 1 次/语言首次 |
| `Router(max_loaded=1)` | 7-10s/语言切换 | 1 次/切换 |
| **`Router(preload=True)`** | **32.8ms (GPU) / 193-464ms (CPU)** | **0** |

([huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya))

**核心经验**：生产环境必须 preload，否则语言切换触发冷重载——CPU 上 7.4s 中位数，T4 上 10.3s。

---

## 4. Laya 部署效果（重点分析）

### 4.1 速度基准（Tesla T4 GPU）

| 每调用问题数 | `laya` | `laya-multilingual` |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 5 | 84.5 ms | **40.1 ms** |
| 10 | 158.6 ms (15.9 ms/q) | **72.3 ms (7.2 ms/q)** |
| 50 | 771 ms | **337 ms (6.8 ms/q)** |

批处理时**单 T4 可达 103–332 问题/秒** ([huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya))。

**对比 Jev**（独立第三方测量）：Jev 236–276 ms p50 vs Laya 32.8 ms → **快 7.8×** ([mer.vin](https://mer.vin/news/laya-the-33ms-open-source-decision-model-beating-jev))。

### 4.2 设备实测

| 设备 | 场景 | 性能 |
|---|---|---|
| Tesla T4（GPU）| 单问题 | 32.8 ms (Laya) |
| CPU（健康硬件）| 单问题 | 193–464 ms (Laya) |
| Apple M3 Max | MLX 贪吃蛇 | **60 decisions/sec** |
| Apple M5 Pro | 单问题 | 15.3 ms (Laya) vs 298.1 ms (Jev) |
| AX8850 端侧 NPU | 嵌入式 | 33 ms/请求 |
| 4 vCPU / 7GB VPS（无 GPU）| 冷启动后预测 | 49.4s 中位数（Flowtivity 实测）|

([flowtivity.ai](https://flowtivity.ai/blog/laya-open-source-jev-alternative), [wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya))

### 4.3 精度对比（关键基准）

| 基准 | Jev 1.13.0 | Laya (routed) | 优势方 |
|---|---|---|---|
| typed-decisions (2000 决策) | 0.727 | **0.766** | Laya（+3.9pts，超过 teacher ceiling 0.735）|
| AG News (4 labels) | 0.910 | **0.950** | Laya |
| DAIR Emotion (6 labels) | 0.480 | **0.595** | Laya（Jev 在 16% 样本上给真实标签分配 0 概率）|
| Banking77 (77 labels) | **0.870** | 0.425 | Jev（label 多时）|
| **ECE（校准误差，越低越好）** | 0.246 | **0.081** | **Laya 3× 更好**（温度拟合后）|
| p50 latency (1 question) | 236–276 ms | **32.8 ms** | Laya 7.8× 快 |
| Languages usable (>3x random) | 无公开数据 | **45/51** | Laya |
| 成本 | $0.042/M tokens | **$0** (自托管) | Laya |

**重要注释**：Laya 的 0.766 来自 `laya-typed-decisions`（专门微调后的版本），基座模型 `laya` 零样本仅 0.362，`laya-multilingual` 仅 0.342，**接近 0.318 随机基线**，低于 0.461 majority-class 基线——**这是 Laya 最需要被诚实承认的局限** ([huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya))。

### 4.4 工业级应用案例

#### A. AI 语音代理（DEV.to 案例研究，[dev.to/msinfotech](https://dev.to/msinfotech/why-build-ai-voice-agents-3-hidden-laya-framework-secrets-5a3n)）

| 指标 | 单体 LLM 流水线 | 级联语音引擎 | Laya 优化栈 |
|---|---|---|---|
| 端到端延迟 | 1,420 ms | 580 ms | **88 ms** |
| GPU 显存 | 24 GB | 16 GB | **2.8 GB** |
| 打断响应时间 | 450 ms | 210 ms | **18 ms** |
| 每千轮成本 | $4.20 | $1.85 | **$0.38** |
| 意图准确率 | 94.2% | 89.1% | **96.8%** |

单张低配 GPU 上可并行运行 8 个语音 worker。

#### B. LLM 守卫/越狱检测

- 离线模式：`laya` 在 held-out ToxicChat 数据集上 **0.755–0.762** 准确率
- 选择性覆盖（confidence 阈值筛掉低分）：50% 选择覆盖下 **准确率升至 0.931** ([dev.to/nandakishor_m](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me))

#### C. 生产部署综合

| 场景 | 选择 Laya 的核心理由 |
|---|---|
| 高频本地推理 | 零 token 成本（Apache 2.0 自托管）|
| 数据不能出境（医疗/金融/政府）| 本地部署，零数据离开 |
| 毫秒级实时决策（游戏、风险控制）| 33ms 本地推理，无网络往返 |
| 有标注数据的垂直场景（垃圾邮件、钓鱼、工单分流）| 微调后超过 Jev（0.766 vs 0.727）|
| 多语言混合流量 | Router 自动选 checkpoint，100+ 语言支持 |

([eesel.ai](https://www.eesel.ai/blog/laya-ai-review))

### 4.5 部署限制与坑

Laya 官方 README 与多个第三方评测列出的局限 ([huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya))：

1. **零样本接近随机**：基座模型必须微调，否则 typed-decisions 任务仅 0.362 准确率
2. **短上下文**：英文版 512 tokens、多语版 1024 tokens（state 实际可用 ~320/~768 tokens）；长上下文需显式设 `max_len=8192`
3. **高基数选项衰减**：>20 选项时精度显著下降；Banking77 (77 labels) 仅 0.425 vs Jev 0.870
4. **校准需自拟合**：出厂 ECE 0.466；不拟合温度直接用概率会过度自信
5. **`score` 是最弱原语**：SST-5 五级情感仅 0.372
6. **`noul` 在英文版上偶尔跟随选项标签而非 state**：建议改为二选项 `choice`（[#156 issue](https://github.com/NandhaKishorM/laya/issues/156)）
7. **`action.act_probability` 暂无可用信号**：建议改用 `confidence`（[#185 issue](https://github.com/NandhaKishorM/laya/issues/185)）
8. **英文 checkpoint 在非拉丁脚本上会"自信地错"**：必须用 Router 或显式选 checkpoint

---

## 5. Jev vs Laya 核心对比

| 维度 | Jev | Laya |
|---|---|---|
| 发布方 | TypeSafe AI（公司，融资 $40M）| ConvAI Innovations（独立研究者）|
| 开源性 | **闭源** API | **Apache 2.0**（权重+训练代码全公开）|
| 部署方式 | 托管云（US West Coast）| 本地 / 私有云 / 离线 / 端侧 |
| 架构披露 | 否 | ModernBERT-large + mmBERT-base + 决策头 |
| 参数量 | 未披露 | 322M–421M |
| 训练方法 | RLCD（细节未披露）| RLCD（严格正确评分规则 + GRPO 基线，已公开）|
| 原语 | choice / score / noul | choice / score / noul（接口几乎一致）|
| 上下文长度 | 64k（state + 问题 ≤ 32k）| 512 / 1024 默认（编码器支持至 8k）|
| 选项上限 | **255** | ~20 后显著衰减 |
| **零样本能力** | **强**（开箱即用）| **弱**（必须微调）|
| **定制能力** | 仅限 state + criteria+ instructions | **微调、温度拟合、自托管** |
| 多语言 | 英语为主 | **多语版 + Router；51 语言中 45 可用** |
| 单题延迟 | 70–500 ms（p50 236 ms）| **33–40 ms on T4** |
| 成本 | $0.042/M tokens | **$0 自托管** |
| 数据合规 | 数据需发送至 TypeSafe（企业版 ZDR 可选）| **数据永不离机** |
| 运维 | 全托管 | 需自管 GPU、扩缩容、监控、版本 |
| 生态 | LangChain、Vercel、Cloudflare、Pydantic AI | LangChain、LangGraph、LlamaIndex、CrewAI、MCP |

([wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya))

---

## 6. 决策路径建议（来自 Wilson Wu 综合评测）

按以下顺序问自己 ([wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya))：

```
1. 数据能否送到第三方云（可能跨境）？
   └─ 不能 → Laya（硬约束）
2. 有标注数据、愿意微调吗？
   └─ 没有 → Jev
3. 需要每秒几十次实时决策或端侧推理？
   └─ 是 → Laya
4. state 很长（>1k tokens）或选项很多（>20）？
   └─ 是 → Jev（或 Laya + chunking/shortlisting）
5. 每月千万级以上调用？
   └─ 是 → 倾向 Laya；否则 Jev 通常 TCO 更低
```

**成本算账**：假设每次决策 1k 输入 tokens，Jev $42 / 百万次决策。T4 持续在线约 $0.50/h = $360/月，等于约 850 万 Jev 决策的价格（还不算高可用与人力）。**自托管 Laya 在高量场景才回本，选择 Laya 通常为延迟、合规、可定制，不是省钱**。

---

## 7. 共同陷阱（无论选谁）

([wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya), [huggingface.co/convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya))

1. **类型安全 ≠ 正确判断**：校准描述的是群体预测的统计行为，不保证单个答案正确——必须用自己的数据校准阈值
2. **拆解问题**：不要问"这工单该怎么处理"，拆为"是否退款？"、"紧急度？"、"属于哪个业务线？"几个原子问题再用代码组合
3. **确定性逻辑交给代码**：计数、算术、日期比较不要让模型做
4. **state 保持精简**：只放与问题相关的内容，无关字段会拉低精度
5. **不可逆动作必须守卫**：支付、删除、医疗合规审批绝不基于单一概率自动执行
6. **警惕注入**：state 中的恶意内容仍能左右判断——决策模型不是唯一防线
7. **固定版本 + 持续监控**：pin `jev-1.13.0` 或具体 Laya checkpoint，升级后重新校准阈值；持续记录概率分布、人工抽查样本、监控校准漂移

---

## 8. 关键洞见

1. **"System 1 vs System 2" 已成 AI 架构共识**：决策模型作为生产 AI 的新一层，与代码（确定性）、LLM（生成式）形成三层分工——决策模型填补了"快速模糊判断"的空白 ([eyestech.in](https://eyestech.in/typesafe-jev-system-1-system-3-ai))

2. **非自回归编码器 + 校准训练 = 毫秒级决策的新范式**：Jev 与 Laya 用 BERT 式编码器替代 GPT 式解码器，绕开 token 生成的开销，把"决策"从计算密集型任务变成内存密集型任务

3. **Laya 的真正价值不在性能**：根据多个独立评测，Laya 在零样本、长上下文、高基数选项等关键场景仍逊于 Jev。它的真正价值在 **数据主权 + 成本可控 + 可微调定制**——这是 Apache 2.0 协议的战略意义

4. **Hugging Face 趋势榜的"开源击败闭源"叙事需打折扣**：Laya 0.766 vs Jev 0.727 的对比中，Laya 的 0.766 来自专门在测试集上微调的版本；基座模型仅 0.362（接近随机）。但这不否认 Laya 作为开源"基座 + 微调工具链"的工程价值

5. **决策模型 + LLM 的混合架构成为新标准**：85% 高频简单判断 → Jev/Laya；15% 复杂推理 → Claude/GPT；中间元认知层动态路由 ([blog.margrop.net](https://blog.margrop.net/en/post/jev-system-one-non-generative-decision-model/))

6. **Jevons 悖论的 AI 翻版**：决策调用成本从 $0.05 降至 $0.00004、延迟从 5s 降至 70ms，让 AI 决策从奢侈品变成 if-else 内的基础设施——决策量将呈指数级爆发 ([blog.margrop.net](https://blog.margrop.net/en/post/jev-system-one-non-generative-decision-model/))

7. **不要混淆 Jev 与 JEPA**：Meta 的 JEPA（Joint Embedding Predictive Architecture，Yann LeCun 推动）是自监督世界模型学术架构；Jev 是商业决策 API。两者共享"不生成 token"的哲学，但技术路线与目标市场无关 ([note.com/retail_shirokuma](https://note.com/retail_shirokuma/n/n88ec35695e5a?hl=en))

---

## 来源清单

### 官方与一手资料
1. [GitHub: NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) — Laya 主仓库（含 README、BENCHMARKS、notebooks）
2. [Hugging Face: convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) — 官方模型卡与基准
3. [Hugging Face: inferenceprince/laya-onnx](https://huggingface.co/inferenceprince/laya-onnx) — ONNX 导出版
4. [GitHub: laya-coreml](https://github.com/mizorewww/laya-coreml/blob/main/docs/USAGE.md) — Apple Silicon Core ML 移植
5. [arXiv:2503.23303](https://arxiv.org/abs/2503.23303) — SalesRLAgent 论文
6. [arXiv:2510.01237](https://arxiv.org/abs/2510.01237) — 置信度感知路由论文
7. [DEV.to: I Built Non-Autoregressive Decision Models a Year Ago](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me) — 作者一手发布博文

### 独立评测与对比
8. [Wilson Wu: Jev vs Laya](https://wilsonwu.me/en/blog/2026/jev-vs-laya) — 25 分钟深度对比（最系统的中立评测）
9. [Mervin Praison: Laya 33ms Open-Source Decision Model Beating Jev](https://mer.vin/news/laya-the-33ms-open-source-decision-model-beating-jev) — 详细基准复现
10. [eesel.ai: Laya AI Review](https://www.eesel.ai/blog/laya-ai-review) — 第三方诚实评测（含 Hacker News 评论）
11. [Flowtivity: Laya Open-Source Jev Alternative](https://flowtivity.ai/blog/laya-open-source-jev-alternative) — 实际部署经验
12. [DEV.to: Why Build AI Voice Agents: 3 Hidden Laya Framework Secrets](https://dev.to/msinfotech/why-build-ai-voice-agents-3-hidden-laya-framework-secrets-5a3n) — 语音代理场景工业级基准
13. [The Left Shift: An Indian AI Model That Doesn't Chat](https://www.theleftshift.com/an-indian-ai-model-that-doesnt-chat-claims-top-spot-on-hugging-face) — 起源报道
14. [EYES TECH: Open-Source Beats Jev: Laya Hits #1](https://eyestech.in/?p=2771) — 行业反应分析

### Jev 专项
15. [XY Space: What Is Jev?](http://xyspace.dev/blog/what-is-jev) — 综合介绍
16. [Blockchain Council: Jev Architecture Explained](https://www.blockchain-council.org/ai/jev-architecture-explained) — 架构解读
17. [Attention Heads: Jev, A System One AI Primitive](https://www.attentionheads.blog/p/jev-a-system-one-ai-primitive) — 设计哲学
18. [Margrop Blog: TypeSafe's Jev Proves Zero-Token Models Are the Future](https://blog.margrop.net/en/post/jev-system-one-non-generative-decision-model/) — 混合代理编排
19. [Note.com: The Full Picture of TypeSafe AI's 'Jev'](https://note.com/retail_shirokuma/n/n88ec35695e5a?hl=en) — 含与 JEPA 的辨析
20. [AI Beat: Before Jev, There Was Laya](https://ai-beat.github.io/news/2026/09/laya-before-jev) — 优先权争议分析

### 中文资源
21. [腾讯云：Laya 使用完整教程](https://cloud.tencent.com.cn/developer/article/2750506) — 中文使用指南（17K Star）
22. [稀土掘金：Laya 使用完整教程](https://juejin.cn/post/7688389114781024319) — 同上镜像
23. [Note.com（中文）：Jev 全文画像](https://note.com/retail_shirokuma/n/n88ec35695e5a?hl=en) — 日文/中文译本

---

## 方法论说明

本次调研在 2026 年 9 月窗口完成，共检索 4 类英文查询 + 1 类中文查询，覆盖 web search（MiniMax web_search）、GitHub README、Hugging Face 模型卡、独立技术博客。关键事实（如 Laya 三个 checkpoint 规格、Jev 价格、训练方法）均在 ≥3 个独立来源中交叉验证。基准数据（如 Laya 0.766 vs Jev 0.727）来自 Laya 作者一手测量，Jev 数据为第三方独立测量（[AbdelStark](https://github.com/AbdelStark/jev-benchmarks)、[nibzard](https://github.com/nibzard/decision-model-benchmark)），存在样本量与提示词差异。

**置信度声明**：
- 核心事实（模型存在性、开发者、架构、发布时间）：**高置信**
- 性能数据（延迟、准确率）：**中置信**——所有数据均为单方测量（多为 Laya 团队）
- Jev 商业细节（融资、生态集成）：**高置信**——多源交叉验证
- 部署实战细节（生产配置、性能调优）：**中置信**——基于用户报告和官方文档

**未覆盖领域**：
- Laya 0.3.21 之后的最新变化
- 真实生产环境的长期稳定性数据
- 中文/CJK 场景下 Laya 的具体表现（仅有测试数据，无大规模生产报告）
- 与新兴决策模型（Together AI 的 Tevv 系列）的对比基准