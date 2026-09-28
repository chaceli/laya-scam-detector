# Laya 反诈检测 Web Playground — 设计文档

**日期**：2026-09-28
**状态**：已批准
**依赖模型**：`models/laya-onnx-multilingual-finetuned/`（本地 LoRA 微调产出）

---

## 1. 目标

提供一个本地 Web 页面，让用户：

1. 填入任意文本（中文/英文）
2. 选择 Laya 的三种决策原语输出：**noul**（是/否）、**score**（评分）、**choice**（多选）
3. 可切换模型（自动路由 / 英文 base / 微调多语版）以对比效果
4. 查看每个原语的判定、置信度分布、延迟、路由信息

用途：使用微调模型、测试模型能力、人工验收。

---

## 2. 用户决策摘要（来自 brainstorming）

| 维度 | 选择 |
|---|---|
| "三种方式" | 三种原语 choice / score / noul |
| 问题定义 | 默认诈骗 schema + 可展开高级编辑 |
| 模型选择 | 可切换（自动路由 / english / multilingual）|
| 技术栈 | FastAPI + 原生 HTML/JS（无构建步骤）|
| 结果展示 | 判定 + 置信度分布 + 延迟 + 路由 |
| 架构方案 | 方案 B：FastAPI API + `web/` 静态目录 |

---

## 3. 架构

```
浏览器 (web/)
  ├─ 文本输入
  ├─ 三原语勾选
  ├─ 模型选择器
  └─ 结果卡片 + 元信息
        │ fetch JSON
        ▼
FastAPI (server/app.py)
  ├─ lifespan: 加载 Router 单例（容错）
  ├─ GET  /              → web/index.html
  ├─ GET  /api/health    → 模型状态
  ├─ GET  /api/samples   → 预设样本
  ├─ POST /api/predict   → 推理
  └─ /static → web/
        │
        ▼
Router (src/router.py, 已有)
  ├─ english      → models/laya-onnx-en
  └─ multilingual → models/laya-onnx-multilingual-finetuned
```

**为什么选方案 B**：
- 保持**纯 ONNX 路径**（不引入 torch/transformers）
- 直接加载**我们微调的** bundle（而非 HF 官方权重）
- 前端无构建步骤，易于迭代
- API 与页面分离，便于自动化测试

---

## 4. API 契约

### 4.1 `GET /api/health`

```json
{
  "ok": true,
  "models": {"english": true, "multilingual": true},
  "multilingual_dir": "models/laya-onnx-multilingual-finetuned",
  "load_error": null
}
```

### 4.2 `GET /api/samples`

```json
{
  "samples": [
    {"id": "zh-scam", "label": "中文·快递理赔诈骗", "text": "您好，我是XX快递客服…", "lang": "zh"},
    {"id": "zh-benign", "label": "中文·家人问候", "text": "妈，我今晚回家吃饭…", "lang": "zh"},
    {"id": "en-scam", "label": "英文·PayPal 钓鱼", "text": "URGENT: Verify your PayPal…", "lang": "en"},
    {"id": "en-benign", "label": "英文·会议确认", "text": "Hi, just confirming…", "lang": "en"}
  ]
}
```

### 4.3 `POST /api/predict`

请求：

```jsonc
{
  "text": "您好，我是XX快递客服…",
  "primitives": ["noul", "score", "choice"],
  "model": "auto",                    // auto | english | multilingual
  "questions": {                      // 可选：高级模式覆盖
    "is_scam":       {"type": "noul",  "instructions": "..."},
    "risk_level":    {"type": "score", "instructions": "...", "criteria": ["..."]},
    "scam_category": {"type": "choice","instructions": "...", "criteria": {"k": "v"}}
  }
}
```

响应：

```jsonc
{
  "answers": {
    "is_scam":       {"noul": 0.999, "probabilities": {"false": 0.001, "true": 0.999}, "confidence": 0.999},
    "risk_level":    {"score": 2.47, "max_score": 4, "distribution": {"1 - ...": 0.15, ...}},
    "scam_category": {"choice": "delivery_fraud", "probabilities": {"...": 0.33}, "confidence": 0.331}
  },
  "routing": {"model": "multilingual", "reason": "script=cjk_han (non-Latin)"},
  "latency_ms": 91.2
}
```

### 4.4 原语 → question key 映射

| primitive | question key | 类型 |
|---|---|---|
| `noul` | `is_scam` | 是/否 |
| `score` | `risk_level` | 1-5 评分 |
| `choice` | `scam_category` | 13 类 |

---

## 5. 前端设计

### 5.1 布局（单页三区）

1. **输入区**：文本框 + 4 个预设样本按钮
2. **配置区**：
   - 三原语勾选框
   - 模型单选（自动/英文/多语）
   - `<details>` 高级编辑（instructions + criteria）
   - "开始分析"按钮
3. **结果区**：按勾选原语动态生成卡片 + 元信息行

### 5.2 结果卡片渲染规则

| 原语 | 主判定展示 | 分布展示 |
|---|---|---|
| noul | 大字「🚨 诈骗 99.9%」/「✅ 正常」| true/false 两条概率条 |
| score | 「期望分 2.47 / 4」| 每个等级一条概率条 |
| choice | 「🏷 delivery_fraud 33.1%」| 13 类按概率降序，前 3 高亮，其余折叠 |

### 5.3 交互行为

- 初始：3 原语全勾，模型=自动
- 预设样本：点击填入（不自动运行）
- 运行前校验：文本非空 + 至少 1 原语，否则按钮禁用
- 运行中：按钮「分析中…」并禁用
- 颜色编码：诈骗/高险=红，可疑=橙，正常=绿
- 错误：顶部红色 toast

### 5.4 视觉风格

- 深色主题（底 `#0f1117`，卡片 `#1a1d29`）
- 无 CSS 框架，单文件 `style.css`
- 概率数字用等宽字体
- 响应式：窄屏卡片堆叠

---

## 6. 数据流

```
POST /api/predict
  → 1. 校验 text / primitives / model           (422 on failure)
  → 2. 构建 questions：请求覆盖 或 schemas/scam.json 按勾选过滤
  → 3. 选择 client：auto→router.predict / 显式 model
  → 4. OnnxLayaClient.predict：tokenize → ONNX → softmax(+temp)
  → 5. 返回 answers + routing + latency_ms
```

---

## 7. 模型加载（容错单例）

```python
@asynccontextmanager
async def lifespan(app):
    app.state.router = None
    app.state.load_error = None
    try:
        app.state.router = Router(english_dir=..., multilingual_dir=...)
    except Exception as e:
        app.state.load_error = str(e)   # 不崩溃；health 报告 ok=false
    yield
    app.state.router = None
```

- 模型加载耗时（1.2GB ONNX），单例避免每请求重复
- 模型缺失时服务仍可启动，`/api/health` 报告原因，`/api/predict` 返回 503

---

## 8. 错误处理矩阵

| 场景 | HTTP | 响应 | 前端 |
|---|---|---|---|
| text 为空 | 422 | `{"detail": "text must be non-empty"}` | toast |
| primitives 为空 | 422 | `{"detail": "select at least one primitive"}` | 按钮禁用 |
| 未知 primitive | 422 | `{"detail": "unknown primitive: x"}` | toast |
| 未知 model | 422 | `{"detail": "model must be one of ..."}` | toast |
| 模型未加载 | 503 | `{"detail": "model not loaded: ..."}` | 顶部横幅 |
| 推理异常 | 500 | `{"detail": "inference failed: ..."}` | toast |
| 文本超长 | 200 | 正常（内部截断） | 标注「已截断」 |

---

## 9. 配置（`server/config.py`）

```python
ENGLISH_DIR      = env("LAYA_ENGLISH_DIR", "models/laya-onnx-en")
MULTILINGUAL_DIR = env("LAYA_MULTILINGUAL_DIR", "models/laya-onnx-multilingual-finetuned")
HOST             = env("LAYA_HOST", "127.0.0.1")
PORT             = int(env("LAYA_PORT", "8000"))
```

---

## 10. 测试策略（TDD）

**后端** `tests/test_server.py`（FastAPI TestClient）：

| # | 测试 | 断言 |
|---|---|---|
| 1 | health_ok | 200 + `ok:true` |
| 2 | samples_returns_presets | ≥4 条，含中英 |
| 3 | predict_all_primitives | 响应含 3 个 answers 键 |
| 4 | predict_subset_primitives | 只勾 noul → 只有 is_scam |
| 5 | predict_auto_routes_chinese | 中文+auto → routing.model=multilingual |
| 6 | predict_force_english | 中文+english → routing.model=english |
| 7 | predict_empty_text_422 | 422 |
| 8 | predict_no_primitives_422 | 422 |
| 9 | predict_unknown_primitive_422 | 422 |
| 10 | predict_custom_questions_override | 使用自定义 questions |
| 11 | predict_returns_latency_and_routing | 含 latency_ms + routing |

**前端**：4 个预设样本手动验收；可选 Playwright 截图验证。

---

## 11. 文件结构

```
laya/
├── server/
│   ├── __init__.py
│   ├── app.py           # FastAPI app + lifespan + endpoints
│   ├── config.py        # env 配置
│   └── schemas.py       # Pydantic 请求/响应模型
├── web/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── tests/
│   └── test_server.py   # 11 个 API 测试
└── pyproject.toml       # 新增 [serve] extra
```

---

## 12. 依赖新增

```toml
[project.optional-dependencies]
serve = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "httpx>=0.27",      # TestClient
]
```

---

## 13. 验收标准

| 检查项 | 目标 |
|---|---|
| 服务启动 | `uvicorn server.app:app` 无错误 |
| `/api/health` | 两模型均就绪 |
| 三原语全跑 | 单请求返回 3 张卡 |
| 原语子集 | 只请求勾选的 |
| 中英路由 | 中文→multilingual，英文→english |
| 高级编辑 | 改 instructions 后结果变化 |
| 后端测试 | 11/11 通过 |
| 页面可用 | 4 预设样本均正常渲染 |

---

## 14. 不在范围内（YAGNI）

- 用户认证 / 多用户
- 历史记录持久化
- 批量文件上传
- 多轮对话输入
- 模型热切换（加载/卸载）
- 前端构建工具链（webpack/vite）
- 生产部署（Docker/K8s/反向代理）
