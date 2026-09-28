# Laya 反诈检测 Web Playground — 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在已微调的 Laya ONNX 模型上实现本地 Web 页面，用户输入文本、勾选三种原语（noul/score/choice）、切换模型，查看判定 + 置信度分布 + 延迟 + 路由。

**Architecture:** FastAPI 后端（`server/`，加载 Router 单例）+ 原生 HTML/CSS/JS 前端（`web/`，无构建步骤）。FastAPI 用 `StaticFiles` 挂载 `web/`，3 个 API 端点封装已有的 `OnnxLayaClient` / `Router`。纯 ONNX 路径，不引入 torch。

**Tech Stack:** Python 3.14, FastAPI, uvicorn, Pydantic v2, pytest + httpx (TestClient), vanilla HTML/JS/CSS

**Design:** `docs/plans/2026-09-28-laya-web-playground-design.md`

---

## 已确认的 API 形状（实现时依赖）

```python
Router(english_dir: str, multilingual_dir: str | None)
Router.predict(state: str, questions: dict, max_len=None, head_max_len=None, model=None) -> dict
#  model=None → 自动路由；model="english" | "multilingual" → 强制

# 返回
{
  "answers": {
    "is_scam":       {"noul": 0.018, "probabilities": {"false":.., "true":..}, "confidence": 0.981},
    "risk_level":    {"score": 0.077, "max_score": 4, "distribution": {"1 - ...": ..}, "confidence": ..},
    "scam_category": {"choice": "benign", "probabilities": {"benign": .., ...}, "confidence": ..},
  },
  "routing": {"model": "multilingual", "reason": "script=cjk_han (non-Latin)"},
  "latency_ms": 106.77,
}
```

原语 → question key：`noul → is_scam`，`score → risk_level`，`choice → scam_category`

---

## 工作分解

```
Phase 1: 项目骨架 + 配置          (Tasks 1-2)
Phase 2: 后端 API（TDD）          (Tasks 3-9)
Phase 3: 前端页面                 (Tasks 10-13)
Phase 4: 集成验证 + 文档          (Tasks 14-15)
```

---

## Phase 1: 项目骨架 + 配置

### Task 1: 新增 [serve] 依赖

**Files:**
- Modify: `pyproject.toml`

**Step 1:** 在 `[project.optional-dependencies]` 中，`train` 之前或之后加入：

```toml
serve = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "httpx>=0.27",
]
```

**Step 2:** 安装

```bash
cd /Users/ks/source_code/laya
.venv/bin/pip install -e ".[serve]"
```

Expected: fastapi, uvicorn, httpx, starlette, pydantic 安装成功。

**Step 3:** 验证

```bash
cd /Users/ks/source_code/laya
.venv/bin/python -c "import fastapi, uvicorn, httpx, pydantic; print('fastapi', fastapi.__version__); print('pydantic', pydantic.__version__)"
```

Expected: 打印版本号。

**Step 4:** Commit

```bash
cd /Users/ks/source_code/laya
git add pyproject.toml
git -c user.email=claude@local -c user.name=claude commit -m "chore(deps): add [serve] extra (fastapi, uvicorn, httpx)"
```

---

### Task 2: server 配置模块

**Files:**
- Create: `server/__init__.py`
- Create: `server/config.py`

**Step 1:** 创建 `server/__init__.py`（空文件）

```python
"""Laya web playground server package."""
```

**Step 2:** 创建 `server/config.py`

```python
"""Server configuration, overridable via environment variables."""
import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
SCHEMA_PATH = REPO_ROOT / "schemas" / "scam.json"

ENGLISH_DIR = _env("LAYA_ENGLISH_DIR", str(REPO_ROOT / "models" / "laya-onnx-en"))
MULTILINGUAL_DIR = _env(
    "LAYA_MULTILINGUAL_DIR",
    str(REPO_ROOT / "models" / "laya-onnx-multilingual-finetuned"),
)
HOST = _env("LAYA_HOST", "127.0.0.1")
PORT = int(_env("LAYA_PORT", "8000"))
```

**Step 3:** 验证导入

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -c "from server.config import ENGLISH_DIR, MULTILINGUAL_DIR, WEB_DIR, SCHEMA_PATH; print(ENGLISH_DIR); print(MULTILINGUAL_DIR); print(WEB_DIR); print(SCHEMA_PATH)"
```

Expected: 打印四个绝对路径。

**Step 4:** Commit

```bash
cd /Users/ks/source_code/laya
git add server/__init__.py server/config.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(server): add config module with env overrides"
```

---

## Phase 2: 后端 API（TDD）

### Task 3: Pydantic 请求/响应模型

**Files:**
- Create: `server/schemas.py`

**Step 1:** 写 `server/schemas.py`

```python
"""Pydantic models for the playground API."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Primitive = Literal["noul", "score", "choice"]
ModelChoice = Literal["auto", "english", "multilingual"]

PRIMITIVE_TO_KEY: dict[str, str] = {
    "noul": "is_scam",
    "score": "risk_level",
    "choice": "scam_category",
}


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")
    primitives: list[Primitive] = Field(..., min_length=1)
    model: ModelChoice = "auto"
    questions: Optional[dict[str, Any]] = None
    max_len: Optional[int] = None


class PredictResponse(BaseModel):
    answers: dict[str, Any]
    routing: dict[str, Any]
    latency_ms: float


class HealthResponse(BaseModel):
    ok: bool
    models: dict[str, bool]
    multilingual_dir: str
    load_error: Optional[str] = None


class Sample(BaseModel):
    id: str
    label: str
    text: str
    lang: str


class SamplesResponse(BaseModel):
    samples: list[Sample]
```

**Step 2:** 验证

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -c "
from server.schemas import PredictRequest, PRIMITIVE_TO_KEY
r = PredictRequest(text='hi', primitives=['noul'])
print('ok:', r.model, PRIMITIVE_TO_KEY)
try:
    PredictRequest(text='', primitives=['noul'])
    print('FAIL: empty text accepted')
except Exception:
    print('✓ empty text rejected')
try:
    PredictRequest(text='hi', primitives=[])
    print('FAIL: empty primitives accepted')
except Exception:
    print('✓ empty primitives rejected')
"
```

Expected: `ok: auto {'noul': 'is_scam', ...}` + 两个 `✓`

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add server/schemas.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(server): add Pydantic request/response models"
```

---

### Task 4: 测试脚手架 + health 测试（失败）

**Files:**
- Create: `tests/test_server.py`

**Step 1:** 写测试骨架与 health 测试

```python
"""API tests for the Laya web playground."""
import pytest
from fastapi.testclient import TestClient

from server.app import create_app  # 尚不存在 → 先失败


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_ok_with_models(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["models"]["english"] is True
        assert body["models"]["multilingual"] is True
        assert "multilingual_dir" in body


class TestSamples:
    def test_samples_returns_presets(self, client):
        r = client.get("/api/samples")
        assert r.status_code == 200
        samples = r.json()["samples"]
        assert len(samples) >= 4
        langs = {s["lang"] for s in samples}
        assert {"zh", "en"}.issubset(langs)
        for s in samples:
            assert s["id"] and s["label"] and s["text"]
```

**Step 2:** 运行确认失败

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/test_server.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'server.app'`

---

### Task 5: FastAPI app 骨架（health + samples）

**Files:**
- Create: `server/app.py`

**Step 1:** 写 `server/app.py`

```python
"""FastAPI app for the Laya scam-detection playground."""
from contextlib import asynccontextmanager
import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config
from server.schemas import (
    HealthResponse,
    PRIMITIVE_TO_KEY,
    PredictRequest,
    PredictResponse,
    Sample,
    SamplesResponse,
)

SAMPLES = [
    Sample(id="zh-scam", label="中文·快递理赔诈骗", lang="zh",
           text="您好，我是XX快递客服，您有一个包裹在运输途中丢失，请点击链接填写个人信息进行理赔。"),
    Sample(id="zh-benign", label="中文·家人问候", lang="zh",
           text="妈，我今晚回家吃饭，大概6点到家。"),
    Sample(id="en-scam", label="英文·PayPal 钓鱼", lang="en",
           text="URGENT: Your PayPal account has been limited. Click here to verify: http://paypa1-secure.tk/login"),
    Sample(id="en-benign", label="英文·会议确认", lang="en",
           text="Hi, just confirming our meeting at 3pm tomorrow. Looking forward to it."),
]


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router = None
        app.state.load_error = None
        try:
            from src.router import Router
            app.state.router = Router(
                english_dir=config.ENGLISH_DIR,
                multilingual_dir=config.MULTILINGUAL_DIR,
            )
        except Exception as e:
            app.state.load_error = str(e)
        yield
        app.state.router = None

    app = FastAPI(title="Laya Scam Playground", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        router = app.state.router
        return HealthResponse(
            ok=router is not None,
            models={
                "english": bool(router and router.english),
                "multilingual": bool(router and getattr(router, "multilingual", None)),
            },
            multilingual_dir=config.MULTILINGUAL_DIR,
            load_error=app.state.load_error,
        )

    @app.get("/api/samples", response_model=SamplesResponse)
    def samples() -> SamplesResponse:
        return SamplesResponse(samples=SAMPLES)

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        router = app.state.router
        if router is None:
            raise HTTPException(status_code=503, detail=f"model not loaded: {app.state.load_error}")

        if req.questions is not None:
            questions = {PRIMITIVE_TO_KEY[p]: req.questions[PRIMITIVE_TO_KEY[p]]
                         for p in req.primitives
                         if PRIMITIVE_TO_KEY[p] in req.questions}
            if not questions:
                raise HTTPException(status_code=422, detail="no matching questions provided")
        else:
            schema = json.loads(config.SCHEMA_PATH.read_text())
            questions = {PRIMITIVE_TO_KEY[p]: schema[PRIMITIVE_TO_KEY[p]] for p in req.primitives}

        model = None if req.model == "auto" else req.model
        try:
            result = router.predict(req.text, questions, max_len=req.max_len, model=model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"inference failed: {e}")

        return PredictResponse(
            answers=result["answers"],
            routing=result["routing"],
            latency_ms=result["latency_ms"],
        )

    # Serve the frontend (index at /, assets under /static)
    if config.WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(config.WEB_DIR / "index.html"))

    return app


app = create_app()
```

**Step 2:** 运行测试

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/test_server.py -v
```

Expected: 2 passed（health + samples）。模型加载约 2-5s。

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add server/app.py tests/test_server.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(server): add FastAPI app with health and samples endpoints"
```

---

### Task 6: predict 端点测试（全部原语）

**Files:**
- Modify: `tests/test_server.py`

**Step 1:** 追加测试

```python
class TestPredict:
    def test_all_primitives_returns_three_answers(self, client):
        r = client.post("/api/predict", json={
            "text": "您好，我是XX快递客服，您有一个包裹丢失需要理赔。",
            "primitives": ["noul", "score", "choice"],
            "model": "auto",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body["answers"].keys()) == {"is_scam", "risk_level", "scam_category"}
        assert "routing" in body and "latency_ms" in body
        assert 0.0 <= body["answers"]["is_scam"]["noul"] <= 1.0

    def test_subset_primitives_only_returns_selected(self, client):
        r = client.post("/api/predict", json={
            "text": "Hello world",
            "primitives": ["noul"],
        })
        assert r.status_code == 200, r.text
        assert set(r.json()["answers"].keys()) == {"is_scam"}

    def test_auto_routes_chinese_to_multilingual(self, client):
        r = client.post("/api/predict", json={
            "text": "您好世界",
            "primitives": ["noul"],
            "model": "auto",
        })
        assert r.json()["routing"]["model"] == "multilingual"

    def test_force_english_overrides_routing(self, client):
        r = client.post("/api/predict", json={
            "text": "您好世界",
            "primitives": ["noul"],
            "model": "english",
        })
        assert r.json()["routing"]["model"] == "english"

    def test_custom_questions_override_schema(self, client):
        r = client.post("/api/predict", json={
            "text": "This is definitely a scam message about money.",
            "primitives": ["noul"],
            "questions": {
                "is_scam": {"type": "noul", "instructions": "Is this malicious?"}
            },
        })
        assert r.status_code == 200, r.text
        assert "is_scam" in r.json()["answers"]
```

**Step 2:** 运行

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/test_server.py -v
```

Expected: 7 passed

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add tests/test_server.py
git -c user.email=claude@local -c user.name=claude commit -m "test(server): add predict endpoint tests"
```

---

### Task 7: 校验错误测试（422）

**Files:**
- Modify: `tests/test_server.py`

**Step 1:** 追加

```python
class TestValidation:
    def test_empty_text_422(self, client):
        r = client.post("/api/predict", json={"text": "", "primitives": ["noul"]})
        assert r.status_code == 422

    def test_empty_primitives_422(self, client):
        r = client.post("/api/predict", json={"text": "hi", "primitives": []})
        assert r.status_code == 422

    def test_unknown_primitive_422(self, client):
        r = client.post("/api/predict", json={"text": "hi", "primitives": ["bogus"]})
        assert r.status_code == 422

    def test_unknown_model_422(self, client):
        r = client.post("/api/predict", json={
            "text": "hi", "primitives": ["noul"], "model": "bogus",
        })
        assert r.status_code == 422
```

**Step 2:** 运行

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/test_server.py -v
```

Expected: 11 passed（2 health/samples + 5 predict + 4 validation）

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add tests/test_server.py
git -c user.email=claude@local -c user.name=claude commit -m "test(server): add validation error tests (422)"
```

---

### Task 8: 验证整个后端测试套件无回归

**Step 1:** 运行全部测试

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/ --tb=short -q
```

Expected: 原 98 passed + 新 11 = **109 passed, 2 skipped**

**Step 2:** 若有失败，修复后重跑。

**Step 3:**（无代码变更则跳过 commit）

---

## Phase 3: 前端页面

### Task 9: index.html

**Files:**
- Create: `web/index.html`

**Step 1:** 写 `web/index.html`

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Laya 反诈检测 Playground</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <header>
    <h1>🛡 Laya 反诈检测 Playground</h1>
    <p class="sub">微调模型 · ONNX 本地推理 · 中英双语</p>
    <div id="health-badge" class="badge badge-loading">加载中…</div>
  </header>

  <main>
    <section class="card">
      <h2>① 输入文本</h2>
      <textarea id="text" rows="4"
        placeholder="粘贴或输入要检测的文本（中文/英文）…"></textarea>
      <div id="samples" class="samples"></div>
    </section>

    <section class="card">
      <h2>② 配置</h2>
      <div class="row">
        <span class="row-label">输出方式</span>
        <label><input type="checkbox" class="primitive" value="noul" checked /> 是否诈骗 (noul)</label>
        <label><input type="checkbox" class="primitive" value="score" checked /> 风险评分 (score)</label>
        <label><input type="checkbox" class="primitive" value="choice" checked /> 诈骗类别 (choice)</label>
      </div>
      <div class="row">
        <span class="row-label">模型</span>
        <label><input type="radio" name="model" value="auto" checked /> 自动路由</label>
        <label><input type="radio" name="model" value="english" /> 英文 base</label>
        <label><input type="radio" name="model" value="multilingual" /> 微调多语版</label>
      </div>
      <details id="advanced">
        <summary>高级设置（可编辑问题与选项）</summary>
        <div id="advanced-body"></div>
      </details>
      <button id="run">开始分析</button>
      <span id="run-hint" class="hint"></span>
    </section>

    <section id="results"></section>
    <div id="meta" class="meta"></div>
  </main>

  <div id="toast" class="toast hidden"></div>
  <script src="/static/app.js"></script>
</body>
</html>
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add web/index.html
git -c user.email=claude@local -c user.name=claude commit -m "feat(web): add index.html"
```

---

### Task 10: style.css

**Files:**
- Create: `web/style.css`

**Step 1:** 写 `web/style.css`

```css
:root {
  --bg: #0f1117; --card: #1a1d29; --border: #2a2f42;
  --fg: #e6e9f0; --muted: #8b93a7;
  --red: #ef4444; --orange: #f59e0b; --green: #22c55e; --accent: #6366f1;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  max-width: 900px; margin: 0 auto;
}
header { margin-bottom: 20px; position: relative; }
h1 { margin: 0 0 4px; font-size: 22px; }
.sub { margin: 0; color: var(--muted); font-size: 13px; }
.badge {
  position: absolute; top: 0; right: 0; font-size: 12px;
  padding: 4px 10px; border-radius: 999px;
}
.badge-ok { background: rgba(34,197,94,.15); color: var(--green); }
.badge-err { background: rgba(239,68,68,.15); color: var(--red); }
.badge-loading { background: rgba(139,147,167,.15); color: var(--muted); }
.card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin-bottom: 16px;
}
.card h2 { margin: 0 0 12px; font-size: 14px; color: var(--muted); font-weight: 600; }
textarea {
  width: 100%; background: var(--bg); color: var(--fg);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 10px; font-size: 14px; resize: vertical; font-family: inherit;
}
.samples { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }
.samples button {
  background: var(--bg); color: var(--muted); border: 1px solid var(--border);
  border-radius: 999px; padding: 5px 12px; font-size: 12px; cursor: pointer;
}
.samples button:hover { color: var(--fg); border-color: var(--accent); }
.row { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-bottom: 10px; }
.row-label { color: var(--muted); font-size: 13px; min-width: 64px; }
.row label { font-size: 13px; cursor: pointer; }
details { margin: 10px 0; }
summary { cursor: pointer; color: var(--muted); font-size: 13px; }
#advanced-body { margin-top: 10px; display: flex; flex-direction: column; gap: 10px; }
.adv-item label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.adv-item input, .adv-item textarea {
  width: 100%; background: var(--bg); color: var(--fg);
  border: 1px solid var(--border); border-radius: 6px; padding: 6px; font-size: 12px;
}
#run {
  background: var(--accent); color: white; border: 0; border-radius: 8px;
  padding: 10px 20px; font-size: 14px; cursor: pointer; font-weight: 600;
}
#run:disabled { opacity: .4; cursor: not-allowed; }
.hint { color: var(--muted); font-size: 12px; margin-left: 10px; }
.result-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; margin-bottom: 12px;
}
.result-card h3 { margin: 0 0 10px; font-size: 13px; color: var(--muted); }
.verdict { font-size: 20px; font-weight: 700; margin-bottom: 12px; }
.bar-row { display: grid; grid-template-columns: 200px 1fr 60px; gap: 8px;
  align-items: center; margin-bottom: 5px; font-size: 12px; }
.bar-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); }
.bar-track { background: var(--bg); border-radius: 4px; height: 14px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 4px; }
.bar-pct { text-align: right; font-family: ui-monospace, monospace; color: var(--muted); }
.meta { color: var(--muted); font-size: 12px; padding: 8px 0; font-family: ui-monospace, monospace; }
.toast {
  position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
  background: var(--red); color: white; padding: 10px 20px;
  border-radius: 8px; font-size: 13px; z-index: 100;
}
.hidden { display: none; }
@media (max-width: 640px) {
  .bar-row { grid-template-columns: 110px 1fr 50px; }
}
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add web/style.css
git -c user.email=claude@local -c user.name=claude commit -m "feat(web): add dark-theme stylesheet"
```

---

### Task 11: app.js（前端逻辑）

**Files:**
- Create: `web/app.js`

**Step 1:** 写 `web/app.js`

```javascript
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const COLORS = { noul: "var(--red)", score: "var(--orange)", choice: "var(--accent)" };

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 4000);
}

function pct(x) { return (x * 100).toFixed(1) + "%"; }

function selectedPrimitives() {
  return $$(".primitive:checked").map(c => c.value);
}
function selectedModel() {
  return $('input[name="model"]:checked').value;
}

function updateRunState() {
  const hasText = $("#text").value.trim().length > 0;
  const hasPrim = selectedPrimitives().length > 0;
  $("#run").disabled = !(hasText && hasPrim);
  $("#run-hint").textContent = !hasText ? "请输入文本"
    : !hasPrim ? "请至少勾选一种输出方式" : "";
}

// ---- health ----
async function loadHealth() {
  const badge = $("#health-badge");
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    if (h.ok) {
      badge.textContent = "模型就绪";
      badge.className = "badge badge-ok";
    } else {
      badge.textContent = "模型未加载";
      badge.className = "badge badge-err";
      toast("模型未加载: " + (h.load_error || "unknown"));
    }
  } catch (e) {
    badge.textContent = "服务不可用";
    badge.className = "badge badge-err";
  }
}

// ---- samples ----
async function loadSamples() {
  const r = await fetch("/api/samples");
  const { samples } = await r.json();
  const box = $("#samples");
  box.innerHTML = "";
  for (const s of samples) {
    const b = document.createElement("button");
    b.textContent = s.label;
    b.onclick = () => { $("#text").value = s.text; updateRunState(); };
    box.appendChild(b);
  }
}

// ---- advanced editor ----
let DEFAULT_QUESTIONS = null;
async function loadDefaults() {
  const r = await fetch("/api/defaults").catch(() => null);
  if (r && r.ok) DEFAULT_QUESTIONS = (await r.json()).questions;
}

function renderAdvanced() {
  const body = $("#advanced-body");
  body.innerHTML = "";
  if (!DEFAULT_QUESTIONS) {
    body.innerHTML = "<p class='hint'>默认问题不可用</p>";
    return;
  }
  const prims = selectedPrimitives();
  for (const p of prims) {
    const key = { noul: "is_scam", score: "risk_level", choice: "scam_category" }[p];
    const q = DEFAULT_QUESTIONS[key];
    if (!q) continue;
    const div = document.createElement("div");
    div.className = "adv-item";
    div.innerHTML = `
      <label>${p} · ${key} instructions</label>
      <input data-key="${key}" data-field="instructions" value="${(q.instructions || "").replace(/"/g, "&quot;")}" />
      <label style="margin-top:8px">criteria (JSON)</label>
      <textarea data-key="${key}" data-field="criteria" rows="3">${JSON.stringify(q.criteria || [], null, 0)}</textarea>
    `;
    body.appendChild(div);
  }
}

function collectAdvancedQuestions() {
  const prims = selectedPrimitives();
  const out = {};
  let edited = false;
  for (const p of prims) {
    const key = { noul: "is_scam", score: "risk_level", choice: "scam_category" }[p];
    const base = DEFAULT_QUESTIONS ? { ...DEFAULT_QUESTIONS[key] } : null;
    if (!base) continue;
    const insEl = $(`input[data-key="${key}"][data-field="instructions"]`);
    const critEl = $(`textarea[data-key="${key}"][data-field="criteria"]`);
    if (insEl) base.instructions = insEl.value;
    if (critEl) {
      try {
        const parsed = JSON.parse(critEl.value);
        base.criteria = parsed;
        if (JSON.stringify(parsed) !== JSON.stringify(DEFAULT_QUESTIONS[key].criteria)) edited = true;
      } catch { /* keep default on invalid JSON */ }
    }
    if (insEl && insEl.value !== DEFAULT_QUESTIONS[key].instructions) edited = true;
    out[key] = base;
  }
  return edited ? out : null;
}

// ---- render results ----
function barRow(label, value, color) {
  const w = Math.max(0, Math.min(1, value)) * 100;
  return `<div class="bar-row">
    <span class="bar-label" title="${label}">${label}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${color}"></div></div>
    <span class="bar-pct">${pct(value)}</span>
  </div>`;
}

function renderNoul(ans) {
  const p = ans.noul;
  const isScam = p >= 0.5;
  const color = isScam ? "var(--red)" : "var(--green)";
  const verdict = isScam ? `🚨 诈骗 ${pct(p)}` : `✅ 正常 ${pct(1 - p)}`;
  return `<div class="result-card">
    <h3>是否诈骗 · noul</h3>
    <div class="verdict" style="color:${color}">${verdict}</div>
    ${barRow("false (正常)", ans.probabilities.false, "var(--green)")}
    ${barRow("true (诈骗)", ans.probabilities.true, "var(--red)")}
  </div>`;
}

function renderScore(ans) {
  const rows = Object.entries(ans.distribution)
    .map(([k, v]) => barRow(k, v, "var(--orange)")).join("");
  return `<div class="result-card">
    <h3>风险评分 · score</h3>
    <div class="verdict">期望分 ${ans.score.toFixed(2)} / ${ans.max_score}</div>
    ${rows}
  </div>`;
}

function renderChoice(ans) {
  const sorted = Object.entries(ans.probabilities).sort((a, b) => b[1] - a[1]);
  const top = sorted[0];
  const rows = sorted.map(([k, v], i) =>
    barRow(k, v, i === 0 ? "var(--accent)" : "var(--bg)")).join("");
  return `<div class="result-card">
    <h3>诈骗类别 · choice</h3>
    <div class="verdict">🏷 ${top[0]} <span style="color:var(--muted);font-size:14px">${pct(top[1])}</span></div>
    ${rows}
  </div>`;
}

const RENDERERS = { is_scam: renderNoul, risk_level: renderScore, scam_category: renderChoice };

function renderResults(body) {
  const box = $("#results");
  box.innerHTML = "";
  for (const [key, ans] of Object.entries(body.answers)) {
    if (RENDERERS[key]) box.insertAdjacentHTML("beforeend", RENDERERS[key](ans));
  }
  $("#meta").textContent =
    `模型: ${body.routing.model} · 延迟: ${body.latency_ms.toFixed(0)}ms · 原因: ${body.routing.reason}`;
}

// ---- run ----
async function run() {
  const btn = $("#run");
  btn.disabled = true;
  btn.textContent = "分析中…";
  try {
    const payload = {
      text: $("#text").value,
      primitives: selectedPrimitives(),
      model: selectedModel(),
    };
    const adv = collectAdvancedQuestions();
    if (adv) payload.questions = adv;

    const r = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await r.json();
    if (!r.ok) { toast(body.detail || `HTTP ${r.status}`); return; }
    renderResults(body);
  } catch (e) {
    toast("请求失败: " + e.message);
  } finally {
    btn.textContent = "开始分析";
    updateRunState();
  }
}

// ---- init ----
async function init() {
  $("#text").addEventListener("input", updateRunState);
  $$(".primitive").forEach(c => c.addEventListener("change", () => {
    updateRunState(); renderAdvanced();
  }));
  $("#run").addEventListener("click", run);
  await loadHealth();
  await loadSamples();
  await loadDefaults();
  renderAdvanced();
  updateRunState();
}
init();
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add web/app.js
git -c user.email=claude@local -c user.name=claude commit -m "feat(web): add frontend logic (fetch, render, advanced editor)"
```

---

### Task 12: 新增 /api/defaults 端点（供前端高级编辑预填）

**Files:**
- Modify: `server/app.py`
- Modify: `tests/test_server.py`

**Step 1:** 追加测试

```python
class TestDefaults:
    def test_defaults_returns_three_questions(self, client):
        r = client.get("/api/defaults")
        assert r.status_code == 200
        q = r.json()["questions"]
        assert set(q.keys()) == {"is_scam", "risk_level", "scam_category"}
        assert q["is_scam"]["type"] == "noul"
        assert q["risk_level"]["type"] == "score"
        assert q["scam_category"]["type"] == "choice"
```

**Step 2:** 运行确认失败

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/test_server.py::TestDefaults -v
```

Expected: FAIL (404)

**Step 3:** 在 `server/app.py` 的 `health` 端点后加入：

```python
    @app.get("/api/defaults")
    def defaults() -> dict:
        return {"questions": json.loads(config.SCHEMA_PATH.read_text())}
```

**Step 4:** 运行

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/test_server.py -v
```

Expected: 12 passed

**Step 5:** Commit

```bash
cd /Users/ks/source_code/laya
git add server/app.py tests/test_server.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(server): add /api/defaults endpoint for the advanced editor"
```

---

## Phase 4: 集成验证 + 文档

### Task 13: 启动服务并手动验证

**Step 1:** 启动

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Expected: 启动日志显示 `Uvicorn running on http://127.0.0.1:8000`；模型加载 2-5s。

**Step 2:** 命令行验证 API

```bash
# 另开终端
curl -s localhost:8000/api/health | python3 -m json.tool
curl -s localhost:8000/api/defaults | python3 -c "import json,sys; print(list(json.load(sys.stdin)['questions'].keys()))"
curl -s -X POST localhost:8000/api/predict -H 'content-type: application/json' \
  -d '{"text":"您好，我是XX快递客服，您有一个包裹丢失需要理赔。","primitives":["noul","score","choice"]}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('routing:', d['routing']); print('latency:', round(d['latency_ms'])); print('is_scam:', round(d['answers']['is_scam']['noul'],3)); print('category:', d['answers']['scam_category']['choice'])"
```

Expected:
- health: `ok: true`
- defaults: `['is_scam', 'risk_level', 'scam_category']`
- predict: `routing: multilingual`, `is_scam: ~0.99`, `category: delivery_fraud`

**Step 3:** 浏览器验证 `http://127.0.0.1:8000`

验收清单：
- [ ] 页面加载，右上角显示"模型就绪"
- [ ] 4 个预设样本按钮可见，点击填入文本框
- [ ] 3 个原语默认勾选，模型默认"自动路由"
- [ ] 点"开始分析"→ 出现 3 张结果卡 + 元信息行
- [ ] noul 卡片显示红色"🚨 诈骗 99.x%" + 两条概率条
- [ ] choice 卡片显示 13 类按概率降序的条形图
- [ ] 取消勾选 score → 结果只剩 2 张卡
- [ ] 选"英文 base" + 中文文本 → 元信息显示 `模型: english`
- [ ] 展开"高级设置"→ 可编辑 instructions → 改动后结果变化
- [ ] 清空文本 → 按钮禁用并提示"请输入文本"

**Step 4:** 停止服务（Ctrl-C）

**Step 5:** 无需 commit（纯验证）

---

### Task 14: README 增加运行说明

**Files:**
- Modify: `README.md`

**Step 1:** 在 README 的 "Usage" 章节后追加：

```markdown
### Web Playground

Interactive UI over the fine-tuned model.

```bash
pip install -e ".[serve]"
PYTHONPATH=. python -m uvicorn server.app:app --port 8000
# open http://127.0.0.1:8000
```

Features:
- Paste text (Chinese/English), pick which of the 3 primitives to run
  (noul / score / choice)
- Switch model: auto-route / English base / fine-tuned multilingual
- Expand "advanced" to edit instructions and criteria inline
- Results show verdict, full probability distribution, latency, and routing

API: `GET /api/health`, `GET /api/samples`, `GET /api/defaults`,
`POST /api/predict`.
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add README.md
git -c user.email=claude@local -c user.name=claude commit -m "docs: add web playground usage to README"
```

---

### Task 15: 最终测试全跑

**Step 1:** 运行全部测试

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/ --tb=short -q
```

Expected: **109 passed, 2 skipped**（原 98 + 新 12，减去重叠计数以实际为准）

**Step 2:** 检查 git 状态

```bash
cd /Users/ks/source_code/laya
git status --short
git log --oneline -8
```

Expected: 无未提交的代码文件（`.omo/` 与 kaggle/upload 的 jsonl 属预期忽略项）。

---

## 验收标准（设计 §13）

| 检查项 | 目标 |
|---|---|
| 服务启动 | `uvicorn server.app:app` 无错误 |
| `/api/health` | 两模型均就绪 |
| 三原语全跑 | 单请求返回 3 张卡 |
| 原语子集 | 只请求勾选的 |
| 中英路由 | 中文→multilingual，英文→english |
| 高级编辑 | 改 instructions 后结果变化 |
| 后端测试 | 12/12 通过 |
| 页面可用 | 4 预设样本均正常渲染 |

---

## 执行方式

计划保存至 `docs/plans/2026-09-28-laya-web-playground-impl.md`。

两个执行选项：
1. **Subagent-Driven**（本会话）— 每任务派新 subagent，任务间审查
2. **Parallel Session**（新会话）— 用 executing-plans 批量执行 + 检查点

注：本环境 subagent dispatch 曾因 API key 失败；建议由主代理直接执行（多数任务为本地文件操作）。
