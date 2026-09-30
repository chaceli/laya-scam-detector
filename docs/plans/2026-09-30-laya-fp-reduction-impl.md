# Laya v2 误报率压降 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 依据 `research/dataset_research.md` 洞察与设计文档，扩充 4 个新数据源、构造八体裁难负样本、14 类体系重训 v2，把难负评测集 FPR 压到 ≤2% 且现有三门槛不回退。

**Architecture:** 现有 fetch→build→train→export→eval 管线保持不动；新增数据源 loader 模块（`scripts/dataset_loaders.py`）与显式配比模块（`scripts/dataset_mix.py`）；难负样本以独立 jsonl（git 跟踪）注入训练；评测协议扩展分体裁 FPR／分类别召回／阈值曲线；FPR>2% 时滚动挖掘闭环（≤2 轮）。

**Tech Stack:** Python ≥3.10（venv 为 3.14）；推理仅 onnxruntime+tokenizers+numpy；训练 PyTorch+laya SDK+peft（M4 Pro MPS）；难负生成走 arkcli（火山方舟多模型）；验证 pytest。

## Global Constraints（所有任务默认遵守）

- 设计文档 `docs/plans/2026-09-30-laya-fp-reduction-design.md`；洞察报告 `research/dataset_research.md`
- **14 类**：`CANONICAL_CATEGORIES` 在 `job_scam` 之后新增 `rebate_scam`；`schemas/scam.json` 的 `scam_category.criteria` 键顺序必须与 `CANONICAL_CATEGORIES` 完全一致（训练 CHOICE_Q 与推理共用此顺序）
- **配比硬约束**（compose_train 断言）：CCL **非刷单返利** 正样本 ≤20% 正样本总量（`rebate_scam` 豁免——spec 勘误：原「CCL 笔录 ≤20%」与「刷单返利 ≥25%」矛盾，以本条为准）；ChiFraud benign ≤10% 负样本总量；任何单类别 ≤30% 正样本总量；`rebate_scam` ≥25% 正样本总量；生成样本 ≤30% 训练总量；FGRC 提示类（fgrc 两 source 的 benign）= 10–15% 负样本总量
- **评测硬门槛**：难负评测集 FPR ≤2%（主目标）；holdout zh acc ≥0.90 / recall ≥0.95；手写集 14 类 acc ≥0.70；`rebate_scam` 召回 ≥0.80
- `datasets/hardneg_eval.jsonl` **永不进训练**；训练/评测 id 不相交是构建期断言
- 所有命令从仓库根目录执行；HF 相关命令前缀 `HF_HUB_DISABLE_XET=1`
- **pytest 是验证门**（ruff 非门槛建，不修不相关文件）；conventional commits
- 挖掘闭环硬上限 2 轮，仍不达标如实报告
- `models/`、`datasets/raw/`、`datasets/training/`、`dist/` 不提交；`datasets/hard_negatives/*.jsonl`、`datasets/hardneg_eval.jsonl`、`scripts/gen_models.json`、种子语料**提交进 git**（数据即证据，训练可复现）
- 训练行格式（勿改）：`{"id", "text", "is_scam": 0|1, "risk": 1-5, "category": "<canonical>", "language", "source"}`；评测行格式：`{"id", "type", "state"|"turns", "language", "expected_risk", "category", "source"}`

---

### Task 1: 14 类 schema 核心（scam_categories.py）

**Files:**
- Modify: `schemas/scam_categories.py`
- Modify: `tests/test_categories.py`

**Interfaces:**
- Produces: `CANONICAL_CATEGORIES`（14 项，`rebate_scam` 紧跟 `job_scam`）；`CCL2023_LABEL_MAP: dict[str,str]`（12 类全覆盖）；`CHIFRAUD_SCAM_MAP`、`CHIFRAUD_BENIGN_LABELS`、`TELE_NORMAL_LABELS`（后续 loader 严格消费）

- [ ] **Step 1: 修改测试（先失败）**

`tests/test_categories.py` 中 `TestCanonicalCategories` 的 13 断言改为 14，并新增：

```python
class TestCanonicalCategories:
    def test_exactly_14_categories(self):
        assert len(CANONICAL_CATEGORIES) == 14

    def test_all_unique(self):
        assert len(set(CANONICAL_CATEGORIES)) == 14

    def test_rebate_scam_after_job_scam(self):
        assert CANONICAL_CATEGORIES.index("rebate_scam") \
            == CANONICAL_CATEGORIES.index("job_scam") + 1


class TestCcl2023Mapping:
    def test_ccl_map_covers_12_classes_and_maps_to_canonical(self):
        from schemas.scam_categories import CCL2023_LABEL_MAP
        assert len(CCL2023_LABEL_MAP) == 12
        for canonical in CCL2023_LABEL_MAP.values():
            assert canonical in CANONICAL_CATEGORIES

    def test_rebate_label_maps_to_rebate_scam(self):
        from schemas.scam_categories import CCL2023_LABEL_MAP
        assert CCL2023_LABEL_MAP["刷单返利类"] == "rebate_scam"

    def test_ccl_labels_reachable_via_normalize_label(self):
        from schemas.scam_categories import normalize_label
        assert normalize_label("刷单返利类") == "rebate_scam"
        assert normalize_label("网黑案件") == "spam_general"


class TestChiFraudMapping:
    def test_only_underground_loan_maps(self):
        from schemas.scam_categories import CHIFRAUD_SCAM_MAP
        assert set(CHIFRAUD_SCAM_MAP.values()) == {"loan_scam"}

    def test_benign_labels_exist(self):
        from schemas.scam_categories import CHIFRAUD_BENIGN_LABELS
        assert len(CHIFRAUD_BENIGN_LABELS) >= 1
```

- [ ] **Step 2: 运行确认失败**

Run: `source .venv/bin/activate && pytest tests/test_categories.py -q`
Expected: FAIL（14 断言、CCL/ChiFraud 映射类全部失败）

- [ ] **Step 3: 实现 scam_categories.py 变更**

(a) `CANONICAL_CATEGORIES` 的 `"job_scam",` 行后插入：

```python
    "rebate_scam",       # v2: 刷单返利（公安口径发案量第一）
```

(b) `CATEGORY_MAP` **保持不动**（用户裁定 2026-09-30：CCL 12 类采用**单一来源注入**，消除与 (c) 的逐字重复 —— CCL 条目由 (c) 末尾的 `CATEGORY_MAP.update(CCL2023_LABEL_MAP)` 注入，`normalize_label` 经由 CATEGORY_MAP 消费）。

(c) 文件末尾（`index_to_label` 之后）新增：

```python
# CCL2023-FCC 严格映射（loader 逐条断言，未覆盖即报错，不走 spam_general 兜底）
CCL2023_LABEL_MAP: dict[str, str] = {
    "刷单返利类": "rebate_scam",
    "贷款代办信用卡类": "loan_scam",
    "虚假征信类": "loan_scam",
    "虚假投资理财类": "investment_scam",
    "虚假购物服务类": "phishing",
    "冒充客服类": "impersonation",
    "冒充公检法类": "impersonation",
    "冒充领导熟人类": "impersonation",
    "网络婚恋交友类": "romance_scam",
    "机票退改签类": "impersonation",
    "网络赌博类": "spam_general",
    "网黑案件": "spam_general",
}

# 单一来源注入：normalize_label 经 CATEGORY_MAP 消费 CCL 12 类（用户裁定 2026-09-30）
CATEGORY_MAP.update(CCL2023_LABEL_MAP)

# ChiFraud（灰产供给侧视角）：仅地下贷款可映射，其余在 loader 中显式丢弃
CHIFRAUD_SCAM_MAP: dict[str, str] = {
    "地下贷款": "loan_scam",
    "地下贷款类": "loan_scam",
}
CHIFRAUD_BENIGN_LABELS: frozenset[str] = frozenset(
    {"正常", "benign", "normal", "合法", "非诈骗"})

# TeleAntiFraud-28k：正常通话标签候选（Task 4 探测后按实际修正）
TELE_NORMAL_LABELS: frozenset[str] = frozenset(
    {"normal", "benign", "正常", "非诈骗", "ham"})
```

> CCL 标签的**精确字符串**以 Task 4 探测为准。若真实标签有差异（如「虚假投资理财」无「类」字），同步改 `CCL2023_LABEL_MAP` 与 `tests/test_categories.py` 两处（`CATEGORY_MAP` 由 update 自动同步；`test_ccl_map_covers_12_classes_and_maps_to_canonical` 强制 12 条全覆盖）。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_categories.py -q`
Expected: PASS（全部）

- [ ] **Step 5: 全量回归**

Run: `pytest -q`
Expected: 基线保持（115+ passed / 2 skipped）。若有测试硬编码 13 类数量而失败，仅把该断言改为 14 或 `len(CANONICAL_CATEGORIES)`，不扩大改动。

- [ ] **Step 6: Commit**

```bash
git add schemas/scam_categories.py tests/test_categories.py
git commit -m "feat(schema): extend taxonomy to 14 classes with rebate_scam + CCL/ChiFraud/Tele maps"
```

---

### Task 2: 14 类传播（scam.json + 训练描述）

**Files:**
- Modify: `schemas/scam.json`
- Modify: `scripts/train_local_lora.py`（`CATEGORY_DESCRIPTIONS` 字典，39-53 行附近）
- Create: `tests/test_schema_14.py`

**Interfaces:**
- Consumes: `CANONICAL_CATEGORIES`（Task 1）
- Produces: `scam.json` 的 `scam_category.criteria` 键 == `CANONICAL_CATEGORIES`（顺序一致）；`CATEGORY_DESCRIPTIONS["rebate_scam"]`（`CHOICE_Q` 构建依赖，缺失即 KeyError）

- [ ] **Step 1: 写失败测试**

```python
"""schemas/scam.json 与 14 类体系一致性。"""
import json

from schemas.scam_categories import CANONICAL_CATEGORIES


def test_scam_json_criteria_match_canonical_order():
    schema = json.loads(open("schemas/scam.json").read())
    keys = list(schema["scam_category"]["criteria"].keys())
    assert keys == CANONICAL_CATEGORIES


def test_train_descriptions_cover_all_categories():
    src = open("scripts/train_local_lora.py").read()
    for c in CANONICAL_CATEGORIES:
        assert f'"{c}":' in src, f"missing description key: {c}"
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_schema_14.py -q`
Expected: FAIL（scam.json 13 键；train 脚本无 rebate_scam 描述）

- [ ] **Step 3: 修改 schemas/scam.json**

`scam_category.criteria` 的 `"job_scam"` 行后插入（位置与 CANONICAL_CATEGORIES 一致）：

```json
      "rebate_scam": "order-brushing rebate fraud, task-based commission scam, upfront deposit",
```

- [ ] **Step 4: 修改 scripts/train_local_lora.py**

`CATEGORY_DESCRIPTIONS` 的 `"job_scam"` 行后插入：

```python
    "rebate_scam": "order-brushing rebate fraud, task-based commission scam, upfront deposit",
```

- [ ] **Step 5: 测试 + 全量回归**

Run: `pytest tests/test_schema_14.py -q && pytest -q`
Expected: PASS（基线保持）

- [ ] **Step 6: Commit**

```bash
git add schemas/scam.json scripts/train_local_lora.py tests/test_schema_14.py
git commit -m "feat(schema): propagate 14-class schema to scam.json + training descriptions"
```

---

### Task 3: fetch_datasets.py 新增 4 个数据源

**Files:**
- Modify: `scripts/fetch_datasets.py`

**Interfaces:**
- Produces: `datasets/raw/ccl2023/`（git clone）、`datasets/raw/chifraud/`（git clone）、`datasets/raw/teleantifraud/`（HF curl，文件清单由 Task 4 探测后回填 `TELE_FILES`）、`datasets/raw/phishing_email/`（kagglehub，可选）

- [ ] **Step 1: 添加常量**

`FBS_SMS_GIT_URL = ...` 之后追加：

```python
CCL2023_GIT_URL = "https://github.com/GJSeason/CCL2023-FCC.git"
CHIFRAUD_GIT_URL = "https://github.com/xuemingxxx/ChiFraud.git"
TELE_ANTIFRAUD_TREE_API = "https://huggingface.co/api/datasets/JimmyMa99/TeleAntiFraud/tree/main"
TELE_ANTIFRAUD_BASE = "https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/resolve/main"

TELE_FILES: list[str] = []  # Task 4 探测后回填，例如 ["data/train.jsonl"]

PHISHING_EMAIL_DATASET = "naserabdullahalam/phishing-email-dataset"
```

- [ ] **Step 2: main() 里 FBS_SMS clone 块之后追加**

```python
    print("\n[ccl2023]")
    git_clone(CCL2023_GIT_URL, RAW_DIR / "ccl2023")

    print("\n[chifraud]")
    git_clone(CHIFRAUD_GIT_URL, RAW_DIR / "chifraud")

    print("\n[teleantifraud]")
    if TELE_FILES:
        for relpath in TELE_FILES:
            curl_download(f"{TELE_ANTIFRAUD_BASE}/{relpath}",
                          RAW_DIR / "teleantifraud" / Path(relpath).name)
    else:
        print("  ! TELE_FILES 未配置。运行: curl -s " + TELE_ANTIFRAUD_TREE_API)
        print("    从返回 JSON 中挑数据文件路径填入 TELE_FILES 后重跑。")

    print("\n[phishing_email]")
    try:
        import kagglehub
        cache = kagglehub.dataset_download(PHISHING_EMAIL_DATASET)
        target = RAW_DIR / "phishing_email"
        target.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in Path(cache).glob("*"):
            if f.is_file():
                shutil.copy(f, target / f.name)
        print("  ✓ kagglehub -> datasets/raw/phishing_email")
    except Exception as e:
        print(f"  ✗ kagglehub 跳过（凭据/网络）: {e}")
```

- [ ] **Step 3: 语法与冒烟**

Run: `source .venv/bin/activate && python -c "import ast; ast.parse(open('scripts/fetch_datasets.py').read())" && python scripts/fetch_datasets.py 2>&1 | head -30`
Expected: 语法通过；ccl2023/chifraud 开始 clone；teleantifraud 打印探测提示；phishing_email 成功或打印跳过原因。

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_datasets.py
git commit -m "feat(data): add CCL2023-FCC, ChiFraud, TeleAntiFraud, PhishingEmail fetchers"
```

---

### Task 4: 下载 + 原始结构探测（inspect_raw.py）

**Files:**
- Create: `scripts/inspect_raw.py`

**Interfaces:**
- Produces: 结构清单（文件、首条记录字段、标签分布）。**探测结论回填三处**：`fetch_datasets.py` 的 `TELE_FILES`；`schemas/scam_categories.py` 的标签常量（若与 Task 1 假设不符）；`dataset_loaders.py`（Task 5）的字段常量。回填后重跑 Task 1 测试。

- [ ] **Step 1: 写 scripts/inspect_raw.py**

```python
"""Dump structure of each raw v2 dataset: files, first-record keys, label distribution."""
import json
import sys
from collections import Counter
from pathlib import Path

RAW = Path("datasets/raw")


def probe_jsonish(p: Path, max_records: int = 2000) -> None:
    if p.suffix == ".json":
        try:
            data = json.load(p.open())
        except Exception as e:
            print(f"    ! JSON parse fail: {e}")
            return
        records = data if isinstance(data, list) else [data]
    else:
        records = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    if not records:
        print("    (empty)")
        return
    print(f"    first record keys: {sorted(records[0].keys())}")
    print(f"    sample: {str(records[0])[:200]}")
    for key in ("label", "label_name", "labels", "类别", "风险类别", "type", "category"):
        if key in records[0]:
            c = Counter(str(r.get(key))[:30] for r in records[:max_records])
            print(f"    label {key!r} dist: {dict(c.most_common(15))}")
            break


def probe_csv(p: Path) -> None:
    import csv
    with p.open(newline="", encoding="utf-8", errors="replace") as f:
        print(f"    header: {next(csv.reader(f), None)}")


def probe_dir(name: str) -> None:
    base = RAW / name
    if not base.exists():
        print(f"[{name}] MISSING (fetch first)")
        return
    print(f"[{name}]")
    files = sorted(p for p in base.rglob("*") if p.is_file()
                   and p.suffix.lower() in (".json", ".jsonl", ".csv", ".txt")
                   and ".git" not in p.parts)
    for p in files[:30]:
        print(f"  {p.relative_to(base)} ({p.stat().st_size / 1e6:.1f} MB)")
        if p.suffix.lower() in (".json", ".jsonl"):
            probe_jsonish(p)
        elif p.suffix.lower() == ".csv":
            probe_csv(p)
    if len(files) > 30:
        print(f"  ... and {len(files) - 30} more files")


def main() -> int:
    for name in sys.argv[1:] or ["ccl2023", "chifraud", "teleantifraud", "phishing_email"]:
        probe_dir(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 运行下载**

Run: `source .venv/bin/activate && python scripts/fetch_datasets.py`
Expected: ccl2023、chifraud clone 完成（各几百 MB–1GB，耐心等待）；teleantifraud 打印探测提示；phishing_email 成功或打印跳过原因。

- [ ] **Step 3: 探测 TeleAntiFraud 文件清单并回填**

Run: `curl -s https://huggingface.co/api/datasets/JimmyMa99/TeleAntiFraud/tree/main | python -m json.tool | head -60`
把数据文件路径（以实际返回为准）填入 `fetch_datasets.py` 的 `TELE_FILES`，重跑 `python scripts/fetch_datasets.py` 下载到 `datasets/raw/teleantifraud/`。

- [ ] **Step 4: 运行探测并记录结论**

Run: `python scripts/inspect_raw.py 2>&1 | tee reports/raw_inventory.txt`
逐源记录：真实文本字段名、标签字段名、CCL 12 类精确字符串、ChiFraud 标签字符串、Tele 正常/诈骗标签字符串。

- [ ] **Step 5: 按探测结论回填常量**

若标签字符串与 Task 1 假设不同：同步修改 `schemas/scam_categories.py`（`CCL2023_LABEL_MAP`、`CHIFRAUD_*`、`TELE_NORMAL_LABELS`；`CATEGORY_MAP` 由 update 自动同步，无需手改）与 `tests/test_categories.py` 的键名。
Run: `pytest tests/test_categories.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/inspect_raw.py scripts/fetch_datasets.py schemas/scam_categories.py tests/test_categories.py reports/raw_inventory.txt
git commit -m "feat(data): raw structure probe + TeleAntiFraud file list from inventory"
```

---

### Task 5: 新数据源 loader（dataset_loaders.py）+ FBS 去空格

**Files:**
- Create: `scripts/dataset_loaders.py`
- Create: `tests/fixtures/ccl2023/train_sample.json`、`tests/fixtures/chifraud/chifraud_sample.csv`、`tests/fixtures/teleantifraud/train_sample.jsonl`、`tests/fixtures/phishing_email/phishing_sample.csv`
- Create: `tests/test_dataset_loaders.py`
- Modify: `scripts/build_dataset.py`（`load_fbs_sms` 去空格）

**Interfaces:**
- Consumes: `CCL2023_LABEL_MAP`、`CHIFRAUD_SCAM_MAP`、`CHIFRAUD_BENIGN_LABELS`、`TELE_NORMAL_LABELS`（Task 1/4）
- Produces: `load_ccl2023() / load_chifraud() / load_teleantifraud() / load_phishing_email() -> list[dict]`、`despace_cjk(text) -> str`、模块常量 `RAW: Path`（测试 monkeypatch 用）。行 shape：`{"text", "is_scam", "risk", "category", "language", "source"}`；source 取值 `ccl2023 / chifraud / teleantifraud / phishing_email`

- [ ] **Step 1: 写 fixture 文件**

`tests/fixtures/ccl2023/train_sample.json`：

```json
[
  {"text": "受害人陈述：刷单做任务先垫付999元后无法提现。", "label_name": "刷单返利类"},
  {"text": "受害人陈述：冒充航空公司客服以机票退改签为名索要费用。", "label_name": "机票退改签类"}
]
```

`tests/fixtures/chifraud/chifraud_sample.csv`：

```csv
text,label
"地下贷款加V办理无需抵押当天放款",地下贷款
"正规棋牌娱乐平台点击下载",赌博
"今天天气不错适合出游",正常
```

`tests/fixtures/teleantifraud/train_sample.jsonl`：

```jsonl
{"text": "您好我是快递客服您的包裹丢失需要理赔", "label": "fraud"}
{"text": "妈我晚上到家吃饭", "label": "normal"}
```

`tests/fixtures/phishing_email/phishing_sample.csv`：

```csv
text,label
"Your invoice for March is attached. Regards",Safe Email
```

- [ ] **Step 2: 写失败测试 tests/test_dataset_loaders.py**

```python
"""Loader tests against small fixtures (no network, no real data)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from dataset_loaders import (  # noqa: E402
    despace_cjk, load_ccl2023, load_chifraud,
    load_teleantifraud, load_phishing_email,
)
from schemas.scam_categories import CCL2023_LABEL_MAP  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures"


class TestDespaceCjk:
    def test_removes_space_between_cjk(self):
        assert despace_cjk("今 天 有 雨") == "今天有雨"

    def test_keeps_cjk_latin_boundary(self):
        assert despace_cjk("验证码 K888，勿泄露") == "验证码 K888，勿泄露"


class TestCclLoader:
    def test_fixture_rows(self, monkeypatch):
        monkeypatch.setattr("dataset_loaders.RAW", FIX)
        rows = load_ccl2023()
        assert len(rows) == 2
        assert any(r["category"] == "rebate_scam" for r in rows)
        assert all(r["is_scam"] == 1 and r["risk"] == 4
                   and r["source"] == "ccl2023" and r["language"] == "zh" for r in rows)


class TestChifraudLoader:
    def test_whitelist_only(self, monkeypatch):
        monkeypatch.setattr("dataset_loaders.RAW", FIX)
        rows = load_chifraud()
        assert any(r["is_scam"] == 1 and r["category"] == "loan_scam" for r in rows)
        assert any(r["is_scam"] == 0 and r["category"] == "benign" for r in rows)
        assert not any("棋牌" in r["text"] for r in rows)  # 赌博显式丢弃


class TestTeleLoader:
    def test_binary_map(self, monkeypatch):
        monkeypatch.setattr("dataset_loaders.RAW", FIX)
        rows = load_teleantifraud()
        assert any(r["is_scam"] == 1 and r["category"] == "spam_general" for r in rows)
        assert any(r["is_scam"] == 0 and r["category"] == "benign" for r in rows)


class TestPhishingLoader:
    def test_only_safe_rows(self, monkeypatch):
        monkeypatch.setattr("dataset_loaders.RAW", FIX)
        rows = load_phishing_email()
        assert len(rows) == 1
        assert rows[0]["is_scam"] == 0 and rows[0]["language"] == "en"


class TestMapCompleteness:
    def test_fixture_labels_in_strict_map(self):
        import json
        for rec in json.loads((FIX / "ccl2023" / "train_sample.json").read_text()):
            assert rec["label_name"] in CCL2023_LABEL_MAP
```

- [ ] **Step 3: 运行确认失败**

Run: `pytest tests/test_dataset_loaders.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 4: 实现 scripts/dataset_loaders.py**

```python
"""V2 data-source loaders. Row shape identical to build_dataset.py:
{"text", "is_scam", "risk", "category", "language", "source"}

Field-name constants are probe-verified (Task 4 inventory); adjust ONLY
these constants plus the scam_categories label maps if real files differ.
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import (
    CCL2023_LABEL_MAP,
    CHIFRAUD_BENIGN_LABELS,
    CHIFRAUD_SCAM_MAP,
    TELE_NORMAL_LABELS,
)

RAW = Path("datasets/raw")

# —— 探测期可调常量（Task 4 inventory 为准）——
CCL_TEXT_KEYS = ("text", "content", "案件描述", "文本", "dialogue")
CCL_LABEL_KEYS = ("label_name", "label", "罪名", "类别", "riskType")
CHIFRAUD_TEXT_KEYS = ("text", "content", "文本")
CHIFRAUD_LABEL_KEYS = ("label", "label_name", "类别")
TELE_TEXT_KEYS = ("text", "transcription", "content", "对话")
TELE_LABEL_KEYS = ("label", "label_name", "type", "风险类别")


def despace_cjk(text: str) -> str:
    """FBS_SMS 逐字加空格存储：'今 天 有 雨' -> '今天有雨'（保留中英边界）。"""
    return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)


def _first(item: dict, keys) -> str:
    for k in keys:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def parse_jsonish(p: Path) -> list[dict]:
    """Parse .json (list-of-dicts) or .jsonl into records."""
    if p.suffix == ".json":
        data = json.load(p.open())
        return data if isinstance(data, list) else [data]
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        out.append(rec)
                except json.JSONDecodeError:
                    continue
    return out


def load_ccl2023() -> list[dict]:
    """CCL2023-FCC 受害人笔录。严格 12 类映射：未知标签收集后一次性报错。"""
    out, unmapped = [], []
    base = RAW / "ccl2023"
    for p in sorted(base.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        if p.suffix.lower() in (".json", ".jsonl"):
            records = parse_jsonish(p)
        elif p.suffix.lower() == ".csv":
            with p.open(newline="", encoding="utf-8", errors="replace") as f:
                records = list(csv.DictReader(f))
        else:
            continue
        for item in records:
            text = _first(item, CCL_TEXT_KEYS)
            label = _first(item, CCL_LABEL_KEYS)
            if not text or not label:
                continue
            if label not in CCL2023_LABEL_MAP:
                unmapped.append(label)
                continue
            out.append({
                "text": text[:1024], "is_scam": 1, "risk": 4,
                "category": CCL2023_LABEL_MAP[label],
                "language": "zh", "source": "ccl2023",
            })
    if unmapped:
        sample = sorted(set(unmapped))[:5]
        raise RuntimeError(
            f"CCL2023 未映射标签 {len(unmapped)} 条（如 {sample}）—— "
            f"按 Task 4 探测结果补 CCL2023_LABEL_MAP")
    return out


def load_chifraud() -> list[dict]:
    """ChiFraud：仅 地下贷款→loan_scam 与 benign 保留，其余灰产类显式丢弃。"""
    out = []
    base = RAW / "chifraud"
    for p in sorted(base.rglob("*.csv")):
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                text = _first(row, CHIFRAUD_TEXT_KEYS)
                label = _first(row, CHIFRAUD_LABEL_KEYS)
                if not text:
                    continue
                if label in CHIFRAUD_BENIGN_LABELS:
                    out.append({
                        "text": text[:1024], "is_scam": 0, "risk": 1,
                        "category": "benign", "language": "zh", "source": "chifraud",
                    })
                elif label in CHIFRAUD_SCAM_MAP:
                    out.append({
                        "text": text[:1024], "is_scam": 1, "risk": 4,
                        "category": CHIFRAUD_SCAM_MAP[label],
                        "language": "zh", "source": "chifraud",
                    })
    return out


def load_teleantifraud() -> list[dict]:
    """TeleAntiFraud-28k：ASR 转写无标点保留原样；二元映射。"""
    out = []
    base = RAW / "teleantifraud"
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".json", ".jsonl"):
            continue
        for item in parse_jsonish(p):
            text = _first(item, TELE_TEXT_KEYS)
            label = _first(item, TELE_LABEL_KEYS)
            if not text or not label:
                continue
            is_normal = label.lower() in TELE_NORMAL_LABELS
            out.append({
                "text": text[:1024],
                "is_scam": 0 if is_normal else 1,
                "risk": 1 if is_normal else 4,
                "category": "benign" if is_normal else "spam_general",
                "language": "zh", "source": "teleantifraud",
            })
    return out


def load_phishing_email() -> list[dict]:
    """Phishing Email Dataset：仅保留合法邮件作英文 benign 补充。"""
    out = []
    base = RAW / "phishing_email"
    if not base.exists():
        return out
    for p in sorted(base.rglob("*.csv")):
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                text = (row.get("text") or "").strip()
                label = (row.get("label") or "").strip()
                if text and "safe" in label.lower():
                    out.append({
                        "text": text[:1024], "is_scam": 0, "risk": 1,
                        "category": "benign", "language": "en",
                        "source": "phishing_email",
                    })
    return out
```

- [ ] **Step 5: build_dataset.py 的 load_fbs_sms 去空格**

`scripts/build_dataset.py` import 区（`from schemas.scam_categories import ...` 之后）加：

```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_loaders import despace_cjk
```

`load_fbs_sms` 中把 `text = line.strip()` 替换为：

```python
                text = despace_cjk(line.strip())
```

- [ ] **Step 6: 运行测试 + 全量回归**

Run: `pytest tests/test_dataset_loaders.py -q && pytest -q`
Expected: PASS（基线保持）

- [ ] **Step 7: 真实数据冒烟**

Run: `python -c "
import sys; sys.path.insert(0, 'scripts')
import dataset_loaders as d
print('ccl', len(d.load_ccl2023()))
print('chifraud', len(d.load_chifraud()))
print('tele', len(d.load_teleantifraud()))
print('phish', len(d.load_phishing_email()))"
Expected: 打印各源真实行数（ccl 万级、tele 万级）。CCL 抛「未映射标签」→ 回 Task 4 Step 5 补映射。

- [ ] **Step 8: Commit**

```bash
git add scripts/dataset_loaders.py scripts/build_dataset.py tests/test_dataset_loaders.py tests/fixtures/
git commit -m "feat(data): v2 source loaders (CCL/ChiFraud/Tele/PhishingEmail) + FBS de-space"
```

---

### Task 6: 显式配比模块（scripts/dataset_mix.py）

**Files:**
- Create: `scripts/dataset_mix.py`
- Create: `tests/test_dataset_mix.py`

**Interfaces:**
- Consumes: 训练行 shape（Task 5）
- Produces: `compose_train(rows: list[dict], eval_ids: set[str], train_target: int = 45000, seed: int = 42) -> tuple[list[dict], dict]`；常量 `HARDNEG_SOURCES / GEN_SOURCES / FGRC_TIP_SOURCES: frozenset[str]`；共享 `stable_id(text, source) -> str`（Task 10-12 复用，公式必须与 build_dataset.stable_id 一致）

- [ ] **Step 1: 写失败测试 tests/test_dataset_mix.py**

```python
"""compose_train ratio invariants (synthetic rows, no real data)."""
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from dataset_mix import compose_train, HARDNEG_SOURCES, stable_id  # noqa: E402


def mk(text, is_scam, source, category=None):
    return {
        "id": f"{source}-{abs(hash(text)) % 10**8}",
        "text": text, "is_scam": is_scam,
        "risk": 4 if is_scam else 1,
        "category": category or ("benign" if not is_scam else "spam_general"),
        "language": "zh", "source": source,
    }


def test_caps_and_rebate_floor():
    rows = []
    rows += [mk(f"ccl-r{i}", 1, "ccl2023", "rebate_scam") for i in range(4000)]
    rows += [mk(f"ccl-nr{i}", 1, "ccl2023", "phishing") for i in range(500)]
    rows += [mk(f"inv{i}", 1, "fgrc_scd_sms", "investment_scam") for i in range(4000)]
    rows += [mk(f"ph{i}", 1, "fbs_sms", "phishing") for i in range(200)]
    rows += [mk(f"tele{i}", 1, "teleantifraud", "spam_general") for i in range(3000)]
    rows += [mk(f"hn{i}", 0, "hn_financial_notice") for i in range(600)]
    rows += [mk(f"nb{i}", 0, "nb_natural_benign") for i in range(200)]
    rows += [mk(f"tip{i}", 0, "fgrc_scd_sms") for i in range(3000)]
    rows += [mk(f"nat{i}", 0, "ealvaradob") for i in range(3000)]

    train, stats = compose_train(rows, eval_ids=set(), train_target=2000, seed=42)

    pos = [r for r in train if r["is_scam"] == 1]
    neg = [r for r in train if r["is_scam"] == 0]
    assert len(train) == 2000
    assert stats["n_pos"] == 900 and stats["n_neg"] == 1100

    # rebate_scam ≥25% 正样本
    assert sum(1 for r in pos if r["category"] == "rebate_scam") >= 0.25 * 900 - 1
    # CCL 非刷单返利 ≤20% 正样本（rebate 豁免——spec 勘误）
    ccl_nr = [r for r in pos if r["source"] == "ccl2023"
              and r["category"] != "rebate_scam"]
    assert len(ccl_nr) <= 0.20 * 900 + 1
    # Tele ≤10% 正样本
    assert sum(1 for r in pos if r["source"] == "teleantifraud") <= 0.10 * 900 + 1
    # 单类别 ≤30% 正样本
    for c, k in Counter(r["category"] for r in pos).items():
        assert k <= 0.30 * 900 + 1, f"{c} over cap"
    # FGRC 提示类 10–15% 负样本
    tips = [r for r in neg if r["source"] in ("fgrc_scd_sms", "fgrc_scd_dialog")]
    assert 0.10 * 1100 - 1 <= len(tips) <= 0.15 * 1100 + 1
    # 难负全保留（不超过负样本预算时）
    assert sum(1 for r in neg if r["source"] in HARDNEG_SOURCES) == 600


def test_rebate_floor_error_when_insufficient():
    rows = [mk(f"p{i}", 1, "fbs_sms", "phishing") for i in range(100)]
    rows += [mk(f"n{i}", 0, "ealvaradob") for i in range(100)]
    with pytest.raises(RuntimeError, match="rebate_scam"):
        compose_train(rows, eval_ids=set(), train_target=1000, seed=42)


def test_eval_ids_excluded():
    rows = [mk(f"hn{i}", 0, "hn_financial_notice") for i in range(50)]
    rows += [mk(f"p{i}", 1, "ccl2023", "rebate_scam") for i in range(100)]
    leak = rows[0]["id"]
    train, _ = compose_train(rows, eval_ids={leak}, train_target=100, seed=42)
    assert all(r["id"] != leak for r in train)


def test_generated_cap_enforced():
    rows = [mk(f"hn{i}", 0, "hn_financial_notice") for i in range(2000)]
    rows += [mk(f"p{i}", 1, "ccl2023", "rebate_scam") for i in range(1000)]
    with pytest.raises(RuntimeError, match="生成样本"):
        compose_train(rows, eval_ids=set(), train_target=2000, seed=42)


def test_stable_id_matches_build_dataset():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bd", Path(__file__).resolve().parent.parent / "scripts" / "build_dataset.py")
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    assert stable_id("样本", "hn_x") == bd.stable_id("样本", "hn_x")
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_dataset_mix.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 scripts/dataset_mix.py**

```python
"""V2 训练集显式配比（FP 压降核心）。设计见 2026-09-30-laya-fp-reduction-design.md §7.1。

compose_train 取代 v1 的 per-source balance_train：难负/简单负样本优先全保留，
FGRC 提示类压到 10–15%，CCL/Tele/单类别按上限裁剪。
"""
import hashlib
import random
from collections import Counter, defaultdict

TRAIN_TARGET = 45_000
POS_FRACTION = 0.45
CCL_NONREBATE_CAP = 0.20   # CCL 非刷单返利 ≤20% 正样本（rebate 豁免——spec 勘误）
TELE_POS_CAP = 0.10        # TeleAntiFraud ≤10% 正样本
PER_CATEGORY_CAP = 0.30    # 任何单类别 ≤30% 正样本
REBATE_FLOOR = 0.25        # rebate_scam ≥25% 正样本
CHIFRAUD_BENIGN_CAP = 0.10
TIPS_FRACTION = 0.125      # FGRC 提示类目标（区间 0.10–0.15 中点）
GEN_CAP = 0.30             # 生成样本 ≤30% 训练总量

HARDNEG_SOURCES = frozenset({
    "hn_antifraud_propaganda", "hn_financial_notice", "hn_ecommerce_logistics",
    "hn_job_ad", "hn_promotion", "hn_gov_notice", "hn_personal_social",
    "hn_traffic_funnel", "contrastive_pair",
})
GEN_SOURCES = frozenset({
    "hn_financial_notice", "hn_ecommerce_logistics", "hn_job_ad",
    "hn_promotion", "hn_gov_notice", "hn_personal_social",
    "hn_traffic_funnel", "nb_natural_benign", "contrastive_pair",
})
FGRC_TIP_SOURCES = frozenset({"fgrc_scd_sms", "fgrc_scd_dialog"})


def stable_id(text: str, source: str) -> str:
    """与 build_dataset.stable_id 公式完全一致（test_stable_id_matches 强制）。"""
    h = hashlib.sha1(f"{source}::{text}".encode("utf-8")).hexdigest()[:12]
    return f"{source}-{h}"


def _sample(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k <= 0:
        return []
    if len(items) <= k:
        return items[:]
    return rng.sample(items, k)


def compose_train(rows: list[dict], eval_ids: set[str],
                  train_target: int = TRAIN_TARGET, seed: int = 42,
                  ) -> tuple[list[dict], dict]:
    """按配比组装训练集。返回 (train_rows, stats)。违反硬约束即抛错。"""
    rng = random.Random(seed)
    pool = [r for r in rows if r.get("id") not in eval_ids]
    pos_all = [r for r in pool if r["is_scam"] == 1]
    neg_all = [r for r in pool if r["is_scam"] == 0]

    n_pos = round(train_target * POS_FRACTION)
    n_neg = train_target - n_pos

    # —— 正样本（预算瀑布）——
    rebate_pool = [r for r in pos_all if r["category"] == "rebate_scam"]
    need_rebate = round(n_pos * REBATE_FLOOR)
    if len(rebate_pool) < need_rebate:
        raise RuntimeError(
            f"rebate_scam 仅 {len(rebate_pool)} 条 < 下限 {need_rebate}"
            f" —— CCL2023 未就绪或被过度去重")
    pos_taken = _sample(rebate_pool, need_rebate, rng)
    taken_ids = {r["id"] for r in pos_taken}
    budget = n_pos - len(pos_taken)

    ccl_rest = [r for r in pos_all if r["source"] == "ccl2023"
                and r["category"] != "rebate_scam" and r["id"] not in taken_ids]
    picked = _sample(ccl_rest, min(round(n_pos * CCL_NONREBATE_CAP), budget), rng)
    pos_taken += picked
    taken_ids |= {r["id"] for r in picked}
    budget -= len(picked)

    tele_rest = [r for r in pos_all if r["source"] == "teleantifraud"
                 and r["id"] not in taken_ids]
    picked = _sample(tele_rest, min(round(n_pos * TELE_POS_CAP), budget), rng)
    pos_taken += picked
    taken_ids |= {r["id"] for r in picked}
    budget -= len(picked)

    others = [r for r in pos_all if r["id"] not in taken_ids]
    by_cat = defaultdict(list)
    for r in others:
        by_cat[r["category"]].append(r)
    cat_cap = round(n_pos * PER_CATEGORY_CAP)
    for cat in sorted(by_cat):
        if budget <= 0:
            break
        picked = _sample(by_cat[cat], min(cat_cap, budget), rng)
        pos_taken += picked
        taken_ids |= {r["id"] for r in picked}
        budget -= len(picked)

    if budget > 0:  # 比例填充剩余（按 source 占比，预算守卫）
        leftovers = [r for r in pos_all if r["id"] not in taken_ids]
        if leftovers:
            src_w = Counter(r["source"] for r in leftovers)
            by_src = defaultdict(list)
            for r in leftovers:
                by_src[r["source"]].append(r)
            for src in sorted(by_src):
                if budget <= 0:
                    break
                share = min(round(budget * src_w[src] / len(leftovers)) + 1, budget)
                picked = _sample(by_src[src], share, rng)
                pos_taken += picked
                taken_ids |= {r["id"] for r in picked}
                budget -= len(picked)
    if budget > 0:  # 最后兜底
        rest_pos = [r for r in pos_all if r["id"] not in taken_ids]
        picked = _sample(rest_pos, min(len(rest_pos), budget), rng)
        pos_taken += picked
        budget -= len(picked)

    # —— 负样本（贪婪瀑布：难负 > 简单 > 提示类 > ChiFraud > 自然 benign）——
    hard = [r for r in neg_all if r["source"] in HARDNEG_SOURCES]
    simple = [r for r in neg_all if r["source"] == "nb_natural_benign"]
    tips_pool = [r for r in neg_all
                 if r["source"] in FGRC_TIP_SOURCES and r["category"] == "benign"]
    natural = [r for r in neg_all
               if r["source"] not in HARDNEG_SOURCES
               and r["source"] != "nb_natural_benign"
               and r["source"] not in FGRC_TIP_SOURCES]

    neg_taken: list[dict] = []
    nbudget = n_neg
    picked = _sample(hard, min(len(hard), nbudget), rng)
    neg_taken += picked
    nbudget -= len(picked)
    hard_kept = len(picked)
    picked = _sample(simple, min(len(simple), nbudget), rng)
    neg_taken += picked
    nbudget -= len(picked)
    simple_kept = len(picked)
    tips_quota = min(round(n_neg * TIPS_FRACTION), nbudget)
    picked = _sample(tips_pool, tips_quota, rng)
    neg_taken += picked
    nbudget -= len(picked)
    tips_kept = len(picked)
    chifraud = [r for r in natural if r["source"] == "chifraud"]
    natural_other = [r for r in natural if r["source"] != "chifraud"]
    picked = _sample(chifraud, min(round(n_neg * CHIFRAUD_BENIGN_CAP), nbudget), rng)
    neg_taken += picked
    nbudget -= len(picked)
    picked = _sample(natural_other, min(len(natural_other), nbudget), rng)
    neg_taken += picked
    nbudget -= len(picked)

    train = pos_taken + neg_taken
    rng.shuffle(train)

    # —— 生成样本上限 ——
    gen_n = sum(1 for r in train if r["source"] in GEN_SOURCES)
    if gen_n > GEN_CAP * len(train):
        raise RuntimeError(
            f"生成样本 {gen_n}/{len(train)} ({gen_n / len(train):.1%}) "
            f"超过 {GEN_CAP:.0%} 上限 —— 降低生成体裁目标量或提高真实数据配额")

    stats = {
        "n": len(train), "n_pos": len(pos_taken), "n_neg": len(neg_taken),
        "hardneg_kept": hard_kept, "simple_kept": simple_kept,
        "tips_kept": tips_kept,
        "gen_share": gen_n / len(train) if train else 0.0,
        "pos_unfilled": budget, "neg_unfilled": nbudget,
        "pos_by_category": dict(Counter(r["category"] for r in pos_taken)),
        "neg_by_source": dict(Counter(r["source"] for r in neg_taken)),
    }
    return train, stats
```

- [ ] **Step 4: 运行测试**

Run: `pytest tests/test_dataset_mix.py -q`
Expected: PASS（5 个测试全部通过）

- [ ] **Step 5: Commit**

```bash
git add scripts/dataset_mix.py tests/test_dataset_mix.py
git commit -m "feat(data): explicit v2 train composition with caps/floors + invariant tests"
```

---

### Task 7: build_dataset.py v2 集成

**Files:**
- Modify: `scripts/build_dataset.py`（main 流程）

**Interfaces:**
- Consumes: `dataset_loaders.load_*`（Task 5）、`dataset_mix.compose_train`（Task 6）、`datasets/hard_negatives/train_pool.jsonl` + `contrastive_pairs.jsonl`（Task 11/12 产出，缺失时警告并以空集继续）、`datasets/hardneg_eval.jsonl` 的 id（Task 12）
- Produces: `datasets/training/{train,val,test}.jsonl`（v2）；stdout 输出 mix stats

- [ ] **Step 1: 扩展 loader**

`main()` 中 loader 元组替换为（`from dataset_loaders import ...` 在 `from dataset_mix import ...` 旁新增 import，置于现有 `from schemas.scam_categories import ...` 之后）：

```python
    from dataset_loaders import (
        load_ccl2023, load_chifraud, load_teleantifraud, load_phishing_email,
    )
    for loader in (
        load_fgrc_scd_sms,
        load_fgrc_scd_dialog,
        load_scamshield,
        load_ealvaradob,
        load_fbs_sms,
        load_uc_irvine,
        load_synthetic_extra,
        load_ccl2023,
        load_chifraud,
        load_teleantifraud,
        load_phishing_email,
    ):
```

- [ ] **Step 2: 替换组装逻辑**

把 main() 中从 `train, val, test = split_dataset(real)` 到 `print(f"Train after balance: {len(train):,}")` 的整段，替换为：

```python
    # —— v2: 难负样本（train-only，不进 val/test）——
    hardneg_rows: list[dict] = []
    for hn_path in (Path("datasets/hard_negatives/train_pool.jsonl"),
                    Path("datasets/hard_negatives/contrastive_pairs.jsonl")):
        if hn_path.exists():
            with hn_path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        hardneg_rows.append(json.loads(line))
            print(f"  ✓ hardneg {hn_path.name}: cumulative {len(hardneg_rows):,}")
        else:
            print(f"  ! hardneg 缺失: {hn_path}（先跑 Task 10-12）")

    hardneg_eval_ids: set[str] = set()
    _eval_path = Path("datasets/hardneg_eval.jsonl")
    if _eval_path.exists():
        with _eval_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    hardneg_eval_ids.add(json.loads(line)["id"])
        print(f"  ✓ hardneg_eval ids: {len(hardneg_eval_ids):,}")

    for r in cleaned:
        r["id"] = stable_id(r["text"], r["source"])
    for r in hardneg_rows:
        r.setdefault("id", stable_id(r["text"], r["source"]))

    from dataset_mix import compose_train
    train, mix_stats = compose_train(real + synth + hardneg_rows,
                                     eval_ids=hardneg_eval_ids)
    print("v2 mix stats:")
    print(json.dumps(mix_stats, ensure_ascii=False, indent=2))

    _, val, test = split_dataset(real)
    for r in val + test:
        r["id"] = stable_id(r["text"], r["source"])
    print(f"Split: train={len(train):,} val={len(val):,} test={len(test):,}")
```

同时**删除** v1 的这段（id 重复赋值会按 build_dataset 公式覆盖难负 id，破坏与 eval 集的不相交保证）：

```python
    for split, rows in [("train", train), ("val", val), ("test", test)]:
        for r in rows:
            r["id"] = stable_id(r["text"], r["source"])
```

**删除 `balance_train` 函数**（用户裁定 2026-09-30：v2 后无调用者、测试不引用；评审规则视死代码为缺陷，git 历史可找回）。

- [ ] **Step 3: 冒烟构建**

Run: `source .venv/bin/activate && python scripts/build_dataset.py 2>&1 | tail -40`
Expected: 各 loader 行数（ccl 万级、tele 万级）；hardneg 缺失警告（正常，Task 10-12 未跑）；mix stats 中 `n=45000`、`pos_by_category["rebate_scam"]/n_pos ≥ 0.25`、`gen_share` 很低；val/test 万级。此步会覆盖 `datasets/training/`（gitignored，v1 产物不再保留——audit 用 v2 文件同样可行）。

- [ ] **Step 4: 全量回归 + Commit**

Run: `pytest -q`
Expected: PASS（基线保持）

```bash
git add scripts/build_dataset.py
git commit -m "feat(data): build_dataset v2 — new loaders + explicit mix composition"
```

---

### Task 8: 评测字段口径统一（category）

**Files:**
- Modify: `src/eval.py`
- Create: `scripts/migrate_eval_fields.py`
- Create: `tests/test_eval_fields.py`

**Interfaces:**
- Consumes: `normalize_label`（已有）
- Produces: `src.eval._expected_category(rec) -> str | None`（category 优先，expected_category 向后兼容）；`datasets/{eval,multi,single,public}.jsonl` 全部行补 `category` 字段

- [ ] **Step 1: 写失败测试 tests/test_eval_fields.py**

```python
"""_expected_category 口径统一测试（design §4.2）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.eval import _expected_category


def test_category_field_wins():
    assert _expected_category(
        {"category": "benign", "expected_category": "spam"}) == "benign"


def test_expected_category_fallback():
    assert _expected_category({"expected_category": "spam"}) == "spam_general"


def test_none_when_absent():
    assert _expected_category({"state": "x"}) is None


def test_normalize_applied():
    assert _expected_category({"category": "ham"}) == "benign"
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_eval_fields.py -q`
Expected: FAIL（`_expected_category` 不存在）

- [ ] **Step 3: 修改 src/eval.py**

在 `_thresholded_is_scam` 之后新增：

```python
def _expected_category(rec: dict) -> str | None:
    """v2 统一口径：category 为准，expected_category 向后兼容。"""
    raw = rec.get("category") or rec.get("expected_category")
    if raw is None:
        return None
    return normalize_label(raw)
```

`run_evaluation` 中替换：

```python
            expected_cat = rec.get("expected_category")
            if expected_cat is not None:
                expected_cat = normalize_label(expected_cat)
```

为：

```python
            expected_cat = _expected_category(rec)
```

- [ ] **Step 4: 写并运行迁移脚本 scripts/migrate_eval_fields.py**

```python
"""统一评测字段口径：为 datasets/*.jsonl 补 category 字段（design §4.2）。"""
import json
from pathlib import Path

FILES = [Path("datasets/eval.jsonl"), Path("datasets/multi.jsonl"),
         Path("datasets/single.jsonl"), Path("datasets/public.jsonl")]

for path in FILES:
    if not path.exists():
        print(f"! 缺失 {path}")
        continue
    rows = [json.loads(l) for l in path.open() if l.strip()]
    changed = 0
    for rec in rows:
        if "category" not in rec:
            derived = rec.get("expected_category")
            if derived is None:
                risk = rec.get("expected_risk")
                derived = ("spam_general"
                           if (risk is not None and risk >= 4) else "benign")
            rec["category"] = derived
            changed += 1
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"✓ {path}: {changed}/{len(rows)} 补 category")
```

Run: `python scripts/migrate_eval_fields.py`
Expected: 四个文件全部 ✓；public.jsonl 400/400（全部缺 category）。

- [ ] **Step 5: 测试 + 全量回归 + Commit**

Run: `pytest tests/test_eval_fields.py -q && pytest -q`
Expected: PASS（基线保持）

```bash
git add src/eval.py scripts/migrate_eval_fields.py tests/test_eval_fields.py datasets/eval.jsonl datasets/multi.jsonl datasets/single.jsonl datasets/public.jsonl
git commit -m "feat(eval): unify category field across eval sets + backward-compat reader"
```

---

### Task 9: FGRC 标签噪声审计（audit_fgrc_labels.py）

**Files:**
- Create: `scripts/audit_fgrc_labels.py`
- Output: `reports/fgrc_audit.md`（tracked）

**Interfaces:**
- Consumes: `OnnxLayaClient`（v1 模型 `models/laya-onnx-multilingual-finetuned`）、`datasets/training/train.jsonl`（Task 7 产出，含 fgrc 行）
- Produces: `reports/fgrc_audit.md` —— 分歧样本清单（label=scam&pred<0.5 与 label=benign&pred≥0.5 各 Top 100）

- [ ] **Step 1: 写 scripts/audit_fgrc_labels.py**

```python
"""用 v1 ONNX 模型审计 FGRC 标签噪声（design §4.2：抽样发现错配，二元用途可控）。

用法：PYTHONPATH=. python scripts/audit_fgrc_labels.py [--sample 2000]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.laya_onnx import OnnxLayaClient

MODEL_DIR = "models/laya-onnx-multilingual-finetuned"  # v1
TRAIN = "datasets/training/train.jsonl"
OUT = "reports/fgrc_audit.md"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    args = ap.parse_args()

    if not Path(args.model_dir, "model.onnx").exists():
        print(f"✗ 模型缺失: {args.model_dir}/model.onnx")
        return 2

    rows = []
    with open(TRAIN) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["source"].startswith("fgrc"):
                    rows.append(r)
    rng = random.Random(42)
    rng.shuffle(rows)
    rows = rows[: args.sample]
    if not rows:
        print(f"✗ {TRAIN} 无 fgrc 行（先跑 Task 7）")
        return 2

    client = OnnxLayaClient(args.model_dir)
    schema = json.loads(Path("schemas/scam.json").read_text())

    noise_flag, hard_fp = [], []  # scam→benign 分歧 / benign→scam 分歧
    for i, r in enumerate(rows):
        pred = client.predict(r["text"], schema)
        noul = pred["answers"]["is_scam"]["noul"]
        if r["is_scam"] == 1 and noul < 0.5:
            noise_flag.append((r, noul))
        elif r["is_scam"] == 0 and noul >= 0.5:
            hard_fp.append((r, noul))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}")

    lines = [
        "# FGRC 标签噪声审计（v1 模型 vs 训练标签）",
        "",
        f"- 样本: {len(rows)} 条 fgrc 行（seed=42）",
        f"- label=scam & pred<0.5（疑似标签噪声）: {len(noise_flag)}",
        f"- label=benign & pred≥0.5（疑似难负/误报源）: {len(hard_fp)}",
        "",
        "## 疑似标签噪声 Top 100（模型判 benign，标签 scam）",
        "",
        "| noul | label category | text |",
        "|---|---|---|",
    ]
    for r, noul in sorted(noise_flag, key=lambda x: x[1])[:100]:
        preview = r["text"][:80].replace("|", "\\|")
        lines.append(f"| {noul:.2f} | {r['category']} | {preview} |")
    lines += ["", "## 疑似难负 Top 100（模型判 scam，标签 benign）", "",
              "| noul | text |", "|---|---|"]
    for r, noul in sorted(hard_fp, key=lambda x: -x[1])[:100]:
        preview = r["text"][:80].replace("|", "\\|")
        lines.append(f"| {noul:.2f} | {preview} |")
    Path(OUT).write_text("\n".join(lines) + "\n")
    print(f"✓ {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 运行（模型门控，~2000 次 × 125ms ≈ 5 分钟）**

Run: `PYTHONPATH=. python scripts/audit_fgrc_labels.py`
Expected: 生成 `reports/fgrc_audit.md`，两个分歧计数打印。

- [ ] **Step 3: 人工检视（checkpoint）**

打开 `reports/fgrc_audit.md` 抽查各 10 条：噪声样本是「标签错」还是「模型弱」？难负样本是否正是报告预言的提示句式误报源？**结论只记录，不改代码**（FGRC 维持二元用途，design 已定）。

- [ ] **Step 4: Commit**

```bash
git add scripts/audit_fgrc_labels.py reports/fgrc_audit.md
git commit -m "feat(data): FGRC label-noise audit against v1 model"
```

---

### Task 10: 反诈宣传抓取 + 种子语料（体裁 1）

**Files:**
- Create: `datasets/hard_negatives/seed_propaganda.jsonl`（20 条人工种子，tracked）
- Create: `scripts/scrape_antifraud.py`
- Output: `datasets/hard_negatives/hn_antifraud_propaganda.jsonl`（tracked）

**Interfaces:**
- Consumes: `dataset_mix.stable_id`（Task 6）
- Produces: `datasets/hard_negatives/hn_antifraud_propaganda.jsonl`（训练格式行，source=`hn_antifraud_propaganda`）；中间产物 `datasets/raw/propaganda/*.txt`（gitignored）

- [ ] **Step 1: 写种子语料 datasets/hard_negatives/seed_propaganda.jsonl**

```jsonl
{"text": "【公安部】凡是网上刷单返利要求垫资的都是诈骗，请勿参与。"}
{"text": "【反诈中心】凡是自称公检法要求转账到安全账户的都是诈骗。"}
{"text": "【12381】国家反诈中心提醒：您可能正在遭遇电信网络诈骗，请提高警惕，切勿向陌生账户转账汇款。"}
{"text": "【96110】反诈专线来电请接听，民警正在对您进行预警劝阻。"}
{"text": "【XX银行】尊敬的客户：我行工作人员绝不会以任何理由索要您的短信验证码，请妥善保管。"}
{"text": "【反诈中心】凡是网络贷款先交手续费的都是诈骗，正规贷款不放款前不收费。"}
{"text": "【反诈中心】凡是自称客服主动退款并要求下载会议软件的都是诈骗。"}
{"text": "【XX街道办】防范电信诈骗十个凡是：凡是自称公检法要求汇款的，都是诈骗。"}
{"text": "【XX公安】凡是索要银行卡密码、验证码、短信验证码的都是诈骗，请立即挂断。"}
{"text": "【工信部】12381涉诈预警劝阻短信已启用，收到请核对来源，谨防受骗。"}
{"text": "【XX派出所】警方提示：网络交友诱导投资就是杀猪盘诈骗，请立即止损并拨打110。"}
{"text": "【反诈中心】凡是刷单做任务要您垫付资金的都是诈骗。"}
{"text": "【反诈中心】凡是冒充领导熟人要求帮忙转账的，务必当面或电话核实身份。"}
{"text": "【XX市场监管局】警惕虚假投资理财平台，承诺保本高息的就是诈骗。"}
{"text": "【96110】您是否正在向陌生账户转账？请暂停操作，这是典型的诈骗手法。"}
{"text": "【XX公安】不要向任何陌生人透露短信验证码，验证码就是钱。"}
{"text": "【反诈中心】凡是虚假购物平台低价促销要求先付款再发货的都是诈骗。"}
{"text": "【XX社区】谨防冒充电商客服退款诈骗：官方客服不会索要验证码，不会要求屏幕共享。"}
{"text": "【反诈中心】谨记三不一多：未知链接不点击，陌生来电不轻信，个人信息不透露，转账汇款多核实。"}
{"text": "【XX公安】凡是网上办理证件先交定金的都是诈骗，正规机构不收取任何前期费用。"}
```

- [ ] **Step 2: 写 scripts/scrape_antifraud.py**

```python
"""抓取公安/政府/运营商反诈宣传文本（难负样本体裁 1）。

URL 来自 dataset_research.md 引文；单 URL 失败优雅跳过（种子语料兜底）。
用法：python scripts/scrape_antifraud.py
"""
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_mix import stable_id

SEED = Path("datasets/hard_negatives/seed_propaganda.jsonl")
RAW_PAGES = Path("datasets/raw/propaganda")
OUT = Path("datasets/hard_negatives/hn_antifraud_propaganda.jsonl")
SOURCE = "hn_antifraud_propaganda"

URLS = [
    # 晋城公安反诈语录「十个凡是」（report 【23】）
    "http://ywtb.gaj.jcgov.gov.cn/site/public/showinfo.aspx?id=2026030609245557470105",
    # 平坝区致群众一封信（report 【29】）
    "https://www.pingba.gov.cn/xzjd/tlz/zfxxgk_5668073/fdzdgknr_5668076/xxgk/202607/t20260701_90572939.html",
    # 临汾移动 12381/96110 误区澄清（report 【24】）
    "https://lf.sxgov.cn/content/2026-08/25/content_13676534.htm",
    # 公安部 2026 宣传手册报道（report 【27】）
    "https://news.ycwb.com/ikimvkmtjj/content_54169914.htm",
]

KEYWORD = re.compile(r"反诈|诈骗|预警|劝阻|96110|12381|凡是")


class TextExtract(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def fetch(url: str) -> str:
    r = subprocess.run(["curl", "-sL", "-m", "60", url],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"curl exit {r.returncode}")
    return r.stdout


def extract_texts(html: str) -> list[str]:
    p = TextExtract()
    p.feed(html)
    return [t for t in p.parts if 15 <= len(t) <= 200 and KEYWORD.search(t)]


def main() -> int:
    RAW_PAGES.mkdir(parents=True, exist_ok=True)
    texts: list[str] = []
    with SEED.open() as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    print(f"  ✓ seed: {len(texts)}")

    for i, url in enumerate(URLS):
        stem = f"page{i}"
        page = RAW_PAGES / f"{stem}.txt"
        try:
            html = fetch(url)
            page.write_text(html)
            got = extract_texts(html)
            texts.extend(got)
            print(f"  ✓ {url[:60]}... -> {len(got)} 段")
        except Exception as e:
            print(f"  ✗ {url[:60]}... 跳过: {e}")

    seen, rows = set(), []
    for t in texts:
        if t in seen:
            continue
        seen.add(t)
        rows.append({
            "id": stable_id(t, SOURCE), "text": t, "is_scam": 0, "risk": 1,
            "category": "benign", "language": "zh", "source": SOURCE,
        })
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {OUT}: {len(rows)} 条（seed 20 + 抓取）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: 运行 + 人工检视（checkpoint）**

Run: `python scripts/scrape_antifraud.py && head -40 datasets/hard_negatives/hn_antifraud_propaganda.jsonl`
Expected: seed 20 条 + 抓取段落数（政府站可达性不定，失败正常）。人工删除混入的导航/无关段落：**从 `datasets/raw/propaganda/pageN.txt` 源文件删或改关键词过滤，重跑脚本**（脚本是确定性的，直接改产物会在下次运行被覆盖）。

**数量下限**：体裁 1 目标 800-1000（design §6.1）。若 seed+抓取 < 115 条（Task 12 会警告）：人工扩充 —— 加大 URL 清单（搜索「反诈宣传 十个凡是 预警短信」类政务页）或直接在 `seed_propaganda.jsonl` 追加人工撰写条目至 ≥115（每条须为真实宣传话术变体，保持官方宣传口吻）。

- [ ] **Step 4: Commit**

```bash
git add datasets/hard_negatives/seed_propaganda.jsonl datasets/hard_negatives/hn_antifraud_propaganda.jsonl scripts/scrape_antifraud.py
git commit -m "feat(data): anti-fraud propaganda hard negatives (curated seed + scrape)"
```

---

### Task 11: arkcli 多模型难负样本生成（体裁 2-8 + 简单负样本 + 对照对）

**Files:**
- Create: `scripts/gen_models.json`（探测后填写，tracked）
- Create: `scripts/generate_negatives.py`
- Output: `datasets/hard_negatives/hn_{financial_notice,ecommerce_logistics,job_ad,promotion,gov_notice,personal_social,traffic_funnel}.jsonl`、`datasets/hard_negatives/nb_natural_benign.jsonl`、`datasets/hard_negatives/contrastive_pairs.jsonl`（全部 tracked）

**Interfaces:**
- Consumes: `dataset_mix.stable_id`（Task 6）；arkcli CLI（火山方舟，多模型）
- Produces: 训练格式难负样本行，source=体裁名；每行附 `"generator": "<model-id>"`（compose 忽略此字段，仅溯源）

- [ ] **Step 1: 探测 arkcli 并写 scripts/gen_models.json**

Run: `arkcli chat --help 2>&1 | head -40` 与 `arkcli models list 2>&1 | head -60`（子命令名以 `arkcli --help` 实际输出为准）
从可用文本模型中选 **≥3 个不同家族**的模型 ID，把真实调用形态写入：

```json
{
  "models": ["<model-id-1>", "<model-id-2>", "<model-id-3>"],
  "argv_template": ["arkcli", "chat", "--model", "{model}", "--prompt", "{prompt}"]
}
```

`argv_template` 的占位符 `{model}`/`{prompt}` 由脚本替换；以 `--help` 显示的真实参数名为准修正模板。

- [ ] **Step 2: 写 scripts/generate_negatives.py**

```python
"""arkcli 多模型生成难负样本（design §6.1 八体裁 + §6.2 对照对）。

多模型轮转避免单一文体；每行记录 generator。QC：长度 10-160、含 CJK、全库去重。
用法：
  python scripts/generate_negatives.py --genre hn_financial_notice --limit 25  # pilot
  python scripts/generate_negatives.py                                          # 全量
  python scripts/generate_negatives.py --genre hn_promotion --append           # 补量
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_mix import stable_id

DATA = Path("datasets/hard_negatives")
CFG = Path("scripts/gen_models.json")
BATCH = 25

PROMPT_TMPL = """你在为中文诈骗检测模型构造「难负样本」：看起来像诈骗、实际完全合法、模型容易误报的文本。

体裁：{desc}
内容要素：{elements}
严禁出现：{must_not}

要求：
1. 每行输出一条 JSON：{{"text": "..."}}，共 {n} 条，不要编号、不要解释、不要 markdown 代码块
2. 文本 20-120 字，像真实中文短信，语气自然
3. 多样化：机构名/人名/长度/正式度/标点都要变化，禁止重复句式
4. 直接输出 JSONL，第一行开始就是 JSON"""

PAIR_TMPL = """对照对构造：生成 {pairs} 对短信，每对一合法一诈骗。

体裁：{desc}
合法版内容要素：{elements}
合法版严禁：{must_not}
两版结构、语气、署名、句式完全一致，唯一差异：诈骗版{scam_delta}

输出格式（每对两行，共 {pairs} 对，不要编号不要解释）：
{{"is_scam": 0, "text": "合法版内容"}}
{{"is_scam": 1, "text": "诈骗版内容"}}"""

GENRES = {
    "hn_financial_notice": {
        "target": 1200,
        "desc": "银行/券商/保险/支付机构发给客户的真实业务通知短信",
        "elements": "机构署名（如【XX银行】【XX证券】）、验证码、额度调整、账单出账、积分到期、密码重置成功",
        "must_not": "让用户把验证码告诉任何人、引导添加私人微信QQ好友、非官方域名链接、索要密码",
    },
    "hn_ecommerce_logistics": {
        "target": 1200,
        "desc": "电商平台/快递公司的正常通知短信",
        "elements": "订单状态、发货、签收、取件码、地址变更确认、退货进度",
        "must_not": "索费、引导点击非官方链接登录、加私人好友",
    },
    "hn_job_ad": {
        "target": 800,
        "desc": "正规公司招聘/兼职广告短信",
        "elements": "岗位名称、薪资范围、面试地点、公司名",
        "must_not": "预付费用、押金、境外高薪、刷单",
    },
    "hn_promotion": {
        "target": 1000,
        "desc": "商家促销/会员权益通知短信",
        "elements": "满减、折扣、积分兑换、会员日、优惠券到账",
        "must_not": "要求先转账再领奖、垫资、缴纳保证金",
    },
    "hn_gov_notice": {
        "target": 1200,
        "desc": "政务/公共机构通知短信",
        "elements": "社保、医保、公积金、ETC、违章提醒、学校家长通知、单位会议、天气预警",
        "must_not": "索要密码验证码、索费、引导转账",
    },
    "hn_personal_social": {
        "target": 1200,
        "desc": "家人朋友之间的日常短信",
        "elements": "问候、约饭、拼团、红包、代付提醒、到家叮嘱",
        "must_not": "任何机构署名、索费",
    },
    "hn_traffic_funnel": {
        "target": 1000,
        "desc": "合法引流/内容推广短信",
        "elements": "交友活动邀请、直播预告、内容更新提醒、公众号推广",
        "must_not": "索钱、投资引导、刷单",
    },
    "nb_natural_benign": {
        "target": 1500,
        "desc": "最普通的日常短信",
        "elements": "快递取件码、验证码、会议通知、家人问候、缴费提醒",
        "must_not": "任何诈骗结构或索费",
    },
}

CONTRASTIVE_GROUPS = [
    {"genre": "hn_financial_notice", "scam_category": "impersonation",
     "scam_delta": "以银行客服口吻要求提供验证码或点击钓鱼链接办理业务"},
    {"genre": "hn_ecommerce_logistics", "scam_category": "delivery_fraud",
     "scam_delta": "声称包裹丢失需点击链接填写银行卡信息理赔"},
    {"genre": "hn_job_ad", "scam_category": "job_scam",
     "scam_delta": "要求先缴报名费押金培训费"},
]
PAIRS_PER_GROUP = 100


def call_model(cfg: dict, model: str, prompt: str) -> str:
    argv = [a.replace("{model}", model).replace("{prompt}", prompt)
            for a in cfg["argv_template"]]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode}: {r.stderr[:200]}")
    return r.stdout


def parse_rows(out: str) -> list[dict]:
    rows = []
    for line in out.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("text"), str):
            rows.append(rec)
    return rows


def qc(text: str) -> bool:
    return 10 <= len(text) <= 160 and any("\u4e00" <= c <= "\u9fff" for c in text)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("a" if path.exists() else "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_existing_texts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["text"] for l in path.open() if l.strip()}


def gen_genre(cfg: dict, name: str, spec: dict, limit: int | None,
              seen: set[str]) -> int:
    out_path = DATA / f"{name}.jsonl"
    seen |= load_existing_texts(out_path)
    need = min(spec["target"], limit) if limit else spec["target"]
    fresh = 0
    call = 0
    while fresh < need and call < (need // BATCH) * 3 + 6:
        model = cfg["models"][call % len(cfg["models"])]
        prompt = PROMPT_TMPL.format(
            desc=spec["desc"], elements=spec["elements"],
            must_not=spec["must_not"], n=min(BATCH, need - fresh))
        try:
            out = call_model(cfg, model, prompt)
        except Exception as e:
            print(f"  ! {name} model {model} 失败，换下一个: {e}")
            call += 1
            time.sleep(2)
            continue
        rows = []
        for rec in parse_rows(out):
            t = rec["text"].strip()
            if qc(t) and t not in seen:
                seen.add(t)
                rows.append({
                    "id": stable_id(t, name), "text": t, "is_scam": 0,
                    "risk": 1, "category": "benign", "language": "zh",
                    "source": name, "generator": model,
                })
                fresh += 1
        write_rows(out_path, rows)
        call += 1
        print(f"  {name}: call {call} (+{len(rows)}, total fresh {fresh}/{need})")
    return fresh


def gen_contrastive(cfg: dict, seen: set[str]) -> int:
    out_path = DATA / "contrastive_pairs.jsonl"
    seen |= load_existing_texts(out_path)
    total = 0
    for g in CONTRASTIVE_GROUPS:
        spec = GENRES[g["genre"]]
        pairs_done = 0
        call = 0
        while pairs_done < PAIRS_PER_GROUP and call < PAIRS_PER_GROUP * 2 + 4:
            model = cfg["models"][call % len(cfg["models"])]
            prompt = PAIR_TMPL.format(
                pairs=min(BATCH, PAIRS_PER_GROUP - pairs_done),
                desc=spec["desc"], elements=spec["elements"],
                must_not=spec["must_not"], scam_delta=g["scam_delta"])
            try:
                out = call_model(cfg, model, prompt)
            except Exception as e:
                print(f"  ! contrastive {g['genre']} 失败: {e}")
                call += 1
                time.sleep(2)
                continue
            rows = []
            for rec in parse_rows(out):
                t = rec["text"].strip()
                is_scam = int(rec.get("is_scam", -1))
                if not qc(t) or t in seen or is_scam not in (0, 1):
                    continue
                seen.add(t)
                rows.append({
                    "id": stable_id(t, "contrastive_pair"), "text": t,
                    "is_scam": is_scam, "risk": 4 if is_scam else 1,
                    "category": g["scam_category"] if is_scam else "benign",
                    "language": "zh", "source": "contrastive_pair",
                    "generator": model,
                })
                if is_scam == 1:
                    pairs_done += 1
            write_rows(out_path, rows)
            total += len(rows)
            call += 1
            print(f"  contrastive {g['genre']}: call {call} (+{len(rows)})")
    return total


def main() -> int:
    if not CFG.exists():
        print("✗ scripts/gen_models.json 缺失 —— 先完成 Step 1 探测")
        return 2
    cfg = json.loads(CFG.read_text())
    if len(cfg.get("models", [])) < 3:
        print("✗ gen_models.json 需 ≥3 个模型（多模型分散文体）")
        return 2

    ap = argparse.ArgumentParser()
    ap.add_argument("--genre", help="只跑指定体裁（pilot 用）")
    ap.add_argument("--limit", type=int, help="每体裁上限（pilot 用）")
    args = ap.parse_args()

    seen: set[str] = set()
    targets = {k: v for k, v in GENRES.items()
               if not args.genre or k == args.genre}
    for name, spec in targets.items():
        got = gen_genre(cfg, name, spec, args.limit, seen)
        print(f"✓ {name}: {got} 条")
    if not args.genre:
        got = gen_contrastive(cfg, seen)
        print(f"✓ contrastive_pairs: {got} 行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Pilot（人工 checkpoint）**

Run: `python scripts/generate_negatives.py --genre hn_financial_notice --limit 25 && head -25 datasets/hard_negatives/hn_financial_notice.jsonl`
Expected: 25 条生成。**人工抽检 20 条**：是否像真实银行短信？是否混入索验证码引导（must_not 违例）？不合格 → 调 GENRES spec 措辞或换模型 ID 后删除该文件重试。

- [ ] **Step 4: 全量生成（后台，~460 次调用 ≈ 2-3 小时）**

Run: `nohup python scripts/generate_negatives.py > reports/gen_negatives.log 2>&1 &`
监控: `tail -f reports/gen_negatives.log`
Expected: 8 体裁各达 target×0.9 以上；contrastive 三组各 ~100 对。

- [ ] **Step 5: 数量与质量验证**

```bash
for f in datasets/hard_negatives/hn_*.jsonl datasets/hard_negatives/nb_natural_benign.jsonl datasets/hard_negatives/contrastive_pairs.jsonl; do echo "$f $(wc -l < $f)"; done
```
Expected: financial/ecommerce/gov/social ≥1080、promotion/funnel ≥900、job ≥720、nb ≥1350、contrastive ≥540（300 对×2 边缘±）；每体裁再抽 5 条人工过目。

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_negatives.py scripts/gen_models.json datasets/hard_negatives/ reports/gen_negatives.log
git commit -m "feat(data): multi-model generated hard negatives (8 genres + contrastive pairs)"
```

---

### Task 12: 难负评测集切分（build_hardneg_evalset.py）

**Files:**
- Create: `scripts/build_hardneg_evalset.py`
- Create: `tests/test_hardneg_evalset.py`
- Output: `datasets/hardneg_eval.jsonl`（评测格式，tracked）+ `datasets/hard_negatives/train_pool.jsonl`（训练格式，tracked）

**Interfaces:**
- Consumes: `datasets/hard_negatives/hn_*.jsonl` + `nb_natural_benign.jsonl`（Task 10/11，行含 `id`/`text`/`source`）；`contrastive_pairs.jsonl` **不参与切分**（train-only）
- Produces: `datasets/hardneg_eval.jsonl`（每体裁 75 条 held-out，评测格式）；`datasets/hard_negatives/train_pool.jsonl`（剩余全部，训练格式）；**id 不相交断言**

- [ ] **Step 1: 写失败测试 tests/test_hardneg_evalset.py**

```python
"""build_hardneg_evalset 切分不变量测试（fixtures，无真实数据）。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_hardneg_evalset as bhe  # noqa: E402


@pytest.fixture()
def genre_dir(tmp_path, monkeypatch):
    data = tmp_path / "hard_negatives"
    data.mkdir()
    for genre in ("hn_financial_notice", "nb_natural_benign"):
        with (data / f"{genre}.jsonl").open("w") as f:
            for i in range(100):
                f.write(json.dumps({
                    "id": f"{genre}-{i}", "text": f"{genre} 文本 {i}",
                    "is_scam": 0, "risk": 1, "category": "benign",
                    "language": "zh", "source": genre,
                }, ensure_ascii=False) + "\n")
    monkeypatch.setattr(bhe, "DATA", data)
    monkeypatch.setattr(bhe, "OUT_EVAL", tmp_path / "hardneg_eval.jsonl")
    monkeypatch.setattr(bhe, "OUT_TRAIN", data / "train_pool.jsonl")
    return tmp_path


def test_split_disjoint_and_counts(genre_dir):
    assert bhe.main() == 0
    eval_rows = [json.loads(l) for l in
                 (genre_dir / "hardneg_eval.jsonl").open() if l.strip()]
    train_rows = [json.loads(l) for l in
                  (genre_dir / "hard_negatives" / "train_pool.jsonl").open()
                  if l.strip()]
    assert len(eval_rows) == 150          # 75 × 2 体裁
    assert len(train_rows) == 50          # 25 × 2
    assert {r["id"] for r in eval_rows}.isdisjoint(
        {r["id"] for r in train_rows})
    r = eval_rows[0]
    assert r["type"] == "single" and r["expected_risk"] == 1
    assert r["category"] == "benign" and r["source"].startswith(("hn_", "nb_"))
    assert "state" in r
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_hardneg_evalset.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 scripts/build_hardneg_evalset.py**

```python
"""切分难负评测集（永不进训练）与训练池。design §7.2。

评测集：8 体裁 + 自然 benign 各 75 条（不足则全取并警告）。
输出：datasets/hardneg_eval.jsonl（评测格式）
     + datasets/hard_negatives/train_pool.jsonl（训练格式）。
contrastive_pairs.jsonl 不参与切分（train-only）。
用法：python scripts/build_hardneg_evalset.py
"""
import json
import random
import sys
from pathlib import Path

DATA = Path("datasets/hard_negatives")
OUT_EVAL = Path("datasets/hardneg_eval.jsonl")
OUT_TRAIN = DATA / "train_pool.jsonl"
PER_GENRE = 75
SEED = 42


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def main() -> int:
    rng = random.Random(SEED)
    eval_rows: list[dict] = []
    train_rows: list[dict] = []
    genre_files = sorted(DATA.glob("hn_*.jsonl"))
    nb = DATA / "nb_natural_benign.jsonl"
    if nb.exists():
        genre_files.append(nb)
    for p in genre_files:
        if p.name == "train_pool.jsonl":
            continue
        rows = load_jsonl(p)
        if not rows:
            print(f"  ! {p.name} 为空，跳过")
            continue
        rng.shuffle(rows)
        k = min(PER_GENRE, len(rows))
        if len(rows) < PER_GENRE + 40:
            print(f"  ! {p.stem} 仅 {len(rows)} 条（建议 ≥115：75 eval + 40 train）")
        source = rows[0]["source"]
        for r in rows[:k]:
            eval_rows.append({
                "id": r["id"], "type": "single", "state": r["text"],
                "language": r.get("language", "zh"), "expected_risk": 1,
                "category": "benign", "source": source,
            })
        train_rows.extend(rows[k:])
        print(f"  ✓ {p.stem}: eval {k} / train {len(rows) - k}")

    eval_ids = {r["id"] for r in eval_rows}
    train_ids = {r["id"] for r in train_rows}
    overlap = eval_ids & train_ids
    assert not overlap, f"评测/训练 id 相交 {len(overlap)} 条 —— 切分 bug"

    with OUT_EVAL.open("w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_TRAIN.open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {OUT_EVAL}: {len(eval_rows)} 条")
    print(f"✓ {OUT_TRAIN}: {len(train_rows)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 运行测试 + 真实切分**

Run: `pytest tests/test_hardneg_evalset.py -q && python scripts/build_hardneg_evalset.py`
Expected: 测试 PASS；真实输出 hardneg_eval ~675 条（9 体裁×75）、train_pool ~万级；无相交断言失败。

- [ ] **Step 5: Commit**

```bash
git add scripts/build_hardneg_evalset.py tests/test_hardneg_evalset.py datasets/hardneg_eval.jsonl datasets/hard_negatives/train_pool.jsonl
git commit -m "feat(data): carve out held-out hard-negative eval set (75/genre, disjoint)"
```

---

### Task 13: v2 数据集全量构建

**Files:**
- 运行 Task 7 改造后的 `scripts/build_dataset.py`（无新代码）

**Interfaces:**
- Consumes: 全部 raw 数据（Task 4）+ 难负 train_pool / contrastive_pairs / hardneg_eval ids（Task 10-12）
- Produces: `datasets/training/{train,val,test}.jsonl`（v2 最终版，gitignored 不提交）

- [ ] **Step 1: 构建**

Run: `source .venv/bin/activate && python scripts/build_dataset.py 2>&1 | tee reports/build_v2.log | tail -60`
Expected stdout：hardneg train_pool 与 contrastive 加载行数；hardneg_eval ids ~675；mix stats `n=45000`、`hardneg_kept` ≈ train_pool 全量、`tips_kept`/n_neg ∈ [10%, 15%]、`gen_share` ≤ 0.30；`pos_by_category["rebate_scam"]` ≥ 5063（25% × 20250）。

- [ ] **Step 2: 独立验证不变量**

```bash
python -c "
import json
from collections import Counter
train = [json.loads(l) for l in open('datasets/training/train.jsonl')]
eval_ids = {json.loads(l)['id'] for l in open('datasets/hardneg_eval.jsonl')}
pos = [r for r in train if r['is_scam'] == 1]
c = Counter(r['category'] for r in pos)
gen = sum(1 for r in train if r.get('source', '').startswith(('hn_', 'nb_', 'contrastive')))
leak = [r['id'] for r in train if r['id'] in eval_ids]
print('n', len(train), '| pos', len(pos), '| rebate', round(c['rebate_scam']/len(pos), 3), '| gen', round(gen/len(train), 3), '| leak', len(leak))
assert len(train) == 45000 and not leak
assert c['rebate_scam']/len(pos) >= 0.25 - 0.001
assert gen/len(train) <= 0.30 + 0.001
print('✓ invariants hold')"
```
Expected: `✓ invariants hold`

- [ ] **Step 3: 提交构建日志（datasets/training gitignored 不提交）**

```bash
git add reports/build_v2.log
git commit -m "docs(reports): v2 dataset build log (45k, hard negatives mixed)"
```

---

### Task 14: 评测协议扩展（src/report.py）

**Files:**
- Modify: `src/report.py`
- Create: `tests/test_report_v2.py`

**Interfaces:**
- Consumes: enriched results（`src.eval.run_evaluation` 输出行；`source` 字段已透传）
- Produces: `_per_category_recall(results) -> list[tuple[str, int, int, float]]`；`_hardneg_fpr(results) -> tuple[list[tuple[str,int,int,float]], tuple[int,int,float]] | None`；`_threshold_table(results) -> list[str]`；报告中新增三个 section（`## Per-Category Recall`、`## Hard-Negative FPR by Genre` 含 `| **overall** |` 行、`## Threshold-Recall Curve`）—— Task 16 的解析器按这些精确格式读取

- [ ] **Step 1: 写失败测试 tests/test_report_v2.py**

```python
"""report.py v2 扩展：分类别召回 / 分体裁 FPR / 阈值曲线。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.report import (
    _hardneg_fpr, _per_category_recall, _threshold_table, render_report,
)


def _res(noul, exp_risk, cat_pred=None, cat_exp=None, source=None):
    return {
        "id": "x", "type": "single", "language": "zh",
        "expected_risk": exp_risk, "expected_category": cat_exp,
        "source": source, "state_preview": "",
        "predicted": {
            "is_scam_noul": noul, "is_scam_label": int(noul >= 0.5),
            "risk_level": 1, "scam_category": cat_pred,
            "category_probs": None,
        },
        "routing": {}, "latency_ms": 10.0, "error": None,
    }


class TestPerCategoryRecall:
    def test_math(self):
        rs = [
            _res(0.9, 4, "rebate_scam", "rebate_scam", "ccl2023"),
            _res(0.9, 4, "phishing", "rebate_scam", "ccl2023"),
            _res(0.1, 1, "benign", "benign", "fgrc_scd_sms"),
        ]
        out = dict((c, (n, k, r)) for c, n, k, r in _per_category_recall(rs))
        assert out["rebate_scam"] == (2, 1, 0.5)
        assert out["benign"][2] == 1.0


class TestHardnegFpr:
    def test_overall_and_genres(self):
        rs = [
            _res(0.6, 1, source="hn_financial_notice"),
            _res(0.3, 1, source="hn_financial_notice"),
            _res(0.9, 1, source="hn_promotion"),
        ]
        genres, overall = _hardneg_fpr(rs)
        assert overall == (3, 2, 2 / 3)
        assert dict((s, fpr) for s, _, _, fpr in genres) == {
            "hn_financial_notice": 0.5, "hn_promotion": 1.0}

    def test_none_when_no_hardneg(self):
        assert _hardneg_fpr([_res(0.9, 4, source="ccl2023")]) is None


class TestThresholdTable:
    def test_has_mid_threshold_row(self):
        lines = _threshold_table([
            _res(0.9, 5), _res(0.2, 1), _res(0.8, 4), _res(0.4, 2),
        ])
        assert any(l.startswith("| 0.50 |") for l in lines)


class TestRenderReport:
    def test_sections_written(self, tmp_path):
        rs = [
            _res(0.6, 1, "benign", "benign", "hn_financial_notice"),
            _res(0.9, 4, "rebate_scam", "rebate_scam", "ccl2023"),
        ]
        md_path, _ = render_report(rs, {}, tmp_path)
        text = md_path.read_text()
        assert "## Per-Category Recall" in text
        assert "rebate_scam" in text
        assert "## Hard-Negative FPR by Genre" in text
        assert "| **overall** |" in text
        assert "## Threshold-Recall Curve" in text
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_report_v2.py -q`
Expected: FAIL（三个函数不存在）

- [ ] **Step 3: 实现 src/report.py 扩展**

`_category_table` 之前新增三个函数：

```python
def _per_category_recall(results: list[dict]) -> list[tuple[str, int, int, float]]:
    stats: dict[str, list[int]] = {}
    for r in results:
        p = r["predicted"]["scam_category"]
        e = r.get("expected_category")
        if p and e:
            s = stats.setdefault(e, [0, 0])
            s[1] += 1
            if p == e:
                s[0] += 1
    return [(c, s[1], s[0], s[0] / s[1]) for c, s in sorted(stats.items())]


def _hardneg_fpr(results: list[dict]):
    """分体裁 FPR；无难负行（source 不以 hn_/nb_ 开头）返回 None。"""
    rows = [r for r in results
            if (r.get("source") or "").startswith(("hn_", "nb_"))
            and r.get("expected_risk") is not None and r["expected_risk"] < 4
            and r["predicted"]["is_scam_noul"] is not None]
    if not rows:
        return None
    by_src: dict[str, list[int]] = {}
    for r in rows:
        s = by_src.setdefault(r["source"], [0, 0])
        s[1] += 1
        if r["predicted"]["is_scam_noul"] >= 0.5:
            s[0] += 1
    genres = [(src, s[1], s[0], s[0] / s[1]) for src, s in sorted(by_src.items())]
    n = len(rows)
    fp = sum(1 for r in rows if r["predicted"]["is_scam_noul"] >= 0.5)
    return genres, (n, fp, fp / n)


def _threshold_table(results: list[dict]) -> list[str]:
    pairs = [(r["predicted"]["is_scam_noul"], int(r["expected_risk"] >= 4))
             for r in results
             if r["predicted"]["is_scam_noul"] is not None
             and r.get("expected_risk") is not None]
    if not pairs:
        return ["_No binary data._"]
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    lines = ["| Threshold | Recall | FPR | Flagged% |", "|---|---|---|---|"]
    for t in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 0.95):
        tp = sum(1 for p, y in pairs if p >= t and y == 1)
        fp = sum(1 for p, y in pairs if p >= t and y == 0)
        lines.append(f"| {t:.2f} | {tp / max(n_pos, 1):.3f} | "
                     f"{fp / max(n_neg, 1):.3f} | {(tp + fp) / len(pairs):.1%} |")
    return lines
```

`render_report` 中把这一行：

```python
    md += ["", "## Scam Category Confusion", "", _category_table(results), "",
```

替换为（在其前插入三个 section）：

```python
    per_cat = _per_category_recall(results)
    if per_cat:
        md += ["", "## Per-Category Recall", "",
               "| Category | n | Correct | Recall |", "|---|---|---|---|"]
        for cat, n, correct, recall in per_cat:
            md.append(f"| {cat} | {n} | {correct} | {recall:.3f} |")
    hnfpr = _hardneg_fpr(results)
    if hnfpr is not None:
        genres, overall = hnfpr
        md += ["", "## Hard-Negative FPR by Genre", "",
               "| Source | n | FP | FPR |", "|---|---|---|---|"]
        for src, n, fp, fpr in genres:
            md.append(f"| {src} | {n} | {fp} | {fpr:.3f} |")
        md.append(f"| **overall** | {overall[0]} | {overall[1]} | {overall[2]:.3f} |")
    md += ["", "## Threshold-Recall Curve", ""] + _threshold_table(results)
    md += ["", "## Scam Category Confusion", "", _category_table(results), "",
```

- [ ] **Step 4: 运行测试 + 全量回归**

Run: `pytest tests/test_report_v2.py -q && pytest -q`
Expected: PASS（基线保持——现有 test_report.py 不受影响，只增不改）

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_report_v2.py
git commit -m "feat(eval): per-category recall + hard-negative FPR + threshold curve in reports"
```

---

### Task 15: 训练 v2 + ONNX 导出（runbook，长任务）

**Files:**
- 无新代码；输出 `models/laya-lora-finetuned-v2/`、`models/laya-onnx-multilingual-finetuned-v2/`（gitignored）；`reports/train_v2.log`（tracked）

**Interfaces:**
- Consumes: `datasets/training/{train,val}.jsonl`（Task 13 v2 最终版）；现有 `scripts/train_local_lora.py` / `scripts/export_local_onnx.py`（不修改）
- Produces: v2 ONNX bundle（14 类 categories.json 由 export 脚本 import CANONICAL_CATEGORIES 自动生成）

- [ ] **Step 1: 训练（后台 ~4-5 小时）**

```bash
source .venv/bin/activate
nohup env HF_HUB_DISABLE_XET=1 python -u scripts/train_local_lora.py \
  --train datasets/training/train.jsonl \
  --val datasets/training/val.jsonl \
  --max-samples 45000 --epochs 4 --batch-size 8 --grad-accum 2 \
  --output models/laya-lora-finetuned-v2 > reports/train_v2.log 2>&1 &
```

监控：`tail -f reports/train_v2.log`（v1 为 15k/73min/3epoch；45k/4epoch 预估 4-5h）
Expected：每 epoch 一行 `[epoch N] loss=… val_zh_scam_acc=… val_cat_acc=…` + `✓ new best; saving checkpoint`。
**训练门槛**：最终 val_zh_scam_acc ≥ 0.93 且逐 epoch 上升（v1 达 0.949）。若 < 0.90：检查 mix stats 与 val 构成，修数据后重训，不带病进入导出。

- [ ] **Step 2: ONNX 导出 + 验证**

```bash
HF_HUB_DISABLE_XET=1 python scripts/export_local_onnx.py \
  --state models/laya-lora-finetuned-v2/model_state.pt \
  --output models/laya-onnx-multilingual-finetuned-v2
```
Expected：`max diff … ✓ (阈值 1e-3)`；bundle 内 categories.json 含 14 类。

- [ ] **Step 3: 冒烟预测（刷单返利 + 反诈宣传两条）**

```bash
python main.py --predict "您好，动动手指刷单做任务，先垫付货款就能返佣金高额回报，加我了解详情" \
  --questions schemas/scam.json \
  --multilingual-dir models/laya-onnx-multilingual-finetuned-v2
python main.py --predict "【12381】国家反诈中心提醒：您可能正在遭遇电信网络诈骗，请提高警惕，切勿向陌生账户转账汇款。" \
  --questions schemas/scam.json \
  --multilingual-dir models/laya-onnx-multilingual-finetuned-v2
```
Expected：第一条 is_scam noul 高（>0.5）、category 含 rebate_scam 信号；第二条 noul < 0.5（v1 大概率误报的体裁）。第二条若仍 ≥0.5 不必惊慌——四门槛看的是评测集整体 FPR。

- [ ] **Step 4: 全量回归 + Commit 训练日志**

Run: `pytest -q`
Expected: 基线保持（模型依赖测试走默认目录，v1 模型仍在，不受影响）

```bash
git add reports/train_v2.log
git commit -m "docs(reports): v2 training log (45k samples, 4 epochs, MPS)"
```

---

### Task 16: 四门槛验收（run_full_eval.py + check_acceptance.py v2）

**Files:**
- Modify: `scripts/check_acceptance.py`
- Create: `scripts/run_full_eval.py`
- Create: `tests/test_acceptance.py`
- Output: `datasets/eval_holdout_v2.jsonl`（600 条，tracked）、3 份 eval 报告 + 验收输出（reports/，tracked）

**Interfaces:**
- Consumes: 报告格式来自 Task 14（`| **overall** |` FPR 行、`## Per-Category Recall` 表）；`main.py --eval`（打印 `✓ Report: <path>` 到 stderr）；v2 模型目录
- Produces: `check_acceptance.parse_overall_fpr(text) -> float | None`、`parse_category_recall(text, category) -> float | None`；4 门槛 exit code（0=全过）；`run_full_eval.py --model-dir <dir>` 一键四门槛

- [ ] **Step 1: 写失败测试 tests/test_acceptance.py**

```python
"""check_acceptance v2 解析函数测试（合成报告，无模型）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from check_acceptance import parse_category_recall, parse_overall_fpr  # noqa: E402

FPR_MD = """## Hard-Negative FPR by Genre

| Source | n | FP | FPR |
|---|---|---|---|
| hn_financial_notice | 75 | 1 | 0.013 |
| **overall** | 600 | 9 | 0.015 |
"""

RECALL_MD = """## Per-Category Recall

| Category | n | Correct | Recall |
|---|---|---|---|
| rebate_scam | 120 | 108 | 0.900 |
| benign | 80 | 72 | 0.900 |
"""


def test_parse_overall_fpr():
    assert parse_overall_fpr(FPR_MD) == 0.015


def test_parse_overall_fpr_missing():
    assert parse_overall_fpr("no table here") is None


def test_parse_category_recall():
    assert parse_category_recall(RECALL_MD, "rebate_scam") == 0.900


def test_parse_category_recall_missing():
    assert parse_category_recall(RECALL_MD, "crypto_scam") is None
```

- [ ] **Step 2: 运行确认失败**

Run: `pytest tests/test_acceptance.py -q`
Expected: FAIL（两个函数不存在）

- [ ] **Step 3: 改造 scripts/check_acceptance.py**

改动规则：`parse_headline_metrics`、`parse_language_accuracy`、`find_metric` 三个函数体**原样保留不动**；文件头加 `import re`。新增两个解析函数（`find_metric` 之后）：

```python
def parse_overall_fpr(text: str) -> float | None:
    """从 ## Hard-Negative FPR by Genre 表读 overall 行。"""
    m = re.search(
        r"\|\s*\*\*overall\*\*\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([0-9.]+)\s*\|",
        text)
    return float(m.group(1)) if m else None


def parse_category_recall(text: str, category: str) -> float | None:
    """从 ## Per-Category Recall 表读指定类别召回。"""
    m = re.search(
        rf"\|\s*{re.escape(category)}\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([0-9.]+)\s*\|",
        text)
    return float(m.group(1)) if m else None
```

把现有 `main()` 整体改名为 `run_legacy(report_arg: str | None) -> int`（函数体原样：开头两行改为 `if report_arg: report = Path(report_arg)` / `else:` 分支不变），然后新增新的 `main()`：

```python
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", help="legacy 模式：单报告路径（缺省取最新）")
    ap.add_argument("--hardneg", help="难负评测集报告（Gate 1）")
    ap.add_argument("--holdout", help="600 中文 holdout 报告（Gate 2/4）")
    ap.add_argument("--handwritten", help="手写集报告（Gate 3）")
    args = ap.parse_args()

    if args.report or not (args.hardneg or args.holdout or args.handwritten):
        return run_legacy(args.report)

    hardneg_text = Path(args.hardneg).read_text() if args.hardneg else ""
    holdout_text = Path(args.holdout).read_text() if args.holdout else ""
    hand_text = Path(args.handwritten).read_text() if args.handwritten else ""
    flags: list[bool] = []

    # Gate 1: 难负评测集 FPR ≤ 2%（主目标）
    fpr = parse_overall_fpr(hardneg_text)
    ok = fpr is not None and fpr <= 0.02
    print(f"{'✓' if ok else '✗'} Gate1 难负评测集 FPR: "
          f"{fpr if fpr is None else f'{fpr:.3f}'} (need ≤ 0.02)")
    flags.append(ok)

    # Gate 2: holdout zh acc ≥0.90 / recall ≥0.95（绝对门槛）
    zh_acc = parse_language_accuracy(holdout_text, "zh")
    recall = find_metric(parse_headline_metrics(holdout_text), "recall")
    ok = (zh_acc is not None and zh_acc >= 0.90
          and recall is not None and recall >= 0.95)
    print(f"{'✓' if ok else '✗'} Gate2 holdout zh acc/recall: "
          f"{zh_acc} / {recall} (need ≥0.90 / ≥0.95)")
    flags.append(ok)

    # Gate 3: 手写集 14 类 acc ≥0.70（固定可比）
    cat_acc = find_metric(parse_headline_metrics(hand_text), "category accuracy")
    ok = cat_acc is not None and cat_acc >= 0.70
    print(f"{'✓' if ok else '✗'} Gate3 手写集 scam_category acc: "
          f"{cat_acc} (need ≥ 0.70)")
    flags.append(ok)

    # Gate 4: rebate_scam 召回 ≥0.80（holdout 分类别召回表）
    rebate = parse_category_recall(holdout_text, "rebate_scam")
    ok = rebate is not None and rebate >= 0.80
    print(f"{'✓' if ok else '✗'} Gate4 rebate_scam 召回: "
          f"{rebate} (need ≥ 0.80)")
    flags.append(ok)

    print("=" * 44)
    print(f"{sum(flags)}/{len(flags)} criteria met")
    return 0 if all(flags) else 1
```

- [ ] **Step 4: 运行解析测试 + legacy 回归**

Run: `pytest tests/test_acceptance.py -q && PYTHONPATH=. python scripts/check_acceptance.py reports/eval-20260928-221345.md`
Expected: 新测试 PASS；legacy 模式输出原 3 门槛结果（v1 报告 zh 0.967/recall 0.957/13类 0.810 全 ✓）。

- [ ] **Step 5: 写 scripts/run_full_eval.py**

```python
"""跑 v2 三套评测并调用四门槛验收（design §1、§7.2）。

用法：python scripts/run_full_eval.py [--model-dir models/laya-onnx-multilingual-finetuned-v2]
"""
import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

V2_DEFAULT = "models/laya-onnx-multilingual-finetuned-v2"
HOLDOUT = Path("datasets/eval_holdout_v2.jsonl")
TEST = Path("datasets/training/test.jsonl")
EVAL_INPUTS = [
    ("hardneg", "datasets/hardneg_eval.jsonl"),
    ("holdout", str(HOLDOUT)),
    ("handwritten", "datasets/eval.jsonl"),
    # 第 4 套：英文侧不回退证据（design §7.2）——无门槛消费，仅出报告
    ("public", "datasets/public.jsonl"),
]


def ensure_holdout() -> None:
    """600 条中文 holdout（test.jsonl 采样，seed=42，评测格式）。"""
    if HOLDOUT.exists():
        return
    rows = [json.loads(l) for l in TEST.open() if l.strip()]
    zh = [r for r in rows if r.get("language") == "zh"]
    rng = random.Random(42)
    rng.shuffle(zh)
    picked = zh[:600]
    assert len(picked) == 600, f"test.jsonl 中文行不足 600: {len(zh)}"
    with HOLDOUT.open("w") as f:
        for r in picked:
            f.write(json.dumps({
                "id": r["id"], "type": "single", "state": r["text"],
                "language": "zh", "expected_risk": r["risk"],
                "category": r["category"], "source": r["source"],
            }, ensure_ascii=False) + "\n")
    print(f"✓ {HOLDOUT}: 600 条")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=V2_DEFAULT)
    args = ap.parse_args()

    if not Path(args.model_dir, "model.onnx").exists():
        print(f"✗ 模型缺失: {args.model_dir}")
        return 2

    ensure_holdout()
    paths: dict[str, Path] = {}
    for name, inp in EVAL_INPUTS:
        cmd = [sys.executable, "main.py", "--eval", "--input", inp,
               "--output", "reports/", "--multilingual-dir", args.model_dir]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            print(f"✗ {name} eval 失败:\n{r.stderr[-500:]}")
            return 2
        m = re.search(r"✓ Report: (.+)", r.stderr)
        if not m:
            print(f"✗ {name} eval 未输出报告路径")
            return 2
        paths[name] = Path(m.group(1).strip())
        print(f"✓ {name}: {paths[name]}")

    cmd = [sys.executable, "scripts/check_acceptance.py",
           "--hardneg", str(paths["hardneg"]),
           "--holdout", str(paths["holdout"]),
           "--handwritten", str(paths["handwritten"])]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: 运行四门槛验收**

Run: `source .venv/bin/activate && python scripts/run_full_eval.py 2>&1 | tee reports/v2_acceptance.txt`
Expected：4 份新 eval 报告生成（含 public 英文侧证据报告，无门槛消费）；`Gate1-4` 逐项 ✓/✗ + `N/4 criteria met`。
- **4/4** → 计划完成，跳过 Task 17，写收尾报告（见完成标准）
- **Gate1 ✗（FPR>2%）** → 进入 Task 17 挖掘闭环
- 其他 Gate ✗ → 先查对应报告的 Failure analysis 表定位（数据配比/类别缺口），不盲目重训

- [ ] **Step 7: 全量回归 + Commit**

Run: `pytest -q`
Expected: 基线保持

```bash
git add scripts/check_acceptance.py scripts/run_full_eval.py tests/test_acceptance.py datasets/eval_holdout_v2.jsonl reports/v2_acceptance.txt
git add reports/eval-*.md reports/results-*.jsonl
git commit -m "feat(eval): 4-gate acceptance (hardneg FPR + no-regress + rebate recall) + full-eval runner"
```

---

### Task 17: 挖掘闭环（FPR>2% 时，硬上限 2 轮）

**Files:**
- Modify: `scripts/generate_negatives.py`（加 `--out` 旗标）
- Create: `scripts/mine_hard_negatives.py`
- Output: `datasets/hard_negatives/mining_candidates.jsonl`、`mining_round{N}_review.jsonl`（tracked）

**Interfaces:**
- Consumes: `OnnxLayaClient`（v2 模型）、Task 11 生成脚本
- Produces: `mine_hard_negatives.py mine --round N --top K`（Top-K 复核清单，verdict 留空）；`merge --round N`（verdict=="keep" 行追加进 train_pool.jsonl）

- [ ] **Step 1: generate_negatives.py 加 --out 旗标**

三处小改：

```python
    ap.add_argument("--out", help="输出到指定文件（挖掘候选池），而非按体裁分文件")
```

`gen_genre` 签名与首行：

```python
def gen_genre(cfg: dict, name: str, spec: dict, limit: int | None,
              seen: set[str], out_override: str | None = None) -> int:
    out_path = Path(out_override) if out_override else DATA / f"{name}.jsonl"
```

`main()` 调用处传 `args.out`：

```python
        got = gen_genre(cfg, name, spec, args.limit, seen, args.out)
```

并把 `main()` 里的 `if not args.genre:` 改为 `if not args.genre and not args.out:`（候选池模式跳过对照对生成，防止向训练文件 `contrastive_pairs.jsonl` 追加新对）。

- [ ] **Step 2: 写 scripts/mine_hard_negatives.py**

```python
"""滚动挖掘最难负样本（design §6.4；FPR>2% 时启动，硬上限 2 轮）。

mine  : v2 模型给候选池打分 → Top-K 复核清单（verdict 留空待人工）
merge : 人工填 verdict=keep/drop 后，把 keep 行追加进 train_pool.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.laya_onnx import OnnxLayaClient

DATA = Path("datasets/hard_negatives")
SCHEMA = json.loads(Path("schemas/scam.json").read_text())


def mode_mine(args) -> int:
    if not Path(args.model_dir, "model.onnx").exists():
        print(f"✗ 模型缺失: {args.model_dir}")
        return 2
    cand = DATA / "mining_candidates.jsonl"
    if not cand.exists():
        print(f"✗ {cand} 缺失 —— 先跑 generate_negatives.py --out 生成候选池")
        return 2
    rows = [json.loads(l) for l in cand.open() if l.strip()]
    print(f"候选池 {len(rows)} 条，打分中…")
    client = OnnxLayaClient(args.model_dir)
    scored = []
    for i, r in enumerate(rows):
        pred = client.predict(r["text"], SCHEMA)
        scored.append((pred["answers"]["is_scam"]["noul"], r))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)}")
    scored.sort(key=lambda x: -x[0])
    out = DATA / f"mining_round{args.round}_review.jsonl"
    with out.open("w") as f:
        for noul, r in scored[: args.top]:
            rec = dict(r)
            rec["noul"] = noul
            rec["verdict"] = ""
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✓ {out}: Top {min(args.top, len(scored))} 复核清单")
    print("  人工操作：把每行 verdict 填 keep（确认真 benign）或 drop（实为诈骗/假负例）")
    return 0


def mode_merge(args) -> int:
    review = DATA / f"mining_round{args.round}_review.jsonl"
    pool = DATA / "train_pool.jsonl"
    keep = [json.loads(l) for l in review.open() if l.strip()
            and json.loads(l).get("verdict") == "keep"]
    with pool.open("a") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ {pool} += {len(keep)} keep 行（drop 行不入池）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["mine", "merge"])
    ap.add_argument("--model-dir",
                    default="models/laya-onnx-multilingual-finetuned-v2")
    ap.add_argument("--round", type=int, required=True, choices=[1, 2])
    ap.add_argument("--top", type=int, default=1000)
    args = ap.parse_args()
    return mode_mine(args) if args.mode == "mine" else mode_merge(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: 挖掘轮 runbook（每轮重复，硬上限 2 轮）**

```bash
# 1. 生成新候选池（~5k，独立文件不进训练）
python scripts/generate_negatives.py --out datasets/hard_negatives/mining_candidates.jsonl

# 2. v2 模型打分 → 复核清单（~5k × 125ms ≈ 10 分钟）
PYTHONPATH=. python scripts/mine_hard_negatives.py mine --round 1 --top 1000

# 3. 【人工】复核 mining_round1_review.jsonl：按 noul 降序逐条填 verdict
#    keep = 确认真 benign（模型误报样本，正是最难的难负）
#    drop = 实为诈骗/低质量（假负例，剔除）
#    建议：复核前 500-1000 条即可，低分样本模型本就不误报、无挖掘价值

# 4. 合并 keep 行进训练池
PYTHONPATH=. python scripts/mine_hard_negatives.py merge --round 1

# 5. 重建数据集（train_pool 变化 → 配比更新）
python scripts/build_dataset.py 2>&1 | tail -20

# 6. 重训 → 导出 → 四门槛（round 1 版本号后缀 -v2r1）
nohup env HF_HUB_DISABLE_XET=1 python -u scripts/train_local_lora.py \
  --train datasets/training/train.jsonl --val datasets/training/val.jsonl \
  --max-samples 45000 --epochs 4 --batch-size 8 --grad-accum 2 \
  --output models/laya-lora-finetuned-v2r1 > reports/train_v2r1.log 2>&1 &
# （训练完成后）
HF_HUB_DISABLE_XET=1 python scripts/export_local_onnx.py \
  --state models/laya-lora-finetuned-v2r1/model_state.pt \
  --output models/laya-onnx-multilingual-finetuned-v2r1
python scripts/run_full_eval.py --model-dir models/laya-onnx-multilingual-finetuned-v2r1
```

round 2 同构（`--round 2`，输出后缀 `-v2r2`），**仅当 round 1 后 Gate1 仍 ✗**。

- [ ] **Step 4: 停止条件 + 收尾报告**

- FPR ≤ 2% 在任一轮达成 → 停，写 `reports/v2-acceptance-summary.md`（四门槛终值 + 各评测集报告链接 + 采纳的模型目录名）
- 2 轮后仍 > 2% → **如实报告**，写 `reports/v2-acceptance-summary.md`：门槛差距（如 2.4%）、分体裁 FPR 表指出最差体裁、根因假设（生成样本文体单一 / 体裁覆盖不足 / 提示类占比仍高）、下一步建议（生产误报日志反哺 —— 报告 §一结论：终极误报率只能靠真实误报数据）。**不无限循环**。

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_negatives.py scripts/mine_hard_negatives.py datasets/hard_negatives/mining_candidates.jsonl datasets/hard_negatives/mining_round*_review.jsonl datasets/hard_negatives/train_pool.jsonl reports/v2-acceptance-summary.md
git commit -m "feat(data): hard-negative mining loop (2-round cap) + acceptance summary"
```

---

## 完成标准（Definition of Done）

全部满足即计划完成：

1. `python scripts/check_acceptance.py --hardneg … --holdout … --handwritten …` 输出 **4/4** —— 或 2 轮挖掘后 FPR 仍 >2%，差距如实写入 `reports/v2-acceptance-summary.md`
2. `pytest -q` 全量通过（基线 115+ passed / 2 skipped）
3. v2 ONNX bundle 导出验证 logit diff < 1e-3
4. 数据产物（难负样本 jsonl、hardneg_eval、gen_models.json、训练/构建/验收日志）全部提交进 git
5. 若 FPR≤2% 首轮即达成：跳过 Task 17，直接写 `reports/v2-acceptance-summary.md` 收尾

## 任务依赖图

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 ─┐
Task 6 ─────────────────────────────────────┤→ Task 7 → Task 13
Task 8（独立，可并行）                        │
Task 9（依赖 Task 7 产物）→ Task 10 → Task 11 → Task 12 ─┘
Task 13 → Task 15 → Task 16 → [Task 17 仅当 Gate1 ✗]
```
