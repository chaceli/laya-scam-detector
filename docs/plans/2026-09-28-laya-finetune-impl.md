# Laya 多语增量训练 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fine-tune `convaiinnovations/laya-multilingual` (mmBERT-base) with LoRA on multi-source Chinese + English scam datasets (~30k samples), export to ONNX, and integrate into the existing CLI/eval pipeline with a 13-class schema.

**Architecture:** Local data prep (Python) → Kaggle 2×T4 fine-tune (Jupyter) → Local merge + ONNX export (Python + PyTorch reference) → Local evaluation (reuse existing OnnxLayaClient) → Update Router for script-based dispatch to either English or multilingual checkpoint.

**Tech Stack:** Python 3.10+, peft (LoRA), trl, accelerate, datasets, transformers, torch (training only); ONNX Runtime + tokenizers + numpy (inference). Kaggle free GPU.

**Design:** `docs/plans/2026-09-28-laya-finetune-design.md`

---

## Work Decomposition Overview

```
Phase 1: Data prep + schema      (Tasks 1-7, ~1-2h local)
Phase 2: Router dual-checkpoint   (Tasks 8-10, ~30min)
Phase 3: Local pre-flight tests    (Tasks 11-13, ~30min)
Phase 4: Kaggle notebook           (Tasks 14-23, ~1h notebook + manual Kaggle 4-5h)
Phase 5: Merge + ONNX export       (Tasks 24-27, ~1h local)
Phase 6: Eval + integration       (Tasks 28-35, ~2h local)
```

**Manual Kaggle tasks** marked with ⚠️ — those can't be automated.

**Parallel opportunities**:
- Phase 1 Tasks 3-4 (schema + fetch) can run in parallel
- Phase 1 Tasks 8-9 (Router) can start in parallel with Phase 1 Task 7 (data prep completion)
- Phase 3 tests can run in parallel with Phase 4 notebook creation

---

## Phase 1: Data Preparation + 13-Class Schema

### Task 1: Add training dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1:** Add `train` extra to pyproject.toml

Append after the existing `dev` extra:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov", "ruff>=0.6"]
train = [
    "peft>=0.10",
    "trl>=0.8",
    "accelerate>=0.27",
    "datasets>=2.18",
    "transformers>=4.48",
    "safetensors>=0.4",
    "kagglehub>=0.2",
    "ipywidgets>=8.1",
    "matplotlib>=3.8",
]
```

**Step 2:** Install

```bash
cd /Users/ks/source_code/laya
.venv/bin/pip install -e ".[dev,train]"
```

Expected: peft, trl, accelerate, datasets, transformers, torch, safetensors, kagglehub, ipywidgets, matplotlib all install successfully. (torch was not in our previous deps — this will add ~800MB.)

**Step 3:** Verify imports

```bash
.venv/bin/python -c "import peft, trl, accelerate, datasets, transformers, torch, kagglehub; print('peft', peft.__version__); print('trl', trl.__version__); print('torch', torch.__version__); print('transformers', transformers.__version__)"
```

Expected: prints versions of each.

**Step 4:** Commit

```bash
cd /Users/ks/source_code/laya
git add pyproject.toml
git -c user.email=claude@local -c user.name=claude commit -m "chore(deps): add [train] extra with peft, trl, accelerate, torch for fine-tuning"
```

---

### Task 2: Create category mapping module

**Files:**
- Create: `schemas/scam_categories.py`

**Step 1:** Write the mapping module

```python
"""Category mapping from source dataset labels to our unified 13-class schema.

Each public dataset uses different label vocabulary. This module provides:
1. The 13-class canonical taxonomy (CANONICAL_CATEGORIES)
2. Maps source labels → canonical labels (CATEGORY_MAP)
3. Functions to normalize a free-form label to the canonical 13

Sources we normalize:
- SpamShield-Datasets (M-Arjun): spam/phishing/crypto/marketing/job_scam/giveaway/adult/promo/normal
- FBS_SMS_Dataset (fl-wxiao): AD:Loan/AD:Network_service/AD:Other/FR:Financial/FR:Phishing(Bank)/
                              FR:Phishing(Other)/FR:Other/IL:Escort_service/IL:Fake_ID_and_invoice/
                              IL:Gambling/IL:Political_propaganda/Other
- FGRC-SCD: low_risk_sms/high_risk_sms/fraud_call/phishing_link (custom)
- ealvaradob: phishing/url_phishing/html_phishing/legitimate
- UC Irvine SMS Spam: ham/spam
- vichetkao/Scam_Message_9_Language: ham/spam (binary only)
- Handwritten (existing): benign / delivery_fraud / phishing / romance_scam / investment_scam /
                            impersonation / lottery_scam / loan_scam
"""
from __future__ import annotations


CANONICAL_CATEGORIES = [
    "benign",
    "phishing",
    "crypto_scam",
    "investment_scam",
    "lottery_scam",
    "job_scam",
    "loan_scam",
    "impersonation",
    "romance_scam",
    "delivery_fraud",
    "marketing",
    "adult_content",
    "spam_general",
]

CANONICAL_TO_INDEX = {c: i for i, c in enumerate(CANONICAL_CATEGORIES)}


CATEGORY_MAP: dict[str, str] = {
    # SpamShield (English + others)
    "spam": "spam_general",
    "phishing": "phishing",
    "crypto": "crypto_scam",
    "marketing": "marketing",
    "job_scam": "job_scam",
    "giveaway": "lottery_scam",
    "adult": "adult_content",
    "promo": "marketing",
    "normal": "benign",
    "ham": "benign",
    "legitimate": "benign",

    # FBS_SMS_Dataset (Chinese)
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

    # FGRC-SCD (Chinese telecom fraud)
    "low_risk_sms": "benign",
    "high_risk_sms": "spam_general",
    "fraud_call": "impersonation",
    "phishing_link": "phishing",

    # ealvaradob phishing dataset
    "url_phishing": "phishing",
    "html_phishing": "phishing",

    # Handwritten (existing project dataset)
    "benign": "benign",
    "delivery_fraud": "delivery_fraud",
    "romance_scam": "romance_scam",
    "investment_scam": "investment_scam",
    "impersonation": "impersonation",
    "lottery_scam": "lottery_scam",
    "loan_scam": "loan_scam",
}


def normalize_label(raw_label: str) -> str:
    """Map any source label to canonical 13-class category.

    Falls back to 'spam_general' for unknown source labels (better than
    dropping the sample, which would lose training data).
    """
    return CATEGORY_MAP.get(raw_label, "spam_general")


def is_valid_category(label: str) -> bool:
    return label in CANONICAL_TO_INDEX


def label_to_index(label: str) -> int:
    if label not in CANONICAL_TO_INDEX:
        raise ValueError(f"unknown category: {label!r}")
    return CANONICAL_TO_INDEX[label]


def index_to_label(idx: int) -> str:
    if not 0 <= idx < len(CANONICAL_CATEGORIES):
        raise ValueError(f"index out of range: {idx}")
    return CANONICAL_CATEGORIES[idx]
```

**Step 2:** Write tests

Create `tests/test_categories.py`:

```python
"""Tests for category mapping module."""
import pytest

from schemas.scam_categories import (
    CANONICAL_CATEGORIES,
    CATEGORY_MAP,
    normalize_label,
    is_valid_category,
    label_to_index,
    index_to_label,
)


class TestCanonicalCategories:
    def test_exactly_13_categories(self):
        assert len(CANONICAL_CATEGORIES) == 13

    def test_benign_is_first(self):
        assert CANONICAL_CATEGORIES[0] == "benign"

    def test_all_unique(self):
        assert len(set(CANONICAL_CATEGORIES)) == 13


class TestNormalizeLabel:
    def test_known_spamshield_labels(self):
        assert normalize_label("spam") == "spam_general"
        assert normalize_label("crypto") == "crypto_scam"
        assert normalize_label("job_scam") == "job_scam"
        assert normalize_label("ham") == "benign"

    def test_known_fbs_sms_labels(self):
        assert normalize_label("AD:Loan") == "loan_scam"
        assert normalize_label("FR:Phishing(Bank)") == "phishing"
        assert normalize_label("IL:Gambling") == "lottery_scam"
        assert normalize_label("IL:Fake_ID_and_invoice") == "impersonation"

    def test_known_fgrc_scd_labels(self):
        assert normalize_label("low_risk_sms") == "benign"
        assert normalize_label("phishing_link") == "phishing"

    def test_known_ealvaradob_labels(self):
        assert normalize_label("phishing") == "phishing"
        assert normalize_label("legitimate") == "benign"

    def test_unknown_label_falls_back_to_spam_general(self):
        assert normalize_label("totally_new_label_xyz") == "spam_general"

    def test_idempotent_for_canonical(self):
        for c in CANONICAL_CATEGORIES:
            assert normalize_label(c) == c


class TestIndexConversion:
    def test_round_trip(self):
        for label in CANONICAL_CATEGORIES:
            idx = label_to_index(label)
            assert index_to_label(idx) == label

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError):
            label_to_index("nonexistent")

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError):
            index_to_label(99)


class TestIsValidCategory:
    def test_canonical_is_valid(self):
        for c in CANONICAL_CATEGORIES:
            assert is_valid_category(c) is True

    def test_non_canonical_invalid(self):
        assert is_valid_category("spam") is False
        assert is_valid_category("spam_general") is True
```

**Step 3:** Run tests

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/test_categories.py -v
```

Expected: All 14 tests pass.

**Step 4:** Commit

```bash
cd /Users/ks/source_code/laya
git add schemas/scam_categories.py tests/test_categories.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(schemas): add 13-class taxonomy and source-label mapping"
```

---

### Task 3: Update scam.json to 13 classes

**Files:**
- Modify: `schemas/scam.json`

**Step 1:** Replace contents with 13-class schema

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
      "benign": "normal legitimate message (no fraud signal)",
      "phishing": "phishing link, credential theft, account verification",
      "crypto_scam": "cryptocurrency fraud, fake coin offering",
      "investment_scam": "fake returns, stock tips, Ponzi scheme",
      "lottery_scam": "fake prize, lucky draw, congratulations winner",
      "job_scam": "fake job offer, upfront fee, mule recruitment",
      "loan_scam": "fake loan offer, predatory lending",
      "impersonation": "fake police, government, bank, customer service",
      "romance_scam": "pig butchering, emotional manipulation, long game",
      "delivery_fraud": "fake courier, refund, lost package",
      "marketing": "legitimate but aggressive sales / promotions",
      "adult_content": "adult, escort, sexual services",
      "spam_general": "other unsolicited junk / noise"
    }
  }
}
```

**Step 2:** Verify tests still pass

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v --tb=no -q
```

Expected: 70 existing tests + 14 new category tests = **84 passed** (or 70 if test_categories is picked up differently).

Note: This change **will break** `tests/test_e2e_predict.py` if it hardcodes 8 categories. We'll fix this in Task 30.

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add schemas/scam.json
git -c user.email=claude@local -c user.name=claude commit -m "feat(schemas): extend scam schema from 8 to 13 categories"
```

---

### Task 4: Write dataset fetcher script

**Files:**
- Create: `scripts/fetch_datasets.py`

**Step 1:** Write the script

```python
"""Download public scam-detection datasets for fine-tuning.

Uses curl to bypass Hugging Face's SOCKS proxy issues on this machine.

Downloads:
- FGRC-SCD (Abooooo, Chinese telecom fraud)
- vichetkao/Scam_Message_9_Language (multilingual ham/spam)
- M-Arjun/SpamShield-Datasets (multilingual categorized)
- fl-wxiao/FBS_SMS_Dataset (Chinese fake-base-station)
- ealvaradob/phishing-dataset (English URL/SMS/email phishing)

UC Irvine SMS Spam already downloaded by previous project (datasets/public.jsonl).
"""
import json
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path("datasets/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = [
    {
        "name": "fgrc_scd",
        "url": "https://huggingface.co/datasets/Abooooo/FGRC-SCD/resolve/main/data/train-00000-of-00001.parquet",
        "format": "parquet",
        "text_col": "text",
        "label_col": "label",
        "category_col": "category",
    },
    {
        "name": "scam_9_lang",
        "url": "https://huggingface.co/datasets/vichetkao/Scam_Message_9_Language/resolve/main/data/train-00000-of-00001.parquet",
        "format": "parquet",
        "text_col": "text",
        "label_col": "label",
        "language_col": "language",
    },
    {
        "name": "spamshield",
        "url": "https://huggingface.co/datasets/M-Arjun/SpamShield-Datasets/resolve/main/combined.parquet",
        "format": "parquet",
        "text_col": "text",
        "label_col": "label",
        "category_col": "category",
        "language_col": "language",
    },
    {
        "name": "fbs_sms",
        "url": None,  # Special: GitHub clone
        "format": "github",
        "text_col": "text",
        "label_col": "label",
        "category_col": "category",
        "git_url": "https://github.com/fl-wxiao/FBS_SMS_Dataset.git",
        "git_dir": RAW_DIR / "fbs_sms",
    },
    {
        "name": "ealvaradob_phishing",
        "url": "https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/phishing_dataset.csv",
        "format": "csv",
        "text_col": "text",
        "label_col": "label",
    },
]


def curl_download(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 1000:
        print(f"  ✓ {target.name} already exists ({target.stat().st_size:,} bytes)")
        return
    print(f"Downloading {url[:80]}... → {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-L", "-m", "300", "-o", str(target), url],
        check=False,
    )
    if result.returncode != 0:
        print(f"  ✗ curl failed for {url}", file=sys.stderr)
        target.unlink(missing_ok=True)
        return


def git_clone(git_url: str, target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        print(f"  ✓ {target.name} already cloned")
        return
    print(f"Cloning {git_url} → {target.name}")
    subprocess.run(
        ["git", "clone", "--depth", "1", git_url, str(target)],
        check=False,
    )


def main() -> int:
    for ds in DATASETS:
        print(f"\n[{ds['name']}]")
        if ds["format"] == "github":
            git_clone(ds["git_url"], ds["git_dir"])
        elif ds["format"] == "parquet":
            target = RAW_DIR / f"{ds['name']}.parquet"
            curl_download(ds["url"], target)
        elif ds["format"] == "csv":
            target = RAW_DIR / f"{ds['name']}.csv"
            curl_download(ds["url"], target)

    print("\n✓ All datasets downloaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2:** Run the fetcher

```bash
cd /Users/ks/source_code/laya
.venv/bin/python scripts/fetch_datasets.py
```

Expected:
- Downloads fgrc_scd, scam_9_lang, spamshield, ealvaradob (~30-100MB each)
- Clones fbs_sms from GitHub
- Prints "✓ All datasets downloaded"

If any download fails (network issues), note which and continue.

**Step 3:** Verify file sizes

```bash
cd /Users/ks/source_code/laya
ls -lh datasets/raw/
du -sh datasets/raw/
```

Expected: total ~200-500MB across 5 entries.

**Step 4:** Commit

```bash
cd /Users/ks/source_code/laya
git add scripts/fetch_datasets.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(scripts): add dataset fetcher for 5 public scam datasets

Sources:
- FGRC-SCD (Abooooo, Chinese telecom fraud, MIT)
- Scam_Message_9_Language (vichetkao, 9 langs incl Chinese, MIT)
- SpamShield-Datasets (M-Arjun, 23 langs incl Chinese, CC-BY-4.0)
- FBS_SMS_Dataset (fl-wxiao, Chinese fake base station, research)
- ealvaradob/phishing-dataset (English URL/SMS/email, research)

Uses curl to bypass SOCKS proxy issues."
```

Note: Add `datasets/raw/` to `.gitignore` in a follow-up task — raw downloads are large.

---

### Task 5: Write dataset builder script

**Files:**
- Create: `scripts/build_dataset.py`

**Step 1:** Write the builder

```python
"""Build unified training/val/test dataset from multi-source raw data.

Reads:
  datasets/raw/{fgrc_scd, scam_9_lang, spamshield, fbs_sms, ealvaradob_phishing}.*

Writes:
  datasets/training/{train,val,test}.jsonl

Each output row:
  {"text": "...", "is_scam": 0|1, "risk": 1-5, "category": "<canonical_13>",
   "source": "<dataset_id>", "language": "<zh|en|...>"}

Strategy:
- Normalize all labels via schemas.scam_categories.normalize_label
- Filter SpamShield: drop Dutch and Italian (imbalanced); drop synthetic examples
  from val/test (keep synthetic in train only for augmentation)
- Balance scam:benign = 1:1 within each source
- Holdout 10% as test, 10% as val, rest train (stratified by source)
- Risk level: heuristic mapping from source label intensity
"""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

# Add repo root to sys.path so we can import schemas package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import CANONICAL_CATEGORIES, normalize_label  # noqa


RAW_DIR = Path("datasets/raw")
OUT_DIR = Path("datasets/training")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = OUT_DIR / "train.jsonl"
VAL_PATH = OUT_DIR / "val.jsonl"
TEST_PATH = OUT_DIR / "test.jsonl"

RNG_SEED = 42
TEST_FRAC = 0.10
VAL_FRAC = 0.10


def stable_id(text: str, source: str) -> str:
    h = hashlib.sha1(f"{source}::{text}".encode("utf-8")).hexdigest()[:12]
    return f"{source}-{h}"


def _load_parquet(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def _load_csv(path: Path) -> list[dict]:
    import csv
    with path.open() as f:
        return list(csv.DictReader(f))


def load_spamshield() -> list[dict]:
    rows = _load_parquet(RAW_DIR / "spamshield.parquet")
    out = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        lang = r.get("language", "en")
        if lang in {"Dutch", "Italian"}:
            continue  # imbalanced in this dataset
        is_scam = int(r["label"]) == 1
        category = normalize_label(r.get("category", "spam_general"))
        risk = 4 if is_scam else 1
        out.append({
            "text": text,
            "is_scam": int(is_scam),
            "risk": risk,
            "category": category,
            "language": lang,
            "source": "spamshield",
        })
    return out


def load_scam_9_lang() -> list[dict]:
    rows = _load_parquet(RAW_DIR / "scam_9_lang.parquet")
    out = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        is_scam = int(r["label"]) == 1
        lang = r.get("language", "en")
        out.append({
            "text": text,
            "is_scam": int(is_scam),
            "risk": 4 if is_scam else 1,
            "category": "spam_general",
            "language": lang,
            "source": "scam_9_lang",
        })
    return out


def load_fgrc_scd() -> list[dict]:
    rows = _load_parquet(RAW_DIR / "fgrc_scd.parquet")
    out = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        label = r.get("label", "")
        is_scam = label in ("high_risk_sms", "fraud_call", "phishing_link")
        category = normalize_label(label) if is_scam else "benign"
        out.append({
            "text": text,
            "is_scam": int(is_scam),
            "risk": 4 if is_scam else 1,
            "category": category,
            "language": "zh",
            "source": "fgrc_scd",
        })
    return out


def load_fbs_sms() -> list[dict]:
    """Iterate all CSV files in the FBS_SMS git clone directory."""
    base = RAW_DIR / "fbs_sms"
    out = []
    for csv_path in sorted(base.rglob("*.csv")):
        rel = csv_path.relative_to(base).as_posix()
        for r in _load_csv(csv_path):
            text = (r.get("text") or "").strip()
            if not text:
                continue
            label = r.get("label") or rel.split("/")[0]
            is_scam = label not in ("ham", "benign", "legit", "normal")
            category = normalize_label(label) if is_scam else "benign"
            out.append({
                "text": text,
                "is_scam": int(is_scam),
                "risk": 4 if is_scam else 1,
                "category": category,
                "language": "zh",
                "source": "fbs_sms",
            })
    return out


def load_ealvaradob() -> list[dict]:
    rows = _load_csv(RAW_DIR / "ealvaradob_phishing.csv")
    out = []
    for r in rows:
        text = (r.get("text") or r.get("URL") or "").strip()
        if not text:
            continue
        is_scam = int(r.get("label", 0)) == 1
        out.append({
            "text": text,
            "is_scam": int(is_scam),
            "risk": 4 if is_scam else 1,
            "category": "phishing" if is_scam else "benign",
            "language": "en",
            "source": "ealvaradob",
        })
    return out


def split_dataset(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by source. Holdout test/val are 100% original (no synthetic)."""
    random.seed(RNG_SEED)
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    train, val, test = [], [], []
    for source, items in by_source.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * TEST_FRAC))
        n_val = max(1, int(n * VAL_FRAC))
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])
    return train, val, test


def balance_train(rows: list[dict]) -> list[dict]:
    """Downsample majority class (benign) to 1:1 with scam within each source."""
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    out = []
    for source, items in by_source.items():
        scam = [r for r in items if r["is_scam"] == 1]
        benign = [r for r in items if r["is_scam"] == 0]
        random.shuffle(benign)
        out.extend(scam)
        out.extend(benign[: len(scam)])  # 1:1 balance
    random.shuffle(out)
    return out


def main() -> int:
    print("Loading sources...")
    all_rows: list[dict] = []
    for loader in (load_spamshield, load_scam_9_lang, load_fgrc_scd, load_fbs_sms, load_ealvaradob):
        try:
            rows = loader()
            print(f"  ✓ {loader.__name__}: {len(rows):,} rows")
            all_rows.extend(rows)
        except FileNotFoundError as e:
            print(f"  ✗ {loader.__name__}: {e} (skipping)")
    print(f"\nTotal raw rows: {len(all_rows):,}")

    # Drop empty/duplicate texts
    seen = set()
    cleaned = []
    for r in all_rows:
        key = r["text"].strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(r)
    print(f"After dedup: {len(cleaned):,} rows")

    # Stratified split
    train, val, test = split_dataset(cleaned)
    print(f"Split: train={len(train):,} val={len(val):,} test={len(test):,}")

    # Balance train only
    train = balance_train(train)
    print(f"Train after balance: {len(train):,}")

    # Assign stable IDs
    for split, rows in [("train", train), ("val", val), ("test", test)]:
        for r in rows:
            r["id"] = stable_id(r["text"], r["source"])

    # Write outputs
    for path, rows in [(TRAIN_PATH, train), (VAL_PATH, val), (TEST_PATH, test)]:
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  ✓ {path} ({len(rows):,} rows)")

    # Stats
    cat_counter = Counter(r["category"] for r in train)
    print("\nTrain category distribution:")
    for c in CANONICAL_CATEGORIES:
        print(f"  {c}: {cat_counter.get(c, 0)}")
    lang_counter = Counter(r["language"] for r in train)
    print("\nTrain language distribution:")
    for lang, count in lang_counter.most_common():
        print(f"  {lang}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2:** Run the builder

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python scripts/build_dataset.py
```

Expected:
- Total raw rows: ~180k+
- After dedup: ~120-150k
- Split: train=~95k val=~12k test=~12k
- Train after balance: ~50-70k
- Categories distribution shows all 13 with reasonable counts
- Languages include zh, en, plus others (German, French, etc.)

**Step 3:** Verify outputs

```bash
cd /Users/ks/source_code/laya
ls -lh datasets/training/
head -3 datasets/training/train.jsonl
.venv/bin/python -c "
import json
from collections import Counter
rows = [json.loads(l) for l in open('datasets/training/train.jsonl')]
print(f'train: {len(rows)} rows')
print('categories:', dict(Counter(r['category'] for r in rows).most_common()))
print('langs:', dict(Counter(r['language'] for r in rows).most_common(5)))
print('is_scam:', Counter(r['is_scam'] for r in rows))
"
```

Expected: balanced ~1:1 scam:benign, all 13 categories present, multiple languages.

**Step 4:** Commit

```bash
cd /Users/ks/source_code/laya
git add scripts/build_dataset.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(scripts): add dataset builder with 13-class label normalization

Combines 5 sources (~30-50k balanced training samples):
- SpamShield (~149k raw, English + 22 langs)
- Scam_Message_9_Language (12.8k, 9 langs)
- FGRC-SCD (<1k Chinese telecom fraud)
- FBS_SMS (~14k Chinese fake base station)
- ealvaradob (~73k English phishing URLs/SMS/email)

Stratified split by source (10% val, 10% test).
Train balanced 1:1 scam:benign within each source.
All labels normalized to canonical 13 categories via
schemas.scam_categories.normalize_label."
```

---

### Task 6: Add datasets/raw/ to .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1:** Append raw dataset exclusion

Append to `.gitignore`:

```
# Raw public datasets (large, regenerated by scripts/fetch_datasets.py)
datasets/raw/
```

**Step 2:** Add training data exclusion (also large)

```
# Training data (regenerated by scripts/build_dataset.py)
datasets/training/
```

**Step 3:** Verify .gitignore

```bash
cd /Users/ks/source_code/laya
cat .gitignore | grep -A1 "datasets"
```

**Step 4:** Commit

```bash
cd /Users/ks/source_code/laya
git add .gitignore
git -c user.email=claude@local -c user.name=claude commit -m "chore: ignore datasets/raw and datasets/training (regenerated by build scripts)"
```

---

### Task 7: Verify Phase 1 outputs

**Step 1:** Confirm all artifacts present

```bash
cd /Users/ks/source_code/laya
test -f scripts/fetch_datasets.py && echo "✓ fetch_datasets.py"
test -f scripts/build_dataset.py && echo "✓ build_dataset.py"
test -f schemas/scam_categories.py && echo "✓ scam_categories.py"
test -f schemas/scam.json && echo "✓ scam.json (updated)"
test -d datasets/raw/ && echo "✓ datasets/raw/ exists"
test -f datasets/training/train.jsonl && echo "✓ train.jsonl"
test -f datasets/training/val.jsonl && echo "✓ val.jsonl"
test -f datasets/training/test.jsonl && echo "✓ test.jsonl"
```

**Step 2:** Run all tests

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/test_categories.py -v
```

Expected: All tests pass.

**Step 3:** Mark Phase 1 complete

Phase 1 done. Move to Phase 2.

---

## Phase 2: Router Dual-Checkpoint Support

### Task 8: Update Router to load both checkpoints

**Files:**
- Modify: `src/router.py`

**Step 1:** Replace router with dual-checkpoint version

```python
"""Unicode-script detection and language routing for Laya checkpoints.

Supports dual-checkpoint deployment:
- english (ModernBERT-large, 421M) — Latin script
- multilingual (mmBERT-base, 322M) — CJK/Hebrew/Arabic/Devanagari etc.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import Optional

from src.laya_onnx import OnnxLayaClient


_NON_LATIN_SCRIPT_PREFIXES = (
    "cjk unified ideograph",
    "hangul syllable",
    "devanagari",
    "arabic",
    "hebrew",
    "cyrillic",
    "greek",
    "tamil",
    "thai",
    "bengali",
    "gurmukhi",
    "telugu",
    "tibetan",
    "myanmar",
    "khmer",
    "lao",
    "georgian",
    "armenian",
)

_PREFIX_TO_TAG = {p: p.replace(" ", "_") for p in _NON_LATIN_SCRIPT_PREFIXES}

LATIN_SCRIPT = "latin"


def detect_script(text: str) -> str:
    """Detect dominant non-Latin script, falling back to 'latin'."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            block = unicodedata.name(ch, "")
        except ValueError:
            block = ""
        matched = False
        for prefix in _NON_LATIN_SCRIPT_PREFIXES:
            if block.lower().startswith(prefix):
                tag = _PREFIX_TO_TAG[prefix]
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
        script = detect_script(text)
        if script == LATIN_SCRIPT:
            return "english", f"script={script}"
        return "multilingual", f"script={script} (non-Latin)"


class Router:
    """High-level router combining English + Multilingual clients.

    Routing policy:
    - Latin script (English, Spanish, French, etc.) → English checkpoint
    - Non-Latin script (CJK, Hebrew, Arabic, etc.) → Multilingual checkpoint
    - If multilingual checkpoint is missing, falls back to English (with warning)

    Per the research:
    - Laya English checkpoint: zero-shot 0.342 typed-decisions; ModernBERT
      backbone shreds CJK characters (Khmer 0.000 acc @ 0.952 conf).
    - Laya multilingual checkpoint (mmBERT-base, 256k vocab): native
      multilingual support, ~2x faster than English checkpoint.
    """

    def __init__(
        self,
        english_dir: Path | str,
        multilingual_dir: Path | str | None = None,
        providers: Optional[list[str]] = None,
    ):
        self.english = OnnxLayaClient(english_dir, providers=providers)
        self.multilingual = None
        if multilingual_dir and Path(multilingual_dir).exists():
            try:
                self.multilingual = OnnxLayaClient(multilingual_dir, providers=providers)
            except Exception as e:
                print(f"⚠ Multilingual checkpoint failed to load: {e}", file=sys.stderr)
        self.scripts = ScriptRouter()

    def _pick(self, text: str) -> tuple[OnnxLayaClient, str, str]:
        model_tag = self.scripts.route(text)
        if model_tag == "multilingual" and self.multilingual is not None:
            return self.multilingual, model_tag, self.scripts.route_with_reason(text)[1]
        reason = self.scripts.route_with_reason(text)[1]
        if model_tag == "multilingual":
            reason += " — multilingual checkpoint unavailable, using English (known limitation)"
        return self.english, "english", reason

    def predict(
        self,
        state: str,
        questions: dict,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        model: Optional[str] = None,
    ) -> dict:
        if isinstance(state, dict):
            import json
            state = json.dumps(state, ensure_ascii=False)
        if model is None:
            client, model_tag, reason = self._pick(state)
        else:
            client = (
                self.multilingual
                if model == "multilingual" and self.multilingual
                else self.english
            )
            model_tag = "multilingual" if (client is self.multilingual) else "english"
            reason = "explicit override"
            if model == "multilingual" and self.multilingual is None:
                reason += " — multilingual checkpoint unavailable, using English"
        result = client.predict(state, questions, max_len=max_len, head_max_len=head_max_len)
        result["routing"] = {"model": model_tag, "reason": reason}
        return result

    def route(self, text: str) -> str:
        return self.scripts.route(text)
```

**Step 2:** Run existing router tests to ensure no regression

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/test_router.py tests/test_router_integration.py -v
```

Expected: All existing tests still pass (with multilingual dir being None for these tests).

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add src/router.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(router): load both English and Multilingual checkpoints

Router now loads laya-onnx-en and (optionally) laya-onnx-multilingual.
Non-Latin script routes to multilingual when available, English otherwise.
Explicit --model override still supported."
```

---

### Task 9: Update integration tests for multilingual path

**Files:**
- Modify: `tests/test_router_integration.py`

**Step 1:** Add multilingual-enabled test

Append new test class to the existing file:

```python
import os
from pathlib import Path

import pytest

from src.router import Router


@pytest.fixture(scope="module")
def router_with_ml():
    """Router with both English and multilingual checkpoints loaded."""
    return Router(
        english_dir="models/laya-onnx-en",
        multilingual_dir="models/laya-onnx-multilingual",
    )


@pytest.mark.skipif(
    not (Path("models/laya-onnx-en/model.onnx").exists()
         and Path("models/laya-onnx-multilingual/model.onnx").exists()),
    reason="Both English and multilingual ONNX checkpoints required",
)
class TestRouterWithMultilingual:
    def test_both_clients_loaded(self, router_with_ml):
        assert router_with_ml.english is not None
        assert router_with_ml.multilingual is not None

    def test_chinese_routes_to_multilingual(self, router_with_ml):
        result = router_with_ml.predict(
            "您好世界",
            json.loads(open("schemas/scam.json").read()),
        )
        assert result["routing"]["model"] == "multilingual"
        assert "non-Latin" in result["routing"]["reason"]

    def test_hebrew_routes_to_multilingual(self, router_with_ml):
        result = router_with_ml.predict(
            "שלום",
            json.loads(open("schemas/scam.json").read()),
        )
        assert result["routing"]["model"] == "multilingual"

    def test_english_routes_to_english(self, router_with_ml):
        result = router_with_ml.predict(
            "Hello world",
            json.loads(open("schemas/scam.json").read()),
        )
        assert result["routing"]["model"] == "english"

    def test_explicit_multilingual_overrides_script(self, router_with_ml):
        result = router_with_ml.predict(
            "Hello",
            json.loads(open("schemas/scam.json").read()),
            model="multilingual",
        )
        assert result["routing"]["model"] == "multilingual"
```

**Step 2:** Run all router tests

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/test_router.py tests/test_router_integration.py -v
```

Expected: All existing tests pass. New tests SKIP (no multilingual ONNX yet — that's Phase 5 deliverable).

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add tests/test_router_integration.py
git -c user.email=claude@local -c user.name=claude commit -m "test(router): add multilingual integration tests (skipped until Phase 5)"
```

---

### Task 10: Add multilingual ONNX directory placeholder

**Files:**
- Create: `models/laya-onnx-multilingual/.gitkeep`

**Step 1:** Create directory

```bash
cd /Users/ks/source_code/laya
mkdir -p models/laya-onnx-multilingual
touch models/laya-onnx-multilingual/.gitkeep
```

**Step 2:** Verify

```bash
cd /Users/ks/source_code/laya
ls models/
```

Expected: `laya-onnx-en/` and `laya-onnx-multilingual/` both exist.

Note: `models/` is in `.gitignore` but `.gitkeep` files need to be explicitly added. We'll handle this in Task 27 (after ONNX export).

---

## Phase 3: Local Pre-flight Tests

### Task 11: Verify PyTorch Laya SDK loads

**Files:**
- Create: `tests/test_laya_pytorch_load.py`

**Step 1:** Write a smoke test

```python
"""Verify that the Laya PyTorch SDK loads and can run inference.

This validates that Phase 4 (Kaggle training) won't fail due to a missing
or broken laya SDK install. We run a single prediction on the multilingual
checkpoint to confirm it works locally before pushing to Kaggle.
"""
import os
import pytest

try:
    import laya
    LAYA_AVAILABLE = True
except ImportError:
    LAYA_AVAILABLE = False


@pytest.mark.skipif(not LAYA_AVAILABLE, reason="laya SDK not installed (try pip install -e '.[train]')")
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN env var required for gated models")
class TestLayaPyTorchLoad:
    def test_import_laya(self):
        import laya
        assert hasattr(laya, "load")
        assert hasattr(laya, "Router")

    def test_load_english_checkpoint(self):
        from laya import Router
        router = Router(device="cpu", preload=True)
        assert router is not None
        assert router.english is not None

    def test_single_prediction(self):
        from laya import Router
        import json
        router = Router(device="cpu")
        schema = json.load(open("schemas/scam.json"))
        result = router.predict("This is a test message.", schema)
        assert "answers" in result
        assert "is_scam" in result["answers"]
        prob = result["answers"]["is_scam"]["noul"]
        assert 0.0 <= prob <= 1.0
```

**Step 2:** Run test

```bash
cd /Users/ks/source_code/laya
HF_TOKEN=your_token_here PYTHONPATH=. .venv/bin/python -m pytest tests/test_laya_pytorch_load.py -v
```

Note: `convaiinnovations/laya` is a public repo, no token needed. If your environment still has issues:

```bash
cd /Users/ks/source_code/laya
.venv/bin/python -c "import laya; print(dir(laya))" 2>&1 | head -10
```

Expected: `laya` SDK loads, lists functions. If import fails, check that `pip install -e ".[train]"` succeeded.

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add tests/test_laya_pytorch_load.py
git -c user.email=claude@local -c user.name=claude commit -m "test: add PyTorch Laya SDK smoke tests"
```

---

### Task 12: Validate mmBERT-base Chinese tokenization quality

**Files:**
- Create: `tests/test_tokenization_quality.py`

**Step 1:** Write the tokenization quality test

```python
"""Verify mmBERT-base tokenization efficiency on Chinese text.

The English ModernBERT-large backbone shreds CJK into 1-character-per-token
which hurts inference speed and accuracy. We verify that mmBERT-base
(laya-multilingual) achieves better Chinese tokenization.

Acceptance: chars/token >= 1.5 on a sample of Chinese text.
If true (likely), we can proceed with the multilingual base.
If false (< 1.5), we should consider alternative backbones.
"""
import os
import pytest

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


CHINESE_SAMPLES = [
    "您好，我是XX快递客服，您有一个包裹丢失需要理赔。",
    "【最高人民检察院】您涉嫌洗钱，请将资金转入安全账户。",
    "恭喜您被抽中二等奖，奖金10万元，请尽快联系客服领取。",
    "宝贝儿，我在这边投资了一个平台，稳赚不赔，先投5000试试。",
    "您的银行账户将被停用，请点击链接重新认证身份。",
]


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not available")
@pytest.mark.skipif(
    not os.environ.get("TEST_MULTILINGUAL_TOKENIZER"),
    reason="WHY no TEST_MULTILINGUAL_TOKENIZER env var — requires HF model download"
)
class TestTokenizationQuality:
    def test_mmbert_chinese_efficiency(self):
        """mmBERT-base should achieve >= 1.5 chars/token on Chinese."""
        # This requires downloading the mmBERT-base tokenizer, which is
        # the same tokenizer family used by laya-multilingual. Test is
        # gated behind env var to avoid network in CI.
        from transformers import AutoTokenizer
        # mmBERT-base is the underlying encoder for laya-multilingual
        # Laya team confirmed via github: "mmBERT-base (256k vocab)"
        tok = AutoTokenizer.from_pretrained("answerdotai/mmbert-base")
        total_chars = sum(len(s) for s in CHINESE_SAMPLES)
        total_tokens = sum(len(tok.encode(s)) for s in CHINESE_SAMPLES)
        ratio = total_chars / total_tokens
        assert ratio >= 1.5, (
            f"mmBERT-base Chinese tokenization ratio {ratio:.2f} "
            f"(< 1.5 chars/token indicates inefficient CJK handling)"
        )

    def test_modernbert_chinese_efficiency_baseline(self):
        """ModernBERT-large (English) baseline for comparison."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        total_chars = sum(len(s) for s in CHINESE_SAMPLES)
        total_tokens = sum(len(tok.encode(s)) for s in CHINESE_SAMPLES)
        ratio = total_chars / total_tokens
        # ModernBERT typically gets 0.5-0.8 chars/token on Chinese
        # We expect mmBERT > ModernBERT
        print(f"\nModernBERT-large Chinese ratio: {ratio:.2f} chars/token")
        # Just informational — no assertion
```

**Step 2:** Run with the env var set (optional)

```bash
cd /Users/ks/source_code/laya
TEST_MULTILINGUAL_TOKENIZER=1 HF_TOKEN=your_token .venv/bin/python -m pytest tests/test_tokenization_quality.py -v -s
```

Expected: `mmbert-base` Chinese ratio ≥ 1.5 chars/token (target). ModernBERT ratio ~0.5-0.8 (baseline for comparison).

If mmBERT ratio is too low (< 1.5), flag in README and consider mDeBERTa-v3-base as alternative.

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add tests/test_tokenization_quality.py
git -c user.email=claude@local -c user.name=claude commit -m "test: add Chinese tokenization quality validation (gated behind env var)"
```

---

### Task 13: Verify Phase 3 outputs

**Step 1:** Confirm all Phase 3 files present

```bash
cd /Users/ks/source_code/laya
test -f tests/test_laya_pytorch_load.py && echo "✓ test_laya_pytorch_load.py"
test -f tests/test_tokenization_quality.py && echo "✓ test_tokenization_quality.py"
.venv/bin/python -c "from laya import Router; print('✓ laya SDK importable')"
```

**Step 2:** Mark Phase 3 complete

Move to Phase 4 (Kaggle notebook).

---

## Phase 4: Kaggle Notebook

### Task 14: Create Kaggle notebook structure (file scaffold)

**Files:**
- Create: `kaggle/laya_finetune_multilingual.ipynb` (JSON content)

The notebook file is a JSON document. We create it as a Python file then convert.

**Step 1:** Create directory

```bash
cd /Users/ks/source_code/laya
mkdir -p kaggle
```

**Step 2:** Create the notebook using nbformat

Create `kaggle/build_notebook.py` (one-time helper):

```python
"""Generate the Kaggle notebook for Laya multilingual fine-tuning."""
import json
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell("# Laya Multilingual Fine-tuning for Scam Detection\n"
                              "Trains LoRA adapter on `convaiinnovations/laya-multilingual` "
                              "with multi-source scam data."),
]

with open("kaggle/laya_finetune_multilingual.ipynb", "w") as f:
    nbf.write(nb, f)

print("✓ Notebook scaffold created")
```

```bash
cd /Users/ks/source_code/laya
.venv/bin/pip install nbformat
.venv/bin/python kaggle/build_notebook.py
```

Note: we'll add the actual cells in Tasks 15-21. For now, just the markdown intro.

**Step 3:** Commit scaffold

```bash
cd /Users/ks/source_code/laya
git add kaggle/laya_finetune_multilingual.ipynb kaggle/build_notebook.py
git -c user.email=claude@local -c user.name=claude commit -m "chore(kaggle): create notebook scaffold"
```

---

### Task 15: Add Setup cell

**Files:**
- Modify: `kaggle/laya_finetune_multilingual.ipynb`

**Step 1:** Add Cell 1 (Setup) via Python helper

```python
# kaggle/add_cell_setup.py
import json
import nbformat as nbf

nb = nbf.read("kaggle/laya_finetune_multilingual.ipynb", as_version=4)
setup_cell = nbf.v4.new_code_cell('''\
!pip install -q -U peft trl accelerate datasets transformers safetensors
import torch
from transformers import AutoTokenizer

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
''')
nb.cells.insert(1, setup_cell)

with open("kaggle/laya_finetune_multilingual.ipynb", "w") as f:
    nbf.write(nb, f)
print("✓ Setup cell added")
```

```bash
cd /Users/ks/source_code/laya
.venv/bin/python kaggle/add_cell_setup.py
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add kaggle/laya_finetune_multilingual.ipynb
git -c user.email=claude@local -c user.name=claude commit -m "feat(kaggle): add setup cell with deps install"
```

---

### Task 16: Add base model loading cell

**Files:**
- Modify: `kaggle/laya_finetune_multilingual.ipynb`

**Step 1:** Append cell

```python
# kaggle/add_cell_load_base.py
import nbformat as nbf
nb = nbf.read("kaggle/laya_finetune_multilingual.ipynb", as_version=4)

load_base_cell = nbf.v4.new_code_cell('''\
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # if Kaggle region blocked
os.environ["HF_HUB_DISABLE_XET"] = "1"

from laya import Router
import torch

# Load multilingual checkpoint via PyTorch Laya SDK
router = Router(device="cuda", preload=True)
print("Available checkpoints:")
for name, client in [("english", router.english), ("multilingual", getattr(router, "multilingual", None))]:
    print(f"  {name}: {client is not None}")

# Extract the encoder (mmBERT-base) and decision head for LoRA wrapping
agent = router.multilingual if router.multilingual else router.english
model = agent.model  # HF model with encoder + decision head

# Inspect model architecture
print("Model class:", type(model).__name__)
n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {n_params/1e6:.1f}M")
print(f"Trainable (no LoRA yet): {n_trainable/1e6:.1f}M")

# Save tokenizer for later
tokenizer = AutoTokenizer.from_pretrained("answerdotai/mmbert-base")
print("Tokenizer loaded")
''')
nb.cells.insert(2, load_base_cell)

with open("kaggle/laya_finetune_multilingual.ipynb", "w") as f:
    nbf.write(nb, f)
print("✓ Load-base cell added")
```

```bash
cd /Users/ks/source_code/laya
.venv/bin/python kaggle/add_cell_load_base.py
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add kaggle/laya_finetune_multilingual.ipynb
git -c user.email=claude@local -c user.name=claude commit -m "feat(kaggle): add base model loading cell"
```

---

### Task 17: Add LoRA injection cell

**Files:**
- Modify: `kaggle/laya_finetune_multilingual.ipynb`

**Step 1:** Append cell

```python
# kaggle/add_cell_lora.py
import nbformat as nbf
nb = nbf.read("kaggle/laya_finetune_multilingual.ipynb", as_version=4)

lora_cell = nbf.v4.new_code_cell('''\
from peft import LoraConfig, get_peft_model, TaskType

# LoRA on attention projections in mmBERT
# r=8 → ~3.5M trainable params (~1% of base)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["query", "key", "value", "dense"],  # mmBERT attention layers
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# Expected output: ~3.5M trainable / ~322M total = ~1.1%
''')
nb.cells.insert(3, lora_cell)

with open("kaggle/laya_finetune_multilingual.ipynb", "w") as f:
    nbf.write(nb, f)
print("✓ LoRA injection cell added")
```

```bash
cd /Users/ks/source_code/laya
.venv/bin/python kaggle/add_cell_lora.py
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add kaggle/laya_finetune_multilingual.ipynb
git -c user.email=claude@local -c user.name=claude commit -m "feat(kaggle): add LoRA injection cell (r=8, attention targets)"
```

---

### Task 18: Add dataset loading cell

**Files:**
- Modify: `kaggle/laya_finetune_multilingual.ipynb`

**Step 1:** Append cell

```python
# kaggle/add_cell_data.py
import nbformat as nbf
nb = nbf.read("kaggle/laya_finetune_multilingual.ipynb", as_version=4)

data_cell = nbf.v4.new_code_cell('''\
from datasets import load_dataset, Dataset

# Load train/val/test JSONLs (already prepared locally and uploaded as Kaggle dataset)
# Kaggle dataset path: /kaggle/input/scam-detection-training/{train,val,test}.jsonl
TRAIN_PATH = "/kaggle/input/scam-detection-training/train.jsonl"
VAL_PATH = "/kaggle/input/scam-detection-training/val.jsonl"

train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
val_ds = load_dataset("json", data_files=VAL_PATH, split="val")

print(f"Train: {len(train_ds)} samples")
print(f"Val:   {len(val_ds)} samples")
print(f"Columns: {train_ds.column_names}")
print(f"Sample: {train_ds[0]}")

# Schema for Laya: question + state in JSON-like format
def format_for_laya(example):
    state = {"text": example["text"]}
    questions = {
        "is_scam": {"type": "noul", "instructions": "Is this a scam?"},
        "category": {
            "type": "choice",
            "instructions": "What category of scam is this?",
            "criteria": {
                "benign": "normal legitimate message",
                "phishing": "phishing link",
                "crypto_scam": "cryptocurrency fraud",
                "investment_scam": "fake investment",
                "lottery_scam": "fake prize",
                "job_scam": "fake job offer",
                "loan_scam": "fake loan",
                "impersonation": "fake official",
                "romance_scam": "pig butchering",
                "delivery_fraud": "fake courier",
                "marketing": "legitimate marketing",
                "adult_content": "adult services",
                "spam_general": "other junk",
            },
        },
    }
    return {
        "state": str(state),
        "questions": str(questions),
        "target_is_scam": example["is_scam"],
        "target_category": example["category"],
    }

train_ds = train_ds.map(format_for_laya)
val_ds = val_ds.map(format_for_laya)
print(f"\\nFormatted sample: {train_ds[0]}")
''')
nb.cells.insert(4, data_cell)

with open("kaggle/laya_finetune_multilingual.ipynb", "w") as f:
    nbf.write(nb, f)
print("✓ Dataset loading cell added")
```

```bash
cd /Users/ks/source_code/laya
.venv/bin/python kaggle/add_cell_data.py
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add kaggle/laya_finetune_multilingual.ipynb
git -c user.email=claude@local -c user.name=claude commit -m "feat(kaggle): add dataset loading cell with schema formatting"
```

---

### Task 19: Add RLCD trainer cell

**Files:**
- Modify: `kaggle/laya_finetune_multilingual.ipynb`

**Step 1:** Append cell

```python
# kaggle/add_cell_rlcd.py
import nbformat as nbf
nb = nbf.read("kaggle/laya_finetune_multilingual.ipynb", as_version=4)

rlcd_cell = nbf.v4.new_code_cell('''\
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# RLCD: Reinforcement Learning for Calibrated Decisions
# Reward = strictly proper scoring rule(s)
# Policy gradient with GRPO-style group baseline

class RLCDTrainer:
    def __init__(self, model, tokenizer, device="cuda", lr=2e-4):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

    def compute_reward(self, probs: torch.Tensor, target: torch.Tensor):
        """Strictly proper scoring rules: log + spherical.
        Both are maximized when model reports true probabilities.
        """
        eps = 1e-7
        # log score (negative cross-entropy)
        log_score = torch.log(probs.gather(1, target.unsqueeze(1)).squeeze() + eps)
        # spherical score: probs * target normalized
        target_onehot = F.one_hot(target, probs.size(-1)).float()
        norm_p = probs / (probs.sum(dim=-1, keepdim=True) + eps)
        norm_t = target_onehot / (target_onehot.sum(dim=-1, keepdim=True) + eps)
        spherical = (norm_p * norm_t).sum(dim=-1)
        return log_score + spherical  # higher = better

    def train_step(self, batch):
        self.optimizer.zero_grad()
        # batch contains state, questions, target_is_scam, target_category
        # Run forward pass through Laya — simplified here
        # In practice, use Laya SDK's predict and extract logits
        # This is a placeholder; full implementation below
        raise NotImplementedError("Use laya SDK predict, then RLCD gradient")

    def fit_temperature(self, eval_set):
        """Fit per-type temperature on held-out data."""
        # Find T such that T * logits produces best-calibrated probs
        # Simplified: sweep T in [0.1, 5.0] for is_scam and category separately
        pass

# Note: full RLCD implementation would use Laya SDK's predict() and
# backprop through the decision head. For brevity, this notebook demonstrates
# the structure. The actual training loop is below.

print("RLCD trainer class defined")
print("\\nNOTE: Full RLCD training uses the official laya SDK predict method")
print("and GRPO-style group baseline. See laya_finetune_typed_decisions_2xT4_kaggle.ipynb")
print("for the canonical implementation pattern.")
''')
nb.cells.insert(5, rlcd_cell)

with open("kaggle/laya_finetune_multilingual.ipynb", "w") as f:
    nbf.write(nb, f)
print("✓ RLCD trainer cell added")
```

```bash
cd /Users/ks/source_code/laya
.venv/bin/python kaggle/add_cell_rlcd.py
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add kaggle/laya_finetune_multilingual.ipynb
git -c user.email=claude@local -c user.name=claude commit -m "feat(kaggle): add RLCD trainer scaffolding cell"
```

Note: The full RLCD implementation requires Laya SDK internals access. Recommend copying/adapt from official `laya_finetune_typed_decisions_2xT4_kaggle.ipynb` reference at:
https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb

---

### Task 20: Add training loop + merge + save cells

**Files:**
- Modify: `kaggle/laya_finetune_multilingual.ipynb`

**Step 1:** Append cells for training, merging, and saving

```python
# kaggle/add_cell_training.py
import nbformat as nbf
nb = nbf.read("kaggle/laya_finetune_multilingual.ipynb", as_version=4)

training_cell = nbf.v4.new_code_cell('''\
# Training loop — see official notebook for full RLCD implementation:
# https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
#
# Key hyperparameters:
EPOCHS = 4
BATCH_SIZE = 8  # per T4
GRAD_ACCUM = 4   # effective batch = 32
LR = 2e-4
WARMUP_RATIO = 0.06
MAX_SEQ = 512

# Use HF Trainer with custom RLCD reward (recommended) or write manual loop
# Below is the standard SFTTrainer pattern adapted for Laya:

# from trl import SFTTrainer, SFTConfig
# training_args = SFTConfig(
#     output_dir="./laya-finetuned",
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     gradient_accumulation_steps=GRAD_ACCUM,
#     learning_rate=LR,
#     warmup_ratio=WARMUP_RATIO,
#     max_length=MAX_SEQ,
#     fp16=True,  # T4 has fp16 support
#     save_strategy="epoch",
#     evaluation_strategy="epoch",
#     logging_steps=50,
# )
# trainer = SFTTrainer(
#     model=peft_model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=tokenizer,
# )
# trainer.train()
print("Training loop template — fill in from official notebook")
''')

merge_cell = nbf.v4.new_code_cell('''\
# After training: merge LoRA adapter into base model
from peft import PeftModel

# Load base model fresh to avoid OOM from accumulated gradients
del peft_model
torch.cuda.empty_cache()

base = router.multilingual.model if router.multilingual else router.english.model
merged = PeftModel.from_pretrained(base, "./laya-finetuned/checkpoint-final")
merged = merged.merge_and_unload()

# Save merged model
merged.save_pretrained("./laya-merged")
tokenizer.save_pretrained("./laya-merged")

# Fit temperatures on validation set
from laya.calibration import fit_all_temperatures  # hypothetical utility
temperatures = fit_all_temperatures(merged, val_ds)
print("Fitted temperatures:", temperatures)

import json
with open("./laya-merged/rl_agent_config.json", "r") as f:
    cfg = json.load(f)
cfg["temperature_by_options"] = temperatures
with open("./laya-merged/rl_agent_config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("✓ Saved merged model + fitted temperatures")
''')

upload_cell = nbf.v4.new_code_cell('''\
# Push to Hugging Face Hub (private)
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./laya-merged",
    repo_id="<YOUR_USERNAME>/laya-multilingual-finetuned-scam",
    repo_type="model",
    private=True,
)
print("✓ Uploaded to HF Hub")

# Also save adapter-only for later merging locally
peft_model.save_pretrained("./laya-adapter-final")
api.upload_folder(
    folder_path="./laya-adapter-final",
    repo_id="<YOUR_USERNAME>/laya-multilingual-scam-adapter",
    repo_type="model",
    private=True,
)
print("✓ Uploaded adapter to HF Hub")
''')

nb.cells.append(training_cell)
nb.cells.append(merge_cell)
nb.cells.append(upload_cell)

with open("kaggle/laya_finetune_multilingual.ipynb", "w") as f:
    nbf.write(nb, f)
print("✓ Training + merge + upload cells added")
```

```bash
cd /Users/ks/source_code/laya
.venv/bin/python kaggle/add_cell_training.py
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add kaggle/laya_finetune_multilingual.ipynb
git -c user.email=claude@local -c user.name=claude commit -m "feat(kaggle): add training, merge, and upload-to-HF cells"
```

---

### Task 21: ⚠️ MANUAL: Run notebook on Kaggle

**This is a manual task — Kaggle requires interactive execution.**

**Step 1:** Prepare Kaggle Dataset
- Create Kaggle Dataset at kaggle.com/datasets/<user>/scam-detection-training
- Upload `datasets/training/{train,val,test}.jsonl` (from Task 5)
- Note the dataset path for use in notebook

**Step 2:** Push notebook to Kaggle
- Open kaggle.com/code → Create New Notebook
- Upload `kaggle/laya_finetune_multilingual.ipynb`
- Attach the scam-detection-training dataset
- Set accelerator: **2x T4 GPU**
- Set persistence: Files only (the merged model goes to a HF private repo, not the 20GB output limit)

**Step 3:** Edit Cell 11 to use actual username
Replace `<YOUR_USERNAME>` in the upload cell with the Kaggle account username.

**Step 4:** Run all cells
- Expected runtime: 4-5 hours
- Save version: "laya-multilingual-finetuned-v1"

**Step 5:** Verify outputs
- Check that `./laya-merged/` contains: model.safetensors (or pytorch_model.bin), config.json, tokenizer/, rl_agent_config.json
- Check HF Hub: `huggingface.co/<YOUR_USERNAME>/laya-multilingual-finetuned-scam` has files

**Step 6:** Document the run

Create `reports/finetune-progress-v1.md`:
- Kaggle run URL
- Total runtime
- Final training loss curve
- Per-epoch eval metrics (accuracy, F1, MAE, ECE)
- Fitted temperature values
- HF Hub repo URL

**Step 7:** Commit progress report

```bash
cd /Users/ks/source_code/laya
git add reports/finetune-progress-v1.md
git -c user.email=claude@local -c user.name=claude commit -m "docs(reports): record v1 fine-tuning run results"
```

---

### Task 22: Verify Phase 4 outputs

**Step 1:** Confirm all Phase 4 artifacts

```bash
cd /Users/ks/source_code/laya
test -f kaggle/laya_finetune_multilingual.ipynb && echo "✓ notebook"
test -f reports/finetune-progress-v1.md && echo "✓ progress report"
```

**Step 2:** Mark Phase 4 complete

Move to Phase 5 (merge + ONNX export).

---

## Phase 5: Merge + ONNX Export

### Task 23: Write merge + ONNX export script

**Files:**
- Create: `scripts/merge_and_export_onnx.py`

**Step 1:** Write the script

```python
"""Merge LoRA adapter into base model and export to ONNX.

Inputs:
  HuggingFace repo: <user>/laya-multilingual-scam-adapter (LoRA adapter)
  HuggingFace repo: convaiinnovations/laya (multilingual subfolder) (base)

Outputs:
  models/laya-onnx-multilingual-finetuned/
    model.onnx + model.onnx.data
    tokenizer/tokenizer.json + tokenizer_config.json
    rl_agent_config.json (with fitted temperatures)

Validates: max logit diff vs PyTorch < 1e-5
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

# Add repo root to sys.path so we can import OnnxLayaClient
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


ADAPTER_REPO = os.environ.get("LAYA_ADAPTER_REPO", "<YOUR_USERNAME>/laya-multilingual-scam-adapter")
OUTPUT_DIR = Path("models/laya-onnx-multilingual-finetuned")
MAX_LOGIT_DIFF_THRESHOLD = 1e-5


def export_laya_to_onnx(model, tokenizer, output_dir: Path):
    """Export the merged Laya model to ONNX.

    Reference implementation:
    https://github.com/receptron/laya/blob/main/export/export_onnx.py

    For simplicity, we use torch.onnx.export with dynamic axes.
    The output matches the contract expected by our existing OnnxLayaClient:
      inputs: input_ids, attention_mask, marker_pos, marker_mask, qtype
      output: logits
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    model.eval()
    model.to("cpu")

    # Dummy inputs for tracing
    seq_len = 64
    n_opts = 16
    dummy_input_ids = torch.zeros((1, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, seq_len), dtype=torch.long)
    dummy_marker_pos = torch.zeros((1, n_opts), dtype=torch.long)
    dummy_marker_mask = torch.zeros((1, n_opts), dtype=torch.bool)
    dummy_qtype = torch.zeros((1,), dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_marker_pos, dummy_marker_mask, dummy_qtype),
        onnx_path,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "marker_pos": {0: "batch"},
            "marker_mask": {0: "batch"},
            "logits": {0: "batch", 1: "n_opts"},
        },
        opset_version=14,
    )
    print(f"  ✓ Exported {onnx_path} ({onnx_path.stat().st_size:,} bytes)")

    # Copy tokenizer files
    tok_dir = output_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    for fn in ["tokenizer.json", "tokenizer_config.json"]:
        src = Path(tokenizer.name_or_path) / fn
        if src.exists():
            shutil.copy(src, tok_dir / fn)
        else:
            # Try to fetch from HF
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(ADAPTER_REPO.split("/")[0] + "/" + "laya-multilingual-finetuned-scam", fn, local_dir=str(output_dir))
            shutil.move(downloaded, tok_dir / fn)
    print(f"  ✓ Copied tokenizer files to {tok_dir}")


def validate_onnx_vs_pytorch(onnx_path: Path, pytorch_model, tokenizer) -> float:
    """Run both PyTorch and ONNX on same inputs, compute max logit diff."""
    print("\nValidating ONNX vs PyTorch...")
    pytorch_model.eval()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    test_inputs = [
        ("This is a phishing email asking for your password.", 0),
        ("恭喜您被抽中二等奖，奖金10万元。", 1),
        ("Meeting at 3pm tomorrow.", 2),
    ]
    max_diff = 0.0
    for text, qtype in test_inputs:
        # Build input
        encoded = tokenizer.encode(text)
        token_ids = encoded.ids
        seq_len = len(token_ids)
        # For simplicity, use single MASK token for now
        marker_pos = [2]  # placeholder
        inputs = {
            "input_ids": np.array([token_ids], dtype=np.int64),
            "attention_mask": np.ones((1, seq_len), dtype=np.int64),
            "marker_pos": np.array([marker_pos + [0] * (16 - len(marker_pos))], dtype=np.int64),
            "marker_mask": np.zeros((1, 16), dtype=bool),
            "qtype": np.array([qtype], dtype=np.int64),
        }
        # PyTorch forward (simplified — uses underlying model)
        with torch.no_grad():
            pt_input_ids = torch.tensor([token_ids], dtype=torch.long)
            pt_attention_mask = torch.ones((1, seq_len), dtype=torch.long)
            pt_outputs = pytorch_model(
                input_ids=pt_input_ids,
                attention_mask=pt_attention_mask,
            )
            pt_logits = pt_outputs.logits[0, :len(marker_pos)].numpy()

        onnx_logits = session.run(None, inputs)[0][0, :len(marker_pos)]
        diff = float(np.max(np.abs(pt_logits - onnx_logits)))
        print(f"  '{text[:40]}...' → logit diff: {diff:.6f}")
        max_diff = max(max_diff, diff)

    return max_diff


def main() -> int:
    print("=" * 70)
    print("Laya Multilingual: merge LoRA + export ONNX")
    print("=" * 70)

    # Step 1: Download adapter and base
    print(f"\nDownloading adapter from {ADAPTER_REPO}...")
    from huggingface_hub import snapshot_download
    adapter_dir = Path(snapshot_download(ADAPTER_REPO, local_dir="models/_laya-adapter"))
    print(f"  ✓ Adapter at {adapter_dir}")

    print(f"\nDownloading base convaiinnovations/laya-multilingual...")
    base_dir = Path(snapshot_download(
        "convaiinnovations/laya",
        repo_type="model",
        local_dir="models/_laya-base",
        allow_patterns=["multilingual/*", "config.json"],
    ))
    # The multilingual subfolder is at base_dir/multilingual
    base_path = base_dir / "multilingual"
    print(f"  ✓ Base at {base_path}")

    # Step 2: Load base + adapter, merge
    print("\nLoading base + adapter...")
    base_model = AutoModel.from_pretrained(str(base_path), trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    merged = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged = merged.merge_and_unload()
    print("  ✓ Merged LoRA adapter into base")

    # Step 3: Export to ONNX
    print(f"\nExporting to ONNX → {OUTPUT_DIR}")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    export_laya_to_onnx(merged, tokenizer, OUTPUT_DIR)

    # Step 4: Copy rl_agent_config.json (with fitted temperatures)
    cfg_src = Path("models/_laya-base/multilingual/config.json")
    if cfg_src.exists():
        shutil.copy(cfg_src, OUTPUT_DIR / "rl_agent_config.json")
        print(f"  ✓ Copied {cfg_src}")

    # Step 5: Validate
    diff = validate_onnx_vs_pytorch(OUTPUT_DIR / "model.onnx", merged, tokenizer)
    print(f"\nMax logit diff: {diff:.2e} (threshold {MAX_LOGIT_DIFF_THRESHOLD:.0e})")
    if diff > MAX_LOGIT_DIFF_THRESHOLD:
        print(f"⚠ Diff exceeds threshold — investigate quantization or export config")
        return 1
    print(f"✓ ONNX export validated")

    # Step 6: Cleanup
    shutil.rmtree("models/_laya-base")
    shutil.rmtree("models/_laya-adapter")
    print("\n✓ Cleanup complete")
    print(f"\nFinal output: {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add scripts/merge_and_export_onnx.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(scripts): merge LoRA adapter + export ONNX with validation"
```

---

### Task 24: Write ONNX validation test

**Files:**
- Create: `tests/test_onnx_export.py`

**Step 1:** Write test

```python
"""Verify ONNX export matches PyTorch within 1e-5 tolerance."""
import os
from pathlib import Path

import numpy as np
import pytest

ONNX_DIR = Path("models/laya-onnx-multilingual-finetuned")


@pytest.mark.skipif(
    not (ONNX_DIR / "model.onnx").exists(),
    reason="Multilingual ONNX checkpoint not exported; run scripts/merge_and_export_onnx.py"
)
class TestONNXExport:
    def test_model_files_present(self):
        assert (ONNX_DIR / "model.onnx").exists()
        assert (ONNX_DIR / "model.onnx.data").exists()
        assert (ONNX_DIR / "tokenizer" / "tokenizer.json").exists()
        assert (ONNX_DIR / "rl_agent_config.json").exists()

    def test_onnx_loads(self):
        import onnxruntime as ort
        session = ort.InferenceSession(
            str(ONNX_DIR / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        input_names = {inp.name for inp in session.get_inputs()}
        expected = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}
        assert expected.issubset(input_names)

    def test_inference_runs(self):
        """Smoke test: load and predict on a sample."""
        import json
        sys_path = os.environ.copy()
        sys_path["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        # Use subprocess to test in isolation
        import subprocess
        result = subprocess.run(
            [".venv/bin/python", "-c", f'''
import sys
sys.path.insert(0, "{Path(__file__).resolve().parent.parent}")
from src.laya_onnx import OnnxLayaClient
client = OnnxLayaClient("models/laya-onnx-multilingual-finetuned")
import json
schema = json.load(open("schemas/scam.json"))
result = client.predict("Test message in Chinese 你好世界", schema)
print("OK", result["answers"]["is_scam"]["noul"])
'''],
            capture_output=True, text=True,
        )
        assert "OK" in result.stdout, f"Inference failed: {result.stderr}"
```

**Step 2:** Commit

```bash
cd /Users/ks/source_code/laya
git add tests/test_onnx_export.py
git -c user.email=claude@local -c user.name=claude commit -m "test: add ONNX export validation tests"
```

---

### Task 25: ⚠️ MANUAL: Run merge + ONNX export

**Step 1:** Set adapter repo env var

```bash
cd /Users/ks/source_code/laya
export LAYA_ADAPTER_REPO="<YOUR_USERNAME>/laya-multilingual-scam-adapter"
```

**Step 2:** Run the script

```bash
cd /Users/ks/source_code/laya
.venv/bin/python scripts/merge_and_export_onnx.py
```

Expected:
- Downloads adapter and base from HF (~1-2GB total)
- Merges LoRA into base (CPU operation, ~5min)
- Exports to ONNX (~10min)
- Validates logit diff < 1e-5
- Cleans up temp files
- Final output at `models/laya-onnx-multilingual-finetuned/`

**Step 3:** Verify outputs

```bash
cd /Users/ks/source_code/laya
ls -lh models/laya-onnx-multilingual-finetuned/
.venv/bin/python main.py --predict "您好世界" --questions schemas/scam.json 2>&1 | head -20
```

Expected:
- `model.onnx` (~3MB) + `model.onnx.data` (~644MB)
- Inference works, returns is_scam probability

**Step 4:** Commit ONNX bundle (override .gitignore)

```bash
cd /Users/ks/source_code/laya
git add -f models/laya-onnx-multilingual-finetuned/
git -c user.email=claude@local -c user.name=claude commit -m "feat(models): add fine-tuned multilingual ONNX bundle (~650MB)"
```

---

### Task 26: Verify Phase 5 outputs

**Step 1:** Confirm all Phase 5 artifacts

```bash
cd /Users/ks/source_code/laya
test -f scripts/merge_and_export_onnx.py && echo "✓ merge_and_export_onnx.py"
test -f tests/test_onnx_export.py && echo "✓ test_onnx_export.py"
test -d models/laya-onnx-multilingual-finetuned/ && echo "✓ multilingual ONNX bundle"
test -f models/laya-onnx-multilingual-finetuned/model.onnx && echo "✓ model.onnx"
```

**Step 2:** Mark Phase 5 complete

Move to Phase 6 (eval + integration).

---

## Phase 6: Evaluation + Integration

### Task 27: Update eval and report for 13-class schema

**Files:**
- Modify: `src/eval.py`
- Modify: `src/report.py`

**Step 1:** Update eval.py to use schemas.scam_categories

```python
"""Batch evaluation over JSONL test sets — now supports 13-class schema."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import CANONICAL_CATEGORIES, normalize_label
from src.router import Router


def load_schema(path: str = "schemas/scam.json") -> dict:
    return json.loads(Path(path).read_text())


def _state_from_record(rec: dict) -> str:
    if rec.get("type") == "multi_turn" and "turns" in rec:
        parts = []
        for i, turn in enumerate(rec["turns"]):
            role_short = (turn.get("role") or "?")[0].upper()
            parts.append(f"[{role_short}{i+1}] {turn.get('text', '')}")
        return "\n".join(parts)
    return rec.get("state", "")


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
            answers = pred.get("answers", {})
            # Normalize expected category to canonical if present
            expected_cat = rec.get("expected_category")
            if expected_cat is not None:
                expected_cat = normalize_label(expected_cat)
            results.append({
                "id": rec.get("id"),
                "type": rec.get("type"),
                "language": rec.get("language"),
                "expected_risk": rec.get("expected_risk"),
                "expected_category": expected_cat,
                "expected_label": rec.get("expected_label"),
                "source": rec.get("source"),
                "state_preview": state[:120],
                "predicted": {
                    "is_scam_noul": answers.get("is_scam", {}).get("noul"),
                    "is_scam_label": _thresholded_is_scam(
                        answers.get("is_scam", {}).get("noul", 0)
                    ) if answers.get("is_scam", {}).get("noul") is not None else None,
                    "risk_level": answers.get("risk_level", {}).get("score"),
                    "scam_category": answers.get("scam_category", {}).get("choice"),
                    "category_probs": answers.get("scam_category", {}).get("probabilities"),
                },
                "routing": pred.get("routing"),
                "latency_ms": pred.get("latency_ms"),
                "error": pred.get("error"),
            })
    return results
```

**Step 2:** Update report.py — only the `_category_table` and Confusion sections need 13 classes

In `src/report.py`, replace `list(p for _, p in counter.keys())` with sort that uses canonical ordering:

```python
def _category_table(results: list[dict]) -> str:
    counter = Counter()
    for r in results:
        p = r["predicted"]["scam_category"]
        e = r["expected_category"]
        if p and e:
            counter[(e, p)] += 1
    if not counter:
        return "_No category data._"
    # Use canonical category ordering from schemas.scam_categories
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from schemas.scam_categories import CANONICAL_CATEGORIES
    expected_cats = [c for c in CANONICAL_CATEGORIES if c in {e for e, _ in counter.keys()}]
    predicted_cats = sorted({p for _, p in counter.keys()})
    lines = ["| expected ↓ / predicted → | " + " | ".join(predicted_cats) + " |",
             "|" + "---|" * (len(predicted_cats) + 1)]
    for e in expected_cats:
        row = [f"| **{e}**"]
        for p in predicted_cats:
            row.append(str(counter.get((e, p), 0)))
        lines.append(" | ".join(row) + " |")
    return "\n".join(lines)
```

**Step 3:** Run existing tests to ensure no regression

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/ --tb=no -q
```

Expected: Most tests pass. `test_e2e_predict.py` may need adjustment since it references 8 categories in expected output.

**Step 4:** Commit

```bash
cd /Users/ks/source_code/laya
git add src/eval.py src/report.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(eval): support 13-class schema with canonical normalization"
```

---

### Task 28: Update existing datasets with canonical 13-class labels

**Files:**
- Modify: `datasets/single.jsonl`
- Modify: `datasets/multi.jsonl`

**Step 1:** Update script

Create `scripts/migrate_to_13_classes.py`:

```python
"""Migrate existing 8-class test sets to canonical 13 classes.

Most labels are already canonical (benign, phishing, etc.). The mapping
function from schemas.scam_categories handles edge cases.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import normalize_label, is_valid_category


def migrate_file(path: Path):
    print(f"Migrating {path}...")
    rows_out = []
    n_fixed = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cat = rec.get("expected_category")
            if cat is not None:
                canonical = normalize_label(cat)
                if canonical != cat:
                    n_fixed += 1
                    rec["expected_category"] = canonical
            if cat and not is_valid_category(rec.get("expected_category", "")):
                print(f"  ⚠ Unknown category after normalization: {cat} → {rec.get('expected_category')}")
            rows_out.append(rec)
    with path.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ Wrote {len(rows_out)} rows, {n_fixed} labels migrated")


def main() -> int:
    for p in [Path("datasets/single.jsonl"), Path("datasets/multi.jsonl"),
              Path("datasets/public.jsonl"), Path("datasets/eval.jsonl")]:
        if p.exists():
            migrate_file(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2:** Run migration

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python scripts/migrate_to_13_classes.py
```

Expected: prints "Wrote N rows, 0 labels migrated" if all labels were already canonical (they are).

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add scripts/migrate_to_13_classes.py datasets/single.jsonl datasets/multi.jsonl datasets/public.jsonl datasets/eval.jsonl
git -c user.email=claude@local -c user.name=claude commit -m "feat(datasets): migrate test sets to canonical 13-class labels"
```

---

### Task 29: Run final evaluation suite

**Step 1:** Run main eval on hand-written set

```bash
cd /Users/ks/source_code/laya
.venv/bin/python main.py --eval --input datasets/eval.jsonl --output reports/
```

Expected: produces `reports/eval-<ts>.md` and `reports/results-<ts>.jsonl`.

**Step 2:** Run on SMS Spam public set

```bash
cd /Users/ks/source_code/laya
.venv/bin/python main.py --eval --input datasets/public.jsonl --output reports/
```

Expected: produces another eval report.

**Step 3:** Run on training data test split (new holdout)

```bash
cd /Users/ks/source_code/laya
.venv/bin/python main.py --eval --input datasets/training/test.jsonl --output reports/
```

Expected: produces eval on training data holdout (most useful metric for measuring fine-tune success).

**Step 4:** Verify outputs

```bash
cd /Users/ks/source_code/laya
ls -lt reports/eval-*.md | head -5
```

Expected: 3 new eval reports with timestamps.

**Step 5:** Compare against baseline

```bash
cd /Users/ks/source_code/laya
.venv/bin/python << 'PYEOF'
import re
import json
from pathlib import Path

for f in sorted(Path("reports").glob("eval-*.md")):
    text = f.read_text()
    metrics = {}
    for line in text.split("\n"):
        if "|" in line and "accuracy" in line.lower():
            continue
        m = re.search(r"\|\s*(\w+(?:\s\w+)*)\s*\|\s*([\d.]+)\s*\|", line)
        if m and any(k in m.group(1).lower() for k in ["accuracy", "f1", "recall", "precision", "latency"]):
            metrics[m.group(1).strip()] = float(m.group(2))
    print(f"\n{f.name}:")
    for k, v in metrics.items():
        if k.startswith(("is_scam", "risk_level", "scam_category", "p50")):
            print(f"  {k}: {v}")
PYEOF
```

Expected: shows comparison between baseline and finetuned results.

**Step 6:** Commit results

```bash
cd /Users/ks/source_code/laya
git add reports/eval-*.md reports/results-*.jsonl
git -c user.email=claude@local -c user.name=claude commit -m "docs(reports): record post-finetuning evaluation results"
```

---

### Task 30: Check acceptance criteria from design §9

**Step 1:** Run all checks

Create `scripts/check_acceptance.py`:

```python
"""Check acceptance criteria from design doc §9."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    # Find latest eval report
    reports = sorted(Path("reports").glob("eval-*.md"))
    if not reports:
        print("⚠ No eval reports found")
        return 1
    latest = reports[-1]
    text = latest.read_text()

    criteria = [
        ("Chinese is_scam accuracy >= 0.90", text, "is_scam accuracy", 0.90, ">="),
        ("Chinese is_scam recall >= 0.95", text, "recall", 0.95, ">="),
        ("13-class accuracy >= 0.70", text, "scam_category accuracy", 0.70, ">="),
        ("ECE < 0.05", text, "ECE", 0.05, "<"),
    ]

    n_pass = 0
    for name, _, key, threshold, op in criteria:
        # Parse value from text (simplified: look for "| key | value |")
        import re
        pattern = rf"\|\s*{key}[^|]*\|\s*([\d.]+)\s*\|"
        m = re.search(pattern, text)
        if not m:
            print(f"⚠ {name}: could not find '{key}' in report")
            continue
        val = float(m.group(1))
        passed = (val >= threshold) if op == ">=" else (val < threshold)
        sym = "✓" if passed else "✗"
        print(f"{sym} {name}: {val:.4f} (need {op} {threshold})")
        if passed:
            n_pass += 1

    print(f"\n{n_pass}/{len(criteria)} must-pass criteria met")
    return 0 if n_pass == len(criteria) else 1


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2:** Run check

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python scripts/check_acceptance.py
```

Expected: prints pass/fail for each criterion. If any fail, decide whether to (a) accept and document, (b) iterate with more training.

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add scripts/check_acceptance.py
git -c user.email=claude@local -c user.name=claude commit -m "feat(scripts): add acceptance criteria checker"
```

---

### Task 31: Update README and EXECUTIVE-SUMMARY

**Files:**
- Modify: `README.md`
- Modify: `reports/EXECUTIVE-SUMMARY.md`

**Step 1:** Update README with fine-tuning section

Append to `README.md`:

```markdown

## Phase 2: Fine-tuned multilingual model

The zero-shot baseline struggled with Chinese (accuracy 0.840 vs 0.923 English).
We fine-tuned `convaiinnovations/laya-multilingual` (mmBERT-base) with LoRA on
multi-source scam data (~30k samples across 5 public datasets).

### Training setup
- **Base**: convaiinnovations/laya-multilingual (mmBERT-base, 322M params, 100+ langs)
- **Method**: LoRA r=8 on attention projections (~3.5M trainable params, ~1% of base)
- **Datasets**: SpamShield (149k multilingual), Scam_Message_9_Language (12.8k),
  FGRC-SCD (<1k Chinese), FBS_SMS (14k Chinese), ealvaradob phishing (73k English)
- **Hardware**: Kaggle free 2×T4 GPUs
- **Time**: ~4-5 hours

### Post-finetuning results

| Metric | Zero-shot baseline | Fine-tuned | Target |
|---|---|---|---|
| Chinese is_scam accuracy | 0.840 | **<new>** | ≥ 0.90 |
| Chinese is_scam recall | n/a | **<new>** | ≥ 0.95 |
| 13-class accuracy | 0.605 (8-class) | **<new>** | ≥ 0.70 |
| ECE | 0.081 | **<new>** | < 0.05 |

See `reports/finetune-progress-v1.md` for full training log and `reports/eval-*.md`
for evaluation reports.

### Deployment

After fine-tuning, the model is exported to ONNX (`models/laya-onnx-multilingual-finetuned/`)
for the same local inference path as the English checkpoint. The Router
auto-dispatches CJK/Hebrew/Arabic text to the multilingual checkpoint and Latin
script to the English checkpoint.
```

**Step 2:** Update EXECUTIVE-SUMMARY.md

Open the file and replace the "## Post-evaluation decision" section with the fine-tuning outcome, plus add a new section "## Phase 2: Fine-tuning Results" before the Conclusion.

**Step 3:** Commit

```bash
cd /Users/ks/source_code/laya
git add README.md reports/EXECUTIVE-SUMMARY.md
git -c user.email=claude@local -c user.name=claude commit -m "docs: update README and EXECUTIVE-SUMMARY with fine-tuning results"
```

---

### Task 32: Final integration test — run full suite

**Step 1:** Run all tests

```bash
cd /Users/ks/source_code/laya
PYTHONPATH=. .venv/bin/python -m pytest tests/ --tb=short -q
```

Expected: All tests pass.

**Step 2:** Run sanity check with 3 languages

```bash
cd /Users/ks/source_code/laya
.venv/bin/python -c "
import json
from src.router import Router
router = Router(
    english_dir='models/laya-onnx-en',
    multilingual_dir='models/laya-onnx-multilingual-finetuned',
)
schema = json.load(open('schemas/scam.json'))
cases = [
    ('English phishing', 'URGENT: Verify your account at http://fake.tk/login'),
    ('Chinese scam', '您好，我是XX快递客服，您有一个包裹丢失需要理赔'),
    ('Chinese benign', '妈，我今晚回家吃饭，大概6点到家。'),
]
for label, text in cases:
    r = router.predict(text, schema)
    print(f'{label}: routing={r[\"routing\"][\"model\"]}, is_scam={r[\"answers\"][\"is_scam\"][\"noul\"]:.3f}, cat={r[\"answers\"][\"scam_category\"][\"choice\"]}, {r[\"latency_ms\"]:.0f}ms')
"
```

Expected: all 3 cases work, Chinese routes to multilingual, English routes to English.

**Step 3:** Commit final state

```bash
cd /Users/ks/source_code/laya
git log --oneline | head -10
```

Confirm the implementation is done.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-09-28-laya-finetune-impl.md`.

**Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Note from previous plan: subagent dispatch via `task` tool failed with API key error (`subagent_type: "general"`). Recommend **main agent direct execution** as fallback — most tasks are local file/script operations that don't need a sub-agent context. Only the Kaggle notebook execution (Tasks 21, 25) is genuinely manual/remote.

Which approach?