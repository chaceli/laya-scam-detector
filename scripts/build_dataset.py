"""Build unified training/val/test dataset from multi-source raw data.

Reads:
  datasets/raw/{fgrc_scd_sms,fgrc_scd_dialog,scamshield,ealvaradob_phishing,fbs_sms}.*

Writes:
  datasets/training/{train,val,test}.jsonl

Each output row:
  {"id": ..., "text": "...", "is_scam": 0|1, "risk": 1-5, "category": "<canonical_13>",
   "source": "<dataset_id>", "language": "<zh|en|...>"}
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import CANONICAL_CATEGORIES, normalize_label


RAW_DIR = Path("datasets/raw")
OUT_DIR = Path("datasets/training")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = OUT_DIR / "train.jsonl"
VAL_PATH = OUT_DIR / "val.jsonl"
TEST_PATH = OUT_DIR / "test.jsonl"

RNG_SEED = 42
TEST_FRAC = 0.10
VAL_FRAC = 0.10
RISK_BY_INTENSITY = {"benign": 1, "low": 2, "medium": 3, "high": 4, "definitive": 5}


def stable_id(text: str, source: str) -> str:
    h = hashlib.sha1(f"{source}::{text}".encode("utf-8")).hexdigest()[:12]
    return f"{source}-{h}"


# ─────────────────────────────────────────────────────────────────
# FGRC-SCD (Chinese telecom fraud)
# ─────────────────────────────────────────────────────────────────
# SMS JSON: [{"文本": "...", "风险类别": "..."}], 10 categories
# - 无风险 (benign) vs all others (scam)
# - Risk category mapping (Chinese → canonical):
FGRC_RISK_MAP = {
    "无风险": "benign",
    "虚假网络投资理财类": "investment_scam",
    "冒充电商物流客服类": "phishing",
    "虚假购物、服务类": "phishing",
    "冒充军警购物类诈骗": "impersonation",
    "冒充公检法及政府机关类": "impersonation",
    "冒充领导、熟人类": "impersonation",
    "网络婚恋、交友类": "romance_scam",
    "虚假信用服务类": "loan_scam",
    "网黑案件": "spam_general",
}


def load_fgrc_scd_sms() -> list[dict]:
    out = []
    for json_path in sorted((RAW_DIR / "fgrc_scd_sms" / "message").glob("*.json")):
        with json_path.open() as f:
            data = json.load(f)
        for item in data:
            text = (item.get("文本") or "").strip()
            if not text:
                continue
            risk_label = item.get("风险类别", "")
            is_scam = risk_label != "无风险"
            canonical = FGRC_RISK_MAP.get(risk_label, "spam_general") if is_scam else "benign"
            risk = 4 if is_scam else 1
            out.append({
                "text": text,
                "is_scam": int(is_scam),
                "risk": risk,
                "category": canonical,
                "language": "zh",
                "source": "fgrc_scd_sms",
            })
    return out


def load_fgrc_scd_dialog() -> list[dict]:
    """FGRC-SCD dialog: list of {"text": "【坐席】...【客户】...", "riskType": "..."}."""
    out = []
    for json_path in sorted((RAW_DIR / "fgrc_scd_dialog" / "dialog").glob("*.json")):
        with json_path.open() as f:
            data = json.load(f)
        for item in data:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            risk_type = item.get("riskType", "")
            is_scam = risk_type != "无风险"
            canonical = FGRC_RISK_MAP.get(risk_type, "spam_general") if is_scam else "benign"
            risk = 4 if is_scam else 1
            out.append({
                "text": text,
                "is_scam": int(is_scam),
                "risk": risk,
                "category": canonical,
                "language": "zh",
                "source": "fgrc_scd_dialog",
            })
    return out


# ─────────────────────────────────────────────────────────────────
# ScamShield (Him1304) - English+job scam, ~23k rows
# ─────────────────────────────────────────────────────────────────
# CSV: text,label,source — label: 0=safe, 1=scam
# source: 'sms', 'job', 'synthetic'
def load_scamshield() -> list[dict]:
    out = []
    with (RAW_DIR / "scamshield.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            is_scam = int(row.get("label", 0)) == 1
            src = row.get("source", "")
            # ScamShield doesn't have detailed category; use source as hint
            if is_scam:
                if src == "job":
                    cat = "job_scam"
                elif src == "sms":
                    cat = "spam_general"
                else:
                    cat = "spam_general"
            else:
                cat = "benign"
            out.append({
                "text": text,
                "is_scam": int(is_scam),
                "risk": 4 if is_scam else 1,
                "category": cat,
                "language": "en",
                "source": "scamshield",
            })
    return out


# ─────────────────────────────────────────────────────────────────
# ealvaradob phishing - English URL/SMS/email
# ─────────────────────────────────────────────────────────────────
# JSON list: [{"text": "...", "label": 0|1}], 77k rows
def load_ealvaradob() -> list[dict]:
    out = []
    with (RAW_DIR / "ealvaradob_phishing.json").open() as f:
        data = json.load(f)
    for item in data:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        is_scam = int(item.get("label", 0)) == 1
        out.append({
            "text": text,
            "is_scam": int(is_scam),
            "risk": 4 if is_scam else 1,
            "category": "phishing" if is_scam else "benign",
            "language": "en",
            "source": "ealvaradob",
        })
    return out


# ─────────────────────────────────────────────────────────────────
# FBS_SMS (fl-wxiao) - Chinese fake-base-station spam
# ─────────────────────────────────────────────────────────────────
# Plain text files: filename = category, content = one message per line
def load_fbs_sms() -> list[dict]:
    out = []
    base = RAW_DIR / "fbs_sms"
    for txt_path in sorted(base.glob("*")):
        if not txt_path.is_file() or txt_path.name == "README.md":
            continue
        category_raw = txt_path.name
        # All messages in FBS_SMS are spam (fake base station = all spam)
        with txt_path.open() as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                canonical = normalize_label(category_raw)
                out.append({
                    "text": text,
                    "is_scam": 1,
                    "risk": 4,
                    "category": canonical,
                    "language": "zh",
                    "source": "fbs_sms",
                })
    return out


# ─────────────────────────────────────────────────────────────────
# UCI SMS Spam (ucirvine/sms_spam) - already in datasets/public.jsonl
# ─────────────────────────────────────────────────────────────────
def load_uc_irvine() -> list[dict]:
    """Read pre-formatted public.jsonl (already 200 spam + 200 ham)."""
    out = []
    src = Path("datasets/public.jsonl")
    if not src.exists():
        return out
    with src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = (rec.get("state") or "").strip()
            if not text:
                continue
            is_scam = int(rec.get("expected_risk", 1)) >= 4
            out.append({
                "text": text,
                "is_scam": int(is_scam),
                "risk": 4 if is_scam else 1,
                "category": "spam_general" if is_scam else "benign",
                "language": "en",
                "source": "uci_sms_spam",
            })
    return out


# ─────────────────────────────────────────────────────────────────
# Split + balance
# ─────────────────────────────────────────────────────────────────
def split_dataset(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
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
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    out = []
    for source, items in by_source.items():
        scam = [r for r in items if r["is_scam"] == 1]
        benign = [r for r in items if r["is_scam"] == 0]
        random.shuffle(benign)
        out.extend(scam)
        out.extend(benign[: len(scam)])
    random.shuffle(out)
    return out


def main() -> int:
    print("Loading sources...")
    all_rows: list[dict] = []
    for loader in (
        load_fgrc_scd_sms,
        load_fgrc_scd_dialog,
        load_scamshield,
        load_ealvaradob,
        load_fbs_sms,
        load_uc_irvine,
    ):
        try:
            rows = loader()
            print(f"  ✓ {loader.__name__}: {len(rows):,} rows")
            all_rows.extend(rows)
        except FileNotFoundError as e:
            print(f"  ✗ {loader.__name__}: {e} (skipping)")
    print(f"\nTotal raw rows: {len(all_rows):,}")

    seen = set()
    cleaned = []
    for r in all_rows:
        key = r["text"].strip()
        if not key or key in seen:
            continue
        seen.add(key)
        cleaned.append(r)
    print(f"After dedup: {len(cleaned):,} rows")

    train, val, test = split_dataset(cleaned)
    print(f"Split: train={len(train):,} val={len(val):,} test={len(test):,}")

    train = balance_train(train)
    print(f"Train after balance: {len(train):,}")

    for split, rows in [("train", train), ("val", val), ("test", test)]:
        for r in rows:
            r["id"] = stable_id(r["text"], r["source"])

    for path, rows in [(TRAIN_PATH, train), (VAL_PATH, val), (TEST_PATH, test)]:
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  ✓ {path} ({len(rows):,} rows)")

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