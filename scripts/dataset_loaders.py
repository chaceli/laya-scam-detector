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

# Detection key candidates for flexible source schemas.
CCL_TEXT_KEYS = ("案情描述", "案件描述", "text", "content", "文本", "dialogue")
CCL_LABEL_KEYS = ("案件类别", "类别", "label_name", "label", "罪名", "riskType")
CHIFRAUD_TEXT_KEYS = ("Text", "text", "content", "文本")
CHIFRAUD_LABEL_KEYS = ("Label_id", "label", "label_name", "类别")
TELE_TEXT_KEYS = ("text", "transcription", "content", "dialogue", "conversation", "对话")
TELE_LABEL_KEYS = ("label", "label_name", "is_fraud", "fraud_type", "type", "风险类别")


def despace_cjk(text: str) -> str:
    """FBS_SMS stores CJK with spaces between chars; collapse only CJK-CJK gaps."""
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
    """CCL2023-FCC victim records. Strict 12-class mapping raises on unmapped labels."""
    out, unmapped = [], []
    files_seen = 0
    base = RAW / "ccl2023"
    if not base.exists():
        return out
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
        files_seen += 1
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
            f"CCL2023 unmapped labels: {len(unmapped)} (e.g. {sample}) — "
            f"update CCL2023_LABEL_MAP per Task 4 probe")
    if files_seen > 0 and not out:
        raise RuntimeError(
            f"CCL2023: {files_seen} data file(s) under {base} yielded 0 rows — "
            f"field-name mismatch? expected text in {CCL_TEXT_KEYS}, label in {CCL_LABEL_KEYS}")
    return out


def _load_chifraud_classes(base: Path) -> dict[str, str]:
    """Load numeric Label_id -> class name mapping from class.txt."""
    mapping: dict[str, str] = {}
    for class_file in sorted(base.rglob("class.txt")):
        with class_file.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    mapping[parts[0]] = parts[1]
        break
    return mapping


def load_chifraud() -> list[dict]:
    """ChiFraud: keep only underground loan (loan_scam) and benign rows."""
    out = []
    base = RAW / "chifraud"
    if not base.exists():
        return out
    label_map = _load_chifraud_classes(base)
    for p in sorted(base.rglob("*.csv")):
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                text = _first(row, CHIFRAUD_TEXT_KEYS)
                label_id = (row.get("Label_id") or "").strip()
                if not text or not label_id:
                    continue
                label = label_map.get(label_id, label_id)
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
    """TeleAntiFraud-28k: ASR transcripts with binary fraud/normal labels."""
    out = []
    files_seen = 0
    base = RAW / "teleantifraud"
    if not base.exists():
        return out
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".json", ".jsonl"):
            continue
        files_seen += 1
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
    if files_seen > 0 and not out:
        raise RuntimeError(
            f"TeleAntiFraud: {files_seen} data file(s) under {base} yielded 0 rows — "
            f"field-name mismatch? expected text in {TELE_TEXT_KEYS}, label in {TELE_LABEL_KEYS}")
    return out


def load_phishing_email() -> list[dict]:
    """Phishing Email Dataset: keep only legitimate (label 0) English benign rows."""
    out = []
    base = RAW / "phishing_email"
    if not base.exists():
        return out

    combined = base / "phishing_email.csv"
    if combined.exists():
        paths = [combined]
    else:
        # Fallback: per-source CSVs only when combined file is absent;
        # combined and per-source are never read together, so no double-count.
        paths = sorted(base.rglob("*.csv"))

    # Some email bodies exceed csv's default 128KB field limit.
    _prior = csv.field_size_limit()
    try:
        csv.field_size_limit(16 * 1024 * 1024)
        for p in paths:
            with p.open(newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    text = (row.get("text_combined") or "").strip()
                    label = (row.get("label") or "").strip()
                    if text and label == "0":
                        out.append({
                            "text": text[:1024], "is_scam": 0, "risk": 1,
                            "category": "benign", "language": "en",
                            "source": "phishing_email",
                        })
    finally:
        csv.field_size_limit(_prior)
    return out
