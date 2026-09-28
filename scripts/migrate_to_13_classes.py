"""Migrate existing test sets to canonical 13-class labels.

Most existing labels are already canonical (benign, phishing, etc.).
The mapping function from schemas.scam_categories handles edge cases.

Idempotent: safe to run multiple times.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import normalize_label, is_valid_category


TARGETS = [
    Path("datasets/single.jsonl"),
    Path("datasets/multi.jsonl"),
    Path("datasets/public.jsonl"),
    Path("datasets/eval.jsonl"),
]


def migrate_file(path: Path) -> None:
    if not path.exists():
        print(f"  ⚠ {path} not found, skipping")
        return
    rows_out = []
    n_fixed = 0
    n_invalid = 0
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
                if not is_valid_category(rec["expected_category"]):
                    n_invalid += 1
            rows_out.append(rec)
    with path.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  ✓ {path}: {len(rows_out)} rows, {n_fixed} labels migrated, {n_invalid} invalid")


def main() -> int:
    print("Migrating test sets to canonical 13-class labels...")
    for p in TARGETS:
        migrate_file(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())