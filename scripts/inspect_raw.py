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
