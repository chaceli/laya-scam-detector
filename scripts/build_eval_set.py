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