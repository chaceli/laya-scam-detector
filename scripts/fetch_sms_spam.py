"""Download SMS Spam Collection from Hugging Face and sample into public.jsonl.

Uses curl to bypass SOCKS proxy issues with hf CLI / huggingface_hub.
"""
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq


SOURCE_URL = (
    "https://huggingface.co/datasets/ucirvine/sms_spam/resolve/main/"
    "plain_text/train-00000-of-00001.parquet"
)
TMP = Path("/tmp/sms_spam.parquet")
TARGET = Path("datasets/public.jsonl")
SAMPLE_PER_LABEL = 200


def main() -> int:
    if not TMP.exists():
        print(f"Downloading SMS Spam Collection from {SOURCE_URL}...", file=sys.stderr)
        import subprocess
        result = subprocess.run(
            ["curl", "-s", "-L", "-m", "300", "-o", str(TMP), SOURCE_URL],
            check=True,
        )
    print(f"Loading {TMP}...", file=sys.stderr)
    table = pq.read_table(TMP)
    print(f"Columns: {table.column_names}", file=sys.stderr)

    # SMS Spam uses 'sms' column for text and 'label' for label (0=ham, 1=spam)
    rows = table.to_pylist()
    spam = [r for r in rows if r.get("label") == 1]
    ham = [r for r in rows if r.get("label") == 0]
    print(f"Total: {len(rows)} ({len(spam)} spam, {len(ham)} ham)", file=sys.stderr)

    random.seed(42)
    sample = (
        random.sample(spam, min(SAMPLE_PER_LABEL, len(spam)))
        + random.sample(ham, min(SAMPLE_PER_LABEL, len(ham)))
    )
    random.shuffle(sample)

    with TARGET.open("w") as f:
        for i, r in enumerate(sample):
            obj = {
                "id": f"public-sms-{i:04d}",
                "type": "single",
                "language": "en",
                "expected_label": "spam" if r["label"] == 1 else "benign",
                "expected_risk": 5 if r["label"] == 1 else 1,
                "expected_category": "phishing" if r["label"] == 1 else "benign",
                "state": r["sms"],
                "source": "SMS Spam Collection (ucirvine)",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {len(sample)} samples to {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())