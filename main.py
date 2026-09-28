"""CLI for local Laya scam-phrase evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.router import Router, ScriptRouter


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="laya-scam-detector",
        description="Local Laya decision model — scam-phrase risk evaluation",
    )
    parser.add_argument("--predict", help="Single state string to evaluate")
    parser.add_argument("--questions", help="JSON file with question schema")
    parser.add_argument("--model", choices=["english", "multilingual"],
                        help="Force checkpoint (multilingual falls back to English if unavailable)")
    parser.add_argument("--route", help="Just detect script and route (no inference)")
    parser.add_argument("--eval", action="store_true", help="Run batch evaluation")
    parser.add_argument("--input", help="Input JSONL file for --eval")
    parser.add_argument("--output", default="reports/",
                        help="Output directory for reports (default: reports/)")
    parser.add_argument("--english-dir", default="models/laya-onnx-en")
    parser.add_argument("--multilingual-dir", default="models/laya-onnx-multilingual")

    args = parser.parse_args()

    # Mode: --route only (no model load required)
    if args.route:
        sr = ScriptRouter()
        model, reason = sr.route_with_reason(args.route)
        print(json.dumps({
            "input": args.route,
            "routed_to": model,
            "reason": reason,
        }, ensure_ascii=False))
        return 0

    # Load Router (or fail fast with download instructions)
    try:
        router = Router(
            english_dir=args.english_dir,
            multilingual_dir=args.multilingual_dir,
        )
    except Exception as e:
        print(f"✗ Failed to load model: {e}", file=sys.stderr)
        print(f"  Run: bash scripts/download_models.sh", file=sys.stderr)
        return 2

    # Mode: --eval
    if args.eval:
        if not args.input:
            print("✗ --eval requires --input <file.jsonl>", file=sys.stderr)
            return 2
        from src.eval import run_evaluation, load_schema
        from src.report import render_report
        schema = load_schema()
        results = run_evaluation(router, args.input, schema)
        out_dir = Path(args.output)
        report_path, jsonl_path = render_report(results, schema, out_dir)
        print(f"✓ Report: {report_path}", file=sys.stderr)
        print(f"✓ Results: {jsonl_path}", file=sys.stderr)
        return 0

    # Mode: --predict
    if args.predict:
        if not args.questions:
            print("✗ --predict requires --questions <file.json>", file=sys.stderr)
            return 2
        schema = json.loads(Path(args.questions).read_text())
        try:
            result = router.predict(args.predict, schema, model=args.model)
        except Exception as e:
            print(f"✗ Inference failed: {e}", file=sys.stderr)
            return 3
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())