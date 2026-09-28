"""Check acceptance criteria from design doc section 9.

Reads the latest eval report and checks hard criteria:
- Chinese is_scam accuracy >= 0.90
- is_scam recall >= 0.95
- 13-class accuracy >= 0.70

Usage:
  PYTHONPATH=. python scripts/check_acceptance.py [report_path]

If no report_path given, uses the most recent reports/eval-*.md.
"""
import sys
from pathlib import Path


def parse_headline_metrics(text: str) -> dict[str, float]:
    """Extract '| metric name | value |' rows from the Headline Metrics table."""
    metrics: dict[str, float] = {}
    in_headline = False
    for line in text.split("\n"):
        if line.startswith("## Headline Metrics"):
            in_headline = True
            continue
        if in_headline and line.startswith("## "):
            break
        if not in_headline or not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 2:
            continue
        key = parts[0]
        val_raw = parts[1].split()[0] if parts[1] else ""
        try:
            metrics[key] = float(val_raw)
        except (ValueError, TypeError):
            continue
    return metrics


def parse_language_accuracy(text: str, lang: str) -> float | None:
    """Parse per-language Acc column for a given lang code."""
    for line in text.split("\n"):
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 6 and parts[0] == lang:
            try:
                return float(parts[-1])
            except (ValueError, TypeError):
                return None
    return None


def find_metric(metrics: dict[str, float], needle: str) -> float | None:
    """Substring match on metric keys."""
    for k, v in metrics.items():
        if needle.lower() in k.lower():
            return v
    return None


def main() -> int:
    if len(sys.argv) > 1:
        report = Path(sys.argv[1])
    else:
        reports = sorted(Path("reports").glob("eval-*.md"))
        if not reports:
            print("⚠ No eval reports found in reports/")
            return 1
        report = reports[-1]
    if not report.exists():
        print(f"⚠ Report not found: {report}")
        return 1

    print(f"Checking acceptance criteria against {report}")
    print("=" * 60)
    text = report.read_text()
    metrics = parse_headline_metrics(text)

    results: list[tuple[str, float | None, float, str]] = [
        ("Chinese is_scam accuracy", parse_language_accuracy(text, "zh"), 0.90, ">="),
        ("is_scam recall", find_metric(metrics, "recall"), 0.95, ">="),
        ("13-class accuracy", find_metric(metrics, "category accuracy"), 0.70, ">="),
    ]

    n_pass = 0
    n_total = 0
    for name, val, threshold, op in results:
        n_total += 1
        if val is None:
            print(f"⚠ {name}: not found in report")
            continue
        passed = (val >= threshold) if op == ">=" else (val < threshold)
        print(f"{'✓' if passed else '✗'} {name}: {val:.4f} (need {op} {threshold})")
        n_pass += int(passed)

    print("=" * 60)
    print(f"{n_pass}/{n_total} criteria met")

    ece = find_metric(metrics, "ece")
    if ece is not None:
        print(f"   (ECE: {ece:.4f}, target < 0.05)")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())