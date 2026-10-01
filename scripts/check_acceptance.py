"""Check acceptance criteria from design doc section 9.

Reads the latest eval report and checks hard criteria:
- Chinese is_scam accuracy >= 0.90
- is_scam recall >= 0.95
- 13-class accuracy >= 0.70

Usage:
  PYTHONPATH=. python scripts/check_acceptance.py [report_path]

If no report_path given, uses the most recent reports/eval-*.md.
"""
import re
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


def run_legacy(report_arg: str | None) -> int:
    if report_arg:
        report = Path(report_arg)
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


def main() -> int:
    ap_args = sys.argv[1:]
    # legacy 模式：直接传 report 路径作为唯一位置参数
    if ap_args and not ap_args[0].startswith("-"):
        return run_legacy(ap_args[0])

    # 新 argparse 用于四门槛模式
    import argparse
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


if __name__ == "__main__":
    sys.exit(main())