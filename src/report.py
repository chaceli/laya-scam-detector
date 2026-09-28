"""Render Markdown evaluation report from evaluation results."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median


def _summary(results: list[dict]) -> dict:
    """Compute aggregate metrics from enriched eval results."""
    pairs = [
        (r["predicted"]["is_scam_noul"], int(r["expected_risk"] >= 4))
        for r in results
        if r["predicted"]["is_scam_noul"] is not None and r.get("expected_risk") is not None
    ]
    if not pairs:
        return {}

    y_true = [y for _, y in pairs]
    y_pred = [1 if p >= 0.5 else 0 for p, _ in pairs]
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    risk_pairs = [
        (r["predicted"]["risk_level"], r["expected_risk"])
        for r in results
        if r["predicted"]["risk_level"] is not None and r.get("expected_risk") is not None
    ]
    risk_mae = mean(abs(p - e) for p, e in risk_pairs) if risk_pairs else None

    cat_pairs = [
        (r["predicted"]["scam_category"], r["expected_category"])
        for r in results
        if r["predicted"]["scam_category"] and r.get("expected_category")
    ]
    cat_acc = sum(1 for p, e in cat_pairs if p == e) / max(len(cat_pairs), 1) if cat_pairs else None

    # Per-language breakdown
    by_lang = {}
    for r in results:
        lang = r.get("language", "?")
        if r["predicted"]["is_scam_noul"] is None or r.get("expected_risk") is None:
            continue
        by_lang.setdefault(lang, {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0})
        pred_label = 1 if r["predicted"]["is_scam_noul"] >= 0.5 else 0
        true_label = 1 if r["expected_risk"] >= 4 else 0
        by_lang[lang]["n"] += 1
        if pred_label == 1 and true_label == 1:
            by_lang[lang]["tp"] += 1
        elif pred_label == 1 and true_label == 0:
            by_lang[lang]["fp"] += 1
        elif pred_label == 0 and true_label == 1:
            by_lang[lang]["fn"] += 1
        else:
            by_lang[lang]["tn"] += 1

    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    p50_lat = median(latencies) if latencies else None
    p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else None

    return {
        "n": len(pairs),
        "is_scam_accuracy": accuracy,
        "is_scam_precision": precision,
        "is_scam_recall": recall,
        "is_scam_f1": f1,
        "is_scm_tp": tp, "is_scm_fp": fp, "is_scm_fn": fn, "is_scm_tn": tn,
        "risk_level_mae": risk_mae,
        "scam_category_accuracy": cat_acc,
        "by_language": by_lang,
        "p50_latency_ms": p50_lat,
        "p95_latency_ms": p95_lat,
        "n_errors": sum(1 for r in results if r.get("error")),
    }


def _category_table(results: list[dict]) -> str:
    counter = Counter()
    for r in results:
        p = r["predicted"]["scam_category"]
        e = r["expected_category"]
        if p and e:
            counter[(e, p)] += 1
    if not counter:
        return "_No category data._"
    expected_cats = sorted({e for e, _ in counter.keys()})
    predicted_cats = sorted({p for _, p in counter.keys()})
    lines = ["| expected ↓ / predicted → | " + " | ".join(predicted_cats) + " |",
             "|" + "---|" * (len(predicted_cats) + 1)]
    for e in expected_cats:
        row = [f"| **{e}**"]
        for p in predicted_cats:
            row.append(str(counter.get((e, p), 0)))
        lines.append(" | ".join(row) + " |")
    return "\n".join(lines)


def _failure_table(results: list[dict], n: int = 10) -> str:
    fn = []
    fp = []
    for r in results:
        p = r["predicted"]["is_scam_noul"]
        e = r["expected_risk"]
        if p is None or e is None:
            continue
        if e >= 4 and p < 0.5:
            fn.append((e - p, r))
        if e <= 2 and p >= 0.5:
            fp.append((p - e, r))

    fn.sort(key=lambda x: -x[0])
    fp.sort(key=lambda x: -x[0])

    lines = ["## Failure analysis", ""]
    lines.append(f"### False Negatives (missed scams, n={len(fn)})")
    lines.append("")
    if not fn:
        lines.append("_None_")
    else:
        lines.append("| ID | Lang | Expected | Pred noul | Preview |")
        lines.append("|---|---|---|---|---|")
        for _, r in fn[:n]:
            preview = (r.get("state_preview") or "").replace("|", "\\|")[:60]
            lines.append(
                f"| {r['id']} | {r.get('language')} | {r['expected_risk']} | "
                f"{r['predicted']['is_scam_noul']:.2f} | {preview} |"
            )
    lines.append("")
    lines.append(f"### False Positives (benign flagged as scam, n={len(fp)})")
    lines.append("")
    if not fp:
        lines.append("_None_")
    else:
        lines.append("| ID | Lang | Expected | Pred noul | Preview |")
        lines.append("|---|---|---|---|---|")
        for _, r in fp[:n]:
            preview = (r.get("state_preview") or "").replace("|", "\\|")[:60]
            lines.append(
                f"| {r['id']} | {r.get('language')} | {r['expected_risk']} | "
                f"{r['predicted']['is_scam_noul']:.2f} | {preview} |"
            )
    return "\n".join(lines)


def render_report(results: list[dict], schema: dict, out_dir: Path) -> tuple[Path, Path]:
    """Render Markdown + JSONL to out_dir; return (md_path, jsonl_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    md_path = out_dir / f"eval-{ts}.md"
    jsonl_path = out_dir / f"results-{ts}.jsonl"

    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n"
    )

    s = _summary(results)
    md = [
        "# Laya Scam-Phrase Evaluation Report",
        f"_Generated: {datetime.now().isoformat()}_",
        "",
        f"**Total samples:** {len(results)}",
        "",
        "## Headline Metrics",
        "",
    ]
    if s:
        md += [
            "| Metric | Value |",
            "|---|---|",
            f"| is_scam accuracy (threshold 0.5) | {s['is_scam_accuracy']:.3f} |",
            f"| is_scam precision | {s['is_scam_precision']:.3f} |",
            f"| is_scam recall | {s['is_scam_recall']:.3f} |",
            f"| is_scam F1 | {s['is_scam_f1']:.3f} |",
        ]
        if s['risk_level_mae'] is not None:
            md.append(f"| risk_level MAE | {s['risk_level_mae']:.2f} |")
        else:
            md.append("| risk_level MAE | n/a |")
        if s['scam_category_accuracy'] is not None:
            md.append(f"| scam_category accuracy | {s['scam_category_accuracy']:.3f} |")
        else:
            md.append("| scam_category accuracy | n/a |")
        if s['p50_latency_ms'] is not None:
            md.append(f"| p50 latency | {s['p50_latency_ms']:.0f} ms |")
        else:
            md.append("| p50 latency | n/a |")
        if s['p95_latency_ms'] is not None:
            md.append(f"| p95 latency | {s['p95_latency_ms']:.0f} ms |")
        else:
            md.append("| p95 latency | n/a |")
        md.append(f"| errors | {s['n_errors']} |")
    md += ["", "## Confusion Matrix (is_scam, threshold 0.5)", ""]
    if s:
        md += [
            "|  | pred=scam | pred=benign |",
            "|---|---|---|",
            f"| actual=scam | {s['is_scm_tp']} (TP) | {s['is_scm_fn']} (FN) |",
            f"| actual=benign | {s['is_scm_fp']} (FP) | {s['is_scm_tn']} (TN) |",
        ]
    if s and s.get("by_language"):
        md += ["", "## Per-Language Breakdown", ""]
        md += [
            "| Lang | n | TP | FP | FN | TN | Acc |",
            "|---|---|---|---|---|---|---|",
        ]
        for lang, stats in sorted(s["by_language"].items()):
            n = stats["n"]
            acc = (stats["tp"] + stats["tn"]) / max(n, 1)
            md.append(
                f"| {lang} | {n} | {stats['tp']} | {stats['fp']} | "
                f"{stats['fn']} | {stats['tn']} | {acc:.3f} |"
            )
    md += ["", "## Scam Category Confusion", "", _category_table(results), "",
           _failure_table(results), "", "## All samples", "",
           "| ID | Lang | Exp risk | Pred noul | Exp cat | Pred cat | Latency |",
           "|---|---|---|---|---|---|---|"]
    for r in results:
        pred = r["predicted"]
        noul_str = (
            f"{pred['is_scam_noul']:.2f}"
            if pred['is_scam_noul'] is not None
            else "NA"
        )
        md.append(
            f"| {r['id']} | {r.get('language')} | {r.get('expected_risk')} | "
            f"{noul_str} | "
            f"{r.get('expected_category') or ''} | {pred['scam_category'] or ''} | "
            f"{r.get('latency_ms', 0):.0f}ms |"
        )
    md.append("")
    md_path.write_text("\n".join(md))
    return md_path, jsonl_path