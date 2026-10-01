"""Batch evaluation over JSONL test sets (13-class schema)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from schemas.scam_categories import normalize_label

from src.router import Router


def load_schema(path: str = "schemas/scam.json") -> dict:
    return json.loads(Path(path).read_text())


def _state_from_record(rec: dict) -> str:
    """Normalize a test record into a state string."""
    if rec.get("type") == "multi_turn" and "turns" in rec:
        parts = []
        for i, turn in enumerate(rec["turns"]):
            role_short = (turn.get("role") or "?")[0].upper()
            parts.append(f"[{role_short}{i+1}] {turn.get('text', '')}")
        return "\n".join(parts)
    return rec.get("state", "")


def _thresholded_is_scam(p_noul: float, threshold: float = 0.5) -> int:
    return 1 if p_noul >= threshold else 0


def _expected_category(rec: dict) -> str | None:
    """v2 统一口径：category 为准，expected_category 向后兼容。"""
    raw = rec.get("category") or rec.get("expected_category")
    if raw is None:
        return None
    return normalize_label(raw)


def run_evaluation(router: Router, input_jsonl: str, schema: dict) -> list[dict]:
    """Run all samples through router.predict; return enriched records."""
    results = []
    with open(input_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = _state_from_record(rec)
            try:
                pred = router.predict(state, schema)
            except Exception as e:
                pred = {"error": str(e), "answers": {}}
            answers = pred.get("answers", {})
            expected_cat = _expected_category(rec)
            enriched = {
                "id": rec.get("id"),
                "type": rec.get("type"),
                "language": rec.get("language"),
                "expected_risk": rec.get("expected_risk"),
                "expected_category": expected_cat,
                "expected_label": rec.get("expected_label"),
                "source": rec.get("source"),
                "state_preview": state[:120],
                "predicted": {
                    "is_scam_noul": answers.get("is_scam", {}).get("noul"),
                    "is_scam_label": _thresholded_is_scam(
                        answers.get("is_scam", {}).get("noul", 0)
                    ) if answers.get("is_scam", {}).get("noul") is not None else None,
                    "risk_level": answers.get("risk_level", {}).get("score"),
                    "scam_category": answers.get("scam_category", {}).get("choice"),
                    "category_probs": answers.get("scam_category", {}).get("probabilities"),
                },
                "routing": pred.get("routing"),
                "latency_ms": pred.get("latency_ms"),
                "error": pred.get("error"),
            }
            results.append(enriched)
    return results