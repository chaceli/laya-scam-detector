"""schemas/scam.json 与 14 类体系一致性。"""
import json

from schemas.scam_categories import CANONICAL_CATEGORIES


def test_scam_json_criteria_match_canonical_order():
    schema = json.loads(open("schemas/scam.json").read())
    keys = list(schema["scam_category"]["criteria"].keys())
    assert keys == CANONICAL_CATEGORIES


def test_train_descriptions_cover_all_categories():
    src = open("scripts/train_local_lora.py").read()
    for c in CANONICAL_CATEGORIES:
        assert f'"{c}":' in src, f"missing description key: {c}"
