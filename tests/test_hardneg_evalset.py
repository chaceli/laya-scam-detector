"""build_hardneg_evalset 切分不变量测试（fixtures，无真实数据）。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_hardneg_evalset as bhe  # noqa: E402


@pytest.fixture()
def genre_dir(tmp_path, monkeypatch):
    data = tmp_path / "hard_negatives"
    data.mkdir()
    for genre in ("hn_financial_notice", "nb_natural_benign"):
        with (data / f"{genre}.jsonl").open("w") as f:
            for i in range(100):
                f.write(json.dumps({
                    "id": f"{genre}-{i}", "text": f"{genre} 文本 {i}",
                    "is_scam": 0, "risk": 1, "category": "benign",
                    "language": "zh", "source": genre,
                }, ensure_ascii=False) + "\n")
    monkeypatch.setattr(bhe, "DATA", data)
    monkeypatch.setattr(bhe, "OUT_EVAL", tmp_path / "hardneg_eval.jsonl")
    monkeypatch.setattr(bhe, "OUT_TRAIN", data / "train_pool.jsonl")
    return tmp_path


def test_split_disjoint_and_counts(genre_dir):
    assert bhe.main() == 0
    eval_rows = [json.loads(l) for l in
                 (genre_dir / "hardneg_eval.jsonl").open() if l.strip()]
    train_rows = [json.loads(l) for l in
                  (genre_dir / "hard_negatives" / "train_pool.jsonl").open()
                  if l.strip()]
    assert len(eval_rows) == 150          # 75 × 2 体裁
    assert len(train_rows) == 50          # 25 × 2
    assert {r["id"] for r in eval_rows}.isdisjoint(
        {r["id"] for r in train_rows})
    r = eval_rows[0]
    assert r["type"] == "single" and r["expected_risk"] == 1
    assert r["category"] == "benign" and r["source"].startswith(("hn_", "nb_"))
    assert "state" in r