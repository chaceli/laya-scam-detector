"""build_dataset 的 rebate 接线：真实 CCL 取代合成，CCL 留出必排除。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dataset_mix import _norm_text  # noqa: E402


def test_synthetic_rebate_file_removed():
    assert not (ROOT / "datasets/synthetic/rebate_scam.jsonl").exists(), \
        "合成 rebate 应已删除（真实 CCL 取代）"


def test_ccl_holdout_excluded_from_train():
    """若构建产物已存在，CCL 留出文本不得出现在训练集里。"""
    hold_path = ROOT / "datasets/ccl_rebate_eval.jsonl"
    assert hold_path.exists(), "CCL 留出集缺失"
    hold_norm = {_norm_text(json.loads(l)["state"]) for l in hold_path.open()}
    train_path = ROOT / "datasets/training/train.jsonl"
    if not train_path.exists():
        return  # 构建未跑则跳过（Task 4 会实跑）
    train_norm = {_norm_text(json.loads(l)["text"]) for l in train_path.open()}
    leak = hold_norm & train_norm
    assert not leak, f"CCL 留出泄漏进训练 {len(leak)} 条"


def test_cross_register_eval_still_excluded():
    """42 条跨源集同样必须被排除。"""
    ev_path = ROOT / "datasets/rebate_eval_real.jsonl"
    ev_norm = {_norm_text(json.loads(l)["text"]) for l in ev_path.open()}
    train_path = ROOT / "datasets/training/train.jsonl"
    if not train_path.exists():
        return
    train_norm = {_norm_text(json.loads(l)["text"]) for l in train_path.open()}
    assert not (ev_norm & train_norm), "42 条跨源集泄漏进训练"
