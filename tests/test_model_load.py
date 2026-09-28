"""Smoke test for ONNX model loading."""
import json
from pathlib import Path

import onnxruntime as ort
import pytest
from tokenizers import Tokenizer


CHECKPOINT_DIR = Path("models/laya-onnx-en")


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.onnx").exists(),
    reason="ONNX checkpoint not downloaded; run scripts/download_models.sh",
)
def test_model_onnx_loads():
    session = ort.InferenceSession(
        CHECKPOINT_DIR / "model.onnx",
        providers=["CPUExecutionProvider"],
    )
    assert session is not None
    inputs = [inp.name for inp in session.get_inputs()]
    assert "input_ids" in inputs
    assert "marker_pos" in inputs


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "tokenizer/tokenizer.json").exists(),
    reason="tokenizer not downloaded",
)
def test_tokenizer_loads():
    tok = Tokenizer.from_file(str(CHECKPOINT_DIR / "tokenizer/tokenizer.json"))
    encoded = tok.encode("Hello world")
    # ModernBERT tokenizer auto-prepends [CLS] and appends [SEP] by default,
    # so "Hello world" produces 4 token ids: CLS, hello, world, SEP.
    assert len(encoded.ids) == 4
    assert encoded.ids[0] == tok.token_to_id("[CLS]")


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "rl_agent_config.json").exists(),
    reason="rl_agent_config not downloaded",
)
def test_config_loads():
    cfg = json.loads((CHECKPOINT_DIR / "rl_agent_config.json").read_text())
    assert "max_len" in cfg or "head_max_len" in cfg


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.onnx").exists(),
    reason="ONNX checkpoint not downloaded",
)
def test_onnx_has_all_required_inputs():
    """Confirm ONNX graph signature matches plan §6 contract."""
    session = ort.InferenceSession(
        CHECKPOINT_DIR / "model.onnx",
        providers=["CPUExecutionProvider"],
    )
    input_names = {inp.name for inp in session.get_inputs()}
    expected = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}
    assert expected.issubset(input_names), (
        f"Missing inputs. Expected {expected}, got {input_names}"
    )


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "tokenizer/tokenizer.json").exists(),
    reason="tokenizer not downloaded",
)
def test_special_tokens_present():
    """Laya requires [CLS], [SEP], [MASK] in tokenizer."""
    tok = Tokenizer.from_file(str(CHECKPOINT_DIR / "tokenizer/tokenizer.json"))
    for special in ["[CLS]", "[SEP]", "[MASK]"]:
        assert tok.token_to_id(special) is not None, f"Missing {special}"