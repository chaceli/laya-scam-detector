"""Verify ONNX export of the fine-tuned multilingual model.

Runs once Phase 5 (Task 25) produces the ONNX bundle. Until then, all
tests skip. The bundle is produced by scripts/merge_and_export_onnx.py.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ONNX_DIR = Path("models/laya-onnx-multilingual-finetuned")
REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_REASON = (
    "Fine-tuned ONNX bundle not present; run "
    "LAYA_ADAPTER_REPO=<user>/laya-multilingual-scam-adapter "
    "python scripts/merge_and_export_onnx.py"
)


@pytest.mark.skipif(
    not (ONNX_DIR / "model.onnx").exists(),
    reason=SKIP_REASON,
)
class TestONNXExport:
    def test_model_files_present(self):
        assert (ONNX_DIR / "model.onnx").exists()
        assert (ONNX_DIR / "tokenizer" / "tokenizer.json").exists()
        assert (ONNX_DIR / "categories.json").exists()

    def test_categories_json_has_13_classes(self):
        from schemas.scam_categories import CANONICAL_CATEGORIES
        data = json.loads((ONNX_DIR / "categories.json").read_text())
        assert data["categories"] == CANONICAL_CATEGORIES
        assert len(data["categories"]) == len(CANONICAL_CATEGORIES)

    def test_onnx_loads(self):
        import onnxruntime as ort
        session = ort.InferenceSession(
            str(ONNX_DIR / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        input_names = {inp.name for inp in session.get_inputs()}
        expected = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"}
        assert expected.issubset(input_names)

    def test_inference_runs_via_onxx_client(self):
        """End-to-end: load bundle with OnnxLayaClient and predict in Chinese."""
        code = f'''
import json, sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from src.laya_onnx import OnnxLayaClient
client = OnnxLayaClient({str(ONNX_DIR)!r})
schema = json.load(open("schemas/scam.json"))
result = client.predict("您好，我是XX快递客服，您有一个包裹丢失需要理赔", schema)
assert "is_scam" in result["answers"]
print("OK", result["answers"]["is_scam"]["noul"])
'''
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        assert "OK" in result.stdout, f"Inference failed:\n{result.stderr}"