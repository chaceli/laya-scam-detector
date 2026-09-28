"""Verify the Laya PyTorch SDK is importable and instantiable.

This validates that Phase 4 (Kaggle training) won't fail due to a missing
or broken laya SDK install. We don't run full inference here because the
official SDK downloads checkpoints from Hugging Face on first use — that's
a 700MB+ side effect we want to avoid in preflight checks. The actual
inference smoke test runs on Kaggle.

Our project also has its own OnnxLayaClient (src/laya_onnx.py) that runs
ONNX Runtime locally — that's what production uses, not the PyTorch SDK.
"""
import pytest

try:
    import laya
    LAYA_AVAILABLE = True
except ImportError:
    LAYA_AVAILABLE = False


@pytest.mark.skipif(not LAYA_AVAILABLE, reason="laya SDK not installed (try pip install laya)")
class TestLayaPyTorchLoad:
    def test_import_laya(self):
        import laya
        assert hasattr(laya, "load")
        assert hasattr(laya, "Router")

    def test_laya_version(self):
        import laya
        # laya 0.3.21 confirmed working in Phase 4
        assert laya.__version__ >= "0.3.0"

    def test_router_can_be_constructed(self):
        from laya import Router
        # Construct Router but don't preload (avoid downloading HF weights).
        # This verifies the Python class loads, not that inference works.
        router = Router(device="cpu", preload=False)
        assert router is not None
        assert router.device == "cpu"

    def test_router_models_registered(self):
        """The Router should know about english/multilingual/typed-decisions."""
        from laya import Router
        router = Router(device="cpu", preload=False)
        # The Router has a `models` dict with available checkpoints
        assert hasattr(router, "models")
        assert "english" in router.models
        assert "multilingual" in router.models

    def test_decide_method_exists(self):
        """decide() is the high-level schema-based entrypoint."""
        from laya import Router
        router = Router(device="cpu", preload=False)
        assert hasattr(router, "decide")
        assert hasattr(router, "predict")
        assert hasattr(router, "predict_batch")