"""Server configuration, overridable via environment variables."""
import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
SCHEMA_PATH = REPO_ROOT / "schemas" / "scam.json"

ENGLISH_DIR = _env("LAYA_ENGLISH_DIR", str(REPO_ROOT / "models" / "laya-onnx-en"))
MULTILINGUAL_DIR = _env(
    "LAYA_MULTILINGUAL_DIR",
    str(REPO_ROOT / "models" / "laya-onnx-multilingual-finetuned"),
)
HOST = _env("LAYA_HOST", "127.0.0.1")
PORT = int(_env("LAYA_PORT", "8000"))