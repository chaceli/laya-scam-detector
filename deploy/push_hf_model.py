"""Publish the fine-tuned fp16 ONNX bundle to the Hugging Face Hub.

Model repositories are free for public models (unlike compute Spaces).

Usage:
  HF_TOKEN=hf_xxx .venv/bin/python deploy/push_hf_model.py \
      --repo-id <user>/laya-scam-detector-onnx [--private]
  # v3 bundle:
  HF_TOKEN=hf_xxx .venv/bin/python deploy/push_hf_model.py \
      --repo-id <user>/laya-scam-detector-onnx-v3 \
      --model-dir models/laya-onnx-multilingual-finetuned-v3-fp16
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "laya-onnx-multilingual-finetuned-fp16"
MODEL_CARD = REPO_ROOT / "deploy" / "model-card-README.md"

INCLUDE = [
    "model.onnx",
    "model.onnx.data",
    "categories.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                    help="fp16 bundle dir to upload (default: v1 fp16)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "model.onnx").exists():
        print(f"✗ {model_dir}/model.onnx not found — run scripts/quantize_onnx_fp16.py")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="laya-hf-model-"))
    total = 0
    for rel in INCLUDE:
        src = model_dir / rel
        if not src.exists():
            print(f"  ⚠ missing {rel}")
            continue
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        total += src.stat().st_size
        print(f"  + {rel}  ({src.stat().st_size/1e6:.1f} MB)")
    shutil.copy(MODEL_CARD, tmp / "README.md")
    print(f"  + README.md (model card)")
    print(f"  total: {total/1e6:.0f} MB  (from {model_dir})")

    if args.dry_run:
        print(f"\n(dry run) staged at {tmp}")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("✗ HF_TOKEN not set")
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"\nCreating model repo {args.repo_id} (private={args.private})...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    print("Uploading...")
    api.upload_folder(
        folder_path=str(tmp),
        repo_id=args.repo_id,
        repo_type="model",
    )
    shutil.rmtree(tmp, ignore_errors=True)

    url = f"https://huggingface.co/{args.repo_id}"
    print(f"\n✓ Model published: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())