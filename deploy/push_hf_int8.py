"""Upload the int8 checkpoint to the HF model repo under int8/.

The browser app fetches these files, so they must live on the Hub: HF
resolve URLs send permissive CORS headers, GitHub Release assets do not.

Usage:
  HF_TOKEN=hf_xxx .venv/bin/python deploy/push_hf_int8.py \
      --repo-id LiChace/laya-scam-detector-onnx
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INT8_DIR = REPO_ROOT / "models" / "laya-onnx-multilingual-finetuned-int8"

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
    ap.add_argument("--prefix", default="int8")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (INT8_DIR / "model.onnx").exists():
        print(f"✗ {INT8_DIR}/model.onnx not found — run scripts/quantize_onnx_int8.py")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="laya-hf-int8-"))
    root = tmp / args.prefix
    total = 0
    for rel in INCLUDE:
        src = INT8_DIR / rel
        if not src.exists():
            print(f"  ⚠ missing {rel}")
            continue
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        total += src.stat().st_size
        print(f"  + {args.prefix}/{rel}  ({src.stat().st_size/1e6:.1f} MB)")
    print(f"  total: {total/1e6:.0f} MB")

    if args.dry_run:
        print(f"\n(dry run) staged at {tmp}")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("✗ HF_TOKEN not set")
        return 2

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    print(f"\nUploading to {args.repo_id}/{args.prefix}/ ...")
    api.upload_folder(
        folder_path=str(root),
        path_in_repo=args.prefix,
        repo_id=args.repo_id,
        repo_type="model",
    )
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n✓ int8 checkpoint at https://huggingface.co/{args.repo_id}/tree/main/{args.prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())