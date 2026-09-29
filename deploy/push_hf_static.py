"""Publish the browser-inference frontend as a Hugging Face Static Space.

Static Spaces are free for everyone (Gradio/Docker Spaces require a paid
plan). The Space ships only HTML/CSS/JS; the model is fetched from the
public HF model repo at runtime, so no compute is involved.

Usage:
  HF_TOKEN=hf_xxx .venv/bin/python deploy/push_hf_static.py \
      --repo-id <user>/laya-scam-detector
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web-static"
SPACE_README = REPO_ROOT / "deploy" / "hf-static-README.md"

INCLUDE = ["index.html", "app.js", "style.css"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for name in INCLUDE:
        if not (WEB_DIR / name).exists():
            print(f"✗ missing web-static/{name}")
            return 1

    tmp = Path(tempfile.mkdtemp(prefix="laya-static-"))
    for name in INCLUDE:
        shutil.copy(WEB_DIR / name, tmp / name)
        print(f"  + {name}  ({(WEB_DIR / name).stat().st_size/1024:.1f} KB)")
    shutil.copy(SPACE_README, tmp / "README.md")
    print("  + README.md (static Space card)")

    if args.dry_run:
        print(f"\n(dry run) staged at {tmp}")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("✗ HF_TOKEN not set")
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"\nCreating Static Space {args.repo_id} (private={args.private})...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="static",
        private=args.private,
        exist_ok=True,
    )
    print("Uploading...")
    api.upload_folder(
        folder_path=str(tmp),
        repo_id=args.repo_id,
        repo_type="space",
    )
    shutil.rmtree(tmp, ignore_errors=True)

    url = f"https://huggingface.co/spaces/{args.repo_id}"
    print(f"\n✓ Space published: {url}")
    print(f"  Live app: https://{args.repo_id.replace('/', '-').lower()}.hf.space")
    return 0


if __name__ == "__main__":
    sys.exit(main())