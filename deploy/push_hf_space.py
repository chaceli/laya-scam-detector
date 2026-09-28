"""Publish this project as a Hugging Face Docker Space.

Assembles a minimal tree (Dockerfile + server/ + src/ + schemas/ + web/),
swaps in the Space README with HF frontmatter, creates the Space and
uploads everything.

Requires an HF token with write access, read from HF_TOKEN or the
huggingface_hub cache.

Usage:
  HF_TOKEN=hf_xxx .venv/bin/python deploy/push_hf_space.py \
      --repo-id <user>/laya-scam-detector --private
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_README = REPO_ROOT / "deploy" / "hf-space-README.md"

INCLUDE_FILES = ["Dockerfile", ".dockerignore", "main.py"]
INCLUDE_DIRS = ["server", "src", "schemas", "web"]


def assemble(target: Path) -> None:
    for name in INCLUDE_FILES:
        src = REPO_ROOT / name
        if src.exists():
            shutil.copy(src, target / name)
    for name in INCLUDE_DIRS:
        src = REPO_ROOT / name
        if not src.exists():
            print(f"  ⚠ missing {name}, skipping")
            continue
        shutil.copytree(
            src, target / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]", "*.egg-info"),
        )
    shutil.copy(SPACE_README, target / "README.md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="e.g. <user>/laya-scam-detector")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble the tree but skip create/upload")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("✗ HF_TOKEN not set (or pass --dry-run to only assemble)")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="laya-space-"))
    print(f"Assembling Space tree at {tmp}")
    assemble(tmp)

    total = sum(f.stat().st_size for f in tmp.rglob("*") if f.is_file())
    files = [p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file()]
    print(f"  {len(files)} files, {total/1e6:.1f} MB")
    for f in sorted(files):
        print(f"    {f}")

    if args.dry_run:
        print(f"\n(dry run) tree left at {tmp}")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"\nCreating Space {args.repo_id} (docker, private={args.private})...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    print("Uploading files...")
    api.upload_folder(
        folder_path=str(tmp),
        repo_id=args.repo_id,
        repo_type="space",
        ignore_patterns=["**/__pycache__/**", "*.pyc"],
    )
    shutil.rmtree(tmp, ignore_errors=True)

    url = f"https://huggingface.co/spaces/{args.repo_id}"
    print(f"\n✓ Space published: {url}")
    print(f"  Build logs: {url}?logs=build")
    return 0


if __name__ == "__main__":
    sys.exit(main())