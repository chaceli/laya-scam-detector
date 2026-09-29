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
TOKENIZER_SRC = REPO_ROOT / "models" / "laya-onnx-multilingual-finetuned-fp16" / "tokenizer"
INCLUDE_OPTIONAL = ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]


def pack(zip_path: Path) -> int:
    """Bundle the Space files for drag-and-drop upload via the HF web UI.

    Includes the tokenizer subdir (~34 MB) so the Space doesn't depend on
    transformers.js's model-id string parsing for the tokenizer.
    """
    import zipfile

    for name in INCLUDE:
        if not (WEB_DIR / name).exists():
            print(f"✗ missing web-static/{name}")
            return 1
    if not TOKENIZER_SRC.is_dir():
        print(f"✗ tokenizer dir missing: {TOKENIZER_SRC}")
        return 1
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = list(INCLUDE) + ["README.md"] + INCLUDE_OPTIONAL
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in INCLUDE:
            zf.write(WEB_DIR / name, name)
        zf.write(SPACE_README, "README.md")
        for rel in INCLUDE_OPTIONAL:
            src = TOKENIZER_SRC / Path(rel).name
            zf.write(src, rel)
    size_kb = zip_path.stat().st_size / 1024
    print(f"✓ {zip_path}  ({size_kb:.1f} KB, {len(files)} files)")
    print("  Upload at https://huggingface.co/new-space (SDK: Static)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", help="e.g. <user>/laya-scam-detector")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pack", metavar="ZIP",
                    help="write a drag-and-drop zip for the HF web UI and exit (no token needed)")
    args = ap.parse_args()

    if args.pack:
        return pack(Path(args.pack))
    if not args.repo_id:
        print("✗ --repo-id is required unless --pack is used")
        return 2

    for name in INCLUDE:
        if not (WEB_DIR / name).exists():
            print(f"✗ missing web-static/{name}")
            return 1
    if not TOKENIZER_SRC.is_dir():
        print(f"✗ tokenizer dir missing: {TOKENIZER_SRC}")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="laya-static-"))
    total = 0
    for name in INCLUDE:
        src = WEB_DIR / name
        dst = tmp / name
        shutil.copy(src, dst)
        size = src.stat().st_size
        total += size
        print(f"  + {name}  ({size/1024:.1f} KB)")
    shutil.copy(SPACE_README, tmp / "README.md")
    print(f"  + README.md (static Space card) ({len(SPACE_README.read_text())/1024:.1f} KB)")
    # Bundle the tokenizer so the Space doesn't depend on transformers.js
    # parsing a remote URL as a model id.
    for rel in INCLUDE_OPTIONAL:
        src = TOKENIZER_SRC / Path(rel).name
        if not src.exists():
            print(f"  ⚠ missing tokenizer/{Path(rel).name}")
            continue
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        size = src.stat().st_size
        total += size
        print(f"  + {rel}  ({size/1024:.1f} KB)")
    print(f"  total: {total/1e6:.1f} MB")

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