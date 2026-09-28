#!/usr/bin/env bash
# Download Laya ONNX checkpoints (English + Multilingual) from Hugging Face.
# Idempotent: skips already-downloaded files.

set -euo pipefail

REPO_EN="inferenceprince/laya-onnx"
REPO_ML="inferenceprince/laya-multilingual-onnx"  # may not exist yet
TARGET_EN="models/laya-onnx-en"
TARGET_ML="models/laya-onnx-multilingual"

download() {
    local repo="$1"
    local target="$2"
    if [ -d "$target" ] && [ -f "$target/model.onnx" ]; then
        echo "✓ $target already exists, skipping"
        return 0
    fi
    echo "↓ Downloading $repo → $target"
    mkdir -p "$target"
    if ! hf download "$repo" --local-dir "$target"; then
        echo "✗ Failed to download $repo"
        return 1
    fi
}

download "$REPO_EN" "$TARGET_EN"

if hf download "$REPO_ML" --local-dir "$TARGET_ML" 2>/dev/null; then
    echo "✓ Multilingual checkpoint downloaded"
else
    echo "⚠ Multilingual checkpoint unavailable; fallback to English for non-Latin"
fi