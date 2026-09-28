#!/usr/bin/env bash
# Download Laya ONNX checkpoint files via curl.
# Bypasses the hf CLI / huggingface_hub SOCKS proxy issue.

set -euo pipefail

REPO_BASE="https://huggingface.co/inferenceprince/laya-onnx/resolve/main"
TARGET="models/laya-onnx-en"

mkdir -p "$TARGET/tokenizer"

download() {
    local relpath="$1"
    local outpath="$2"
    if [ -f "$outpath" ] && [ "$(stat -f%z "$outpath" 2>/dev/null || stat -c%s "$outpath")" -gt 1000 ]; then
        echo "✓ $outpath exists ($(du -h $outpath | cut -f1))"
        return 0
    fi
    echo "↓ $relpath → $outpath"
    mkdir -p "$(dirname "$outpath")"
    curl -L -m 1200 --retry 3 --connect-timeout 60 -o "$outpath" "${REPO_BASE}/${relpath}"
}

download "config.json"                            "$TARGET/config.json"
download "model.onnx"                            "$TARGET/model.onnx"
download "model.onnx.data"                       "$TARGET/model.onnx.data"
download "rl_agent_config.json"                  "$TARGET/rl_agent_config.json"
download "tokenizer/tokenizer.json"              "$TARGET/tokenizer/tokenizer.json"
download "tokenizer/tokenizer_config.json"       "$TARGET/tokenizer/tokenizer_config.json"
download "README.md"                             "$TARGET/README.md"

echo ""
echo "✓ Downloaded all files"
ls -lh "$TARGET"
ls -lh "$TARGET/tokenizer"