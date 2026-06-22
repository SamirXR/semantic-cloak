#!/usr/bin/env bash
# download_weights.sh
# ===================
# Download the pre-trained VLM checkpoints needed by Semantic Cloak.
#
# Usage: bash scripts/download_weights.sh [weights_dir]
#
# Downloads:
#   - SAM ViT-H (~2.5 GB): https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
#   - CLIP ViT-B/32 (~600 MB, downloaded automatically by HuggingFace on first use)
#
# CLIP weights are cached by HuggingFace at ~/.cache/huggingface/ on first
# use, so we only need to download SAM explicitly here.

set -euo pipefail

WEIGHTS_DIR="${1:-weights}"
mkdir -p "$WEIGHTS_DIR"

SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_PATH="$WEIGHTS_DIR/sam_vit_h_4b8939.pth"

if [ -f "$SAM_PATH" ]; then
    echo "[skip] SAM ViT-H already exists at $SAM_PATH"
else
    echo "[download] Fetching SAM ViT-H (~2.5 GB)..."
    wget -O "$SAM_PATH" "$SAM_URL"
    echo "[done] SAM saved to $SAM_PATH"
fi

# Verify
if [ -f "$SAM_PATH" ]; then
    SIZE=$(stat -c%s "$SAM_PATH" 2>/dev/null || stat -f%z "$SAM_PATH")
    if [ "$SIZE" -lt 1000000000 ]; then
        echo "[error] SAM checkpoint is suspiciously small ($SIZE bytes). Download may have failed."
        exit 1
    fi
fi

echo ""
echo "Weights ready. Set in your config:"
echo "  sam_checkpoint: $SAM_PATH"
echo ""
echo "CLIP weights will be downloaded automatically on first use by HuggingFace."
