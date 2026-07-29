#!/bin/bash
# Download all model checkpoints
# ================================
# Run from project root: bash scripts/download_checkpoints.sh
#
# This script is the Linux/WSL companion to download_checkpoints.ps1.
# Both scripts download the same checkpoints to the same directory structure.

set -euo pipefail

CHECKPOINT_DIR="checkpoints"

echo "=== Downloading Model Checkpoints ==="
echo ""

# Ensure all checkpoint directories exist
for dir in sam2 crm unique3d sampart3d; do
    mkdir -p "$CHECKPOINT_DIR/$dir"
done

# ── SAM 2 (ViT-H) ──
echo "[1/4] Downloading SAM 2 (ViT-H)..."
SAM2_PATH="$CHECKPOINT_DIR/sam2/sam2_hiera_large.pt"
if [ ! -f "$SAM2_PATH" ]; then
    SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large.pt"
    echo "  Downloading from $SAM2_URL ..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$SAM2_URL" -O "$SAM2_PATH" && \
            echo "  ✓ SAM 2 downloaded" || \
            echo "  ✗ SAM 2 download failed. Manual download: $SAM2_URL"
    elif command -v curl &> /dev/null; then
        curl -# -L "$SAM2_URL" -o "$SAM2_PATH" && \
            echo "  ✓ SAM 2 downloaded" || \
            echo "  ✗ SAM 2 download failed. Manual download: $SAM2_URL"
    else
        echo "  ✗ Neither wget nor curl found. Manual download: $SAM2_URL"
    fi
else
    echo "  ✓ SAM 2 already exists"
fi

# ── CRM ──
echo "[2/4] Downloading CRM checkpoints..."
CRM_MODEL_DIR="$CHECKPOINT_DIR/crm/model"
if [ ! -d "$CRM_MODEL_DIR" ]; then
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download Zhengyi/CRM --local-dir "$CHECKPOINT_DIR/crm" && \
            echo "  ✓ CRM downloaded" || \
            echo "  ✗ CRM download failed"
    else
        echo "  ⚠ huggingface-cli not found. Install it with:"
        echo "    pip install huggingface-hub"
        echo "    Then run: huggingface-cli download Zhengyi/CRM --local-dir $CHECKPOINT_DIR/crm"
    fi
else
    echo "  ✓ CRM already exists"
fi

# ── Unique3D ──
echo "[3/4] Downloading Unique3D checkpoints..."
UNIQUE3D_MODEL_DIR="$CHECKPOINT_DIR/unique3d/model"
if [ ! -d "$UNIQUE3D_MODEL_DIR" ]; then
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download aiuni/Unique3D --local-dir "$CHECKPOINT_DIR/unique3d" && \
            echo "  ✓ Unique3D downloaded" || \
            echo "  ✗ Unique3D download failed"
    else
        echo "  ⚠ huggingface-cli not found. Install it with:"
        echo "    pip install huggingface-hub"
        echo "    Then run: huggingface-cli download aiuni/Unique3D --local-dir $CHECKPOINT_DIR/unique3d"
    fi
else
    echo "  ✓ Unique3D already exists"
fi

# ── SAMPart3D ──
echo "[4/4] SAMPart3D checkpoints..."
SAMPART3D_MODEL_DIR="$CHECKPOINT_DIR/sampart3d/model"
if [ ! -d "$SAMPART3D_MODEL_DIR" ]; then
    echo "  ⚠ SAMPart3D checkpoints must be downloaded manually."
    echo "    Check: external/SAMPart3D/README.md for download instructions"
    echo "    Place files in: $CHECKPOINT_DIR/sampart3d/"
else
    echo "  ✓ SAMPart3D already exists"
fi

echo ""
echo "=== Download Complete ==="
echo "Verify with: python -c \"from pxr import Usd; import torch; print('OK')\""
