#!/bin/bash
# Download all model checkpoints
# ================================
# Run from project root: bash scripts/download_checkpoints.sh

set -e

CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "=== Downloading Model Checkpoints ==="
echo ""

# ── SAM 2 ──
echo "[1/4] Downloading SAM 2 (ViT-H)..."
mkdir -p "$CHECKPOINT_DIR/sam2"
if [ ! -f "$CHECKPOINT_DIR/sam2/sam2_hiera_large.pt" ]; then
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large.pt" \
        -O "$CHECKPOINT_DIR/sam2/sam2_hiera_large.pt"
    echo "  ✓ SAM 2 downloaded"
else
    echo "  ✓ SAM 2 already exists"
fi

# ── CRM ──
echo "[2/4] Downloading CRM checkpoints..."
mkdir -p "$CHECKPOINT_DIR/crm"
if [ ! -d "$CHECKPOINT_DIR/crm/model" ]; then
    huggingface-cli download Zhengyi/CRM --local-dir "$CHECKPOINT_DIR/crm" 2>/dev/null || \
        echo "  ⚠ CRM download failed — install huggingface-cli or download manually"
    echo "  ✓ CRM downloaded"
else
    echo "  ✓ CRM already exists"
fi

# ── Unique3D ──
echo "[3/4] Downloading Unique3D checkpoints..."
mkdir -p "$CHECKPOINT_DIR/unique3d"
if [ ! -d "$CHECKPOINT_DIR/unique3d/model" ]; then
    huggingface-cli download aiuni/Unique3D --local-dir "$CHECKPOINT_DIR/unique3d" 2>/dev/null || \
        echo "  ⚠ Unique3D download failed — install huggingface-cli or download manually"
    echo "  ✓ Unique3D downloaded"
else
    echo "  ✓ Unique3D already exists"
fi

# ── SAMPart3D ──
echo "[4/4] Downloading SAMPart3D checkpoints..."
mkdir -p "$CHECKPOINT_DIR/sampart3d"
if [ ! -d "$CHECKPOINT_DIR/sampart3d/model" ]; then
    echo "  ⚠ SAMPart3D checkpoints must be downloaded manually."
    echo "    Check: external/SAMPart3D/README.md for download instructions"
    echo "    Place files in: $CHECKPOINT_DIR/sampart3d/"
else
    echo "  ✓ SAMPart3D already exists"
fi

echo ""
echo "=== Download Complete ==="
echo "Verify with: python -c \"from pxr import Usd; import torch; print('OK')\""
