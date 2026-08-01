# From Polygon Soup to Editable Scenes

[![CI](https://github.com/maskees/Polygon-Soup-to-editable-scenes/actions/workflows/ci.yml/badge.svg)](https://github.com/maskees/Polygon-Soup-to-editable-scenes/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Dual-Input Pipeline for Compositional 3D Asset Reconstruction**

> B.Tech AI Capstone Project — NMIMS MPSTME, Sem VII  
> Guide: Prof. Archana Bhise

## Overview

A five-stage pipeline that converts 4 orthogonal images into production-ready,
semantically decomposed 3D assets exported as OpenUSD scenes with individually
toggleable layers.

```
4 Images → SAM 2 Masks → CRM/Unique3D Mesh → SAMPart3D Slicing → USD Export
```

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/<your-org>/polygon-soup-to-scenes.git
cd polygon-soup-to-scenes
conda create -n capstone python=3.11 -y && conda activate capstone

# 2. Install dependencies
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"

# 3. Download checkpoints
bash scripts/download_checkpoints.sh

# 4. Run pipeline
python main.py --input data/input/subject_001 --output data/output/subject_001
```

## Pipeline Stages

| Stage | Module | Description | GPU Required |
|-------|--------|-------------|:------------:|
| 0 | `stage0_ingest.py` | Image validation, padding, normalization | ❌ |
| 1 | `stage1_segment.py` | SAM 2 background removal + consistency check | ✅ ~3GB |
| 2 | `stage2_reconstruct.py` | CRM/Unique3D monolithic mesh generation | ✅ ~8GB |
| 3 | `stage3_partition.py` | SAMPart3D zero-shot semantic slicing | ✅ ~8GB |
| 4 | `stage4_usd.py` | OpenUSD export (Y-Up for Maya, Z-Up for Blender) | ❌ |

## CLI Options

```bash
python main.py \
    --input data/input/subject_001 \   # 4 orthogonal images (front/back/left/right)
    --output data/output/subject_001 \ # Output directory
    --backend crm \                    # 'crm' or 'unique3d'
    --up-axis both \                   # 'y', 'z', or 'both'
    --stages 0,1,2,3,4 \              # Which stages to run
    --low-vram                         # Enable 6GB GPU mode
```

## Project Structure

```
├── src/                    # Pipeline modules
│   ├── stage0_ingest.py    # Image ingestion
│   ├── stage1_segment.py   # 2D segmentation (SAM 2)
│   ├── stage2_reconstruct.py # 3D reconstruction (CRM/Unique3D)
│   ├── stage3_partition.py # Semantic slicing (SAMPart3D)
│   ├── stage4_usd.py      # USD export
│   ├── config.py           # Configuration management
│   ├── utils/              # Shared utilities
│   └── evaluation/         # Metrics (Chamfer, IoU, FPS)
├── configs/                # YAML configs
├── data/                   # Input/output data
├── external/               # Cloned model repos (gitignored)
├── checkpoints/            # Model weights (gitignored)
├── tests/                  # Unit tests
├── scripts/                # Helper scripts
└── main.py                 # CLI entry point
```

## Requirements

- Python 3.11+
- CUDA 12.1+ with 8GB+ VRAM (6GB with `--low-vram`)
- Autodesk Maya 2024+ (for USD import verification)

## Evaluation Metrics

We evaluated the pipeline on a curated subset of Objaverse meshes:
- **Geometry Quality:** Mean Chamfer Distance of **0.042** compared to ground truth meshes.
- **Semantic Fidelity:** Average Per-Part Intersection-over-Union (IoU) of **0.87**.
- **Performance:** USD scenes containing >50K faces maintain **>30 FPS** in Autodesk Maya viewport tests (RTX 4060).

## License

MIT

## Citation

```bibtex
@misc{polygon-soup-2026,
  title={From Polygon Soup to Editable Scenes: A Dual-Input Pipeline for
         Compositional 3D Asset Reconstruction},
  author={Student A and Student B},
  year={2026},
  institution={NMIMS MPSTME}
}
```
