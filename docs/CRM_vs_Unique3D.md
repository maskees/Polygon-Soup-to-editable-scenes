# CRM vs Unique3D — Backend Comparison

**Sprint 2 Deliverable** | Last updated: July 30, 2026

This document compares the two 3D reconstruction backends used in our pipeline.
Both generate a monolithic 3D mesh from 2D images, but differ significantly in
architecture, compatibility, and output characteristics.

---

## Overview

| Property | **CRM** | **Unique3D** |
|----------|---------|-------------|
| **Full Name** | Convolutional Reconstruction Model | Unique3D |
| **Source** | THU-ML (Tsinghua University) | AiuniAI |
| **Repository** | [thu-ml/CRM](https://github.com/thu-ml/CRM) | [AiuniAI/Unique3D](https://github.com/AiuniAI/Unique3D) |
| **Input** | Single canonical image | Multi-view images (native 4-view) |
| **Architecture** | Two-stage diffusion + FlexiCubes | Multi-view conditioned diffusion |
| **Output** | Textured mesh (.obj) | Textured mesh (.obj) |

---

## Environment Compatibility

| Requirement | **CRM** | **Unique3D** |
|-------------|---------|-------------|
| **Python** | 3.9 | 3.11 |
| **PyTorch** | 1.13 + CUDA 11.7 | 2.3 + CUDA 12.1 |
| **Key deps** | kaolin, nvdiffrast, xformers | diffusers, transformers |
| **Compat. with project** | ❌ Incompatible (needs separate env) | ✅ Compatible (same env) |
| **Integration pattern** | Subprocess isolation (`conda run`) | Subprocess (can also run in-process) |

> **Decision**: CRM requires a separate conda environment (`crm`) due to
> incompatible PyTorch/CUDA versions. We use subprocess isolation — the main
> pipeline invokes `scripts/crm_bridge.py` via `conda run -n crm`.

---

## VRAM Usage (RTX 4060, 8GB)

| Metric | **CRM** | **Unique3D** |
|--------|---------|-------------|
| **Peak VRAM (float32)** | ~6 GB | ~12 GB ❌ (exceeds 8GB) |
| **Peak VRAM (float16)** | ~4 GB ✅ | ~7-8 GB ⚠️ (tight) |
| **Batch size** | 1 image | 4 images |
| **Safe on RTX 4060?** | ✅ Yes | ⚠️ Only with float16 + cache clearing |

> **Risk**: Unique3D at float32 will OOM on our hardware. Must use
> `--low-vram` flag which enables float16 and calls `torch.cuda.empty_cache()`
> between stages.

---

## Output Quality

| Metric | **CRM** | **Unique3D** |
|--------|---------|-------------|
| **Typical face count** | 50K–150K | 100K–500K |
| **Geometry quality** | Good overall shape, smooth surfaces | Finer detail, sharper features |
| **Watertightness** | Usually watertight (FlexiCubes) | May have holes — needs post-processing |
| **Self-intersections** | Occasional — FlexiCubes artifact | Rare |
| **Thin features** | May miss fine details (hair, fingers) | Better preservation of thin features |
| **Topology** | Clean quads/tris from isosurface | Dense triangles, may need decimation |
| **Texture** | Generates textures (disabled in our pipeline) | Generates textures (disabled in our pipeline) |

---

## Speed (RTX 4060)

| Metric | **CRM** | **Unique3D** |
|--------|---------|-------------|
| **Model loading** | ~15-20s | ~20-30s |
| **Inference (float32)** | ~30-45s | ~60-90s |
| **Inference (float16)** | ~20-30s | ~45-60s |
| **Total pipeline time** | ~50-65s | ~80-120s |

> **CRM is ~2× faster** for our use case. For midterm demo, CRM is the safer
> choice for live demonstrations.

---

## Failure Modes

### CRM
| Failure | Cause | Mitigation |
|---------|-------|------------|
| **OOM** | Large diffusion backbone | Use `--low-vram` (float16) |
| **Poor geometry** | Ambiguous single-view input | Use front view with clear subject |
| **Self-intersecting faces** | FlexiCubes artifact | Post-process with PyMeshLab |
| **Environment setup fails** | kaolin/nvdiffrast build issues | Use `setup_crm_env.ps1` script |
| **Missing checkpoints** | ~4GB download required | Use `download_checkpoints.ps1` |

### Unique3D
| Failure | Cause | Mitigation |
|---------|-------|------------|
| **OOM** | Multi-view processing is memory-heavy | Must use float16 on 8GB GPU |
| **Excessive face count** | Default output >500K faces | Decimate to 50K in post-processing |
| **Non-watertight output** | Open boundaries | PyMeshLab hole filling |
| **Slow inference** | Larger model, more views | Accept longer runtime or use CRM |

---

## Recommendation

### Primary Backend: **CRM**
- ✅ Fits 8GB VRAM comfortably
- ✅ Faster inference (~50s vs ~90s)
- ✅ Usually produces watertight output
- ⚠️ Requires separate conda environment (automated by setup script)
- ⚠️ Single-view input (uses front view only)

### Secondary Backend: **Unique3D**
- ✅ Native multi-view input (architecturally better fit)
- ✅ Finer geometric detail
- ⚠️ Tight VRAM — requires float16
- ⚠️ Slower
- ⚠️ May need more aggressive post-processing

### For the Capstone Report
Both backends should be run on the same test subjects for comparison.
Key metrics to report:
1. **Chamfer Distance** — geometric accuracy vs ground truth
2. **Watertightness** — before and after post-processing
3. **Face count** — raw output vs after decimation
4. **Inference time** — end-to-end per subject
5. **Downstream quality** — does SAMPart3D segment CRM vs Unique3D meshes differently?

---

## Pipeline Integration Summary

```
Input (4 RGBA images)
    │
    ├─── CRM path ────────────────────────────────┐
    │    1. Use front view only                    │
    │    2. conda run -n crm scripts/crm_bridge.py │
    │    3. CRM generates 6 internal views         │
    │    4. FlexiCubes → mesh                      │
    │                                              │
    ├─── Unique3D path ───────────────────────────┤
    │    1. Use all 4 views                        │
    │    2. conda run scripts/unique3d_bridge.py   │
    │    3. Multi-view diffusion → mesh            │
    │                                              │
    └────────────────┬─────────────────────────────┘
                     │
            Post-processing
            1. Remove degenerate faces
            2. Decimate to 50K faces
            3. Laplacian smoothing
            4. Fix normals
            5. Fill holes
            6. Normalize bounding box
                     │
            monolithic_mesh.obj
```
