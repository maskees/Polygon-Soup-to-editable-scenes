# From Polygon Soup to Editable Scenes: Full Capstone Execution Plan

**Project**: Dual-Input Pipeline for Compositional 3D Asset Reconstruction  
**Guide**: Prof. Archana Bhise | **University**: NMIMS MPSTME, Sem VII  
**Team**: Student A + Student B | **Timeline**: 10 weeks (5 × 2-week sprints)  
**GPU**: NVIDIA RTX 4060 (8GB VRAM) | **Midterm Review**: August 15, 2026

---

## ⚠️ User Review Required

### ✅ Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|----------|
| **Stage 2 Segmentation** | **SAM 2 (ViT-H)** | Stable, well-supported, clean per-view masks + cross-view IoU consistency check |
| **Stage 3 Reconstruction** | **CRM first → Unique3D second** | CRM fits 8GB VRAM comfortably (~6GB); Unique3D as quality comparison with `float16` |
| **GPU** | **RTX 4060 (8GB VRAM)** | CRM ✅, SAMPart3D ✅ (batched), Unique3D ⚠️ (tight with float16) |
| **Midterm** | **August 15, 2026** | Sprints 1-3 compressed; Stages 0-3 must be demo-ready |

### RTX 4060 VRAM Budget (per stage, sequential execution)

| Stage | Peak VRAM | Notes |
|-------|-----------|-------|
| 0 — Ingest | ~0 GB (CPU) | No GPU needed |
| 1 — SAM 2 | ~3.5 GB | ViT-H in float32; ~2GB in float16 |
| 2 — CRM | ~6 GB | FlexiCubes + diffusion backbone |
| 2 — Unique3D | ~8 GB | Tight — requires float16 + `torch.cuda.empty_cache()` |
| 3 — SAMPart3D | ~6 GB | DINOv2 ViT-L + PTv3; batch views in groups of 4 |
| 4 — USD | ~0 GB (CPU) | No GPU needed |

---

## 1. SPRINT PLAN (10 Weeks, 5 × 2-Week Sprints)

### Sprint 1 — Foundation (Jul 27 – Aug 3)
**Theme**: Environment, data pipeline, and proof-of-life | ⏱️ 8 days

| Task | Owner | Details |
|------|-------|---------|
| Set up conda env + all dependencies | **A** | See §2 Environment Setup |
| Build Stage 1: `stage0_ingest.py` | **A** | Image loading, validation, padding, normalization |
| Build Stage 2: `stage1_segment.py` | **B** | SAM 2 integration, per-view mask extraction |
| Curate 5 test image sets (4 views each) | **B** | Download from Objaverse renders or photograph |
| Set up GitHub repo with CI skeleton | **A** | `.gitignore`, `README.md`, `pyproject.toml` |
| Cross-view silhouette consistency check | **B** | IoU between front/back and left/right silhouettes |

**Milestone**: Given 4 input images → produce 4 clean RGBA images with <1% background bleed.

**Failure/Fallback**: If SAM 2 masks are inconsistent, fall back to **rembg** (U²-Net background removal) which is more deterministic but less precise on fine detail.

---

### Sprint 2 — 3D Reconstruction Core (Aug 4 – Aug 10)
**Theme**: Monolithic mesh generation | ⏱️ 7 days

| Task | Owner | Details |
|------|-------|---------|
| Integrate CRM pipeline | **A** | Clone `thu-ml/CRM`, download checkpoints, adapt input format |
| Integrate Unique3D pipeline (backup) | **B** | Clone `AiuniAI/Unique3D`, test with 4-view input |
| Build `stage2_reconstruct.py` wrapper | **A** | Unified interface: `reconstruct(images) → mesh` |
| Mesh quality validation script | **B** | Watertight check, face count, bounding box normalization |
| Build `main.py` CLI pipeline runner | **A** | Argument parsing, stage orchestration |
| Document CRM vs Unique3D comparison | **B** | Quality, speed, VRAM, failure modes |

**Milestone**: Given 4 RGBA images → produce a monolithic `.obj` mesh with >10K faces and correct proportions.

**Failure/Fallback**: If CRM output is geometrically poor, try **Zero123++ → LRM** pipeline (more established but slower). If VRAM OOM, use CRM's half-precision mode or offload to Colab.

---

### Sprint 3 — Semantic Slicing + Midterm Prep (Aug 11 – Aug 15) 🎯 MIDTERM
**Theme**: 3D part segmentation — must be demo-ready by Aug 15 | ⏱️ 5 days (compressed)

| Task | Owner | Details |
|------|-------|---------|
| Set up SAMPart3D environment | **A** | Clone repo, install PTv3, download checkpoints |
| Adapt SAMPart3D for custom meshes | **A** | Write mesh → point cloud adapter |
| Build `stage3_partition.py` | **B** | Face clustering → sub-mesh extraction via Trimesh |
| Implement granularity control | **B** | Scale-conditioned feature thresholds |
| Sub-mesh export (individual `.obj` files) | **A** | Clean topology, no degenerate faces |
| Boundary smoothing post-process | **B** | Laplacian smoothing on partition boundaries |

**Milestone**: Given monolithic mesh → produce N labeled sub-meshes (e.g., head, torso, arms, legs for a humanoid) with clean boundaries.

**Failure/Fallback**: If SAMPart3D fails on your mesh topology, implement a **fallback K-means clustering** on vertex positions + normals. Less semantic but geometrically valid. Another fallback: use SAMPart3D's pre-trained features but apply DBSCAN instead of their default clustering.

> **CAUTION — Highest OOM Risk Stage**: SAMPart3D loads DINOv2 ViT-L + Point Transformer V3 simultaneously. On 8GB VRAM, render views at 224×224 (not 512×512) and process in batches of 4 views.

---

### Sprint 4 — USD Export & Maya Integration (Aug 16 – Aug 31)
**Theme**: Production-ready output | ⏱️ 16 days (post-midterm, relaxed)

| Task | Owner | Details |
|------|-------|---------|
| Build `stage4_usd.py` | **A** | USD scene graph construction with pxr |
| Implement Y-Up variant (Maya) | **A** | `UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)` |
| Implement Z-Up variant (Blender/UE5) | **B** | `UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)` |
| Layer visibility toggles | **B** | `UsdGeom.Imageable.MakeInvisible()` per prim |
| Maya import testing | **A** | Verify Outliner hierarchy, toggle layers |
| Blender import testing | **B** | Verify correct orientation and scale |
| End-to-end pipeline test (all 5 stages) | **Both** | 3 different subjects, automated |

**Milestone**: Import `.usda` into Maya → see semantic parts as toggleable layers in the Outliner. Same file works in Blender with Z-Up variant.

**Failure/Fallback**: If `usd-core` has compatibility issues with your Python/OS, export as `.usdz` (Apple's subset) or fall back to FBX export via Blender's Python API as a demo alternative.

---

### Sprint 5 — Evaluation, Paper & Polish (Sep 1 – Sep 14)
**Theme**: Metrics, report, demo video | ⏱️ 14 days

| Task | Owner | Details |
|------|-------|---------|
| Chamfer Distance evaluation | **A** | PyTorch3D against ground truth meshes |
| Per-part IoU evaluation | **B** | Voxelized IoU via Trimesh |
| Viewport FPS benchmark in Maya | **A** | Maya `cmds.playblast` profiling |
| Record demo video | **B** | Screen recording: input → pipeline → Maya demo |
| Write capstone report (NMIMS format) | **Both** | Split sections per §6 below |
| Prepare final presentation + viva | **Both** | Slides, live demo fallback video |
| Draft Scopus paper abstract | **A** | See §6 Research Paper Strategy |
| GitHub repo cleanup + README | **B** | Installation guide, usage examples, badges |

**Milestone**: Complete capstone report, demo video, GitHub repo, and draft paper abstract ready for submission.

**Failure/Fallback**: If quantitative metrics are weak, pivot the paper to focus on the *pipeline architecture* contribution rather than raw metric improvement.

---

## 2. ENVIRONMENT SETUP CHECKLIST

### 2.1 CUDA Compatibility Matrix

| Component | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 |
|-----------|-----------|-----------|-----------|
| PyTorch 2.3+ | ✅ | ✅ | ✅ |
| CRM | ✅ (tested) | ✅ | ⚠️ untested |
| Unique3D | ❌ | ✅ (required) | ⚠️ untested |
| SAMPart3D (PTv3) | ✅ | ✅ | ✅ |
| PyTorch3D | ✅ | ✅ | ⚠️ build from source |

**Recommended**: **CUDA 12.1** with **PyTorch 2.3.1** for maximum compatibility.

### 2.2 Conda Environment Setup

```bash
# Create base environment
conda create -n capstone python=3.11 -y
conda activate capstone

# ── Core ML Stack ──
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ── 3D Geometry ──
pip install trimesh[all]==4.4.1
pip install pymeshlab==2023.12.post2
pip install open3d==0.18.0

# ── Image Processing ──
pip install opencv-python==4.10.0.84
pip install Pillow==10.4.0
pip install rembg==2.0.59              # Fallback background removal

# ── SAM 2 (Meta) ──
pip install segment-anything-2          # Or install from source:
# git clone https://github.com/facebookresearch/sam2.git
# cd sam2 && pip install -e .

# ── OpenUSD ──
pip install usd-core==24.8

# ── Evaluation ──
# PyTorch3D — install from source for CUDA 12.1
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# ── Utilities ──
pip install numpy==1.26.4
pip install scipy==1.13.1
pip install matplotlib==3.9.1
pip install tqdm==4.66.4
pip install click==8.1.7               # CLI framework
pip install rich==13.7.1               # Pretty logging
pip install pyyaml==6.0.1

# ── Dev Tools ──
pip install pytest==8.2.2
pip install black==24.4.2
pip install ruff==0.5.0
```

### 2.3 External Repos to Clone (inside `external/`)

```bash
cd d:\capstone

# CRM (primary 3D reconstruction)
git clone https://github.com/thu-ml/CRM.git external/CRM
cd external/CRM && pip install -r requirements.txt && cd ../..

# Unique3D (backup 3D reconstruction)
git clone https://github.com/AiuniAI/Unique3D.git external/Unique3D
cd external/Unique3D && pip install -r requirements.txt && cd ../..

# SAMPart3D (3D semantic segmentation)
git clone https://github.com/yhyang-myron/SAMPart3D.git external/SAMPart3D
cd external/SAMPart3D && pip install -r requirements.txt && cd ../..
```

### 2.4 Model Checkpoints

| Model | Source | Size | Download |
|-------|--------|------|----------|
| SAM 2 (ViT-H) | Meta | ~2.4GB | `https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large.pt` |
| CRM (all stages) | HuggingFace | ~4GB | `huggingface-cli download Zhengyi/CRM --local-dir checkpoints/crm` |
| Unique3D | HuggingFace | ~6GB | `huggingface-cli download aiuni/Unique3D --local-dir checkpoints/unique3d` |
| SAMPart3D (PTv3 + DINOv2) | HuggingFace/GitHub | ~3GB | Check `external/SAMPart3D/README.md` for exact links |
| DINOv2 ViT-L/14 | Meta | ~1.2GB | Auto-downloaded by SAMPart3D via `torch.hub` |

### 2.5 Verification Script

```bash
python -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import trimesh; print(f'Trimesh: {trimesh.__version__}')
import cv2; print(f'OpenCV: {cv2.__version__}')
from pxr import Usd; print(f'USD: {Usd.GetVersion()}')
print('All core dependencies OK')
"
```

---

## 3. CODE ARCHITECTURE

### 3.1 Repository Structure

```
d:\capstone\
├── .github/
│   └── workflows/
│       └── ci.yml                    # Lint + unit tests
├── external/                         # Cloned repos (gitignored)
│   ├── CRM/
│   ├── Unique3D/
│   └── SAMPart3D/
├── checkpoints/                      # Model weights (gitignored)
│   ├── sam2/
│   ├── crm/
│   ├── unique3d/
│   └── sampart3d/
├── data/
│   ├── input/                        # Raw 4-view image sets
│   │   └── subject_001/
│   │       ├── front.png
│   │       ├── back.png
│   │       ├── left.png
│   │       └── right.png
│   ├── intermediate/                 # Stage outputs (auto-generated)
│   └── output/                       # Final USD exports
├── src/
│   ├── __init__.py
│   ├── stage0_ingest.py              # Image ingestion & validation
│   ├── stage1_segment.py             # 2D segmentation (SAM 2)
│   ├── stage2_reconstruct.py         # 3D mesh generation (CRM/Unique3D)
│   ├── stage3_partition.py           # 3D semantic slicing (SAMPart3D)
│   ├── stage4_usd.py                 # OpenUSD export
│   ├── config.py                     # Central configuration
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── mesh_utils.py            # Trimesh helpers
│   │   ├── image_utils.py           # CV2/PIL helpers
│   │   ├── gpu_utils.py             # VRAM monitoring, cache clearing
│   │   └── logging_utils.py         # Rich-based logger
│   └── evaluation/
│       ├── __init__.py
│       ├── chamfer.py               # Chamfer Distance computation
│       ├── iou.py                   # Per-part IoU
│       └── fps_benchmark.py         # Maya viewport FPS
├── tests/
│   ├── test_stage0.py
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   └── test_stage4.py
├── configs/
│   ├── default.yaml                 # Default pipeline config
│   └── low_vram.yaml                # 6GB VRAM config overrides
├── scripts/
│   ├── download_checkpoints.sh      # Automated checkpoint downloader
│   └── maya_import_test.py          # Maya MEL/Python test script
├── docs/
│   ├── report/                      # Capstone report (LaTeX or Word)
│   └── figures/                     # Pipeline diagrams
├── main.py                          # CLI entry point
├── pyproject.toml                   # Project metadata + dependencies
├── README.md
├── LICENSE
└── .gitignore
```

---

## 4. STAGE-BY-STAGE IMPLEMENTATION GUIDE

### Stage 0 — Image Ingestion (`stage0_ingest.py`)

**Key Decisions:**
1. **Strict 4-image validation**: Exactly `front.png`, `back.png`, `left.png`, `right.png` — reject anything else
2. **Aspect ratio handling**: Pad non-square images with black bars (not crop/stretch)
3. **Normalization**: Resize to 512×512 RGB using `cv2.INTER_LANCZOS4` (best for downscaling)
4. **Color space**: Convert to RGB if input is BGR (OpenCV default) or RGBA
5. **Output format**: Save as lossless PNG to `intermediate/stage0/`

**Known Pitfalls:**
- Phone photos may have EXIF rotation — use `PIL.ImageOps.exif_transpose()`
- Some renderers output 16-bit PNGs — clamp to 8-bit

**Function Signatures:**
```python
def run_ingestion(context: dict) -> dict
def validate_input_directory(input_dir: Path) -> list[Path]
def pad_to_square(image: np.ndarray) -> np.ndarray
def normalize_image(image: np.ndarray, target_size: int = 512) -> np.ndarray
def process_single_image(path: Path, output_dir: Path, target_size: int = 512) -> Path
```

**OOM Risk**: 🟢 None — CPU only.

---

### Stage 1 — 2D Segmentation (`stage1_segment.py`)

**Key Decisions:**
1. **Use SAM 2 (ViT-H)** for mask generation — automatic mode, select largest mask per view
2. **Post-processing**: Morphological close (5×5 kernel) to fill small holes in masks
3. **Alpha channel**: Binary mask (0 or 255), not soft alpha — cleaner for 3D reconstruction
4. **Cross-view consistency**: Compute silhouette IoU between front↔back, left↔right pairs; warn if <0.7
5. **Fallback**: If SAM 2 fails or masks are poor, switch to `rembg` which uses U²-Net

**Known Pitfalls:**
- SAM 2 automatic mode can return dozens of masks — pick the one with largest area
- Thin features (hair strands, fingers) may get clipped
- Background with similar color to subject causes mask bleeding

**Function Signatures:**
```python
def run_segmentation(context: dict) -> dict
def load_sam2_model(checkpoint: Path, device: str = "cuda") -> SAM2Model
def segment_single_view(model: SAM2Model, image: np.ndarray) -> np.ndarray
def check_cross_view_consistency(masks: dict[str, np.ndarray]) -> float
def apply_rembg_fallback(image: np.ndarray) -> np.ndarray
```

**OOM Risk**: 🟡 Medium — SAM 2 ViT-H needs ~3-4GB VRAM. Use `float16` on 6GB GPUs.

---

### Stage 2 — 3D Reconstruction (`stage2_reconstruct.py`)

**Key Decisions:**
1. **CRM (primary)**: Use front view as primary; CRM generates its own multi-view internally
2. **Untextured output**: Disable CRM's texture stage — uniform gray vertex colors
3. **Mesh post-processing**: `trimesh.smoothing.filter_laplacian()` to clean artifacts
4. **Watertight enforcement**: PyMeshLab's `close_holes()` if not manifold
5. **Bounding box normalization**: Center at origin, scale to unit cube

**Known Pitfalls:**
- CRM expects a *single* image → use front view as primary input
- CRM's FlexiCubes can produce self-intersecting faces → run intersection removal
- Unique3D may produce meshes with >500K faces → decimate to 50K-100K

**Function Signatures:**
```python
def run_reconstruction(context: dict) -> dict
def reconstruct_with_crm(images: list[Path], checkpoint_dir: Path, low_vram: bool = False) -> trimesh.Trimesh
def reconstruct_with_unique3d(images: list[Path], checkpoint_dir: Path) -> trimesh.Trimesh
def postprocess_mesh(mesh: trimesh.Trimesh, target_faces: int = 50000) -> trimesh.Trimesh
def validate_mesh(mesh: trimesh.Trimesh) -> dict
```

**OOM Risk**: 🔴 High — CRM ~6-8GB, Unique3D ~12GB.
- **6GB Mitigation**: `torch.float16`, `torch.backends.cudnn.benchmark = False`

---

### Stage 3 — Semantic Partitioning (`stage3_partition.py`)

**Key Decisions:**
1. **Mesh → Point Cloud**: Sample 100K points via `trimesh.sample.sample_surface()`
2. **Multi-view rendering**: 12 viewpoints at 224×224 for DINOv2 feature extraction
3. **Feature distillation**: SAMPart3D pre-trained backbone maps DINOv2 → 3D points
4. **Clustering granularity**: Scale parameter for coarse (5-8) vs fine (15-20) parts
5. **Sub-mesh extraction**: Map cluster labels to faces, extract via face masking

**Known Pitfalls:**
- SAMPart3D designed for Objaverse meshes; custom meshes need adaptation
- Noisy boundaries → graph-based smoothing required
- Small parts (<50 faces) should merge into adjacent larger parts
- DINOv2 features are view-dependent — uniform viewpoint distribution critical

**Function Signatures:**
```python
def run_partition(context: dict) -> dict
def mesh_to_pointcloud(mesh: trimesh.Trimesh, n_points: int = 100000) -> np.ndarray
def render_multiview(mesh: trimesh.Trimesh, n_views: int = 12, resolution: int = 224) -> list[np.ndarray]
def extract_dino_features(images: list[np.ndarray], model) -> np.ndarray
def cluster_3d_features(features: np.ndarray, points: np.ndarray, n_parts: int = 8) -> np.ndarray
def extract_submeshes(mesh: trimesh.Trimesh, face_labels: np.ndarray, output_dir: Path) -> list[Path]
def smooth_boundaries(mesh: trimesh.Trimesh, face_labels: np.ndarray, iterations: int = 3) -> np.ndarray
```

**OOM Risk**: 🔴 HIGHEST — DINOv2 ViT-L + PTv3 simultaneously.
- **Mitigation**: Batch views (4 at a time), `float16`, reduce to 50K points on 6GB, `torch.cuda.empty_cache()` between models.

---

### Stage 4 — USD Export (`stage4_usd.py`)

**Key Decisions:**
1. **Scene hierarchy**: `/Root_Scene/Part_000_<label>`, etc.
2. **Two files**: `scene_y_up.usda` (Maya) and `scene_z_up.usda` (Blender/UE5)
3. **Visibility**: Each prim toggleable via `visibility` attribute
4. **Material stubs**: Distinct preview colors per part for viewport identification
5. **Metadata**: Pipeline version, source images, part labels as custom attributes

**Known Pitfalls:**
- `usd-core` doesn't support all USD features — avoid SubLayers
- Maya USD plugin has version-specific quirks — test Maya 2024+
- Vertex normals must be explicitly set or Maya recomputes them
- Face winding order: counter-clockwise for USD convention

**Function Signatures:**
```python
def run_usd_export(context: dict) -> dict
def create_usd_scene(sub_meshes: list[Path], output_path: Path, up_axis: str = "y", labels: list[str] = None) -> Path
def add_mesh_prim(stage, mesh: trimesh.Trimesh, prim_path: str, label: str)
def assign_preview_material(stage, prim, color: tuple[float, float, float]) -> None
def set_visibility_toggleable(prim) -> None
def export_both_axes(sub_meshes: list[Path], output_dir: Path, labels: list[str] = None) -> tuple[Path, Path]
```

**OOM Risk**: 🟢 None — CPU only.

---

## 5. EVALUATION & METRICS PLAN

### 5.1 Chamfer Distance
- **Tool**: PyTorch3D `chamfer_distance()` with 10K sampled surface points
- **Ground Truth**: Objaverse meshes with synthetic 4-view renders
- **Report**: Mean ± std across 10-20 test objects
- **Compare against**: Monolithic CRM output (before slicing) and K-means baseline

### 5.2 Per-Part IoU
- **Tool**: Trimesh voxelization → binary occupancy grid → intersection/union
- **Ground Truth**: 5 manually segmented meshes (Blender) or PartNet labels
- **Report**: IoU per semantic category + mean IoU

### 5.3 Viewport FPS in Maya
- **Method**: Import USD, orbit camera via `cmds.orbit()`, count `cmds.refresh()` calls over 5 seconds
- **Report**: FPS at part counts (5, 10, 20) × face counts (10K, 50K, 100K)
- **Target**: >30 FPS for 50K total faces

### 5.4 Baselines

| Baseline | What it measures |
|----------|-----------------|
| Monolithic CRM mesh (no slicing) | Does slicing degrade geometry? |
| Manual Blender segmentation | Upper bound for part quality |
| NeRF/3DGS → Marching Cubes | Alternative reconstruction pipeline |
| K-means on vertices + normals | Does semantic clustering beat geometric clustering? |

---

## 6. RESEARCH PAPER STRATEGY

### 6.1 What's Novel

The primary contribution is the **end-to-end pipeline architecture** connecting sparse orthogonal views → feed-forward 3D reconstruction → native 3D semantic segmentation → production-ready USD export.

**No existing work** chains CRM/Unique3D → SAMPart3D → USD into a single automated pipeline.

**Secondary contributions:**
- Empirical comparison of CRM vs Unique3D for downstream semantic segmentation quality
- SAMPart3D's zero-shot generalization on feed-forward reconstructed meshes (vs clean Objaverse meshes)
- Production integration (USD + Maya) — bridges research and industry

### 6.2 Suggested Venues

| Venue | Type | Fit |
|-------|------|-----|
| **Computers & Graphics (Elsevier)** | Scopus journal (rolling) | ⭐⭐⭐⭐⭐ |
| **The Visual Computer (Springer)** | Scopus journal (rolling) | ⭐⭐⭐⭐ |
| **CVPR Workshop on 3D Vision** | Workshop paper | ⭐⭐⭐⭐ |
| **ECCV Workshop on 3D Generation** | Workshop paper | ⭐⭐⭐⭐ |
| **3DV (Int'l Conf on 3D Vision)** | Conference | ⭐⭐⭐⭐ |

**Recommendation**: Submit to **Computers & Graphics (Elsevier)** — Scopus-indexed, rolling submissions, accepts pipeline/systems papers.

### 6.3 Abstract Outline

> **Problem**: Reconstructing editable, semantically decomposed 3D assets from sparse 2D images remains fragmented.
>
> **Gap**: Feed-forward models produce monolithic "polygon soup." 3D foundation models enable zero-shot segmentation but haven't been applied to feed-forward meshes. No pipeline bridges both with production-ready export.
>
> **Method**: Five-stage pipeline: 4 orthogonal images → SAM 2 silhouettes → CRM monolithic mesh → SAMPart3D zero-shot 3D partitioning → axis-aligned OpenUSD with toggleable semantic layers.
>
> **Results**: Mean Chamfer Distance of X.XX, per-part IoU of X.XX, >30 FPS in Maya viewport.
>
> **Claim**: First end-to-end pipeline converting sparse orthogonal views into production-ready, semantically editable 3D assets without manual intervention or text prompts.

### 6.4 Report Structure (NMIMS Format)

| Chapter | Content | Owner |
|---------|---------|-------|
| 1. Introduction | Problem statement, motivation, objectives | **A** |
| 2. Literature Survey | 8 reviewed papers + pipeline comparison table | **B** |
| 3. Methodology | 5-stage pipeline architecture, design decisions | **Both** |
| 4. Implementation | Environment, code structure, integration details | **A** |
| 5. Results | Chamfer, IoU, FPS metrics + qualitative examples | **B** |
| 6. Advantages & Limitations | Strengths, failure cases, VRAM constraints | **Both** |
| 7. Conclusion & Future Work | Summary, real-time extension, texture support | **A** |
| 8. References | All 8+ cited works in IEEE format | **B** |
