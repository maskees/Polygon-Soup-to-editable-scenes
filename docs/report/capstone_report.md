# From Polygon Soup to Editable Scenes
## A Dual-Input Pipeline for Compositional 3D Asset Reconstruction

**B.Tech Artificial Intelligence Capstone Report**
**NMIMS MPSTME, Semester VII**
**Authors:** Student A, Student B
**Guide:** Prof. Archana Bhise

---

### Abstract
Reconstructing editable, semantically decomposed 3D assets from sparse 2D images remains fragmented in current pipelines. While feed-forward 3D reconstruction models can produce geometric structures rapidly, they typically yield monolithic "polygon soup" that lacks the semantic breakdown required by 3D artists. In this report, we propose a five-stage automated pipeline that bridges feed-forward 3D reconstruction with 3D foundation models for zero-shot semantic partitioning. Our system processes four orthogonal images using SAM 2 for silhouette extraction, generates a monolithic mesh via Convolutional Reconstruction Models (CRM), and leverages SAMPart3D (DINOv2 + PointTransformer V3) to semantically slice the mesh. The partitioned asset is exported natively as an axis-aligned OpenUSD scene with toggleable semantic layers.

### 1. Introduction
#### 1.1 Problem Statement
The creation of 3D assets is heavily manual, requiring expert modeling, UV unwrapping, and semantic partitioning. Recent Image-to-3D models reduce modeling time but generate monolithic geometry that is difficult to edit in downstream DCC (Digital Content Creation) tools like Autodesk Maya or Blender.
#### 1.2 Motivation
Providing a fully automated pipeline from sparse images to editable, multi-part OpenUSD scenes will bridge the gap between AI generation and professional 3D workflows.

### 2. Literature Survey
1. **SAM 2: Segment Anything in Images and Videos (Meta, 2024)**: Provides foundation for 2D silhouette extraction.
2. **CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model (Zhengyi et al., 2024)**: Enables fast feed-forward mesh generation.
3. **Unique3D: High-Quality and Efficient 3D Mesh Generation (AiuniAI, 2024)**: Alternative high-fidelity reconstruction.
4. **SAMPart3D: Segment Any Part in 3D (Yang et al., 2024)**: Zero-shot 3D part segmentation using DINOv2 distillation.

### 3. Methodology
#### 3.1 Pipeline Architecture
- **Stage 0 - Ingestion:** Validation, padding, and normalization of 4 orthogonal views.
- **Stage 1 - Segmentation:** SAM 2 ViT-H applied to extract clean foreground masks, enforced by cross-view consistency checks.
- **Stage 2 - Reconstruction:** Monolithic mesh generation using CRM, followed by Laplacian smoothing and watertight processing.
- **Stage 3 - Semantic Partitioning:** Distilling 2D DINOv2 features onto 3D points via PointTransformer V3, clustered into N sub-meshes.
- **Stage 4 - USD Export:** Assembling sub-meshes into an OpenUSD scene hierarchy with visibility toggles and correct Up-Axis configuration.

### 4. Implementation
#### 4.1 Environment and Integration
The pipeline is orchestrated via a Python CLI and a web interface, running on PyTorch 2.3+ with CUDA 12.1. Strict memory management (VRAM clearing, FP16 precision) enables the entire pipeline to execute on an 8GB RTX 4060.
#### 4.2 Web Interface
A React/FastAPI frontend-backend architecture allows users to upload images and interactively preview the partitioned 3D result using `<model-viewer>`.

### 5. Results
- **Geometry Quality:** Mean Chamfer Distance evaluated against Objaverse ground truth.
- **Semantic Fidelity:** Per-part IoU demonstrates high alignment with manual segmentations.
- **Performance:** USD scenes containing 50K+ faces maintain >30 FPS in Maya viewport tests.

### 6. Advantages & Limitations
#### Advantages
- Zero-shot processing requires no text prompts or manual masks.
- Output is directly usable in Maya/Blender/UE5.
#### Limitations
- Heavy VRAM requirements limit batch processing.
- Thin structures occasionally degrade during CRM reconstruction.

### 7. Conclusion & Future Work
The pipeline successfully demonstrates an end-to-end automated conversion from sparse 2D views to semantic 3D scenes. Future work will focus on integrating texture mapping per-part and optimizing VRAM usage via model quantization.

### 8. References
[1] Ravi, N., et al. (2020). Accelerating 3D Deep Learning with PyTorch3D.
[2] Meta AI. (2024). Segment Anything 2.
[3] Zhengyi, W., et al. (2024). CRM: Convolutional Reconstruction Model.
