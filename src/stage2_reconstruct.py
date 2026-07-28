"""
Stage 2 — Sparse 3D Reconstruction
====================================
Feed masked orthogonal images into CRM or Unique3D to generate
a monolithic 3D mesh.

Input:  4 RGBA images from Stage 1
Output: Single monolithic .obj mesh (untextured "clay" topology)
"""

import logging
import sys
from pathlib import Path

import numpy as np

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def run_reconstruction(context: dict) -> dict:
    """
    Main entry point for Stage 2.

    Generates a monolithic 3D mesh from 4 segmented RGBA images
    using either CRM or Unique3D backend.

    Parameters
    ----------
    context : dict
        Pipeline context. Must contain 'segmented_images' from Stage 1.

    Returns
    -------
    dict
        Updated context with 'monolithic_mesh' key (Path to .obj).
    """
    import trimesh

    cfg: PipelineConfig = context["cfg"]
    backend = context["backend"]
    output_dir = Path(context["output_dir"]) / "intermediate" / "stage2_reconstruct"
    output_dir.mkdir(parents=True, exist_ok=True)

    segmented = context["segmented_images"]
    image_paths = [segmented[v] for v in ["front", "back", "left", "right"]]

    # Run reconstruction
    logger.info(f"Running 3D reconstruction with backend: {backend}")

    if backend == "crm":
        mesh = reconstruct_with_crm(
            images=image_paths,
            checkpoint_dir=Path(cfg.crm_checkpoint_dir),
            low_vram=cfg.use_float16,
        )
    elif backend == "unique3d":
        mesh = reconstruct_with_unique3d(
            images=image_paths,
            checkpoint_dir=Path(cfg.unique3d_checkpoint_dir),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Post-process mesh
    logger.info("Post-processing mesh...")
    mesh = postprocess_mesh(mesh, target_faces=cfg.target_face_count)

    # Validate
    quality = validate_mesh(mesh)
    logger.info(f"  Mesh quality: {quality}")

    # Save
    mesh_path = output_dir / "monolithic_mesh.obj"
    mesh.export(str(mesh_path))
    logger.info(f"  Saved monolithic mesh to: {mesh_path}")

    context["monolithic_mesh"] = mesh_path
    context["mesh_quality"] = quality
    return context


def reconstruct_with_crm(
    images: list[Path],
    checkpoint_dir: Path,
    low_vram: bool = False,
) -> "trimesh.Trimesh":
    """
    Run CRM (Convolutional Reconstruction Model) pipeline.

    CRM expects a single canonical image and internally generates
    6 orthogonal views. We use the front view as the primary input.

    Parameters
    ----------
    images : list[Path]
        List of 4 RGBA image paths [front, back, left, right].
    checkpoint_dir : Path
        Directory containing CRM model checkpoints.
    low_vram : bool
        If True, use float16 and reduced batch sizes.

    Returns
    -------
    trimesh.Trimesh
        Reconstructed mesh (may have texture — we'll strip it).
    """
    import torch
    import trimesh

    # CRM uses the front view as primary input
    front_image_path = images[0]

    # Add CRM to Python path
    crm_dir = Path("external/CRM")
    if not crm_dir.exists():
        raise FileNotFoundError(
            f"CRM not found at {crm_dir}. "
            "Run: git clone https://github.com/thu-ml/CRM.git external/CRM"
        )
    sys.path.insert(0, str(crm_dir))

    # ---------------------------------------------------------------
    # TODO: Integrate CRM inference pipeline
    #
    # The CRM pipeline has 3 internal stages:
    #   1. Multi-view diffusion: single image → 6 orthogonal views
    #   2. CCM generation: views → Canonical Coordinate Maps
    #   3. Convolutional U-Net + FlexiCubes → textured mesh
    #
    # Key integration points:
    #   - Load CRM model: `from model import CRM; model = CRM(cfg)`
    #   - Prepare input: PIL Image of front view, RGBA with white bg
    #   - Run inference: `mesh = model(input_image)`
    #   - Extract mesh: Access the FlexiCubes output
    #
    # Low VRAM mode:
    #   - Use torch.float16: `model = model.half()`
    #   - Disable cudnn benchmark: `torch.backends.cudnn.benchmark = False`
    #   - Process with torch.no_grad() context
    #
    # PLACEHOLDER: Return a simple test mesh until CRM is integrated
    # ---------------------------------------------------------------

    logger.warning(
        "CRM integration not yet complete — returning placeholder mesh. "
        "See TODO in reconstruct_with_crm() for integration steps."
    )

    # Placeholder: generate a unit sphere as a test mesh
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    return mesh


def reconstruct_with_unique3d(
    images: list[Path],
    checkpoint_dir: Path,
) -> "trimesh.Trimesh":
    """
    Run Unique3D reconstruction pipeline.

    Unique3D natively accepts multi-view input, making it a better
    fit for our 4-view input format.

    Parameters
    ----------
    images : list[Path]
        List of 4 RGBA image paths [front, back, left, right].
    checkpoint_dir : Path
        Directory containing Unique3D model checkpoints.

    Returns
    -------
    trimesh.Trimesh
        Reconstructed mesh.
    """
    import trimesh

    unique3d_dir = Path("external/Unique3D")
    if not unique3d_dir.exists():
        raise FileNotFoundError(
            f"Unique3D not found at {unique3d_dir}. "
            "Run: git clone https://github.com/AiuniAI/Unique3D.git external/Unique3D"
        )
    sys.path.insert(0, str(unique3d_dir))

    # ---------------------------------------------------------------
    # TODO: Integrate Unique3D inference pipeline
    #
    # Unique3D pipeline:
    #   1. Accepts multi-view images (4 views in our case)
    #   2. Generates normal maps and geometric features
    #   3. Produces high-fidelity mesh via ISOMER
    #
    # Key integration:
    #   - Load model: from scripts.inference import Unique3DPipeline
    #   - Prepare views: 4 RGBA images at expected resolution
    #   - Run inference: mesh = pipeline(images)
    #
    # WARNING: Requires 12GB+ VRAM — not viable on 6GB GPUs
    #
    # PLACEHOLDER: Return test mesh until Unique3D is integrated
    # ---------------------------------------------------------------

    logger.warning(
        "Unique3D integration not yet complete — returning placeholder mesh. "
        "See TODO in reconstruct_with_unique3d() for integration steps."
    )

    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    return mesh


def postprocess_mesh(
    mesh: "trimesh.Trimesh",
    target_faces: int = 50000,
    smoothing_iterations: int = 3,
) -> "trimesh.Trimesh":
    """
    Post-process a reconstructed mesh for downstream use.

    Steps:
    1. Decimate to target face count (if over budget)
    2. Laplacian smoothing to remove reconstruction artifacts
    3. Fix normals (consistent winding order)
    4. Attempt watertight closure
    5. Normalize bounding box (center at origin, scale to unit cube)

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Raw mesh from reconstruction.
    target_faces : int
        Maximum face count after decimation.
    smoothing_iterations : int
        Number of Laplacian smoothing passes.

    Returns
    -------
    trimesh.Trimesh
        Cleaned mesh ready for semantic partitioning.
    """
    import trimesh

    logger.info(f"  Input: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")

    # 1. Decimate if over budget
    if len(mesh.faces) > target_faces:
        logger.info(f"  Decimating from {len(mesh.faces)} to {target_faces} faces")
        mesh = mesh.simplify_quadric_decimation(target_faces)

    # 2. Laplacian smoothing
    if smoothing_iterations > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_iterations)

    # 3. Fix normals
    mesh.fix_normals()

    # 4. Fill holes (attempt watertight)
    mesh.fill_holes()

    # 5. Normalize bounding box
    # Center at origin
    centroid = mesh.centroid
    mesh.vertices -= centroid

    # Scale to unit cube
    extents = mesh.extents
    max_extent = max(extents)
    if max_extent > 0:
        mesh.vertices /= max_extent

    logger.info(f"  Output: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    logger.info(f"  Watertight: {mesh.is_watertight}")

    return mesh


def validate_mesh(mesh: "trimesh.Trimesh") -> dict:
    """
    Compute quality metrics for a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to validate.

    Returns
    -------
    dict
        Quality metrics including face_count, vertex_count, is_watertight,
        euler_number, and bounding_box.
    """
    return {
        "face_count": len(mesh.faces),
        "vertex_count": len(mesh.vertices),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "bounding_box": mesh.bounds.tolist(),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
    }
