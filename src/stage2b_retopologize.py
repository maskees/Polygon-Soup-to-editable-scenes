"""
Stage 2b — Quad Mesh Retopology
=================================
Remesh triangular monolithic meshes into clean editable quad meshes using
Blender's Python API (bpy) Quadriflow remesher.

Input:  Monolithic triangle .obj mesh from Stage 2
Output: Monolithic quad .obj mesh
"""

import logging
from pathlib import Path

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def run_retopologize(context: dict) -> dict:
    """
    Main entry point for Stage 2b.

    Converts a triangle mesh into an editable quad mesh using Quadriflow via `bpy`.

    Parameters
    ----------
    context : dict
        Pipeline context. Must contain 'monolithic_mesh' from Stage 2.

    Returns
    -------
    dict
        Updated context with 'monolithic_mesh' pointing to the new quad mesh.
    """
    cfg: PipelineConfig = context["cfg"]
    output_dir = Path(context["output_dir"]) / "intermediate" / "stage2b_retopologize"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_mesh_path = context["monolithic_mesh"]
    quad_mesh_path = output_dir / "monolithic_quad_mesh.obj"

    if context.get("skip_existing") and quad_mesh_path.exists():
        logger.info(f"Skipping Stage 2b — output exists: {quad_mesh_path}")
        context["monolithic_mesh"] = quad_mesh_path
        return context

    target_quads = getattr(cfg, "target_quad_count", 10000)
    logger.info(f"Retopologizing mesh to ~{target_quads} quads using Quadriflow...")

    try:
        retopologize_with_bpy(input_mesh_path, quad_mesh_path, target_faces=target_quads)
        logger.info(f"Successfully generated quad mesh at: {quad_mesh_path}")
        context["monolithic_mesh"] = quad_mesh_path
    except Exception as e:
        logger.warning(f"Quad remeshing failed ({e}). Proceeding with original triangle mesh.")

    return context


def retopologize_with_bpy(input_path: Path, output_path: Path, target_faces: int = 10000) -> Path:
    """
    Execute Quadriflow remesh using Blender's Python API (bpy).

    Parameters
    ----------
    input_path : Path
        Path to input triangle .obj mesh.
    output_path : Path
        Path where output quad .obj mesh should be saved.
    target_faces : int
        Target number of quad faces.

    Returns
    -------
    Path
        Output quad mesh path.
    """
    try:
        import bpy
    except ImportError:
        raise RuntimeError(
            "Blender Python API (`bpy`) is not installed. "
            "Install it via `pip install bpy` to enable quad remeshing."
        )

    # Clear existing mesh objects in Blender scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import OBJ
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(input_path))
    else:
        bpy.ops.import_scene.obj(filepath=str(input_path))

    # Select the imported object
    selected_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not selected_objects:
        raise RuntimeError("No mesh object was imported into Blender scene.")

    mesh_obj = selected_objects[0]
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)

    # Run Quadriflow remesh
    logger.info(f"  Running bpy.ops.object.quadriflow_remesh(target_faces={target_faces})...")
    bpy.ops.object.quadriflow_remesh(
        use_mesh_symmetry=False,
        use_preserve_surface_curvature=True,
        use_preserve_sharp_boundaries=True,
        target_faces=target_faces,
    )

    # Export OBJ with quads (do not triangulate)
    if hasattr(bpy.ops.wm, "obj_export"):
        bpy.ops.wm.obj_export(
            filepath=str(output_path),
            export_triangulate_faces=False,
            export_materials=False,
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=str(output_path),
            use_triangles=False,
            use_materials=False,
        )

    return output_path
