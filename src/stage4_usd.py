"""
Stage 4 — Axis-Aligned OpenUSD Export
=======================================
Package semantic sub-meshes as USD prims under a /Root_Scene parent.
Export two variants: Y-Up (Maya) and Z-Up (Blender/UE5).

Input:  N sub-meshes (.obj per semantic part) from Stage 3
Output: .usda master files with individually toggleable layers
"""

import logging
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    import trimesh
    from pxr import Usd, UsdGeom

from src.config import PipelineConfig
from src.stage3_partition import PART_COLORS

logger = logging.getLogger(__name__)


def run_usd_export(context: dict) -> dict:
    """
    Main entry point for Stage 4.

    Creates USD scenes from sub-meshes with proper hierarchy,
    materials, and visibility controls.

    Parameters
    ----------
    context : dict
        Pipeline context. Must contain 'sub_meshes', 'part_labels', 'part_colors'.

    Returns
    -------
    dict
        Updated context with 'usd_files' key (list of .usda paths).
    """
    cfg: PipelineConfig = context["cfg"]
    output_dir = Path(context["output_dir"]) / "usd"
    output_dir.mkdir(parents=True, exist_ok=True)

    sub_meshes = context["sub_meshes"]
    labels = context.get("part_labels", [f"part_{i:03d}" for i in range(len(sub_meshes))])
    colors = context.get(
        "part_colors", [PART_COLORS[i % len(PART_COLORS)] for i in range(len(sub_meshes))]
    )

    up_axis = context.get("up_axis", "both")

    usd_files = []

    if up_axis in ("y", "both"):
        y_up_path = output_dir / "scene_y_up.usda"
        logger.info(f"Exporting Y-Up USD (Maya): {y_up_path}")
        create_usd_scene(
            sub_meshes=sub_meshes,
            output_path=y_up_path,
            up_axis="y",
            labels=labels,
            colors=colors,
            scene_name=cfg.usd_scene_name,
        )
        usd_files.append(y_up_path)

    if up_axis in ("z", "both"):
        z_up_path = output_dir / "scene_z_up.usda"
        logger.info(f"Exporting Z-Up USD (Blender/UE5): {z_up_path}")
        create_usd_scene(
            sub_meshes=sub_meshes,
            output_path=z_up_path,
            up_axis="z",
            labels=labels,
            colors=colors,
            scene_name=cfg.usd_scene_name,
        )
        usd_files.append(z_up_path)

    logger.info(f"Exported {len(usd_files)} USD file(s)")
    context["usd_files"] = usd_files
    return context


def create_usd_scene(
    sub_meshes: list[Path],
    output_path: Path,
    up_axis: str = "y",
    labels: list[str] | None = None,
    colors: list[tuple[float, float, float]] | None = None,
    scene_name: str = "Root_Scene",
) -> Path:
    """
    Build a complete USD scene from sub-meshes.

    Creates a scene hierarchy:
        /Root_Scene
            /Part_000_<label>    (mesh + material)
            /Part_001_<label>    (mesh + material)
            ...

    Parameters
    ----------
    sub_meshes : list[Path]
        Paths to individual sub-mesh .obj files.
    output_path : Path
        Where to save the .usda file.
    up_axis : str
        'y' for Y-Up (Maya) or 'z' for Z-Up (Blender/UE5).
    labels : list[str] | None
        Semantic labels for each part.
    colors : list[tuple] | None
        RGB preview colors for each part.
    scene_name : str
        Name for the root prim.

    Returns
    -------
    Path
        Path to the saved .usda file.
    """
    import trimesh
    from pxr import Usd, UsdGeom

    # Create stage
    stage = Usd.Stage.CreateNew(str(output_path))

    # Set stage metadata
    if up_axis == "y":
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    else:
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root prim
    root_path = f"/{scene_name}"
    root_xform = UsdGeom.Xform.Define(stage, root_path)
    stage.SetDefaultPrim(root_xform.GetPrim())

    # Add custom metadata
    root_prim = stage.GetPrimAtPath(root_path)
    root_prim.SetCustomDataByKey("pipeline", "polygon-soup-to-editable-scenes")
    root_prim.SetCustomDataByKey("pipeline_version", "0.1.0")
    root_prim.SetCustomDataByKey("up_axis", up_axis)
    root_prim.SetCustomDataByKey("n_parts", len(sub_meshes))

    # Create materials scope
    materials_path = f"{root_path}/Materials"
    UsdGeom.Scope.Define(stage, materials_path)

    # Add each sub-mesh as a separate prim
    for i, mesh_path in enumerate(sub_meshes):
        label = labels[i] if labels else f"part_{i:03d}"
        color = colors[i] if colors else PART_COLORS[i % len(PART_COLORS)]

        # Load mesh
        mesh = trimesh.load(str(mesh_path), force="mesh")

        # Prim path
        safe_label = label.replace(" ", "_").replace("-", "_")
        prim_path = f"{root_path}/Part_{i:03d}_{safe_label}"

        # Add mesh prim
        usd_mesh = add_mesh_prim(stage, mesh, prim_path, label)

        # Create and assign material
        mat_path = f"{materials_path}/Mat_{safe_label}"
        assign_preview_material(stage, usd_mesh, color, mat_path)

        # Set visibility as toggleable
        set_visibility_toggleable(usd_mesh)

        logger.info(f"  Added prim: {prim_path} ({len(mesh.faces)} faces)")

    # Save
    stage.GetRootLayer().Save()
    logger.info(f"  Saved USD scene: {output_path}")

    return output_path


def add_mesh_prim(
    stage: "Usd.Stage",
    mesh: "trimesh.Trimesh",
    prim_path: str,
    label: str,
) -> "UsdGeom.Mesh":
    """
    Add a single Trimesh mesh as a USD Mesh prim.

    Parameters
    ----------
    stage : Usd.Stage
        USD stage to add the prim to.
    mesh : trimesh.Trimesh
        Mesh geometry.
    prim_path : str
        USD path for this prim (e.g., '/Root_Scene/Part_000_head').
    label : str
        Semantic label for this part.

    Returns
    -------
    UsdGeom.Mesh
        The created USD mesh prim.
    """
    from pxr import Gf, UsdGeom, Vt

    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)

    # Set vertices (points)
    points = [Gf.Vec3f(*v) for v in mesh.vertices.tolist()]
    usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))

    # Set face topology
    # All faces are triangles (3 vertices each)
    face_vertex_counts = [3] * len(mesh.faces)
    usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(face_vertex_counts))

    face_vertex_indices = mesh.faces.flatten().tolist()
    usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(face_vertex_indices))

    # Set vertex normals
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0:
        normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals.tolist()]
        usd_mesh.GetNormalsAttr().Set(Vt.Vec3fArray(normals))
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # Set subdivision scheme to none (we want exact mesh, no subdivision)
    usd_mesh.GetSubdivisionSchemeAttr().Set("none")

    # Add semantic label as custom data
    prim = usd_mesh.GetPrim()
    prim.SetCustomDataByKey("semantic_label", label)

    return usd_mesh


def assign_preview_material(
    stage: "Usd.Stage",
    mesh_prim: "UsdGeom.Mesh",
    color: tuple[float, float, float],
    material_path: str,
) -> None:
    """
    Assign a UsdPreviewSurface material with a given diffuse color.

    This creates a simple colored material for viewport visualization,
    making each semantic part visually distinct.

    Parameters
    ----------
    stage : Usd.Stage
        USD stage.
    mesh_prim : UsdGeom.Mesh
        Mesh prim to assign material to.
    color : tuple[float, float, float]
        RGB diffuse color (0.0–1.0 range).
    material_path : str
        USD path for the material.
    """
    from pxr import Gf, Sdf, UsdShade

    # Create material
    material = UsdShade.Material.Define(stage, material_path)

    # Create shader
    shader_path = f"{material_path}/PreviewShader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Set diffuse color
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    # Connect shader to material surface output
    material.CreateSurfaceOutput().ConnectToSource(UsdShade.ConnectableAPI(shader), "surface")

    # Bind material to mesh
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
    UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(material)


def set_visibility_toggleable(prim: "UsdGeom.Mesh") -> None:
    """
    Configure prim visibility for layer toggling in DCC tools.

    Sets visibility to 'inherited' (default visible, but toggleable
    via the Maya Outliner or Blender Outliner).

    Parameters
    ----------
    prim : UsdGeom.Mesh
        The mesh prim to configure.
    """
    from pxr import UsdGeom

    imageable = UsdGeom.Imageable(prim.GetPrim())
    imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)


def export_both_axes(
    sub_meshes: list[Path],
    output_dir: Path,
    labels: list[str] | None = None,
    colors: list[tuple[float, float, float]] | None = None,
) -> tuple[Path, Path]:
    """
    Convenience function to export both Y-Up and Z-Up variants.

    Parameters
    ----------
    sub_meshes : list[Path]
        Paths to sub-mesh .obj files.
    output_dir : Path
        Output directory.
    labels : list[str] | None
        Part labels.
    colors : list[tuple] | None
        Part colors.

    Returns
    -------
    tuple[Path, Path]
        (y_up_path, z_up_path)
    """
    y_up = create_usd_scene(
        sub_meshes,
        output_dir / "scene_y_up.usda",
        up_axis="y",
        labels=labels,
        colors=colors,
    )
    z_up = create_usd_scene(
        sub_meshes,
        output_dir / "scene_z_up.usda",
        up_axis="z",
        labels=labels,
        colors=colors,
    )
    return y_up, z_up
