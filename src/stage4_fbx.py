"""
Stage 4 (Alternative) — FBX Export
=======================================
Package semantic sub-meshes as an FBX scene hierarchy.

Input:  N sub-meshes (.obj per semantic part) from Stage 3
Output: .fbx file with parts and simple materials
"""
import logging
from pathlib import Path
import aspose.threed as a3d

from src.config import PipelineConfig
from src.stage3_partition import PART_COLORS

logger = logging.getLogger(__name__)

def run_fbx_export(context: dict) -> dict:
    """
    Creates an FBX scene from sub-meshes.
    """
    cfg: PipelineConfig = context["cfg"]
    output_dir = Path(context["output_dir"]) / "fbx"
    output_dir.mkdir(parents=True, exist_ok=True)

    sub_meshes = context.get("sub_meshes")
    labels = context.get("part_labels")
    colors = context.get("part_colors")

    if not sub_meshes:
        if "monolithic_mesh" in context:
            sub_meshes = [context["monolithic_mesh"]]
            labels = ["Model"]
            colors = [(0.8, 0.8, 0.8)]
            logger.info("Semantic segmentation skipped; exporting un-segmented monolithic mesh.")
        else:
            stage3_dir = Path(context["output_dir"]) / "intermediate" / "stage3_partition"
            sub_meshes = sorted(stage3_dir.glob("part_*.obj"))
            if not sub_meshes:
                raise RuntimeError("No sub-meshes found in context or on disk.")

    if not labels:
        labels = [f"part_{i:03d}" for i in range(len(sub_meshes))]
    if not colors:
        colors = [PART_COLORS[i % len(PART_COLORS)] for i in range(len(sub_meshes))]
    
    scene = a3d.Scene()
    
    for i, mesh_path in enumerate(sub_meshes):
        label = labels[i] if labels else f"part_{i:03d}"
        color = colors[i] if colors else PART_COLORS[i % len(PART_COLORS)]
        
        safe_label = label.replace(" ", "_").replace("-", "_")
        node_name = f"Part_{i:03d}_{safe_label}"
        
        tmp = a3d.Scene.from_file(str(mesh_path))
        mesh_node = None
        for node in tmp.root_node.child_nodes:
            if node.entity is not None:
                mesh_node = node
                break
                
        if mesh_node is not None:
            mesh_node.name = node_name
            
            # Only assign solid material for partitioned parts (preserve vertex colors on monolithic mesh)
            if len(sub_meshes) > 1:
                mat = a3d.shading.LambertMaterial()
                mat.name = f"Mat_{safe_label}"
                mat.diffuse_color = a3d.utilities.Vector3(color[0], color[1], color[2])
                mesh_node.material = mat
            
            scene.root_node.add_child_node(mesh_node)
            logger.info(f"  Added node: {node_name}")
            
    fbx_path = output_dir / "scene.fbx"
    logger.info(f"Exporting FBX: {fbx_path}")
    scene.save(str(fbx_path), a3d.FileFormat.FBX7700_BINARY)
    
    context["fbx_files"] = [fbx_path]
    return context
