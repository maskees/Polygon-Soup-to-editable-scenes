"""
Mesh utility functions.
=======================
Common Trimesh operations for loading, validation, and manipulation.
"""

import typing
from pathlib import Path

import numpy as np

if typing.TYPE_CHECKING:
    import trimesh


def load_mesh(path: Path | str) -> "trimesh.Trimesh":
    """Load a mesh from file, forcing single mesh output."""
    import trimesh

    return trimesh.load(str(path), force="mesh")


def get_mesh_stats(mesh: "trimesh.Trimesh") -> dict:
    """Get comprehensive mesh statistics."""
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "edges": len(mesh.edges),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "bounds": mesh.bounds.tolist(),
        "extents": mesh.extents.tolist(),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "surface_area": float(mesh.area),
    }


def normalize_mesh(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Center mesh at origin and scale to fit in unit cube."""
    mesh.vertices -= mesh.centroid
    max_extent = max(mesh.extents)
    if max_extent > 0:
        mesh.vertices /= max_extent
    return mesh


def ensure_watertight(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Attempt to make a mesh watertight by filling holes."""
    mesh.fill_holes()
    mesh.fix_normals()
    return mesh


def colorize_by_labels(
    mesh: "trimesh.Trimesh",
    face_labels: np.ndarray,
    colors: list[tuple[float, float, float]] | None = None,
) -> "trimesh.Trimesh":
    """
    Assign face colors based on labels for visualization.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    face_labels : np.ndarray
        Integer label per face.
    colors : list[tuple] | None
        RGB colors (0-1 range) per unique label.

    Returns
    -------
    trimesh.Trimesh
        Mesh with face colors set.
    """
    from src.stage3_partition import PART_COLORS

    if colors is None:
        colors = PART_COLORS

    unique_labels = np.unique(face_labels)
    face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)

    for label in unique_labels:
        color_idx = label % len(colors)
        rgb = colors[color_idx]
        rgba = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 255)
        face_colors[face_labels == label] = rgba

    mesh.visual.face_colors = face_colors
    return mesh
