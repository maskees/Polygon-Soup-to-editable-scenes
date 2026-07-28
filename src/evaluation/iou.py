"""
Per-Part IoU evaluation.
=========================
Compute volumetric Intersection over Union between predicted
and ground truth sub-meshes.

Usage:
    python -m src.evaluation.iou --pred-dir parts_pred/ --gt-dir parts_gt/
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_part_iou(
    pred_part_path: str | Path,
    gt_part_path: str | Path,
    voxel_resolution: float = 0.01,
) -> float:
    """
    Compute volumetric IoU between a predicted and ground truth sub-mesh.

    Voxelizes both meshes at the given resolution and computes
    the intersection / union of the resulting occupancy grids.

    Parameters
    ----------
    pred_part_path : str | Path
        Path to predicted sub-mesh (.obj).
    gt_part_path : str | Path
        Path to ground truth sub-mesh (.obj).
    voxel_resolution : float
        Voxel pitch in mesh coordinate units. Smaller = more accurate but slower.

    Returns
    -------
    float
        IoU score (0.0 to 1.0). Higher is better.
    """
    import trimesh

    pred = trimesh.load(str(pred_part_path), force="mesh")
    gt = trimesh.load(str(gt_part_path), force="mesh")

    # Voxelize both meshes
    try:
        pred_vox = pred.voxelized(voxel_resolution).fill()
        gt_vox = gt.voxelized(voxel_resolution).fill()
    except Exception as e:
        logger.warning(f"Voxelization failed: {e}. Returning IoU = 0.0")
        return 0.0

    # Convert to dense boolean grids aligned to same origin
    pred_matrix = pred_vox.matrix
    gt_matrix = gt_vox.matrix

    # Align grids to same coordinate space
    pred_origin = pred_vox.transform[:3, 3]
    gt_origin = gt_vox.transform[:3, 3]

    # For simplicity, compute IoU using point-based occupancy
    pred_points = set(map(tuple, pred_vox.points.round(6)))
    gt_points = set(map(tuple, gt_vox.points.round(6)))

    intersection = len(pred_points & gt_points)
    union = len(pred_points | gt_points)

    return float(intersection) / float(union) if union > 0 else 0.0


def compute_all_parts_iou(
    pred_dir: str | Path,
    gt_dir: str | Path,
    voxel_resolution: float = 0.01,
) -> dict[str, float]:
    """
    Compute IoU for all matching parts between predicted and GT directories.

    Matches parts by filename (e.g., part_000.obj ↔ part_000.obj).

    Parameters
    ----------
    pred_dir : str | Path
        Directory containing predicted sub-meshes.
    gt_dir : str | Path
        Directory containing ground truth sub-meshes.
    voxel_resolution : float
        Voxel pitch.

    Returns
    -------
    dict[str, float]
        Mapping of part name → IoU score.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    results = {}

    for pred_file in sorted(pred_dir.glob("*.obj")):
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            logger.warning(f"No GT match for {pred_file.name}")
            continue

        iou = compute_part_iou(pred_file, gt_file, voxel_resolution)
        results[pred_file.stem] = iou
        logger.info(f"  {pred_file.stem}: IoU = {iou:.4f}")

    # Mean IoU
    if results:
        mean_iou = np.mean(list(results.values()))
        results["mean"] = float(mean_iou)
        logger.info(f"  Mean IoU: {mean_iou:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-part IoU")
    parser.add_argument("--pred-dir", required=True, help="Predicted parts directory")
    parser.add_argument("--gt-dir", required=True, help="Ground truth parts directory")
    parser.add_argument("--resolution", type=float, default=0.01, help="Voxel resolution")
    args = parser.parse_args()

    results = compute_all_parts_iou(args.pred_dir, args.gt_dir, args.resolution)
    for name, iou in results.items():
        print(f"{name}: {iou:.4f}")
