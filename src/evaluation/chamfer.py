"""
Chamfer Distance evaluation.
=============================
Compute symmetric Chamfer Distance between predicted and ground truth meshes.

Usage:
    python -m src.evaluation.chamfer --pred mesh_pred.obj --gt mesh_gt.obj --samples 10000
"""

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_chamfer(
    pred_mesh_path: str | Path,
    gt_mesh_path: str | Path,
    n_samples: int = 10000,
    use_pytorch3d: bool = True,
) -> float:
    """
    Compute Chamfer Distance between predicted and ground truth meshes.

    Samples points uniformly from both mesh surfaces and computes the
    mean bidirectional nearest-neighbor distance.

    Parameters
    ----------
    pred_mesh_path : str | Path
        Path to the predicted mesh (.obj).
    gt_mesh_path : str | Path
        Path to the ground truth mesh (.obj).
    n_samples : int
        Number of points to sample from each mesh surface.
    use_pytorch3d : bool
        If True, use PyTorch3D's GPU-accelerated implementation.
        If False, fall back to scipy KDTree.

    Returns
    -------
    float
        Chamfer Distance (lower is better). Units match mesh coordinate space.
    """
    import trimesh

    # Load and sample meshes
    pred_mesh = trimesh.load(str(pred_mesh_path), force="mesh")
    gt_mesh = trimesh.load(str(gt_mesh_path), force="mesh")

    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, count=n_samples)
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, count=n_samples)

    if use_pytorch3d:
        return _chamfer_pytorch3d(pred_points, gt_points)
    else:
        return _chamfer_scipy(pred_points, gt_points)


def _chamfer_pytorch3d(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    """Chamfer Distance using PyTorch3D (GPU-accelerated)."""
    try:
        import torch
        from pytorch3d.loss import chamfer_distance

        device = "cuda" if torch.cuda.is_available() else "cpu"

        pred_tensor = torch.tensor(pred_points, dtype=torch.float32, device=device).unsqueeze(0)
        gt_tensor = torch.tensor(gt_points, dtype=torch.float32, device=device).unsqueeze(0)

        loss, _ = chamfer_distance(pred_tensor, gt_tensor)
        return loss.item()

    except ImportError:
        logger.warning("PyTorch3D not available, falling back to scipy")
        return _chamfer_scipy(pred_points, gt_points)


def _chamfer_scipy(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    """Chamfer Distance using scipy KDTree (CPU fallback)."""
    from scipy.spatial import KDTree

    tree_pred = KDTree(pred_points)
    tree_gt = KDTree(gt_points)

    # Pred → GT distances
    dist_pred_to_gt, _ = tree_gt.query(pred_points)
    # GT → Pred distances
    dist_gt_to_pred, _ = tree_pred.query(gt_points)

    chamfer = np.mean(dist_pred_to_gt**2) + np.mean(dist_gt_to_pred**2)
    return float(chamfer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Chamfer Distance")
    parser.add_argument("--pred", required=True, help="Predicted mesh path")
    parser.add_argument("--gt", required=True, help="Ground truth mesh path")
    parser.add_argument("--samples", type=int, default=10000, help="Sample count")
    args = parser.parse_args()

    cd = compute_chamfer(args.pred, args.gt, n_samples=args.samples)
    print(f"Chamfer Distance: {cd:.6f}")
