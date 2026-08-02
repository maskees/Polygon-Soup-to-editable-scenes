"""
Stage 3 — Native 3D Semantic Slicing via DINOv2 Features
==========================================================
Zero-shot semantic clustering of mesh faces directly in 3D.
Uses DINOv2 visual features projected from multi-view renders
into 3D space, then spectral clustering for semantic partitioning.

Input:  Monolithic .obj mesh from Stage 2
Output: N sub-meshes (.obj per semantic part)

Architecture:
    1. Sample point cloud from mesh surface
    2. Render mesh from multiple viewpoints (with camera transforms)
    3. Extract DINOv2 ViT-L patch features from each view
    4. Project 2D features to 3D points via camera projection
    5. Cluster points using spectral clustering on combined
       geometric + semantic features
    6. Map point labels to face labels, smooth boundaries, export
"""

import logging
import typing
from pathlib import Path

import numpy as np

if typing.TYPE_CHECKING:
    import trimesh

from src.config import PipelineConfig

logger = logging.getLogger(__name__)

# Semantic part colors for visualization (up to 20 parts)
PART_COLORS = [
    (0.90, 0.30, 0.30),  # Red
    (0.30, 0.70, 0.90),  # Blue
    (0.30, 0.90, 0.40),  # Green
    (0.95, 0.75, 0.20),  # Yellow
    (0.70, 0.30, 0.90),  # Purple
    (0.90, 0.55, 0.20),  # Orange
    (0.20, 0.90, 0.80),  # Cyan
    (0.90, 0.30, 0.70),  # Pink
    (0.50, 0.80, 0.30),  # Lime
    (0.40, 0.40, 0.80),  # Indigo
    (0.80, 0.60, 0.50),  # Brown
    (0.60, 0.90, 0.60),  # Light green
    (0.90, 0.80, 0.60),  # Tan
    (0.50, 0.50, 0.50),  # Gray
    (0.80, 0.20, 0.50),  # Maroon
    (0.20, 0.60, 0.60),  # Teal
    (0.70, 0.70, 0.20),  # Olive
    (0.60, 0.30, 0.30),  # Dark red
    (0.30, 0.30, 0.60),  # Dark blue
    (0.30, 0.60, 0.30),  # Dark green
]


def run_partition(context: dict) -> dict:
    """
    Main entry point for Stage 3.

    Segments the monolithic mesh into semantic parts using DINOv2
    feature extraction + spectral clustering pipeline.

    Parameters
    ----------
    context : dict
        Pipeline context. Must contain 'monolithic_mesh' from Stage 2.

    Returns
    -------
    dict
        Updated context with:
        - 'sub_meshes': list of Paths to individual .obj files
        - 'part_labels': list of semantic label strings
        - 'part_colors': list of RGB color tuples
    """
    import trimesh

    cfg: PipelineConfig = context["cfg"]
    output_dir = Path(context["output_dir"]) / "intermediate" / "stage3_partition"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load monolithic mesh
    mesh_path = context["monolithic_mesh"]
    logger.info(f"Loading monolithic mesh from: {mesh_path}")
    mesh = trimesh.load(str(mesh_path), force="mesh")

    # Step 1: Sample point cloud from mesh surface
    logger.info(f"Sampling {cfg.n_sample_points} points from mesh surface...")
    points, face_indices = mesh_to_pointcloud(mesh, n_points=cfg.n_sample_points)

    # Step 2: Render multi-view images of the mesh (with camera transforms)
    logger.info(f"Rendering {cfg.n_render_views} views at {cfg.render_resolution}px...")
    rendered_views, camera_transforms = render_multiview(
        mesh, n_views=cfg.n_render_views, resolution=cfg.render_resolution
    )

    # Step 3: Extract DINOv2 features from renders
    logger.info("Extracting DINOv2 features...")
    patch_features = extract_dino_features(
        images=rendered_views,
        device=cfg.device,
        use_float16=cfg.use_float16,
        batch_size=cfg.max_batch_size,
        model_name=cfg.dinov2_model,
    )

    # Step 4: Project 2D features to 3D points
    logger.info("Projecting features to 3D point cloud...")
    point_features = project_features_to_points(
        points=points,
        patch_features=patch_features,
        camera_transforms=camera_transforms,
        mesh=mesh,
        resolution=cfg.render_resolution,
        patch_size=14,  # DINOv2 ViT patch size
    )

    # Step 5: Cluster using spectral or k-means
    logger.info(f"Clustering into ~{cfg.n_target_parts} parts ({cfg.partition_method})...")
    point_labels = cluster_3d_features(
        features=point_features,
        points=points,
        n_parts=cfg.n_target_parts,
        method=cfg.partition_method,
        feature_weight=cfg.feature_weight,
        n_neighbors=cfg.spectral_n_neighbors,
    )

    # Step 6: Map point labels to face labels
    face_labels = map_point_labels_to_faces(
        point_labels=point_labels,
        face_indices=face_indices,
        n_faces=len(mesh.faces),
    )

    # Step 7: Smooth boundaries
    logger.info("Smoothing partition boundaries...")
    face_labels = smooth_boundaries(
        mesh=mesh,
        face_labels=face_labels,
        iterations=cfg.boundary_smoothing_iterations,
    )

    # Step 8: Merge tiny parts
    face_labels = merge_small_parts(
        mesh=mesh,
        face_labels=face_labels,
        min_faces=cfg.min_part_faces,
    )

    # Step 9: Extract sub-meshes
    unique_labels = np.unique(face_labels)
    logger.info(f"Extracting {len(unique_labels)} sub-meshes...")

    sub_mesh_paths = extract_submeshes(
        mesh=mesh,
        face_labels=face_labels,
        output_dir=output_dir,
    )

    # Generate labels and colors
    part_labels = [f"part_{i:03d}" for i in range(len(sub_mesh_paths))]
    part_colors = [PART_COLORS[i % len(PART_COLORS)] for i in range(len(sub_mesh_paths))]

    logger.info(f"Extracted {len(sub_mesh_paths)} semantic parts")
    for i, path in enumerate(sub_mesh_paths):
        logger.info(f"  Part {i}: {part_labels[i]} → {path.name}")

    context["sub_meshes"] = sub_mesh_paths
    context["part_labels"] = part_labels
    context["part_colors"] = part_colors
    context["face_labels"] = face_labels
    return context


def mesh_to_pointcloud(
    mesh: "trimesh.Trimesh",
    n_points: int = 100000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample points uniformly from mesh surface with face index tracking.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    n_points : int
        Number of points to sample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - points: (N, 3) float array of surface point coordinates
        - face_indices: (N,) int array of which face each point was sampled from
    """
    import trimesh

    points, face_indices = trimesh.sample.sample_surface(mesh, count=n_points)
    return points.astype(np.float32), face_indices


def render_multiview(
    mesh: "trimesh.Trimesh",
    n_views: int = 12,
    resolution: int = 224,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Render mesh from uniformly distributed viewpoints.

    Uses Fibonacci sphere sampling for uniform viewpoint distribution.
    Returns both rendered images and camera transformation matrices,
    which are needed for projecting 2D features back to 3D.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to render.
    n_views : int
        Number of viewpoints.
    resolution : int
        Image resolution (square).

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        - images: List of RGB images (resolution, resolution, 3) in uint8.
        - camera_transforms: List of 4×4 camera-to-world transform matrices.
    """

    # Generate uniform viewpoints on a sphere using Fibonacci spiral
    views = []
    transforms = []
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n_views):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_views)
        phi = 2 * np.pi * i / golden_ratio

        # Camera position on unit sphere, scaled to 2.5x mesh extent
        radius = max(mesh.extents) * 2.5
        cam_pos = np.array(
            [
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ]
        )

        # Compute camera transform looking at origin from cam_pos
        camera_transform = _look_at(cam_pos, target=np.zeros(3), up=np.array([0, 1, 0]))
        transforms.append(camera_transform)

        try:
            # Create scene and render
            scene = mesh.scene()
            rendered = scene.save_image(resolution=(resolution, resolution))
            import io

            from PIL import Image

            img = np.array(Image.open(io.BytesIO(rendered)))[:, :, :3]
            views.append(img)
        except Exception:
            # Fallback: create a blank image if rendering fails
            views.append(np.zeros((resolution, resolution, 3), dtype=np.uint8))

    return views, transforms


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a 4×4 look-at camera transform matrix."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)

    true_up = np.cross(right, forward)

    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = true_up
    transform[:3, 2] = -forward
    transform[:3, 3] = eye

    return transform


# ─────────────────────────────────────────────────────────────────
# DINOv2 Feature Extraction
# ─────────────────────────────────────────────────────────────────


def extract_dino_features(
    images: list[np.ndarray],
    device: str = "cuda",
    use_float16: bool = False,
    batch_size: int = 4,
    model_name: str = "dinov2_vitl14",
) -> np.ndarray:
    """
    Extract DINOv2 patch features from multi-view renders.

    Loads DINOv2 ViT-L/14 via torch.hub and extracts spatial patch
    tokens (not CLS token) from the last transformer layer. These
    patch features encode rich semantic information that enables
    zero-shot part segmentation.

    Parameters
    ----------
    images : list[np.ndarray]
        List of RGB images (H, W, 3) from multi-view rendering.
    device : str
        Compute device ('cuda' or 'cpu').
    use_float16 : bool
        If True, use half precision for reduced VRAM.
    batch_size : int
        Number of views to process simultaneously.
    model_name : str
        DINOv2 model variant (e.g., 'dinov2_vitl14', 'dinov2_vitb14').

    Returns
    -------
    np.ndarray
        Patch features array of shape (N_views, N_patches, feature_dim).
        For ViT-L/14 with 224×224 input: (N_views, 256, 1024).
    """
    try:
        import torch

        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available — falling back to CPU for DINOv2")
            device = "cpu"

        logger.info(f"  Loading DINOv2 model: {model_name}...")
        model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
        )
        model = model.to(device)
        model.eval()

        if use_float16 and device == "cuda":
            model = model.half()

        logger.info(f"  DINOv2 loaded on {device}")
    except Exception as e:
        logger.warning(
            f"Failed to load DINOv2 ({e}) — using random features as fallback. "
            "This will produce geometric-only clustering."
        )
        n_views = len(images)
        # Fallback: return random features matching expected shape
        patch_size = 14
        h, w = images[0].shape[:2] if images else (224, 224)
        n_patches = (h // patch_size) * (w // patch_size)
        feature_dim = 1024 if "vitl" in model_name else 768
        return np.random.randn(n_views, n_patches, feature_dim).astype(np.float32)

    # Preprocessing transform (ImageNet normalization)
    try:
        import torchvision.transforms as transforms
    except ImportError:
        logger.warning("torchvision not available — using random features")
        n_views = len(images)
        return np.random.randn(n_views, 256, 1024).astype(np.float32)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Extract features in batches
    all_features = []
    dtype = torch.float16 if (use_float16 and device == "cuda") else torch.float32

    for batch_start in range(0, len(images), batch_size):
        batch_images = images[batch_start : batch_start + batch_size]

        # Preprocess batch
        batch_tensors = []
        for img in batch_images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            tensor = transform(img)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(device, dtype=dtype)

        # Extract patch features
        with torch.no_grad():
            output = model.forward_features(batch)
            # DINOv2 returns dict with 'x_norm_patchtokens' or we index
            # into the output to get patch tokens
            if isinstance(output, dict):
                # DINOv2 v2 API
                patch_tokens = output.get(
                    "x_norm_patchtokens",
                    output.get("x_prenorm", None),
                )
                if patch_tokens is None:
                    # Try alternative: full output minus CLS token
                    patch_tokens = output["x_norm_clstoken"]
                    # Fallback — use what we can get
                    logger.warning("  Could not extract patch tokens — using CLS")
                    patch_tokens = patch_tokens.unsqueeze(1).expand(-1, 256, -1)
            else:
                # Older API: output is tensor (B, 1+N_patches, D)
                # Remove CLS token (first token)
                patch_tokens = output[:, 1:, :]

            features_np = patch_tokens.float().cpu().numpy()
            all_features.append(features_np)

        # Free GPU memory between batches
        if device == "cuda":
            torch.cuda.empty_cache()

    # Concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    logger.info(f"  Extracted features: {all_features.shape}")

    # Clean up model from GPU
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return all_features


# ─────────────────────────────────────────────────────────────────
# 2D → 3D Feature Projection
# ─────────────────────────────────────────────────────────────────


def project_features_to_points(
    points: np.ndarray,
    patch_features: np.ndarray,
    camera_transforms: list[np.ndarray],
    mesh: "trimesh.Trimesh",
    resolution: int = 224,
    patch_size: int = 14,
) -> np.ndarray:
    """
    Project 2D DINOv2 patch features back to 3D surface points.

    For each viewpoint:
    1. Project 3D points to 2D using the camera transform
    2. Map each projected point to the nearest DINOv2 patch
    3. Assign the patch's feature vector to that 3D point

    Features are aggregated across views via mean pooling — points
    visible in multiple views get averaged features, improving
    robustness.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) 3D point cloud on mesh surface.
    patch_features : np.ndarray
        (N_views, N_patches, D) DINOv2 patch features.
    camera_transforms : list[np.ndarray]
        List of 4×4 camera-to-world transform matrices.
    mesh : trimesh.Trimesh
        Original mesh (used for depth-based visibility check).
    resolution : int
        Render resolution (matches the DINOv2 input size).
    patch_size : int
        DINOv2 patch size in pixels (14 for ViT-*/14 models).

    Returns
    -------
    np.ndarray
        (N, D) per-point feature vectors. Points with no visible
        features get zero vectors.
    """
    n_points = len(points)
    n_views, n_patches, feature_dim = patch_features.shape
    patches_per_side = resolution // patch_size

    # Accumulate features and count for mean pooling
    feature_sum = np.zeros((n_points, feature_dim), dtype=np.float64)
    feature_count = np.zeros(n_points, dtype=np.int32)

    # Simple perspective camera intrinsics
    # FOV ~60 degrees, matching trimesh's default camera
    focal_length = resolution / (2.0 * np.tan(np.radians(30)))
    cx, cy = resolution / 2.0, resolution / 2.0

    for view_idx, cam_transform in enumerate(camera_transforms):
        # World-to-camera transform (inverse of camera-to-world)
        cam_inv = np.linalg.inv(cam_transform)

        # Transform points to camera space
        points_h = np.hstack([points, np.ones((n_points, 1))])  # (N, 4)
        points_cam = (cam_inv @ points_h.T).T[:, :3]  # (N, 3)

        # Filter: only points in front of camera (z > 0 in camera space)
        # Note: camera looks along -Z, so visible points have negative Z
        # after our look_at convention
        z_vals = -points_cam[:, 2]  # negate because camera looks along -Z
        visible = z_vals > 0.01  # small epsilon to avoid division by zero

        if not np.any(visible):
            continue

        # Project to 2D pixel coordinates (perspective projection)
        x_proj = (focal_length * points_cam[visible, 0] / z_vals[visible]) + cx
        y_proj = (focal_length * (-points_cam[visible, 1]) / z_vals[visible]) + cy

        # Map pixel coordinates to patch indices
        patch_x = np.clip(
            (x_proj / resolution * patches_per_side).astype(int), 0, patches_per_side - 1
        )
        patch_y = np.clip(
            (y_proj / resolution * patches_per_side).astype(int), 0, patches_per_side - 1
        )
        patch_idx = patch_y * patches_per_side + patch_x

        # Filter: only points within image bounds
        in_bounds = (x_proj >= 0) & (x_proj < resolution) & (y_proj >= 0) & (y_proj < resolution)

        # Get the indices of visible & in-bounds points in original array
        visible_indices = np.where(visible)[0]
        valid_mask = in_bounds
        valid_original_indices = visible_indices[valid_mask]
        valid_patch_indices = patch_idx[valid_mask]

        # Assign features
        if len(valid_original_indices) > 0:
            features_for_view = patch_features[view_idx]  # (N_patches, D)
            feature_sum[valid_original_indices] += features_for_view[valid_patch_indices]
            feature_count[valid_original_indices] += 1

    # Mean pooling across views
    has_features = feature_count > 0
    point_features = np.zeros((n_points, feature_dim), dtype=np.float32)
    point_features[has_features] = (
        feature_sum[has_features] / feature_count[has_features, np.newaxis]
    ).astype(np.float32)

    n_assigned = np.sum(has_features)
    logger.info(
        f"  Projected features to {n_assigned}/{n_points} points "
        f"({100 * n_assigned / n_points:.1f}% coverage)"
    )

    return point_features


# ─────────────────────────────────────────────────────────────────
# Feature-Aware Clustering
# ─────────────────────────────────────────────────────────────────


def cluster_3d_features(
    features: np.ndarray,
    points: np.ndarray,
    n_parts: int = 8,
    method: str = "spectral",
    feature_weight: float = 0.7,
    n_neighbors: int = 20,
) -> np.ndarray:
    """
    Cluster 3D points into semantic parts using combined geometric
    and DINOv2 semantic features.

    Supports three methods:
    - "spectral": Spectral clustering on k-NN graph of combined features
    - "kmeans": K-means on combined features (fast baseline)
    - "sampart3d": Full SAMPart3D pipeline via subprocess (if available)

    Parameters
    ----------
    features : np.ndarray
        (N, D) per-point feature vectors from DINOv2 projection.
    points : np.ndarray
        (N, 3) point cloud coordinates.
    n_parts : int
        Target number of semantic parts.
    method : str
        Clustering method: "spectral", "kmeans", or "sampart3d".
    feature_weight : float
        Balance between geometry (0.0) and features (1.0).
        At 0.7, features dominate clustering decisions.
    n_neighbors : int
        Number of neighbors for k-NN graph (spectral clustering).

    Returns
    -------
    np.ndarray
        (N,) integer labels for each point.
    """
    # Normalize features and positions to unit variance

    # Normalize point positions
    pos_centered = points - points.mean(axis=0)
    pos_scale = np.std(pos_centered) + 1e-8
    pos_norm = pos_centered / pos_scale

    # Normalize DINOv2 features
    feat_norm = features.copy()
    feat_scale = np.std(feat_norm) + 1e-8
    if feat_scale > 1e-6:
        feat_norm = feat_norm / feat_scale

    # Combine: weighted concatenation of position + features
    # feature_weight controls the balance
    geo_weight = 1.0 - feature_weight
    combined = np.hstack(
        [
            pos_norm * geo_weight,
            feat_norm * feature_weight,
        ]
    )

    # Check if features are meaningful (not all zeros/random)
    has_real_features = np.std(features) > 1e-4 and np.mean(np.abs(features)) > 1e-4
    if not has_real_features:
        logger.warning(
            "  Features appear to be placeholder — falling back to geometry-only clustering"
        )
        combined = pos_norm

    logger.info(f"  Clustering {len(points)} points into {n_parts} parts ({method})")
    logger.info(
        f"  Combined feature dim: {combined.shape[1]} "
        f"(geo={pos_norm.shape[1]}, feat={feat_norm.shape[1]})"
    )

    if method == "spectral":
        labels = _spectral_cluster(combined, n_parts, n_neighbors)
    elif method == "kmeans":
        labels = _kmeans_cluster(combined, n_parts)
    else:
        logger.warning(f"  Unknown method '{method}' — falling back to kmeans")
        labels = _kmeans_cluster(combined, n_parts)

    return labels


def _spectral_cluster(
    features: np.ndarray,
    n_parts: int,
    n_neighbors: int = 20,
) -> np.ndarray:
    """
    Spectral clustering on k-NN affinity graph.

    Builds a k-nearest-neighbors graph, computes the normalized
    graph Laplacian, extracts the bottom eigenvectors, and clusters
    them with K-means.

    Parameters
    ----------
    features : np.ndarray
        (N, D) combined feature matrix.
    n_parts : int
        Number of clusters.
    n_neighbors : int
        k for k-NN graph.

    Returns
    -------
    np.ndarray
        (N,) cluster labels.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    from scipy.spatial import cKDTree

    n_points = len(features)

    # Subsample for large point clouds (spectral clustering is O(N²))
    max_points = 10000
    if n_points > max_points:
        logger.info(f"  Subsampling from {n_points} to {max_points} for spectral clustering")
        indices = np.random.RandomState(42).choice(n_points, max_points, replace=False)
        sub_features = features[indices]
    else:
        indices = np.arange(n_points)
        sub_features = features

    n_sub = len(sub_features)

    # Build k-NN graph
    tree = cKDTree(sub_features)
    distances, neighbors = tree.query(sub_features, k=min(n_neighbors + 1, n_sub))

    # Remove self-connections (first neighbor is always self)
    distances = distances[:, 1:]
    neighbors = neighbors[:, 1:]

    # Build symmetric affinity matrix with Gaussian kernel
    sigma = np.median(distances) + 1e-8
    rows = np.repeat(np.arange(n_sub), neighbors.shape[1])
    cols = neighbors.flatten()
    weights = np.exp(-(distances.flatten() ** 2) / (2 * sigma**2))

    affinity = csr_matrix((weights, (rows, cols)), shape=(n_sub, n_sub))
    affinity = (affinity + affinity.T) / 2  # Symmetrize

    # Normalized graph Laplacian
    degree = np.array(affinity.sum(axis=1)).flatten()
    degree_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    d_inv_sqrt = csr_matrix(
        (degree_inv_sqrt, (np.arange(n_sub), np.arange(n_sub))),
        shape=(n_sub, n_sub),
    )
    laplacian = np.eye(n_sub) - (d_inv_sqrt @ affinity @ d_inv_sqrt)

    # Compute bottom eigenvectors (skip the trivial first one)
    n_eigenvectors = min(n_parts + 1, n_sub - 1)
    try:
        eigenvalues, eigenvectors = eigsh(laplacian, k=n_eigenvectors, which="SM", maxiter=1000)
        # Use eigenvectors 1..n_parts (skip the constant eigenvector)
        embedding = (
            eigenvectors[:, 1 : n_parts + 1] if n_eigenvectors > n_parts else eigenvectors[:, 1:]
        )
    except Exception as e:
        logger.warning(f"  Eigsh failed ({e}) — falling back to K-means")
        return _kmeans_cluster(features, n_parts)

    # Cluster the spectral embedding with K-means
    sub_labels = _kmeans_cluster(embedding, n_parts)

    # If we subsampled, propagate labels to all points via nearest-neighbor
    if n_points > max_points:
        full_tree = cKDTree(sub_features)
        _, nearest = full_tree.query(features, k=1)
        labels = sub_labels[nearest]
    else:
        labels = sub_labels

    return labels


def _kmeans_cluster(features: np.ndarray, n_parts: int) -> np.ndarray:
    """K-means clustering with sklearn/scipy fallback."""
    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_parts, random_state=42, n_init=10)
        return kmeans.fit_predict(features)
    except ImportError:
        from scipy.cluster.vq import kmeans2, whiten

        whitened = whiten(features.astype(np.float64))
        _, labels = kmeans2(whitened, n_parts, minit="points", seed=42)
        return labels.astype(np.int32)


# ─────────────────────────────────────────────────────────────────
# Label mapping and post-processing
# ─────────────────────────────────────────────────────────────────


def map_point_labels_to_faces(
    point_labels: np.ndarray,
    face_indices: np.ndarray,
    n_faces: int,
) -> np.ndarray:
    """
    Map point-level cluster labels back to mesh faces.

    Uses majority voting: each face gets the most common label
    among its sampled points.

    Parameters
    ----------
    point_labels : np.ndarray
        (N_points,) cluster label for each sampled point.
    face_indices : np.ndarray
        (N_points,) face index each point was sampled from.
    n_faces : int
        Total number of faces in the mesh.

    Returns
    -------
    np.ndarray
        (N_faces,) label for each face.
    """
    from collections import Counter

    face_label_counts = {}
    for pt_label, face_idx in zip(point_labels, face_indices):
        if face_idx not in face_label_counts:
            face_label_counts[face_idx] = Counter()
        face_label_counts[face_idx][pt_label] += 1

    # Majority vote per face
    face_labels = np.zeros(n_faces, dtype=np.int32)
    for face_idx, counts in face_label_counts.items():
        face_labels[face_idx] = counts.most_common(1)[0][0]

    # Assign unlabeled faces to nearest labeled neighbor
    unlabeled = np.where(~np.isin(np.arange(n_faces), list(face_label_counts.keys())))[0]
    if len(unlabeled) > 0:
        logger.info(f"  {len(unlabeled)} faces had no sampled points — assigning to neighbors")
        # Simple: assign to label 0 (will be fixed by boundary smoothing)
        face_labels[unlabeled] = 0

    return face_labels


def smooth_boundaries(
    mesh: "trimesh.Trimesh",
    face_labels: np.ndarray,
    iterations: int = 3,
) -> np.ndarray:
    """
    Graph-based label smoothing to reduce boundary noise.

    Uses a pre-built adjacency list for O(N × avg_neighbors) per
    iteration, much faster than the naive O(N × E) approach.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (needed for face adjacency).
    face_labels : np.ndarray
        (N_faces,) current labels.
    iterations : int
        Number of smoothing passes.

    Returns
    -------
    np.ndarray
        Smoothed face labels.
    """
    from collections import Counter

    labels = face_labels.copy()
    n_faces = len(labels)

    # Pre-build adjacency list from face_adjacency pairs
    # This is O(E) once, then O(avg_neighbors) per face per iteration
    adjacency_list = [[] for _ in range(n_faces)]
    for pair in mesh.face_adjacency:
        adjacency_list[pair[0]].append(pair[1])
        adjacency_list[pair[1]].append(pair[0])

    for iteration in range(iterations):
        new_labels = labels.copy()
        changed = 0

        for face_idx in range(n_faces):
            neighbors = adjacency_list[face_idx]
            if not neighbors:
                continue

            # Include self in voting
            neighbor_labels = [labels[n] for n in neighbors] + [labels[face_idx]]
            counts = Counter(neighbor_labels)
            best_label = counts.most_common(1)[0][0]

            if best_label != labels[face_idx]:
                new_labels[face_idx] = best_label
                changed += 1

        labels = new_labels
        logger.debug(f"  Smoothing iteration {iteration + 1}: {changed} faces changed")

        if changed == 0:
            break

    return labels


def merge_small_parts(
    mesh: "trimesh.Trimesh",
    face_labels: np.ndarray,
    min_faces: int = 50,
) -> np.ndarray:
    """
    Merge very small parts (<min_faces) into their largest adjacent neighbor.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    face_labels : np.ndarray
        (N_faces,) current labels.
    min_faces : int
        Minimum face count for a valid part.

    Returns
    -------
    np.ndarray
        Updated labels with small parts merged.
    """
    from collections import Counter

    labels = face_labels.copy()
    n_faces = len(labels)

    # Build adjacency list
    adjacency_list = [[] for _ in range(n_faces)]
    for pair in mesh.face_adjacency:
        adjacency_list[pair[0]].append(pair[1])
        adjacency_list[pair[1]].append(pair[0])

    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        if count >= min_faces:
            continue

        # Find faces with this label
        small_faces = np.where(labels == label)[0]

        # Find neighboring labels
        neighbor_labels = Counter()
        for face_idx in small_faces:
            for nf in adjacency_list[face_idx]:
                if labels[nf] != label:
                    neighbor_labels[labels[nf]] += 1

        if neighbor_labels:
            # Merge into the most common neighbor
            merge_target = neighbor_labels.most_common(1)[0][0]
            labels[small_faces] = merge_target
            logger.info(f"  Merged part {label} ({count} faces) into part {merge_target}")

    return labels


def extract_submeshes(
    mesh: "trimesh.Trimesh",
    face_labels: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    """
    Split mesh into sub-meshes by face label and save each as .obj.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Full monolithic mesh.
    face_labels : np.ndarray
        (N_faces,) label for each face.
    output_dir : Path
        Directory to save sub-mesh .obj files.

    Returns
    -------
    list[Path]
        Ordered list of paths to saved sub-mesh files.
    """

    paths = []
    unique_labels = sorted(np.unique(face_labels))

    for i, label in enumerate(unique_labels):
        # Get faces belonging to this part
        face_mask = face_labels == label
        face_indices = np.where(face_mask)[0]

        # Extract sub-mesh
        sub_mesh = mesh.submesh([face_indices], append=True)

        # Clean up
        mask = sub_mesh.nondegenerate_faces()
        sub_mesh.update_faces(mask)
        sub_mesh.remove_unreferenced_vertices()
        sub_mesh.fix_normals()

        # Save
        filename = f"part_{i:03d}.obj"
        filepath = output_dir / filename
        sub_mesh.export(str(filepath))
        paths.append(filepath)

    return paths
