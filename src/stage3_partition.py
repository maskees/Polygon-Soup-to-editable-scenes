"""
Stage 3 — Native 3D Semantic Slicing via SAMPart3D
=====================================================
Zero-shot semantic clustering of mesh faces directly in 3D.
Uses DINOv2 visual features mapped into 3D space via SAMPart3D.

Input:  Monolithic .obj mesh from Stage 2
Output: N sub-meshes (.obj per semantic part)
"""

import logging
from pathlib import Path

import numpy as np

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

    Segments the monolithic mesh into semantic parts using SAMPart3D's
    DINOv2 feature distillation + 3D clustering pipeline.

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

    # Step 2: Render multi-view images of the mesh
    logger.info(f"Rendering {cfg.n_render_views} views at {cfg.render_resolution}px...")
    rendered_views = render_multiview(
        mesh, n_views=cfg.n_render_views, resolution=cfg.render_resolution
    )

    # Step 3: Extract DINOv2 features from renders
    logger.info("Extracting DINOv2 features...")
    features = extract_dino_features(
        images=rendered_views,
        device=cfg.device,
        use_float16=cfg.use_float16,
    )

    # Step 4: Run SAMPart3D feature distillation and clustering
    logger.info(f"Clustering into ~{cfg.n_target_parts} parts...")
    point_labels = cluster_3d_features(
        features=features,
        points=points,
        n_parts=cfg.n_target_parts,
    )

    # Step 5: Map point labels to face labels
    face_labels = map_point_labels_to_faces(
        point_labels=point_labels,
        face_indices=face_indices,
        n_faces=len(mesh.faces),
    )

    # Step 6: Smooth boundaries
    logger.info("Smoothing partition boundaries...")
    face_labels = smooth_boundaries(
        mesh=mesh,
        face_labels=face_labels,
        iterations=cfg.boundary_smoothing_iterations,
    )

    # Step 7: Merge tiny parts
    face_labels = merge_small_parts(
        mesh=mesh,
        face_labels=face_labels,
        min_faces=cfg.min_part_faces,
    )

    # Step 8: Extract sub-meshes
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
) -> list[np.ndarray]:
    """
    Render mesh from uniformly distributed viewpoints.

    Uses Fibonacci sphere sampling for uniform viewpoint distribution.

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
    list[np.ndarray]
        List of RGB images (resolution, resolution, 3) in uint8.
    """
    import trimesh

    # Generate uniform viewpoints on a sphere using Fibonacci spiral
    views = []
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(n_views):
        theta = np.arccos(1 - 2 * (i + 0.5) / n_views)
        phi = 2 * np.pi * i / golden_ratio

        # Camera position on unit sphere, scaled to 2x mesh extent
        radius = max(mesh.extents) * 2.5
        cam_pos = np.array([
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ])

        # Create scene and render
        scene = mesh.scene()
        # Set camera transform looking at origin from cam_pos
        camera_transform = _look_at(cam_pos, target=np.zeros(3), up=np.array([0, 1, 0]))

        try:
            # Use pyrender or trimesh's built-in renderer
            rendered = scene.save_image(resolution=(resolution, resolution))
            from PIL import Image
            import io

            img = np.array(Image.open(io.BytesIO(rendered)))[:, :, :3]
            views.append(img)
        except Exception:
            # Fallback: create a blank image if rendering fails
            views.append(np.zeros((resolution, resolution, 3), dtype=np.uint8))

    return views


def _look_at(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray
) -> np.ndarray:
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


def extract_dino_features(
    images: list[np.ndarray],
    device: str = "cuda",
    use_float16: bool = False,
) -> np.ndarray:
    """
    Extract DINOv2 features from multi-view renders.

    Parameters
    ----------
    images : list[np.ndarray]
        List of RGB images from multi-view rendering.
    device : str
        Compute device.
    use_float16 : bool
        If True, use half precision.

    Returns
    -------
    np.ndarray
        Feature matrix (N_views, feature_dim) or aggregated features.
    """
    import torch
    import torchvision.transforms as T

    # ---------------------------------------------------------------
    # TODO: Load DINOv2 ViT-L/14 and extract patch features
    #
    # Integration steps:
    #   1. Load model: model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    #   2. Preprocess images: resize to 224×224, normalize with ImageNet stats
    #   3. Extract features: features = model.forward_features(batch)
    #   4. Use patch tokens (not CLS) for spatial features
    #
    # For SAMPart3D integration:
    #   - Feed DINOv2 features into SAMPart3D's pre-trained MLP
    #   - Map 2D features to 3D point cloud using camera projection
    #
    # OOM Mitigation:
    #   - Process views in batches of 4
    #   - Use float16 if use_float16=True
    #   - Call torch.cuda.empty_cache() after feature extraction
    #
    # PLACEHOLDER: Return random features until DINOv2 is integrated
    # ---------------------------------------------------------------

    logger.warning(
        "DINOv2 feature extraction not yet integrated — using random features. "
        "See TODO in extract_dino_features() for integration steps."
    )

    n_views = len(images)
    feature_dim = 1024  # DINOv2 ViT-L feature dimension
    return np.random.randn(n_views, feature_dim).astype(np.float32)


def cluster_3d_features(
    features: np.ndarray,
    points: np.ndarray,
    n_parts: int = 8,
) -> np.ndarray:
    """
    Cluster 3D points into semantic parts based on features.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix from DINOv2 / SAMPart3D backbone.
    points : np.ndarray
        (N, 3) point cloud coordinates.
    n_parts : int
        Target number of semantic parts.

    Returns
    -------
    np.ndarray
        (N,) integer labels for each point.
    """
    # ---------------------------------------------------------------
    # TODO: Replace with SAMPart3D's native clustering pipeline
    #
    # SAMPart3D uses:
    #   1. Scale-conditioned features from PTv3 backbone
    #   2. Spectral clustering on feature similarity graph
    #   3. Multi-granularity merging
    #
    # For now, use K-means on point positions + normals as a baseline.
    # This gives geometric clustering, not semantic clustering.
    # ---------------------------------------------------------------

    logger.info(f"  Clustering {len(points)} points into {n_parts} parts (K-means baseline)")

    try:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_parts, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)
    except ImportError:
        # Fallback: use scipy's K-means if sklearn is not installed
        from scipy.cluster.vq import kmeans2, whiten

        logger.warning("  sklearn not installed — using scipy K-means fallback")
        whitened = whiten(points.astype(np.float64))
        _, labels = kmeans2(whitened, n_parts, minit="points", seed=42)
        labels = labels.astype(np.int32)

    return labels


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

    Uses the face adjacency graph: each face's label is updated to
    the majority label among its neighbors.

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

    # Build adjacency: face_adjacency is (M, 2) array of adjacent face pairs
    adjacency = mesh.face_adjacency

    for _ in range(iterations):
        new_labels = labels.copy()
        for face_idx in range(len(labels)):
            # Find neighbors
            neighbors_mask = (adjacency[:, 0] == face_idx) | (adjacency[:, 1] == face_idx)
            neighbor_faces = adjacency[neighbors_mask].flatten()
            neighbor_faces = neighbor_faces[neighbor_faces != face_idx]

            if len(neighbor_faces) == 0:
                continue

            # Include self in voting
            neighbor_labels = list(labels[neighbor_faces]) + [labels[face_idx]]
            counts = Counter(neighbor_labels)
            new_labels[face_idx] = counts.most_common(1)[0][0]

        labels = new_labels

    return labels


def merge_small_parts(
    mesh: "trimesh.Trimesh",
    face_labels: np.ndarray,
    min_faces: int = 50,
) -> np.ndarray:
    """
    Merge very small parts (< min_faces) into their largest adjacent neighbor.

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
    adjacency = mesh.face_adjacency

    unique_labels, counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, counts):
        if count >= min_faces:
            continue

        # Find faces with this label
        small_faces = np.where(labels == label)[0]

        # Find neighboring labels
        neighbor_labels = Counter()
        for face_idx in small_faces:
            neighbors_mask = (adjacency[:, 0] == face_idx) | (adjacency[:, 1] == face_idx)
            neighbor_faces = adjacency[neighbors_mask].flatten()
            neighbor_faces = neighbor_faces[neighbor_faces != face_idx]

            for nf in neighbor_faces:
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
    import trimesh

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
