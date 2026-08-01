"""
Central configuration management.
=================================
Loads YAML config files and applies low-VRAM overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StageConfig:
    """Configuration for a single pipeline stage."""

    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    # Image ingestion
    target_image_size: int = 512
    required_views: list[str] = field(
        default_factory=lambda: ["front", "back", "left", "right"]
    )

    # Segmentation
    sam2_checkpoint: str = "checkpoints/sam2/sam2_hiera_large.pt"
    sam2_model_cfg: str = "sam2_hiera_l.yaml"
    use_rembg_fallback: bool = False
    mask_morph_kernel_size: int = 5
    cross_view_iou_threshold: float = 0.7

    # Reconstruction
    crm_checkpoint_dir: str = "checkpoints/crm"
    unique3d_checkpoint_dir: str = "checkpoints/unique3d"
    target_face_count: int = 50000
    mesh_smoothing_iterations: int = 3
    crm_conda_env: str = "crm"
    unique3d_conda_env: str = "unique3d"
    reconstruction_timeout: int = 300
    use_pymeshlab_postprocess: bool = True

    # Partitioning (DINOv2 + Spectral Clustering)
    sampart3d_checkpoint: str = "checkpoints/sampart3d"
    n_sample_points: int = 100000
    n_render_views: int = 12
    render_resolution: int = 224
    n_target_parts: int = 8
    min_part_faces: int = 50
    boundary_smoothing_iterations: int = 3
    partition_method: str = "spectral"  # "spectral", "kmeans", or "sampart3d"
    spectral_n_neighbors: int = 20
    feature_weight: float = 0.7  # balance: 0=geometry only, 1=features only
    dinov2_model: str = "dinov2_vitl14"
    sampart3d_conda_env: str = "sampart3d"

    # USD Export
    usd_scene_name: str = "Root_Scene"
    export_y_up: bool = True
    export_z_up: bool = True

    # GPU
    device: str = "cuda"
    use_float16: bool = False
    max_batch_size: int = 8


def load_config(config_path: str | Path, low_vram: bool = False) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.
    low_vram : bool
        If True, apply low-VRAM overrides (6GB GPU mode).

    Returns
    -------
    PipelineConfig
        Populated configuration object.
    """
    cfg = PipelineConfig()
    config_path = Path(config_path)

    if config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}

        # Apply YAML values to dataclass
        for key, value in yaml_data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    # Low-VRAM overrides
    if low_vram:
        cfg.use_float16 = True
        cfg.max_batch_size = 2
        cfg.n_sample_points = 50000
        cfg.render_resolution = 224
        cfg.n_render_views = 8
        cfg.target_face_count = 30000

    return cfg
