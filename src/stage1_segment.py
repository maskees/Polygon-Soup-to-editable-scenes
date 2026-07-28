"""
Stage 1 — 2D Segmentation via SAM 2
=====================================
Extract clean subject silhouettes from each of the 4 orthogonal views.
Produces RGBA images with background zeroed out.

Input:  4 normalized RGB images from Stage 0
Output: 4 RGBA images (subject isolated, background = 0)
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def run_segmentation(context: dict) -> dict:
    """
    Main entry point for Stage 1.

    Segments each of the 4 views using SAM 2, producing clean RGBA masks.
    Falls back to rembg if SAM 2 fails or produces poor masks.

    Parameters
    ----------
    context : dict
        Pipeline context. Must contain 'ingested_images' from Stage 0.

    Returns
    -------
    dict
        Updated context with 'segmented_images' key (dict of view → RGBA path).
    """
    cfg: PipelineConfig = context["cfg"]
    output_dir = Path(context["output_dir"]) / "intermediate" / "stage1_segment"
    output_dir.mkdir(parents=True, exist_ok=True)

    ingested = context["ingested_images"]
    device = cfg.device

    # Load SAM 2 model
    logger.info("Loading SAM 2 model...")
    sam2_model = load_sam2_model(
        checkpoint=Path(cfg.sam2_checkpoint),
        model_cfg=cfg.sam2_model_cfg,
        device=device,
    )

    # Segment each view
    segmented = {}
    masks_for_consistency = {}

    for view_name, img_path in ingested.items():
        logger.info(f"  Segmenting: {view_name}")

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Try SAM 2 segmentation
        try:
            rgba = segment_single_view(
                model=sam2_model,
                image=image,
                morph_kernel_size=cfg.mask_morph_kernel_size,
            )
        except Exception as e:
            logger.warning(f"  SAM 2 failed for {view_name}: {e}")
            if cfg.use_rembg_fallback:
                logger.info(f"  Falling back to rembg for {view_name}")
                rgba = apply_rembg_fallback(image)
            else:
                raise

        # Save RGBA image
        output_path = output_dir / f"{view_name}_rgba.png"
        Image.fromarray(rgba).save(str(output_path))
        segmented[view_name] = output_path
        masks_for_consistency[view_name] = rgba[:, :, 3]  # Alpha channel

    # Cross-view consistency check
    consistency_score = check_cross_view_consistency(masks_for_consistency)
    logger.info(f"  Cross-view consistency score: {consistency_score:.3f}")

    if consistency_score < cfg.cross_view_iou_threshold:
        logger.warning(
            f"  Cross-view consistency ({consistency_score:.3f}) is below "
            f"threshold ({cfg.cross_view_iou_threshold}). Masks may be spatially incoherent."
        )

    context["segmented_images"] = segmented
    context["cross_view_consistency"] = consistency_score
    return context


def load_sam2_model(
    checkpoint: Path,
    model_cfg: str = "sam2_hiera_l.yaml",
    device: str = "cuda",
) -> Any:
    """
    Load SAM 2 model with automatic mask generator.

    Parameters
    ----------
    checkpoint : Path
        Path to SAM 2 checkpoint file (.pt).
    model_cfg : str
        Model configuration name (e.g., 'sam2_hiera_l.yaml').
    device : str
        Device to load model on ('cuda' or 'cpu').

    Returns
    -------
    SAM2AutomaticMaskGenerator
        Initialized mask generator.

    Raises
    ------
    ImportError
        If sam2 package is not installed.
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        raise ImportError(
            "SAM 2 not installed. Run: pip install segment-anything-2 "
            "or clone https://github.com/facebookresearch/sam2"
        )

    sam2 = build_sam2(model_cfg, str(checkpoint), device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )

    return mask_generator


def segment_single_view(
    model: Any,
    image: np.ndarray,
    morph_kernel_size: int = 5,
) -> np.ndarray:
    """
    Segment a single view image using SAM 2.

    Generates all masks, selects the one with the largest area,
    applies morphological cleanup, and produces an RGBA image.

    Parameters
    ----------
    model : SAM2AutomaticMaskGenerator
        Loaded SAM 2 mask generator.
    image : np.ndarray
        RGB image (H, W, 3) in uint8.
    morph_kernel_size : int
        Kernel size for morphological closing (fills small holes).

    Returns
    -------
    np.ndarray
        RGBA image (H, W, 4) with background zeroed.
    """
    # Generate all masks
    masks = model.generate(image)

    if len(masks) == 0:
        raise RuntimeError("SAM 2 produced no masks for this image")

    # Select the mask with the largest area (primary subject)
    masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
    best_mask = masks_sorted[0]["segmentation"]  # Boolean array (H, W)

    # Morphological closing to fill small holes
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    mask_uint8 = (best_mask.astype(np.uint8)) * 255
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

    # Create RGBA image
    h, w = image.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = image
    rgba[:, :, 3] = mask_uint8

    return rgba


def check_cross_view_consistency(masks: dict[str, np.ndarray]) -> float:
    """
    Compute cross-view silhouette consistency.

    Checks IoU between opposing view pairs:
    - front ↔ back  (should have similar silhouette)
    - left ↔ right  (should have similar silhouette, mirrored)

    Parameters
    ----------
    masks : dict[str, np.ndarray]
        Mapping of view name → binary mask (H, W) in uint8.

    Returns
    -------
    float
        Average IoU across opposing view pairs (0.0 to 1.0).
    """
    pairs = [("front", "back"), ("left", "right")]
    ious = []

    for view_a, view_b in pairs:
        if view_a not in masks or view_b not in masks:
            continue

        mask_a = (masks[view_a] > 127).astype(bool)
        mask_b = (masks[view_b] > 127).astype(bool)

        # For left/right comparison, flip horizontally
        if view_a in ("left", "right"):
            mask_b = np.fliplr(mask_b)

        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()

        iou = float(intersection) / float(union) if union > 0 else 0.0
        ious.append(iou)

    return float(np.mean(ious)) if ious else 0.0


def apply_rembg_fallback(image: np.ndarray) -> np.ndarray:
    """
    Background removal using rembg (U²-Net) as a fallback.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3) in uint8.

    Returns
    -------
    np.ndarray
        RGBA image (H, W, 4) with background removed.
    """
    try:
        from rembg import remove
    except ImportError:
        raise ImportError("rembg not installed. Run: pip install rembg")

    pil_image = Image.fromarray(image)
    result = remove(pil_image)
    return np.array(result)
