"""
Stage 0 — Image Ingestion
==========================
Load, validate, pad to square, and normalize 4 orthogonal images to 512×512 RGB.

Input:  Directory with front.png, back.png, left.png, right.png
Output: 4 cleaned, standardized images saved to intermediate/stage0/
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.config import PipelineConfig

logger = logging.getLogger(__name__)

# Canonical view names — must match exactly
REQUIRED_VIEWS = ["front", "back", "left", "right"]
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def run_ingestion(context: dict) -> dict:
    """
    Main entry point for Stage 0.

    Reads 4 orthogonal images from context['input_dir'], processes them,
    and saves cleaned versions to the intermediate directory.

    Parameters
    ----------
    context : dict
        Pipeline context. Must contain 'input_dir', 'output_dir', 'cfg'.

    Returns
    -------
    dict
        Updated context with 'ingested_images' key mapping view names to file paths.
    """
    input_dir = context["input_dir"]
    output_dir = context["output_dir"]
    cfg: PipelineConfig = context["cfg"]

    stage_output_dir = Path(output_dir) / "intermediate" / "stage0_ingest"
    stage_output_dir.mkdir(parents=True, exist_ok=True)

    # Check for skip_existing — if all 4 outputs exist, reuse them
    if context.get("skip_existing"):
        existing = {}
        all_exist = True
        for view in REQUIRED_VIEWS:
            out_path = stage_output_dir / f"{view}.png"
            if out_path.exists():
                existing[view] = out_path
            else:
                all_exist = False
                break
        if all_exist:
            logger.info(f"Skipping Stage 0 — outputs already exist in {stage_output_dir}")
            context["ingested_images"] = existing
            return context

    # Validate input
    image_paths = validate_input_directory(input_dir)
    logger.info(f"Validated {len(image_paths)} input images from {input_dir}")

    # Process each image
    ingested = {}
    for view_name, img_path in image_paths.items():
        output_path = process_single_image(
            path=img_path,
            output_dir=stage_output_dir,
            view_name=view_name,
            target_size=cfg.target_image_size,
        )
        ingested[view_name] = output_path
        logger.info(f"  {view_name}: {img_path.name} → {output_path.name}")

    context["ingested_images"] = ingested
    return context


def validate_input_directory(input_dir: Path) -> dict[str, Path]:
    """
    Validate that exactly 4 required orthogonal views exist in the directory.

    Parameters
    ----------
    input_dir : Path
        Directory containing the 4 view images.

    Returns
    -------
    dict[str, Path]
        Mapping of view name → file path.

    Raises
    ------
    FileNotFoundError
        If input_dir doesn't exist.
    ValueError
        If required views are missing.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    found = {}
    for view in REQUIRED_VIEWS:
        # Try common extensions
        matched = None
        for ext in SUPPORTED_EXTENSIONS:
            candidate = input_dir / f"{view}{ext}"
            if candidate.exists():
                matched = candidate
                break
        if matched is None:
            raise ValueError(
                f"Missing required view '{view}' in {input_dir}. "
                f"Expected one of: {[f'{view}{e}' for e in SUPPORTED_EXTENSIONS]}"
            )
        found[view] = matched

    return found


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """
    Pad a non-square image with black bars to make it square.

    Preserves the original aspect ratio by centering the image
    within a square canvas of side length = max(height, width).

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C) in uint8.

    Returns
    -------
    np.ndarray
        Square image (max_dim, max_dim, C) in uint8.
    """
    h, w = image.shape[:2]
    if h == w:
        return image

    max_dim = max(h, w)
    channels = image.shape[2] if image.ndim == 3 else 1

    if image.ndim == 3:
        canvas = np.zeros((max_dim, max_dim, channels), dtype=image.dtype)
    else:
        canvas = np.zeros((max_dim, max_dim), dtype=image.dtype)

    # Center the image
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    canvas[y_offset : y_offset + h, x_offset : x_offset + w] = image

    return canvas


def normalize_image(image: np.ndarray, target_size: int = 512) -> np.ndarray:
    """
    Resize image to target_size×target_size and ensure RGB uint8 format.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W, C). Can be RGB, RGBA, BGR, or grayscale.
    target_size : int
        Target side length in pixels.

    Returns
    -------
    np.ndarray
        Normalized image (target_size, target_size, 3) in RGB uint8.
    """
    # Handle color space
    if image.ndim == 2:
        # Grayscale → RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # RGBA → RGB (composite on white background)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        white_bg = np.ones_like(rgb) * 255.0
        image = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)

    # Handle 16-bit images
    if image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)

    # Resize with high-quality interpolation
    if image.shape[0] != target_size or image.shape[1] != target_size:
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)

    return image


def process_single_image(
    path: Path, output_dir: Path, view_name: str, target_size: int = 512
) -> Path:
    """
    Full processing pipeline for a single input image.

    Steps: load → fix EXIF rotation → pad to square → normalize → save.

    Parameters
    ----------
    path : Path
        Path to the input image file.
    output_dir : Path
        Directory to save the processed image.
    view_name : str
        View name (front, back, left, right).
    target_size : int
        Target image size in pixels.

    Returns
    -------
    Path
        Path to the saved processed image.
    """
    # Load with PIL to handle EXIF rotation
    pil_image = Image.open(path)
    pil_image = ImageOps.exif_transpose(pil_image)

    # Convert to numpy (RGB)
    image = np.array(pil_image)

    # If BGR (shouldn't happen with PIL, but guard against it)
    # PIL always returns RGB, so no conversion needed

    # Pad to square
    image = pad_to_square(image)

    # Normalize to target size
    image = normalize_image(image, target_size=target_size)

    # Save as lossless PNG
    output_path = output_dir / f"{view_name}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return output_path
