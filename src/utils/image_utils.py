"""
Image utility functions.
========================
Common image loading, format conversion, and validation helpers.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image_rgb(path: Path | str) -> np.ndarray:
    """Load an image as RGB numpy array (H, W, 3) uint8."""
    img = Image.open(str(path)).convert("RGB")
    return np.array(img)


def load_image_rgba(path: Path | str) -> np.ndarray:
    """Load an image as RGBA numpy array (H, W, 4) uint8."""
    img = Image.open(str(path)).convert("RGBA")
    return np.array(img)


def save_image(image: np.ndarray, path: Path | str) -> None:
    """Save a numpy array as an image file. Handles RGB and RGBA."""
    if image.shape[2] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    Image.fromarray(image, mode=mode).save(str(path))


def compute_mask_area_ratio(mask: np.ndarray) -> float:
    """Compute the fraction of non-zero pixels in a binary mask."""
    return float(np.count_nonzero(mask)) / float(mask.size)


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a binary mask on an image for visualization."""
    overlay = image.copy()
    mask_bool = mask > 127 if mask.dtype == np.uint8 else mask.astype(bool)

    colored = np.zeros_like(image)
    colored[:] = color
    overlay[mask_bool] = cv2.addWeighted(image[mask_bool], 1 - alpha, colored[mask_bool], alpha, 0)

    return overlay
