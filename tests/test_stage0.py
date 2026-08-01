"""Tests for Stage 0 — Image Ingestion."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.stage0_ingest import (
    normalize_image,
    pad_to_square,
    process_single_image,
    validate_input_directory,
)


@pytest.fixture
def sample_input_dir(tmp_path):
    """Create a temporary directory with 4 test images."""
    for view in ["front", "back", "left", "right"]:
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{view}.png"), img)
    return tmp_path


class TestValidateInputDirectory:
    def test_valid_directory(self, sample_input_dir):
        result = validate_input_directory(sample_input_dir)
        assert len(result) == 4
        assert all(view in result for view in ["front", "back", "left", "right"])

    def test_missing_view(self, tmp_path):
        # Create only 3 views
        for view in ["front", "back", "left"]:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"{view}.png"), img)

        with pytest.raises(ValueError, match="Missing required view"):
            validate_input_directory(tmp_path)

    def test_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError):
            validate_input_directory(Path("/nonexistent/path"))

    def test_jpg_extension(self, tmp_path):
        """Validate that .jpg images are also detected."""
        for view in ["front", "back", "left", "right"]:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"{view}.jpg"), img)

        result = validate_input_directory(tmp_path)
        assert len(result) == 4

    def test_mixed_extensions(self, tmp_path):
        """Mix of .png and .jpg should work."""
        for i, view in enumerate(["front", "back", "left", "right"]):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            ext = ".png" if i % 2 == 0 else ".jpg"
            cv2.imwrite(str(tmp_path / f"{view}{ext}"), img)

        result = validate_input_directory(tmp_path)
        assert len(result) == 4


class TestPadToSquare:
    def test_already_square(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pad_to_square(img)
        assert result.shape == (100, 100, 3)

    def test_landscape(self):
        img = np.ones((100, 200, 3), dtype=np.uint8) * 128
        result = pad_to_square(img)
        assert result.shape == (200, 200, 3)
        # Check padding is black
        assert result[0, 0, 0] == 0  # Top padding

    def test_portrait(self):
        img = np.ones((200, 100, 3), dtype=np.uint8) * 128
        result = pad_to_square(img)
        assert result.shape == (200, 200, 3)

    def test_content_centered(self):
        """Original content should be centered in the padded canvas."""
        img = np.ones((100, 200, 3), dtype=np.uint8) * 200
        result = pad_to_square(img)
        # Image is 100 tall in a 200x200 canvas → starts at y=50
        assert result[50, 100, 0] == 200  # Center of original content
        assert result[0, 0, 0] == 0  # Top-left is padding

    def test_grayscale(self):
        """Grayscale images (2D) should also be padded."""
        img = np.ones((100, 200), dtype=np.uint8) * 128
        result = pad_to_square(img)
        assert result.shape == (200, 200)


class TestNormalizeImage:
    def test_resize(self):
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        result = normalize_image(img, target_size=512)
        assert result.shape == (512, 512, 3)

    def test_rgba_to_rgb(self):
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        img[:, :, :3] = 128
        img[:, :, 3] = 255
        result = normalize_image(img, target_size=100)
        assert result.shape[2] == 3

    def test_grayscale_to_rgb(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        result = normalize_image(img, target_size=100)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_uint16_to_uint8(self):
        img = np.zeros((100, 100, 3), dtype=np.uint16)
        img[:] = 32768  # Mid-range for uint16
        result = normalize_image(img, target_size=100)
        assert result.dtype == np.uint8

    def test_no_resize_if_already_target(self):
        """If image is already the target size, shape should be preserved exactly."""
        img = np.ones((512, 512, 3), dtype=np.uint8) * 100
        result = normalize_image(img, target_size=512)
        assert result.shape == (512, 512, 3)
        assert np.array_equal(result, img)

    def test_small_target_size(self):
        """Non-default target sizes should work."""
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        result = normalize_image(img, target_size=256)
        assert result.shape == (256, 256, 3)


class TestProcessSingleImage:
    def test_end_to_end(self, tmp_path):
        """Full pipeline: create non-square image → process → verify output."""
        # Create a landscape test image (non-square)
        input_img = np.random.randint(0, 255, (300, 500, 3), dtype=np.uint8)
        input_path = tmp_path / "input" / "front.png"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(input_path), input_img)

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result_path = process_single_image(
            path=input_path,
            output_dir=output_dir,
            view_name="front",
            target_size=512,
        )

        # Output file should exist
        assert result_path.exists()
        assert result_path.name == "front.png"

        # Output image should be exactly 512x512 RGB
        output_img = cv2.imread(str(result_path))
        assert output_img.shape == (512, 512, 3)

    def test_rgba_input(self, tmp_path):
        """RGBA input images should be composited onto white and saved as RGB."""
        from PIL import Image as PILImage

        # Create RGBA image
        rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255  # Red
        rgba[:, :, 3] = 128  # Semi-transparent
        pil_img = PILImage.fromarray(rgba, mode="RGBA")
        input_path = tmp_path / "front.png"
        pil_img.save(str(input_path))

        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)

        result_path = process_single_image(
            path=input_path,
            output_dir=output_dir,
            view_name="front",
            target_size=256,
        )

        output_img = cv2.imread(str(result_path))
        assert output_img.shape == (256, 256, 3)

    def test_custom_target_size(self, tmp_path):
        """Custom target sizes should be respected."""
        input_img = np.zeros((200, 200, 3), dtype=np.uint8)
        input_path = tmp_path / "left.png"
        cv2.imwrite(str(input_path), input_img)

        output_dir = tmp_path / "output"
        output_dir.mkdir(exist_ok=True)

        result_path = process_single_image(
            path=input_path,
            output_dir=output_dir,
            view_name="left",
            target_size=128,
        )

        output_img = cv2.imread(str(result_path))
        assert output_img.shape == (128, 128, 3)
