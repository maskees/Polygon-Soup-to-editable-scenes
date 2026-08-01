"""Tests for Stage 2 — 3D Reconstruction."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from src.stage2_reconstruct import (
    _build_conda_command,
    _decimate_mesh,
    _fill_holes,
    _normalize_bounding_box,
    _run_bridge_subprocess,
    postprocess_mesh,
    reconstruct_with_crm,
    reconstruct_with_unique3d,
    validate_mesh,
)


class TestPostprocessMesh:
    def test_decimation(self):
        """Mesh should be decimated to target face count."""
        try:
            import pymeshlab  # noqa: F401
        except ImportError:
            pytest.importorskip("fast_simplification")

        mesh = trimesh.creation.icosphere(subdivisions=5)  # ~20K faces
        assert len(mesh.faces) > 5000
        result = postprocess_mesh(
            mesh, target_faces=5000, smoothing_iterations=0, use_pymeshlab=False
        )
        assert len(result.faces) <= 5000

    def test_normalization(self):
        """Mesh should be centered and scaled to unit cube."""
        mesh = trimesh.creation.box(extents=(10, 20, 30))
        mesh.vertices += [100, 200, 300]  # Offset from origin

        result = postprocess_mesh(
            mesh, target_faces=100000, smoothing_iterations=0, use_pymeshlab=False
        )
        # Check centered
        np.testing.assert_allclose(result.centroid, [0, 0, 0], atol=0.1)
        # Check scaled to unit
        assert max(result.extents) <= 1.1  # Allow small tolerance

    def test_normals_fixed(self):
        """Post-processed mesh should have consistent normals."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        result = postprocess_mesh(
            mesh, target_faces=100000, smoothing_iterations=1, use_pymeshlab=False
        )
        assert result.vertex_normals is not None
        assert len(result.vertex_normals) == len(result.vertices)

    def test_degenerate_removal(self):
        """Post-processing should handle meshes with degenerate faces."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        result = postprocess_mesh(
            mesh, target_faces=100000, smoothing_iterations=0, use_pymeshlab=False
        )
        # Should still produce a valid mesh
        assert len(result.faces) > 0
        assert len(result.vertices) > 0

    def test_smoothing(self):
        """Smoothing should not drastically change vertex count."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        original_verts = len(mesh.vertices)
        result = postprocess_mesh(
            mesh, target_faces=100000, smoothing_iterations=3, use_pymeshlab=False
        )
        # Vertex count should be unchanged by smoothing
        assert len(result.vertices) == original_verts


class TestValidateMesh:
    def test_sphere_validation(self):
        """A sphere should be watertight with known properties."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        stats = validate_mesh(mesh)

        assert stats["is_watertight"] is True
        assert stats["face_count"] > 0
        assert stats["vertex_count"] > 0
        assert stats["euler_number"] == 2  # For a sphere
        assert stats["volume"] is not None
        assert stats["volume"] > 0

    def test_non_watertight_validation(self):
        """A non-watertight mesh should report None volume."""
        # Create a plane (non-watertight)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        stats = validate_mesh(mesh)
        assert stats["is_watertight"] is False
        assert stats["volume"] is None
        assert stats["face_count"] == 2
        assert stats["vertex_count"] == 4

    def test_return_format(self):
        """validate_mesh should return all expected keys."""
        mesh = trimesh.creation.icosphere(subdivisions=2)
        stats = validate_mesh(mesh)

        expected_keys = {
            "face_count",
            "vertex_count",
            "is_watertight",
            "euler_number",
            "bounding_box",
            "volume",
        }
        assert set(stats.keys()) == expected_keys


class TestNormalizeBoundingBox:
    def test_centers_at_origin(self):
        """Mesh should be centered at origin after normalization."""
        mesh = trimesh.creation.box(extents=(2, 2, 2))
        mesh.vertices += [50, 50, 50]

        result = _normalize_bounding_box(mesh)
        np.testing.assert_allclose(result.centroid, [0, 0, 0], atol=1e-6)

    def test_scales_to_unit_cube(self):
        """Mesh should fit within unit cube after normalization."""
        mesh = trimesh.creation.box(extents=(10, 5, 3))
        result = _normalize_bounding_box(mesh)
        assert max(result.extents) <= 1.0 + 1e-6


class TestDecimateMesh:
    def test_trimesh_fallback(self):
        """Decimation should work with trimesh when PyMeshLab is unavailable."""
        pytest.importorskip("fast_simplification", reason="fallback unavailable")

        mesh = trimesh.creation.icosphere(subdivisions=5)
        original_faces = len(mesh.faces)
        assert original_faces > 5000

        result = _decimate_mesh(mesh, target_faces=5000, use_pymeshlab=False)
        # Should have fewer faces than original
        assert len(result.faces) < original_faces


class TestFillHoles:
    def test_watertight_mesh_unchanged(self):
        """A watertight mesh should pass through hole filling unchanged."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        assert mesh.is_watertight

        result = _fill_holes(mesh, use_pymeshlab=False)
        assert result.is_watertight
        assert len(result.faces) == len(mesh.faces)


class TestBuildCondaCommand:
    def test_basic_command(self):
        """Should build a valid conda run command."""
        from pathlib import Path

        cmd = _build_conda_command(
            conda_env="crm",
            script=Path("scripts/crm_bridge.py"),
            args=["--input", "test.png", "--output", "out.obj"],
            low_vram=False,
        )

        assert cmd[0] == "conda"
        assert cmd[1] == "run"
        assert cmd[3] == "crm"
        assert "crm_bridge.py" in " ".join(cmd)
        assert "--low-vram" not in cmd

    def test_low_vram_flag(self):
        """Should include --low-vram when enabled."""
        from pathlib import Path

        cmd = _build_conda_command(
            conda_env="crm",
            script=Path("scripts/crm_bridge.py"),
            args=["--input", "test.png"],
            low_vram=True,
        )

        assert "--low-vram" in cmd


class TestRunBridgeSubprocess:
    def test_timeout_raises_runtime_error(self, tmp_path):
        """Should raise RuntimeError on subprocess timeout."""
        import subprocess
        from unittest.mock import patch

        cmd = ["python", "-c", "import time; time.sleep(100)"]
        expected_output = tmp_path / "nonexistent.obj"

        with patch("src.stage2_reconstruct.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=cmd, timeout=1)

            with pytest.raises(RuntimeError, match="timed out"):
                _run_bridge_subprocess(cmd, expected_output, timeout=1, backend_name="TestBackend")

    def test_missing_output_raises_runtime_error(self, tmp_path):
        """Should raise RuntimeError when subprocess succeeds but no output file."""
        from unittest.mock import MagicMock, patch

        cmd = ["echo", "ok"]
        expected_output = tmp_path / "should_not_exist.obj"

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"status": "success", "message": "done"}\n'
        mock_result.stderr = ""

        with patch("src.stage2_reconstruct.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="output mesh not found"):
                _run_bridge_subprocess(
                    cmd, expected_output, timeout=300, backend_name="TestBackend"
                )

    def test_nonzero_exit_raises_runtime_error(self, tmp_path):
        """Should raise RuntimeError when subprocess exits with nonzero code."""
        from unittest.mock import MagicMock, patch

        cmd = ["python", "-c", "exit(1)"]
        expected_output = tmp_path / "out.obj"

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = '{"status": "error", "message": "something broke"}\n'
        mock_result.stderr = "Traceback (most recent call last): ..."

        with patch("src.stage2_reconstruct.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                _run_bridge_subprocess(
                    cmd, expected_output, timeout=300, backend_name="TestBackend"
                )

    def test_conda_not_found_raises_runtime_error(self, tmp_path):
        """Should raise RuntimeError when conda is not found."""
        from unittest.mock import patch

        cmd = ["conda", "run", "-n", "test_env", "python", "script.py"]
        expected_output = tmp_path / "out.obj"

        with patch("src.stage2_reconstruct.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("conda not found")

            with pytest.raises(RuntimeError, match="conda"):
                _run_bridge_subprocess(
                    cmd, expected_output, timeout=300, backend_name="TestBackend"
                )


class TestReconstructWithCrmErrors:
    def test_missing_bridge_script(self, tmp_path):
        """Should raise FileNotFoundError when bridge script doesn't exist."""
        from unittest.mock import patch

        images = [tmp_path / f"{v}.png" for v in ["front", "back", "left", "right"]]
        for img in images:
            img.touch()

        # Patch Path.exists to return False for the bridge script
        original_exists = Path.exists

        def mock_exists(self):
            if "crm_bridge" in str(self):
                return False
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            with pytest.raises(FileNotFoundError, match="bridge script"):
                reconstruct_with_crm(
                    images=images,
                    checkpoint_dir=tmp_path / "checkpoints",
                )


class TestReconstructWithUnique3dErrors:
    def test_missing_bridge_script(self, tmp_path):
        """Should raise FileNotFoundError when bridge script doesn't exist."""
        from unittest.mock import patch

        images = [tmp_path / f"{v}.png" for v in ["front", "back", "left", "right"]]
        for img in images:
            img.touch()

        original_exists = Path.exists

        def mock_exists(self):
            if "unique3d_bridge" in str(self):
                return False
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            with pytest.raises(FileNotFoundError, match="bridge script"):
                reconstruct_with_unique3d(
                    images=images,
                    checkpoint_dir=tmp_path / "checkpoints",
                )
