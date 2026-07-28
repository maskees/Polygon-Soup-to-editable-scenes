"""Tests for Stage 2 — 3D Reconstruction."""

import numpy as np
import pytest
import trimesh

from src.stage2_reconstruct import postprocess_mesh, validate_mesh


class TestPostprocessMesh:
    def test_decimation(self):
        """Mesh should be decimated to target face count."""
        mesh = trimesh.creation.icosphere(subdivisions=5)  # ~20K faces
        assert len(mesh.faces) > 5000
        result = postprocess_mesh(mesh, target_faces=5000, smoothing_iterations=0)
        assert len(result.faces) <= 5000

    def test_normalization(self):
        """Mesh should be centered and scaled to unit cube."""
        mesh = trimesh.creation.box(extents=(10, 20, 30))
        mesh.vertices += [100, 200, 300]  # Offset from origin

        result = postprocess_mesh(mesh, target_faces=100000, smoothing_iterations=0)
        # Check centered
        np.testing.assert_allclose(result.centroid, [0, 0, 0], atol=0.1)
        # Check scaled to unit
        assert max(result.extents) <= 1.1  # Allow small tolerance

    def test_normals_fixed(self):
        """Post-processed mesh should have consistent normals."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        result = postprocess_mesh(mesh, target_faces=100000, smoothing_iterations=1)
        assert result.vertex_normals is not None
        assert len(result.vertex_normals) == len(result.vertices)


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
