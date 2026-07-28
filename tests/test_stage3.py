"""Tests for Stage 3 — Semantic Partitioning."""

import numpy as np
import pytest
import trimesh

from src.stage3_partition import (
    extract_submeshes,
    map_point_labels_to_faces,
    merge_small_parts,
    mesh_to_pointcloud,
)


class TestMeshToPointcloud:
    def test_point_count(self):
        """Should return requested number of points."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        points, face_indices = mesh_to_pointcloud(mesh, n_points=1000)

        assert points.shape == (1000, 3)
        assert face_indices.shape == (1000,)

    def test_face_indices_valid(self):
        """All face indices should be within valid range."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        _, face_indices = mesh_to_pointcloud(mesh, n_points=500)

        assert np.all(face_indices >= 0)
        assert np.all(face_indices < len(mesh.faces))


class TestMapPointLabelsToFaces:
    def test_majority_voting(self):
        """Face label should reflect majority of sampled points."""
        # Create scenario: face 0 has 5 points with label 1, 2 with label 0
        point_labels = np.array([1, 1, 1, 1, 1, 0, 0])
        face_indices = np.array([0, 0, 0, 0, 0, 0, 0])
        n_faces = 1

        result = map_point_labels_to_faces(point_labels, face_indices, n_faces)
        assert result[0] == 1  # Majority label


class TestMergeSmallParts:
    def test_merge_tiny_part(self):
        """Parts smaller than min_faces should be merged."""
        mesh = trimesh.creation.icosphere(subdivisions=3)

        # Create labels: most faces = 0, a few = 1
        labels = np.zeros(len(mesh.faces), dtype=np.int32)
        labels[:3] = 1  # Only 3 faces for part 1

        result = merge_small_parts(mesh, labels, min_faces=50)
        # Part 1 should have been merged
        assert 1 not in result


class TestExtractSubmeshes:
    def test_submesh_count(self, tmp_path):
        """Should produce one .obj per unique label."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        n_faces = len(mesh.faces)

        labels = np.zeros(n_faces, dtype=np.int32)
        labels[n_faces // 2 :] = 1  # Split into 2 parts

        paths = extract_submeshes(mesh, labels, tmp_path)
        assert len(paths) == 2
        assert all(p.suffix == ".obj" for p in paths)

    def test_face_conservation(self, tmp_path):
        """Total faces across sub-meshes should equal original."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        n_faces = len(mesh.faces)

        labels = np.zeros(n_faces, dtype=np.int32)
        labels[n_faces // 3 :] = 1

        paths = extract_submeshes(mesh, labels, tmp_path)

        total_faces = 0
        for p in paths:
            sub = trimesh.load(str(p), force="mesh")
            total_faces += len(sub.faces)

        assert total_faces == n_faces
