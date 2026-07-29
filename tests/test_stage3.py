"""Tests for Stage 3 — Semantic Partitioning."""

import numpy as np
import pytest
import trimesh

from src.stage3_partition import (
    extract_submeshes,
    map_point_labels_to_faces,
    merge_small_parts,
    mesh_to_pointcloud,
    smooth_boundaries,
    render_multiview,
    extract_dino_features,
    cluster_3d_features,
    _look_at,
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

    def test_points_dtype(self):
        """Points should be float32."""
        mesh = trimesh.creation.icosphere(subdivisions=2)
        points, _ = mesh_to_pointcloud(mesh, n_points=100)
        assert points.dtype == np.float32


class TestMapPointLabelsToFaces:
    def test_majority_voting(self):
        """Face label should reflect majority of sampled points."""
        # Create scenario: face 0 has 5 points with label 1, 2 with label 0
        point_labels = np.array([1, 1, 1, 1, 1, 0, 0])
        face_indices = np.array([0, 0, 0, 0, 0, 0, 0])
        n_faces = 1

        result = map_point_labels_to_faces(point_labels, face_indices, n_faces)
        assert result[0] == 1  # Majority label

    def test_unlabeled_faces_assigned(self):
        """Faces with no sampled points should get a default label."""
        point_labels = np.array([2, 2, 2])
        face_indices = np.array([0, 0, 0])
        n_faces = 3  # Faces 1 and 2 have no points

        result = map_point_labels_to_faces(point_labels, face_indices, n_faces)
        assert result[0] == 2
        # Unlabeled faces get label 0 by default
        assert result[1] == 0
        assert result[2] == 0

    def test_output_shape(self):
        """Output should have one label per face."""
        point_labels = np.array([0, 1, 0, 1])
        face_indices = np.array([0, 1, 0, 1])
        n_faces = 5

        result = map_point_labels_to_faces(point_labels, face_indices, n_faces)
        assert result.shape == (5,)


class TestSmoothBoundaries:
    def test_preserves_uniform_labels(self):
        """A mesh with uniform labels should not change after smoothing."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        labels = np.zeros(len(mesh.faces), dtype=np.int32)

        result = smooth_boundaries(mesh, labels, iterations=3)
        np.testing.assert_array_equal(result, labels)

    def test_reduces_noise(self):
        """Smoothing should reduce the number of isolated label flips."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        n_faces = len(mesh.faces)

        # Create two clean halves
        labels = np.zeros(n_faces, dtype=np.int32)
        labels[n_faces // 2:] = 1

        # Inject noise: flip 5% of labels randomly
        np.random.seed(42)
        noise_mask = np.random.rand(n_faces) < 0.05
        labels[noise_mask] = 1 - labels[noise_mask]
        noisy_unique = len(np.unique(labels))

        result = smooth_boundaries(mesh, labels, iterations=3)

        # After smoothing, there should be fewer or equal label transitions
        # We measure boundary edges (edges where adjacent faces have different labels)
        adj = mesh.face_adjacency
        noisy_boundaries = np.sum(labels[adj[:, 0]] != labels[adj[:, 1]])
        smooth_boundaries_count = np.sum(result[adj[:, 0]] != result[adj[:, 1]])
        assert smooth_boundaries_count <= noisy_boundaries

    def test_output_shape(self):
        """Output should have same shape as input labels."""
        mesh = trimesh.creation.icosphere(subdivisions=2)
        labels = np.zeros(len(mesh.faces), dtype=np.int32)

        result = smooth_boundaries(mesh, labels, iterations=1)
        assert result.shape == labels.shape


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

    def test_large_parts_preserved(self):
        """Parts above min_faces should not be merged."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        n_faces = len(mesh.faces)

        labels = np.zeros(n_faces, dtype=np.int32)
        labels[n_faces // 2:] = 1  # Half the faces

        result = merge_small_parts(mesh, labels, min_faces=50)
        assert 0 in result
        assert 1 in result


class TestRenderMultiview:
    def test_returns_correct_count(self):
        """Should return requested number of views."""
        mesh = trimesh.creation.icosphere(subdivisions=2)
        views = render_multiview(mesh, n_views=4, resolution=64)
        assert len(views) == 4

    def test_image_shape(self):
        """Each rendered image should have the requested resolution and 3 channels."""
        mesh = trimesh.creation.icosphere(subdivisions=2)
        views = render_multiview(mesh, n_views=2, resolution=64)

        for view in views:
            assert view.shape == (64, 64, 3)
            assert view.dtype == np.uint8

    def test_default_parameters(self):
        """Default render should produce 12 views at 224x224."""
        mesh = trimesh.creation.icosphere(subdivisions=2)
        views = render_multiview(mesh)
        assert len(views) == 12
        for view in views:
            assert view.shape[0] == 224
            assert view.shape[1] == 224


class TestLookAt:
    def test_returns_4x4_matrix(self):
        """Should return a 4x4 transformation matrix."""
        eye = np.array([0, 0, 5.0])
        target = np.array([0, 0, 0.0])
        up = np.array([0, 1, 0.0])

        result = _look_at(eye, target, up)
        assert result.shape == (4, 4)

    def test_eye_position_in_transform(self):
        """Translation column should match eye position."""
        eye = np.array([3.0, 4.0, 5.0])
        target = np.array([0, 0, 0.0])
        up = np.array([0, 1, 0.0])

        result = _look_at(eye, target, up)
        np.testing.assert_allclose(result[:3, 3], eye, atol=1e-6)

    def test_orthogonal_axes(self):
        """The rotation part of the matrix should be orthogonal."""
        eye = np.array([1.0, 2.0, 3.0])
        target = np.array([0, 0, 0.0])
        up = np.array([0, 1, 0.0])

        result = _look_at(eye, target, up)
        R = result[:3, :3]

        # R^T * R should be approximately identity
        product = R.T @ R
        np.testing.assert_allclose(product, np.eye(3), atol=1e-6)


class TestExtractDinoFeatures:
    def test_output_shape(self):
        """Should return features with correct dimensions (placeholder)."""
        images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(4)]
        features = extract_dino_features(images, device="cpu", use_float16=False)

        assert features.shape == (4, 1024)  # n_views x DINOv2 ViT-L dim
        assert features.dtype == np.float32


class TestCluster3dFeatures:
    def test_label_count(self):
        """Should produce at most n_parts unique labels."""
        points = np.random.randn(500, 3).astype(np.float32)
        features = np.random.randn(4, 1024).astype(np.float32)

        labels = cluster_3d_features(features, points, n_parts=5)
        assert labels.shape == (500,)
        assert len(np.unique(labels)) <= 5

    def test_label_range(self):
        """Labels should be non-negative integers."""
        points = np.random.randn(200, 3).astype(np.float32)
        features = np.random.randn(4, 1024).astype(np.float32)

        labels = cluster_3d_features(features, points, n_parts=3)
        assert np.all(labels >= 0)


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

    def test_three_way_split(self, tmp_path):
        """Should handle 3-way partition correctly."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        n_faces = len(mesh.faces)

        labels = np.zeros(n_faces, dtype=np.int32)
        labels[n_faces // 3 : 2 * n_faces // 3] = 1
        labels[2 * n_faces // 3 :] = 2

        paths = extract_submeshes(mesh, labels, tmp_path)
        assert len(paths) == 3

