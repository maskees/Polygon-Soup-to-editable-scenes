"""Tests for Stage 4 — USD Export."""

import numpy as np
import pytest
import trimesh


class TestUsdExport:
    """
    Tests for USD scene creation.

    NOTE: These tests require the `usd-core` package.
    Skip if not installed.
    """

    @pytest.fixture
    def sample_submeshes(self, tmp_path):
        """Create 3 test sub-meshes."""
        paths = []
        for i in range(3):
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.3 + i * 0.1)
            mesh.vertices += [i * 0.5, 0, 0]
            path = tmp_path / f"part_{i:03d}.obj"
            mesh.export(str(path))
            paths.append(path)
        return paths

    @pytest.mark.skipif(
        not _usd_available(),
        reason="usd-core not installed",
    )
    def test_y_up_export(self, sample_submeshes, tmp_path):
        """Y-Up USD should have correct up axis metadata."""
        from src.stage4_usd import create_usd_scene

        output = tmp_path / "test_y_up.usda"
        create_usd_scene(
            sub_meshes=sample_submeshes,
            output_path=output,
            up_axis="y",
            labels=["head", "torso", "legs"],
        )

        assert output.exists()

        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(output))
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y

    @pytest.mark.skipif(
        not _usd_available(),
        reason="usd-core not installed",
    )
    def test_z_up_export(self, sample_submeshes, tmp_path):
        """Z-Up USD should have correct up axis metadata."""
        from src.stage4_usd import create_usd_scene

        output = tmp_path / "test_z_up.usda"
        create_usd_scene(
            sub_meshes=sample_submeshes,
            output_path=output,
            up_axis="z",
            labels=["head", "torso", "legs"],
        )

        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(output))
        assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z

    @pytest.mark.skipif(
        not _usd_available(),
        reason="usd-core not installed",
    )
    def test_prim_count(self, sample_submeshes, tmp_path):
        """USD should contain one mesh prim per sub-mesh."""
        from src.stage4_usd import create_usd_scene

        output = tmp_path / "test_prims.usda"
        create_usd_scene(
            sub_meshes=sample_submeshes,
            output_path=output,
            up_axis="y",
        )

        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(output))
        mesh_prims = [
            p for p in stage.Traverse()
            if p.IsA(UsdGeom.Mesh)
        ]
        assert len(mesh_prims) == 3


def _usd_available() -> bool:
    """Check if usd-core is installed."""
    try:
        from pxr import Usd

        return True
    except ImportError:
        return False
