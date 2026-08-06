"""
Generate Test Mesh
===================
Creates a multi-part compound mesh for testing Stages 3 & 4
without requiring CRM/Unique3D reconstruction.

The mesh is a simple humanoid figure composed of geometric primitives:
- Torso (scaled box)
- Head (sphere)
- Arms (cylinders)
- Legs (cylinders)

These distinct geometric parts should be separable by the DINOv2
spectral clustering in Stage 3.

Usage:
    python scripts/generate_test_mesh.py [--output data/test/test_mesh.obj]
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh


def create_humanoid_mesh() -> trimesh.Trimesh:
    """
    Create a simple humanoid-like compound mesh.

    The mesh has visually distinct geometric regions that
    should cluster into separate parts during partitioning.
    """
    parts = []

    # Torso — box
    torso = trimesh.creation.box(extents=[0.4, 0.6, 0.25])
    torso.apply_translation([0, 0, 0])
    parts.append(torso)

    # Head — sphere
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.15)
    head.apply_translation([0, 0.45, 0])
    parts.append(head)

    # Left arm — cylinder
    left_arm = trimesh.creation.cylinder(radius=0.06, height=0.5)
    left_arm.apply_translation([-0.28, 0.1, 0])
    # Tilt slightly
    rot = trimesh.transformations.rotation_matrix(np.radians(15), [0, 0, 1])
    left_arm.apply_transform(rot)
    parts.append(left_arm)

    # Right arm — cylinder
    right_arm = trimesh.creation.cylinder(radius=0.06, height=0.5)
    right_arm.apply_translation([0.28, 0.1, 0])
    rot = trimesh.transformations.rotation_matrix(np.radians(-15), [0, 0, 1])
    right_arm.apply_transform(rot)
    parts.append(right_arm)

    # Left leg — cylinder
    left_leg = trimesh.creation.cylinder(radius=0.07, height=0.55)
    left_leg.apply_translation([-0.12, -0.55, 0])
    parts.append(left_leg)

    # Right leg — cylinder
    right_leg = trimesh.creation.cylinder(radius=0.07, height=0.55)
    right_leg.apply_translation([0.12, -0.55, 0])
    parts.append(right_leg)

    # Concatenate all parts into a single mesh
    combined = trimesh.util.concatenate(parts)

    # Center at origin and normalize to unit cube
    combined.vertices -= combined.centroid
    max_extent = max(combined.extents)
    if max_extent > 0:
        combined.vertices /= max_extent

    return combined


def create_chair_mesh() -> trimesh.Trimesh:
    """
    Create a simple chair mesh as an alternative test subject.

    Parts: seat, backrest, 4 legs — 6 distinct geometric regions.
    """
    parts = []

    # Seat — flat box
    seat = trimesh.creation.box(extents=[0.5, 0.04, 0.5])
    seat.apply_translation([0, 0.4, 0])
    parts.append(seat)

    # Backrest — tall thin box
    backrest = trimesh.creation.box(extents=[0.5, 0.5, 0.04])
    backrest.apply_translation([0, 0.67, -0.23])
    parts.append(backrest)

    # 4 Legs — cylinders
    leg_positions = [
        (-0.2, 0.19, -0.2),
        (0.2, 0.19, -0.2),
        (-0.2, 0.19, 0.2),
        (0.2, 0.19, 0.2),
    ]
    for pos in leg_positions:
        leg = trimesh.creation.cylinder(radius=0.025, height=0.38)
        leg.apply_translation(pos)
        parts.append(leg)

    combined = trimesh.util.concatenate(parts)
    combined.vertices -= combined.centroid
    max_extent = max(combined.extents)
    if max_extent > 0:
        combined.vertices /= max_extent

    return combined


def main():
    parser = argparse.ArgumentParser(description="Generate test mesh for pipeline validation")
    parser.add_argument(
        "--output",
        "-o",
        default="data/test/test_mesh.obj",
        help="Output path for the test mesh",
    )
    parser.add_argument(
        "--shape",
        choices=["humanoid", "chair"],
        default="humanoid",
        help="Shape to generate",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.shape == "humanoid":
        mesh = create_humanoid_mesh()
    else:
        mesh = create_chair_mesh()

    mesh.export(str(output_path))

    print(f"Generated test mesh: {output_path}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Bounds: {mesh.bounds.tolist()}")


if __name__ == "__main__":
    main()
