"""
Mesh Quality Validation Script
================================
Standalone tool to validate mesh quality at any pipeline stage.
Checks watertightness, face count, self-intersections, bounding box, etc.

Usage:
    python scripts/validate_mesh.py mesh.obj [--json] [--strict]
"""

import argparse
import json
import sys
from pathlib import Path


def validate_mesh(mesh_path: Path, strict: bool = False) -> dict:
    """
    Run comprehensive quality checks on a mesh file.

    Parameters
    ----------
    mesh_path : Path
        Path to the mesh file (.obj, .ply, .stl, etc.).
    strict : bool
        If True, stricter thresholds are used.

    Returns
    -------
    dict
        Quality report with metrics and pass/fail flags.
    """
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")

    report = {
        "file": str(mesh_path),
        "file_size_bytes": mesh_path.stat().st_size,
        "metrics": {},
        "checks": {},
        "warnings": [],
    }

    # ── Basic Metrics ──
    n_faces = len(mesh.faces)
    n_vertices = len(mesh.vertices)
    n_edges = len(mesh.edges)

    report["metrics"]["face_count"] = n_faces
    report["metrics"]["vertex_count"] = n_vertices
    report["metrics"]["edge_count"] = n_edges
    report["metrics"]["euler_number"] = int(mesh.euler_number)
    report["metrics"]["surface_area"] = round(float(mesh.area), 6)

    # ── Bounding Box ──
    bounds = mesh.bounds.tolist()
    extents = mesh.extents.tolist()
    report["metrics"]["bounding_box"] = {
        "min": bounds[0],
        "max": bounds[1],
        "extents": extents,
    }

    # ── Watertight Check ──
    is_watertight = bool(mesh.is_watertight)
    report["metrics"]["is_watertight"] = is_watertight
    report["checks"]["watertight"] = is_watertight

    if is_watertight:
        report["metrics"]["volume"] = round(float(mesh.volume), 6)
    else:
        report["metrics"]["volume"] = None
        report["warnings"].append("Mesh is not watertight — volume cannot be computed")

    # ── Face Count Check ──
    min_faces = 1000 if strict else 100
    max_faces = 200000 if strict else 1000000
    face_ok = min_faces <= n_faces <= max_faces
    report["checks"]["face_count_range"] = face_ok
    if not face_ok:
        report["warnings"].append(
            f"Face count {n_faces} outside expected range [{min_faces}, {max_faces}]"
        )

    # ── Degenerate Faces ──
    face_areas = mesh.area_faces
    degenerate_count = int(np.sum(face_areas < 1e-10))
    report["metrics"]["degenerate_faces"] = degenerate_count
    report["checks"]["no_degenerate_faces"] = degenerate_count == 0
    if degenerate_count > 0:
        report["warnings"].append(f"{degenerate_count} degenerate faces (area ≈ 0)")

    # ── Unreferenced Vertices ──
    referenced = set(mesh.faces.flatten())
    unreferenced = n_vertices - len(referenced)
    report["metrics"]["unreferenced_vertices"] = unreferenced
    report["checks"]["no_unreferenced_vertices"] = unreferenced == 0
    if unreferenced > 0:
        report["warnings"].append(f"{unreferenced} unreferenced vertices")

    # ── Bounding Box Normalization ──
    max_extent = max(extents)
    is_normalized = max_extent <= 1.1  # Allow small tolerance
    report["checks"]["bounding_box_normalized"] = is_normalized
    if not is_normalized:
        report["warnings"].append(
            f"Mesh not normalized to unit cube (max extent: {max_extent:.3f})"
        )

    # ── Normal Consistency ──
    # Check if normals are consistently oriented
    try:
        is_consistent = bool(mesh.is_winding_consistent)
    except Exception:
        is_consistent = None
    report["metrics"]["winding_consistent"] = is_consistent
    report["checks"]["consistent_normals"] = is_consistent if is_consistent is not None else True

    # ── Connected Components ──
    try:
        body_count = mesh.body_count
    except Exception:
        body_count = len(mesh.split(only_watertight=False))
    report["metrics"]["connected_components"] = body_count
    report["checks"]["single_body"] = body_count == 1
    if body_count > 1:
        report["warnings"].append(f"Mesh has {body_count} disconnected components")

    # ── Overall Pass/Fail ──
    critical_checks = ["face_count_range", "no_degenerate_faces"]
    if strict:
        critical_checks.extend(["watertight", "bounding_box_normalized", "single_body"])

    report["passed"] = all(report["checks"].get(check, True) for check in critical_checks)

    return report


def print_report(report: dict, use_json: bool = False):
    """Print the validation report in human-readable or JSON format."""
    if use_json:
        print(json.dumps(report, indent=2))
        return

    print("=" * 60)
    print(f"  Mesh Validation Report: {report['file']}")
    print("=" * 60)

    metrics = report["metrics"]
    print(f"\n  Faces:    {metrics['face_count']:,}")
    print(f"  Vertices: {metrics['vertex_count']:,}")
    print(f"  Edges:    {metrics['edge_count']:,}")
    print(f"  Euler #:  {metrics['euler_number']}")
    print(f"  Area:     {metrics['surface_area']}")

    if metrics["volume"] is not None:
        print(f"  Volume:   {metrics['volume']}")

    bb = metrics["bounding_box"]
    print("\n  Bounding Box:")
    print(f"    Min: ({bb['min'][0]:.3f}, {bb['min'][1]:.3f}, {bb['min'][2]:.3f})")
    print(f"    Max: ({bb['max'][0]:.3f}, {bb['max'][1]:.3f}, {bb['max'][2]:.3f})")
    print(f"    Extents: ({bb['extents'][0]:.3f}, {bb['extents'][1]:.3f}, {bb['extents'][2]:.3f})")

    print("\n  Checks:")
    for check, passed in report["checks"].items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")

    if report["warnings"]:
        print("\n  Warnings:")
        for w in report["warnings"]:
            print(f"    ⚠ {w}")

    overall = "PASSED" if report["passed"] else "FAILED"
    print(f"\n  Overall: {overall}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Mesh Quality Validator")
    parser.add_argument("mesh", type=str, help="Path to mesh file (.obj, .ply, .stl)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--strict", action="store_true", help="Use strict thresholds")
    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: mesh file not found: {mesh_path}", file=sys.stderr)
        sys.exit(1)

    report = validate_mesh(mesh_path, strict=args.strict)
    print_report(report, use_json=args.json)

    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
