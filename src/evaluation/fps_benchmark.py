"""
Maya Viewport FPS Benchmark
=============================
Script to run INSIDE Maya's Script Editor to measure viewport performance
after importing a USD scene.

Usage (in Maya Script Editor):
    exec(open("scripts/maya_import_test.py").read())
"""

# NOTE: This script uses maya.cmds which is only available inside Maya.
# It cannot be run from a standard Python environment.


def benchmark_viewport_fps(usd_file: str, duration: float = 5.0) -> dict:
    """
    Import a USD file into Maya and measure viewport FPS during orbit.

    Must be run from Maya's Script Editor (Python tab).

    Parameters
    ----------
    usd_file : str
        Absolute path to the .usda file to import.
    duration : float
        Duration in seconds to benchmark.

    Returns
    -------
    dict
        Benchmark results: fps, frame_count, duration, prim_count.
    """
    import time

    import maya.cmds as cmds

    # Clear scene
    cmds.file(new=True, force=True)

    # Import USD
    print(f"Importing: {usd_file}")
    cmds.file(usd_file, i=True, type="USD Import")

    # Frame all objects
    cmds.select(all=True)
    cmds.viewFit()

    # Count prims
    all_transforms = cmds.ls(type="transform")
    all_meshes = cmds.ls(type="mesh")
    prim_count = len(all_meshes)
    print(f"Imported {prim_count} mesh prims")

    # Test visibility toggling
    print("Testing visibility toggles...")
    for t in all_transforms:
        try:
            cmds.setAttr(f"{t}.visibility", 0)
            cmds.refresh(force=True)
            cmds.setAttr(f"{t}.visibility", 1)
            cmds.refresh(force=True)
        except Exception:
            pass

    # Benchmark: orbit camera and count rendered frames
    print(f"Benchmarking viewport FPS for {duration}s...")
    start = time.time()
    frames = 0

    while time.time() - start < duration:
        cmds.orbit("persp", ra=(2, 0))  # Rotate 2 degrees per frame
        cmds.refresh(force=True)
        frames += 1

    elapsed = time.time() - start
    fps = frames / elapsed

    results = {
        "fps": round(fps, 1),
        "frame_count": frames,
        "duration_seconds": round(elapsed, 2),
        "prim_count": prim_count,
        "mesh_count": len(all_meshes),
    }

    print(f"\n=== Viewport FPS Benchmark ===")
    print(f"  FPS:    {results['fps']}")
    print(f"  Frames: {results['frame_count']}")
    print(f"  Prims:  {results['prim_count']}")
    print(f"  Time:   {results['duration_seconds']}s")

    return results


# Auto-run if executed in Maya
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        usd_path = sys.argv[1]
    else:
        usd_path = "data/output/subject_001/usd/scene_y_up.usda"

    benchmark_viewport_fps(usd_path)
