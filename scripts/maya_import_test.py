import os
import time

import maya.cmds as cmds


def test_usd_import(usda_path):
    """
    Test importing a USD file into Maya and measuring viewport FPS.
    To be run in Maya's Script Editor or via mayapy.
    """
    if not os.path.exists(usda_path):
        cmds.error(f"File not found: {usda_path}")
        return

    # Create a new scene
    cmds.file(new=True, force=True)

    # Make sure USD plugin is loaded
    if not cmds.pluginInfo("mayaUsdPlugin", query=True, loaded=True):
        try:
            cmds.loadPlugin("mayaUsdPlugin")
        except Exception as e:
            cmds.error(f"Could not load mayaUsdPlugin: {e}")
            return

    # Import the USD file
    print(f"Importing {usda_path}...")
    try:
        cmds.file(
            usda_path,
            i=True,
            type="USD Import",
            ignoreVersion=True,
            mergeNamespacesOnClash=False,
            namespace="test",
        )
        print("Import successful.")
    except Exception as e:
        cmds.error(f"Import failed: {e}")
        return

    # Frame the camera on the imported object
    cmds.select(all=True)
    cmds.viewFit(animate=False)
    cmds.select(clear=True)

    # Benchmark Viewport FPS
    print("Starting FPS benchmark...")
    start_time = time.time()
    frames = 0
    duration = 5.0  # Test for 5 seconds

    # Orbit camera slightly each frame
    camera = "persp"

    while (time.time() - start_time) < duration:
        cmds.orbit(camera, horizontalAngle=1.0, pivotPoint=(0, 0, 0))
        cmds.refresh()
        frames += 1

    elapsed = time.time() - start_time
    fps = frames / elapsed

    print("-" * 30)
    print("FPS Benchmark Results:")
    print(f"Elapsed Time : {elapsed:.2f} seconds")
    print(f"Total Frames : {frames}")
    print(f"Average FPS  : {fps:.2f}")
    print("-" * 30)

    # Verify hierarchy
    transforms = cmds.ls(type="transform")
    usd_prims = [t for t in transforms if "test:" in t]
    print(f"Found {len(usd_prims)} imported transform nodes.")
    for p in usd_prims:
        visibility = cmds.getAttr(f"{p}.visibility")
        print(f"  Node: {p} | Visibility: {'ON' if visibility else 'OFF'}")

    return fps


if __name__ == "__main__":
    # Example usage:
    # usd_path = r"C:\\path\\to\\your\\output\\scene_y_up.usda"
    # test_usd_import(usd_path)
    pass
