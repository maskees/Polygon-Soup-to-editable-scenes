"""
Unique3D Subprocess Bridge
============================
Standalone script for Unique3D multi-view 3D reconstruction.
Can run in the same environment as the main pipeline (PyTorch 2.3 + CUDA 12.1)
or in a separate conda environment.

Usage (called by stage2_reconstruct.py, not directly):
    python scripts/unique3d_bridge.py \
        --input-dir data/output/intermediate/stage1_segment \
        --output monolithic_mesh.obj \
        --checkpoint-dir checkpoints/unique3d \
        [--low-vram]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def setup_unique3d_paths():
    """Add Unique3D repo to Python path."""
    unique3d_dir = Path("external/Unique3D")
    if not unique3d_dir.exists():
        raise FileNotFoundError(
            f"Unique3D repo not found at {unique3d_dir}. "
            "Run: git clone https://github.com/AiuniAI/Unique3D.git external/Unique3D"
        )
    sys.path.insert(0, str(unique3d_dir))
    return unique3d_dir


def load_unique3d_pipeline(checkpoint_dir: Path, device: str, use_float16: bool):
    """
    Load the Unique3D reconstruction pipeline.

    Unique3D natively accepts multi-view input images, making it a
    better architectural fit for our 4-view pipeline than CRM.

    Returns
    -------
    object
        Initialized Unique3D pipeline ready for inference.
    """
    import torch

    # Unique3D's pipeline loading
    # The exact import path may vary based on Unique3D version
    try:
        from scripts.inference import Unique3DPipeline
    except ImportError:
        try:
            from app.inference import Unique3DPipeline
        except ImportError:
            # Fallback: try loading from the main app module
            from app import create_pipeline
            pipeline = create_pipeline(
                checkpoint_dir=str(checkpoint_dir),
                device=device,
                dtype=torch.float16 if use_float16 else torch.float32,
            )
            return pipeline

    pipeline = Unique3DPipeline(
        checkpoint_dir=str(checkpoint_dir),
        device=device,
        dtype=torch.float16 if use_float16 else torch.float32,
    )

    return pipeline


def load_images(input_dir: Path):
    """
    Load 4 RGBA images from the segmentation output directory.

    Expected files: front_rgba.png, back_rgba.png, left_rgba.png, right_rgba.png

    Returns
    -------
    list[PIL.Image.Image]
        List of 4 RGBA images in [front, back, left, right] order.
    """
    from PIL import Image

    view_names = ["front", "back", "left", "right"]
    images = []

    for view in view_names:
        # Try both naming conventions
        for pattern in [f"{view}_rgba.png", f"{view}.png"]:
            img_path = input_dir / pattern
            if img_path.exists():
                img = Image.open(str(img_path)).convert("RGBA")
                images.append(img)
                break
        else:
            raise FileNotFoundError(
                f"Could not find {view} view image in {input_dir}. "
                f"Expected: {view}_rgba.png or {view}.png"
            )

    return images


def run_unique3d_inference(pipeline, images, output_path: Path, device: str):
    """
    Run Unique3D inference with multi-view input.

    Unique3D's pipeline:
    1. Processes multi-view images to extract normal maps
    2. Generates geometric features from normals
    3. Reconstructs mesh via ISOMER (Isotropic Surface Meshing)

    Parameters
    ----------
    pipeline : Unique3DPipeline
        Loaded pipeline.
    images : list[PIL.Image.Image]
        4 RGBA input images.
    output_path : Path
        Where to save the output .obj mesh.
    device : str
        Compute device.
    """
    import torch
    import numpy as np

    print(json.dumps({"status": "inference", "message": "Running Unique3D reconstruction..."}))

    with torch.no_grad():
        # Run the full reconstruction pipeline
        # Unique3D accepts a list of PIL images as input
        result = pipeline(images)

    # Extract mesh from result
    # Unique3D may return different formats depending on version
    if hasattr(result, "vertices") and hasattr(result, "faces"):
        vertices = result.vertices
        faces = result.faces
    elif isinstance(result, dict):
        vertices = result.get("vertices", result.get("verts"))
        faces = result.get("faces", result.get("triangles"))
    elif hasattr(result, "export"):
        # It's a trimesh object
        result.export(str(output_path))
        return output_path
    else:
        raise RuntimeError(f"Unexpected Unique3D output type: {type(result)}")

    # Convert tensors to numpy if needed
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()

    # Export as OBJ
    with open(str(output_path), "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Unique3D 3D Reconstruction Bridge")
    parser.add_argument(
        "--input-dir", "-i", required=True,
        help="Directory containing 4 RGBA images (front, back, left, right)",
    )
    parser.add_argument("--output", "-o", required=True, help="Output path for .obj mesh")
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/unique3d",
        help="Unique3D checkpoint directory",
    )
    parser.add_argument("--low-vram", action="store_true", help="Enable float16 mode")
    parser.add_argument("--device", default="cuda", help="Compute device")
    args = parser.parse_args()

    import torch

    start_time = time.time()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)

    if not input_dir.exists():
        print(json.dumps({"status": "error", "message": f"Input directory not found: {input_dir}"}))
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print(json.dumps({"status": "warning", "message": "CUDA not available, falling back to CPU"}))
        device = "cpu"

    try:
        # Setup paths
        setup_unique3d_paths()

        # Load images
        print(json.dumps({"status": "loading_images", "message": "Loading input images..."}))
        images = load_images(input_dir)
        print(json.dumps({"status": "loaded", "message": f"Loaded {len(images)} views"}))

        # Load pipeline
        print(json.dumps({"status": "loading_model", "message": "Loading Unique3D pipeline..."}))
        pipeline = load_unique3d_pipeline(
            checkpoint_dir=checkpoint_dir,
            device=device,
            use_float16=args.low_vram,
        )

        # Clear VRAM before inference
        if device == "cuda":
            torch.cuda.empty_cache()

        # Run inference
        run_unique3d_inference(pipeline, images, output_path, device)

        elapsed = time.time() - start_time
        print(json.dumps({
            "status": "success",
            "message": f"Mesh saved to {output_path}",
            "output_path": str(output_path),
            "elapsed_seconds": round(elapsed, 2),
        }))

    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
        }))
        sys.exit(1)

    finally:
        # Clean up GPU memory
        if "torch" in sys.modules and device == "cuda":
            import torch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
