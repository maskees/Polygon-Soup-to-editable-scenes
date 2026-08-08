"""
TripoSR Subprocess Bridge
==========================
Standalone script that runs inside the crm conda environment.
Invoked by the main pipeline via subprocess to isolate TripoSR's dependencies.

Usage (called by stage2_reconstruct.py, not directly):
    conda run -n crm python scripts/triposr_bridge.py \
        --input front_rgba.png \
        --output monolithic_mesh.obj \
        [--mc-resolution 256] \
        [--chunk-size 8192] \
        [--low-vram]
"""

import argparse
import json
import sys
import time
from pathlib import Path


def setup_triposr_paths():
    """Add TripoSR repo to Python path."""
    triposr_dir = Path("external/TripoSR")
    if not triposr_dir.exists():
        raise FileNotFoundError(
            f"TripoSR repo not found at {triposr_dir}. "
            "Run: git clone https://github.com/VAST-AI-Research/TripoSR.git external/TripoSR"
        )
    sys.path.insert(0, str(triposr_dir))
    return triposr_dir


def preprocess_image(image_path: Path):
    """
    Prepare input image for TripoSR pipeline.

    TripoSR expects an RGBA image with the subject centered,
    resized to ~85% of frame, and composited over gray (0.5).
    """
    import numpy as np
    from PIL import Image
    from tsr.utils import resize_foreground

    img = Image.open(str(image_path)).convert("RGBA")

    # Resize foreground to 85% of bounding box
    img = resize_foreground(img, 0.85)

    # Convert RGBA to RGB with 0.5 gray background
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
    res_img = Image.fromarray((arr * 255.0).astype(np.uint8))

    return res_img


def main():
    parser = argparse.ArgumentParser(description="TripoSR 3D Reconstruction Bridge")
    parser.add_argument("--input", "-i", required=True, help="Path to front-view RGBA image")
    parser.add_argument("--output", "-o", required=True, help="Output path for .obj mesh")
    parser.add_argument(
        "--pretrained-model",
        default="stabilityai/TripoSR",
        help="HuggingFace model ID or local path (default: stabilityai/TripoSR)",
    )
    parser.add_argument(
        "--mc-resolution",
        type=int,
        default=256,
        help="Marching cubes grid resolution (default: 256)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Evaluation chunk size for surface extraction. Smaller = less VRAM (default: 8192)",
    )
    parser.add_argument("--low-vram", action="store_true", help="Enable low-VRAM mode")
    parser.add_argument("--device", default="cuda", help="Compute device")
    args = parser.parse_args()

    import torch

    start_time = time.time()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(json.dumps({"status": "error", "message": f"Input image not found: {input_path}"}))
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print(
            json.dumps({"status": "warning", "message": "CUDA not available, falling back to CPU"})
        )
        device = "cpu"

    # Adjust for low-VRAM
    chunk_size = args.chunk_size
    mc_resolution = args.mc_resolution
    if args.low_vram:
        chunk_size = min(chunk_size, 4096)
        mc_resolution = min(mc_resolution, 192)
        print(
            json.dumps(
                {
                    "status": "info",
                    "message": f"Low-VRAM mode: chunk_size={chunk_size}, mc_resolution={mc_resolution}",
                }
            )
        )

    try:
        # Setup paths
        setup_triposr_paths()

        # Load model
        print(json.dumps({"status": "loading", "message": "Loading TripoSR model..."}))

        from tsr.system import TSR

        model = TSR.from_pretrained(
            args.pretrained_model,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(chunk_size)
        model.to(device)

        print(json.dumps({"status": "loaded", "message": "TripoSR model loaded successfully"}))

        # Preprocess image
        print(json.dumps({"status": "preprocess", "message": "Preprocessing input image..."}))
        image = preprocess_image(input_path)

        # Run inference
        print(json.dumps({"status": "inference", "message": "Running 3D reconstruction..."}))

        with torch.no_grad():
            scene_codes = model([image], device=device)

        # Extract mesh
        print(
            json.dumps(
                {
                    "status": "meshing",
                    "message": f"Extracting mesh (marching cubes resolution={mc_resolution})...",
                }
            )
        )

        meshes = model.extract_mesh(scene_codes, True, resolution=mc_resolution)
        mesh = meshes[0]

        # Save as OBJ
        # TripoSR returns a trimesh.Trimesh object
        mesh.export(str(output_path))

        elapsed = time.time() - start_time
        print(
            json.dumps(
                {
                    "status": "success",
                    "message": f"Mesh saved to {output_path}",
                    "output_path": str(output_path),
                    "elapsed_seconds": round(elapsed, 2),
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces),
                }
            )
        )

    except Exception as e:
        import traceback

        print(
            json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
            )
        )
        sys.exit(1)

    finally:
        # Clean up GPU memory
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
"""
Description: TripoSR bridge script for the 3D reconstruction pipeline.

TripoSR is MIT-licensed, free to use, and runs completely locally.
Model weights are auto-downloaded from HuggingFace on first run
and cached locally for subsequent uses. No API keys or usage limits.
"""
