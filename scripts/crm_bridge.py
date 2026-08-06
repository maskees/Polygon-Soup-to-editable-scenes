"""
CRM Subprocess Bridge
======================
Standalone script that runs inside CRM's conda environment (Python 3.9 + PyTorch 1.13).
Invoked by the main pipeline via subprocess to isolate CRM's incompatible dependencies.

Usage (called by stage2_reconstruct.py, not directly):
    conda run -n crm python scripts/crm_bridge.py \
        --input front_rgba.png \
        --output monolithic_mesh.obj \
        --checkpoint-dir checkpoints/crm \
        [--low-vram]
"""

import argparse
import json
import sys
import time
from pathlib import Path


def setup_crm_paths():
    """Add CRM repo to Python path."""
    crm_dir = Path("external/CRM")
    if not crm_dir.exists():
        raise FileNotFoundError(
            f"CRM repo not found at {crm_dir}. "
            "Run: git clone https://github.com/thu-ml/CRM.git external/CRM"
        )
    sys.path.insert(0, str(crm_dir))
    return crm_dir


def load_crm_pipeline(checkpoint_dir: Path, crm_dir: Path, device: str, dtype):
    """
    Load the CRM two-stage pipeline and reconstruction model.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory with CRM.pth, pixel-diffusion.pth, ccm-diffusion.pth.
    crm_dir : Path
        Root of the CRM repo (external/CRM).
    device : str
        Compute device.
    dtype : torch.dtype
        Data type (float32 or float16).

    Returns
    -------
    tuple
        (pipeline, crm_model) ready for inference.
    """
    import torch
    from model import CRM
    from omegaconf import OmegaConf
    from pipelines import TwoStagePipeline

    # Load the CRM reconstruction head
    crm_path = checkpoint_dir / "CRM.pth"
    if not crm_path.exists():
        raise FileNotFoundError(
            f"CRM checkpoint not found at {crm_path}. "
            "Download from: huggingface-cli download Zhengyi/CRM --local-dir checkpoints/crm"
        )

    specs_path = crm_dir / "configs" / "specs_objaverse_total.json"
    specs = json.load(open(specs_path))
    crm_model = CRM(specs).to(device)
    crm_model.load_state_dict(
        torch.load(str(crm_path), map_location=device), strict=False
    )
    crm_model = crm_model.to(dtype)
    crm_model.eval()

    # Load the two-stage diffusion pipeline using CRM repo's config files
    stage1_config = OmegaConf.load(
        str(crm_dir / "configs" / "nf7_v3_SNR_rd_size_stroke.yaml")
    ).config
    stage2_config = OmegaConf.load(
        str(crm_dir / "configs" / "stage2-v2-snr.yaml")
    ).config

    stage1_sampler_config = stage1_config.sampler
    stage2_sampler_config = stage2_config.sampler
    stage1_model_config = stage1_config.models
    stage2_model_config = stage2_config.models

    # Point checkpoint paths to our local files
    pixel_path = checkpoint_dir / "pixel-diffusion.pth"
    ccm_path = checkpoint_dir / "ccm-diffusion.pth"
    if not pixel_path.exists() or not ccm_path.exists():
        raise FileNotFoundError(
            f"Diffusion checkpoints not found in {checkpoint_dir}. "
            "Expected: pixel-diffusion.pth and ccm-diffusion.pth"
        )
    stage1_model_config.resume = str(pixel_path)
    stage2_model_config.resume = str(ccm_path)

    pipeline = TwoStagePipeline(
        stage1_model_config,
        stage2_model_config,
        stage1_sampler_config,
        stage2_sampler_config,
        device=device,
        dtype=dtype,
    )

    return pipeline, crm_model


def preprocess_image(image_path: Path):
    """
    Prepare input image for CRM pipeline.

    CRM expects a single RGBA image with the subject centered
    and background removed (alpha channel as mask).
    """
    from PIL import Image

    img = Image.open(str(image_path)).convert("RGBA")

    # Ensure square
    w, h = img.size
    if w != h:
        max_dim = max(w, h)
        new_img = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
        paste_x = (max_dim - w) // 2
        paste_y = (max_dim - h) // 2
        new_img.paste(img, (paste_x, paste_y))
        img = new_img

    # Resize to 256x256 (CRM's expected resolution)
    img = img.resize((256, 256), Image.LANCZOS)

    return img


def run_crm_inference(pipeline, crm_model, image, device, output_path: Path):
    """
    Run the full CRM inference pipeline.
    
    Steps:
    1. Two-stage diffusion: single image → 6-view pixel images + CCMs
    2. FlexiCubes mesh reconstruction from triplane features
    3. Export as OBJ
    """
    import shutil

    import numpy as np
    import torch
    from inference import generate3d

    print(json.dumps({"status": "diffusion", "message": "Running two-stage diffusion pipeline..."}))

    # Run the full two-stage pipeline (pixel diffusion → CCM diffusion)
    rt_dict = pipeline(image, scale=5.0, step=50)
    stage1_images = rt_dict["stage1_images"]
    stage2_images = rt_dict["stage2_images"]

    # Concatenate views into single images (CRM's expected format)
    np_imgs = np.concatenate(stage1_images, 1)  # 6-view pixel image strip
    np_xyzs = np.concatenate(stage2_images, 1)  # 6-view CCM strip

    print(json.dumps({"status": "reconstruct", "message": "Running 3D mesh reconstruction..."}))

    # Generate 3D mesh using FlexiCubes
    # generate3d returns (glb_path, zip_path)
    glb_path, zip_path = generate3d(crm_model, np_imgs, np_xyzs, device)

    # Extract the OBJ from the zip or convert GLB
    # The zip contains .obj, .mtl, and .png files
    import zipfile

    if zip_path and Path(zip_path).exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the .obj file inside the zip
            obj_names = [n for n in zf.namelist() if n.endswith(".obj")]
            if obj_names:
                with zf.open(obj_names[0]) as obj_in:
                    with open(str(output_path), "wb") as obj_out:
                        obj_out.write(obj_in.read())
                print(json.dumps({"status": "exported", "message": f"OBJ extracted from CRM output"}))
                return output_path

    # Fallback: try to load GLB and re-export as OBJ
    if glb_path and Path(glb_path).exists():
        import trimesh

        mesh = trimesh.load(glb_path, file_type="glb", force="mesh")
        mesh.export(str(output_path))
        print(json.dumps({"status": "exported", "message": f"OBJ exported from GLB"}))
        return output_path

    raise RuntimeError("CRM did not produce a valid mesh output")


def main():
    parser = argparse.ArgumentParser(description="CRM 3D Reconstruction Bridge")
    parser.add_argument("--input", "-i", required=True, help="Path to front-view RGBA image")
    parser.add_argument("--output", "-o", required=True, help="Output path for .obj mesh")
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints/crm", help="CRM checkpoint directory"
    )
    parser.add_argument("--low-vram", action="store_true", help="Enable low-VRAM mode (float16)")
    parser.add_argument("--device", default="cuda", help="Compute device")
    args = parser.parse_args()

    import torch

    start_time = time.time()
    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir)

    if not input_path.exists():
        print(json.dumps({"status": "error", "message": f"Input image not found: {input_path}"}))
        sys.exit(1)

    dtype = torch.float16 if args.low_vram else torch.float32
    device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print(
            json.dumps({"status": "warning", "message": "CUDA not available, falling back to CPU"})
        )
        device = "cpu"

    try:
        # Setup paths
        crm_dir = setup_crm_paths()

        # Load pipeline
        print(json.dumps({"status": "loading", "message": "Loading CRM pipeline..."}))
        pipeline, crm_model = load_crm_pipeline(checkpoint_dir, crm_dir, device, dtype)

        # Preprocess image
        print(json.dumps({"status": "preprocess", "message": "Preprocessing input image..."}))
        image = preprocess_image(input_path)

        # Run inference
        run_crm_inference(pipeline, crm_model, image, device, output_path)

        elapsed = time.time() - start_time
        print(
            json.dumps(
                {
                    "status": "success",
                    "message": f"Mesh saved to {output_path}",
                    "output_path": str(output_path),
                    "elapsed_seconds": round(elapsed, 2),
                }
            )
        )

    except Exception as e:
        print(
            json.dumps(
                {
                    "status": "error",
                    "message": str(e),
                    "error_type": type(e).__name__,
                }
            )
        )
        sys.exit(1)

    finally:
        # Clean up GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
