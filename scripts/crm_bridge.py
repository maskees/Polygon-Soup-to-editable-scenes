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


def load_crm_pipeline(checkpoint_dir: Path, device: str, dtype):
    """
    Load the CRM two-stage pipeline and reconstruction model.

    Returns
    -------
    tuple
        (pipeline, crm_model) ready for inference.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from model import CRM
    from omegaconf import OmegaConf
    from pipelines import TwoStagePipeline

    # Load CRM reconstruction model
    crm_path = checkpoint_dir / "CRM.pth"
    if not crm_path.exists():
        # Try downloading from HuggingFace
        print(json.dumps({"status": "downloading", "message": "Downloading CRM checkpoint..."}))
        crm_path = hf_hub_download(
            repo_id="Zhengyi/CRM",
            filename="CRM.pth",
            local_dir=str(checkpoint_dir),
        )
        crm_path = Path(crm_path)

    # Load stage configs
    stage1_config_path = checkpoint_dir / "stage1.yaml"
    stage2_config_path = checkpoint_dir / "stage2.yaml"

    # Check for config files, download if missing
    for cfg_name in ["stage1.yaml", "stage2.yaml"]:
        cfg_path = checkpoint_dir / cfg_name
        if not cfg_path.exists():
            hf_hub_download(
                repo_id="Zhengyi/CRM",
                filename=cfg_name,
                local_dir=str(checkpoint_dir),
            )

    # Load the CRM reconstruction head
    specs = json.load(open(checkpoint_dir / "specs_objaverse_total.json"))
    crm_model = CRM(specs).to(device)
    crm_model.load_state_dict(
        torch.load(str(crm_path), map_location=device), strict=False
    )
    crm_model = crm_model.to(dtype)
    crm_model.eval()

    # Load the two-stage diffusion pipeline
    stage1_model_config = OmegaConf.create({
        "config": str(checkpoint_dir / "sd-v2-1-diffusers" / "v2-1_512-ema-pruned.yaml"),
        "resume": str(checkpoint_dir / "pixel-diffusion.ckpt"),
    })
    stage2_model_config = OmegaConf.create({
        "config": str(checkpoint_dir / "sd-v2-1-diffusers" / "v2-1_512-ema-pruned.yaml"),
        "resume": str(checkpoint_dir / "ccm-diffusion.ckpt"),
    })

    # Sampler configs
    stage1_sampler_config = OmegaConf.load(str(stage1_config_path))
    stage2_sampler_config = OmegaConf.load(str(stage2_config_path))

    pipeline = TwoStagePipeline(
        stage1_model_config=stage1_model_config,
        stage2_model_config=stage2_model_config,
        stage1_sampler_config=stage1_sampler_config,
        stage2_sampler_config=stage2_sampler_config,
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
    1. Generate multi-view images from single input (pixel diffusion)
    2. Generate Canonical Coordinate Maps (CCM diffusion)
    3. Reconstruct mesh via FlexiCubes
    """
    import numpy as np
    import torch
    from inference import generate3d

    print(json.dumps({"status": "stage1", "message": "Generating multi-view images..."}))

    # Stage 1: Generate 6 orthogonal views from single image
    stage1_output = pipeline.stage1_sample(image, prompt="3D assets")

    print(json.dumps({"status": "stage2", "message": "Generating CCMs..."}))

    # Stage 2: Generate Canonical Coordinate Maps
    stage2_output = pipeline.stage2_sample(
        image,
        stage1_output,
        prompt="3D assets",
    )

    # stage2_output contains rgb (6-view color) and ccm (coordinate maps)
    rgb = np.array(stage2_output["pixel_images"])
    ccm = np.array(stage2_output["ccm_images"])

    print(json.dumps({"status": "reconstruct", "message": "Running 3D reconstruction..."}))

    # Generate 3D mesh using FlexiCubes
    with torch.no_grad():
        mesh_output = generate3d(crm_model, rgb, ccm, device)

    # Export mesh as OBJ (untextured)
    if hasattr(mesh_output, "export"):
        mesh_output.export(str(output_path))
    else:
        # mesh_output might be a custom Mesh object from CRM
        # Export vertices and faces manually
        vertices = mesh_output.v.detach().cpu().numpy()
        faces = mesh_output.f.detach().cpu().numpy()

        with open(str(output_path), "w") as f:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                # OBJ faces are 1-indexed
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="CRM 3D Reconstruction Bridge")
    parser.add_argument("--input", "-i", required=True, help="Path to front-view RGBA image")
    parser.add_argument("--output", "-o", required=True, help="Output path for .obj mesh")
    parser.add_argument("--checkpoint-dir", default="checkpoints/crm", help="CRM checkpoint directory")
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
        print(json.dumps({"status": "warning", "message": "CUDA not available, falling back to CPU"}))
        device = "cpu"

    try:
        # Setup paths
        setup_crm_paths()

        # Load pipeline
        print(json.dumps({"status": "loading", "message": "Loading CRM pipeline..."}))
        pipeline, crm_model = load_crm_pipeline(checkpoint_dir, device, dtype)

        # Preprocess image
        print(json.dumps({"status": "preprocess", "message": "Preprocessing input image..."}))
        image = preprocess_image(input_path)

        # Run inference
        run_crm_inference(pipeline, crm_model, image, device, output_path)

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
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
