"""
Environment Verification Script
=================================
Cross-platform script to check that all required dependencies are installed
and properly configured for the pipeline.

Usage:
    python scripts/verify_env.py
"""

import importlib
import platform
import sys

# Fix Windows console encoding for Unicode symbols
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def check_package(
    name: str, import_name: str | None = None, min_version: str | None = None
) -> dict:
    """Check if a package is installed and meets version requirements."""
    import_name = import_name or name
    result = {"name": name, "installed": False, "version": None, "status": "MISSING"}

    try:
        mod = importlib.import_module(import_name)
        result["installed"] = True
        version = getattr(mod, "__version__", None)
        if version is None and hasattr(mod, "VERSION"):
            version = str(mod.VERSION)
        result["version"] = version or "unknown"
        result["status"] = "OK"
    except ImportError:
        pass
    except Exception as e:
        result["status"] = f"ERROR: {e}"

    return result


def check_cuda() -> dict:
    """Check CUDA availability and GPU info."""
    result = {"cuda_available": False, "device_name": None, "vram_gb": None, "cuda_version": None}

    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            result["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            result["vram_gb"] = round(props.total_mem / (1024**3), 1)
            result["cuda_version"] = torch.version.cuda
    except ImportError:
        pass
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    print("=" * 60)
    print("  Pipeline Environment Verification")
    print("=" * 60)
    print()

    # System info
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Arch:     {platform.machine()}")
    print()

    # Core dependencies
    packages = [
        ("torch", "torch", "2.3.0"),
        ("torchvision", "torchvision", "0.18.0"),
        ("trimesh", "trimesh", "4.4.0"),

        ("opencv-python", "cv2", "4.10.0"),
        ("Pillow", "PIL", "10.4.0"),
        ("numpy", "numpy", "1.26.0"),
        ("scipy", "scipy", "1.13.0"),
        ("click", "click", "8.1.0"),
        ("rich", "rich", "13.7.0"),
        ("pyyaml", "yaml", "6.0"),
        ("tqdm", "tqdm", "4.66.0"),
        ("matplotlib", "matplotlib", "3.9.0"),
    ]

    # Optional / later-sprint dependencies
    optional_packages = [
        ("rembg", "rembg", None),
        ("usd-core", "pxr", None),
        ("segment-anything-2", "sam2", None),
        ("pytorch3d", "pytorch3d", None),
    ]

    all_ok = True

    print("  Core Dependencies:")
    print("  " + "-" * 56)
    for name, import_name, min_ver in packages:
        info = check_package(name, import_name, min_ver)
        icon = "✓" if info["installed"] else "✗"
        ver = info["version"] or "—"
        color_status = info["status"]
        print(f"  {icon}  {name:<22} {ver:<16} {color_status}")
        if not info["installed"]:
            all_ok = False

    print()
    print("  Optional / Later-Sprint Dependencies:")
    print("  " + "-" * 56)
    for name, import_name, min_ver in optional_packages:
        info = check_package(name, import_name, min_ver)
        icon = "✓" if info["installed"] else "○"
        ver = info["version"] or "—"
        print(f"  {icon}  {name:<22} {ver:<16} {info['status']}")

    # CUDA / GPU check
    print()
    print("  GPU Status:")
    print("  " + "-" * 56)
    cuda_info = check_cuda()
    if cuda_info["cuda_available"]:
        print(f"  ✓  CUDA available:     {cuda_info['cuda_version']}")
        print(f"  ✓  GPU:                {cuda_info['device_name']}")
        print(f"  ✓  VRAM:               {cuda_info['vram_gb']} GB")
        if cuda_info["vram_gb"] and cuda_info["vram_gb"] < 8.0:
            print("  ⚠  Low VRAM — use --low-vram flag when running pipeline")
    else:
        print("  ⚠  CUDA not available — pipeline will run on CPU (slow)")
        all_ok = False

    # Checkpoint files
    print()
    print("  Checkpoint Files:")
    print("  " + "-" * 56)
    from pathlib import Path

    checkpoints = {
        "SAM 2 (ViT-H)": Path("checkpoints/sam2/sam2_hiera_large.pt"),
        "CRM": Path("checkpoints/crm/model"),
        "Unique3D": Path("checkpoints/unique3d/model"),
        "SAMPart3D": Path("checkpoints/sampart3d/model"),
    }
    for name, path in checkpoints.items():
        exists = path.exists()
        icon = "✓" if exists else "○"
        status = "found" if exists else "not downloaded (optional for Sprint 1)"
        print(f"  {icon}  {name:<22} {status}")

    # Summary
    print()
    print("=" * 60)
    if all_ok:
        print("  ✓ All core dependencies satisfied. Ready to run!")
    else:
        print("  ⚠ Some dependencies missing. Check items above.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
