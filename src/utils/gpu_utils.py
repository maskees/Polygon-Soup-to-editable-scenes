"""
GPU utility functions.
======================
VRAM monitoring, cache management, and device selection.
"""

import logging

logger = logging.getLogger(__name__)


def get_device(prefer_cuda: bool = True) -> str:
    """
    Get the best available compute device.

    Parameters
    ----------
    prefer_cuda : bool
        If True, prefer CUDA over CPU.

    Returns
    -------
    str
        Device string ('cuda', 'cuda:0', or 'cpu').
    """
    import torch

    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def log_gpu_status() -> None:
    """Log current GPU status including name, VRAM, and driver version."""
    try:
        import torch

        if not torch.cuda.is_available():
            logger.info("  GPU: No CUDA device available (running on CPU)")
            from rich.console import Console
            Console().print("  [yellow]⚠ No CUDA GPU detected — running on CPU[/]")
            return

        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_mem / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)

        from rich.console import Console
        console = Console()
        console.print(f"  GPU: {name}")
        console.print(f"  VRAM: {allocated:.1f}GB allocated / {total:.1f}GB total")

        if total < 8.0:
            console.print(
                f"  [yellow]⚠ Low VRAM ({total:.0f}GB) — consider using --low-vram flag[/]"
            )

    except Exception as e:
        logger.warning(f"Could not query GPU status: {e}")


def clear_gpu_cache() -> None:
    """Clear GPU memory cache between pipeline stages."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    except Exception:
        pass


def get_vram_usage_mb() -> float:
    """Get current VRAM usage in megabytes."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
    except Exception:
        pass
    return 0.0
