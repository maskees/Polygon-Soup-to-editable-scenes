"""
Stage 2 — Sparse 3D Reconstruction
====================================
Feed masked orthogonal images into CRM, Unique3D, or TripoSR to generate
a monolithic 3D mesh.

Input:  4 RGBA images from Stage 1
Output: Single monolithic .obj mesh (untextured "clay" topology)

Architecture:
    CRM and Unique3D each have specific dependency requirements (nvdiffrast,
    diffusers, transformers at pinned versions). We use subprocess isolation —
    each backend runs in its own conda environment via a bridge script,
    communicating via filesystem (input images → output mesh) and JSON
    status on stdout.
"""

import json
import logging
import subprocess
import typing
from pathlib import Path

if typing.TYPE_CHECKING:
    import trimesh

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


def run_reconstruction(context: dict) -> dict:
    """
    Main entry point for Stage 2.

    Generates a monolithic 3D mesh from 4 segmented RGBA images
    using either CRM or Unique3D backend.

    Parameters
    ----------
    context : dict
        Pipeline context. Must contain 'segmented_images' from Stage 1.

    Returns
    -------
    dict
        Updated context with 'monolithic_mesh' key (Path to .obj).
    """
    import trimesh

    cfg: PipelineConfig = context["cfg"]
    backend = context["backend"]
    output_dir = Path(context["output_dir"]) / "intermediate" / "stage2_reconstruct"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for skip_existing BEFORE accessing segmented_images
    mesh_path = output_dir / "monolithic_mesh.obj"
    if context.get("skip_existing") and mesh_path.exists():
        logger.info(f"Skipping Stage 2 — output exists: {mesh_path}")
        mesh = trimesh.load(str(mesh_path), force="mesh")
        context["monolithic_mesh"] = mesh_path
        context["mesh_quality"] = validate_mesh(mesh)
        return context

    segmented = context["segmented_images"]
    image_paths = [segmented[v] for v in ["front", "back", "left", "right"]]

    # Run reconstruction
    logger.info(f"Running 3D reconstruction with backend: {backend}")

    if backend == "crm":
        mesh = reconstruct_with_crm(
            images=image_paths,
            checkpoint_dir=Path(cfg.crm_checkpoint_dir),
            low_vram=cfg.use_float16,
            conda_env=cfg.crm_conda_env,
            timeout=cfg.reconstruction_timeout,
        )
    elif backend == "unique3d":
        mesh = reconstruct_with_unique3d(
            images=image_paths,
            checkpoint_dir=Path(cfg.unique3d_checkpoint_dir),
            low_vram=cfg.use_float16,
            conda_env=cfg.unique3d_conda_env,
            timeout=cfg.reconstruction_timeout,
        )
    elif backend == "triposr":
        mesh = reconstruct_with_triposr(
            images=image_paths,
            low_vram=cfg.use_float16,
            conda_env=cfg.triposr_conda_env,
            mc_resolution=cfg.triposr_mc_resolution,
            chunk_size=cfg.triposr_chunk_size,
            timeout=cfg.reconstruction_timeout,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Post-process mesh
    logger.info("Post-processing mesh...")
    mesh = postprocess_mesh(
        mesh,
        target_faces=cfg.target_face_count,
        smoothing_iterations=cfg.mesh_smoothing_iterations,
        use_pymeshlab=cfg.use_pymeshlab_postprocess,
    )

    # Validate
    quality = validate_mesh(mesh)
    logger.info(f"  Mesh quality: {quality}")

    # Save
    mesh.export(str(mesh_path))
    logger.info(f"  Saved monolithic mesh to: {mesh_path}")

    context["monolithic_mesh"] = mesh_path
    context["mesh_quality"] = quality
    return context


# ─────────────────────────────────────────────────────────────────
# Backend: CRM (subprocess isolation)
# ─────────────────────────────────────────────────────────────────


def reconstruct_with_crm(
    images: list[Path],
    checkpoint_dir: Path,
    low_vram: bool = False,
    conda_env: str = "crm",
    timeout: int = 300,
) -> "trimesh.Trimesh":
    """
    Run CRM reconstruction via subprocess isolation.

    CRM runs in a separate conda environment via a bridge script to
    isolate its specific dependencies (nvdiffrast, diffusers, etc.).

    CRM expects a single canonical image and internally generates
    6 orthogonal views. We use the front view as the primary input.

    Parameters
    ----------
    images : list[Path]
        List of 4 RGBA image paths [front, back, left, right].
    checkpoint_dir : Path
        Directory containing CRM model checkpoints.
    low_vram : bool
        If True, use float16 and reduced batch sizes.
    conda_env : str
        Name of the conda environment with CRM dependencies.
    timeout : int
        Maximum seconds to wait for the subprocess.

    Returns
    -------
    trimesh.Trimesh
        Reconstructed mesh.

    Raises
    ------
    RuntimeError
        If CRM fails or times out.
    FileNotFoundError
        If the CRM repo or bridge script is not found.
    """
    import trimesh

    # CRM uses the front view as primary input
    front_image_path = images[0]

    # Determine output path
    output_mesh_path = front_image_path.parent.parent / "stage2_reconstruct" / "crm_raw.obj"
    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)

    # Locate bridge script
    bridge_script = Path("scripts/crm_bridge.py")
    if not bridge_script.exists():
        raise FileNotFoundError(
            f"CRM bridge script not found at {bridge_script}. "
            "This file should exist in the scripts/ directory."
        )

    # Build subprocess command
    cmd = _build_conda_command(
        conda_env=conda_env,
        script=bridge_script,
        args=[
            "--input",
            str(front_image_path),
            "--output",
            str(output_mesh_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        low_vram=low_vram,
    )

    logger.info(f"  Launching CRM subprocess: {' '.join(cmd[:5])}...")

    # Run subprocess
    mesh_path = _run_bridge_subprocess(cmd, output_mesh_path, timeout, "CRM")

    # Load the generated mesh
    mesh = trimesh.load(str(mesh_path), force="mesh")
    logger.info(f"  CRM produced mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")

    return mesh


# ─────────────────────────────────────────────────────────────────
# Backend: Unique3D (subprocess or direct import)
# ─────────────────────────────────────────────────────────────────


def reconstruct_with_unique3d(
    images: list[Path],
    checkpoint_dir: Path,
    low_vram: bool = False,
    conda_env: str = "unique3d",
    timeout: int = 300,
) -> "trimesh.Trimesh":
    """
    Run Unique3D reconstruction via subprocess.

    Unique3D natively accepts multi-view input, making it a better
    architectural fit for our 4-view input format. It uses
    PyTorch 2.3 + CUDA 12.1, same as our project.

    Parameters
    ----------
    images : list[Path]
        List of 4 RGBA image paths [front, back, left, right].
    checkpoint_dir : Path
        Directory containing Unique3D model checkpoints.
    low_vram : bool
        If True, use float16 mode.
    conda_env : str
        Name of the conda environment with Unique3D dependencies.
    timeout : int
        Maximum seconds to wait for the subprocess.

    Returns
    -------
    trimesh.Trimesh
        Reconstructed mesh.
    """
    import trimesh

    # Determine input directory (where RGBA images are)
    input_dir = images[0].parent

    # Output path
    output_mesh_path = input_dir.parent / "stage2_reconstruct" / "unique3d_raw.obj"
    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)

    # Locate bridge script
    bridge_script = Path("scripts/unique3d_bridge.py")
    if not bridge_script.exists():
        raise FileNotFoundError(f"Unique3D bridge script not found at {bridge_script}.")

    # Build subprocess command
    cmd = _build_conda_command(
        conda_env=conda_env,
        script=bridge_script,
        args=[
            "--input-dir",
            str(input_dir),
            "--output",
            str(output_mesh_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        low_vram=low_vram,
    )

    logger.info(f"  Launching Unique3D subprocess: {' '.join(cmd[:5])}...")

    # Run subprocess
    mesh_path = _run_bridge_subprocess(cmd, output_mesh_path, timeout, "Unique3D")

    # Load the generated mesh
    mesh = trimesh.load(str(mesh_path), force="mesh")
    logger.info(f"  Unique3D produced mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")

    return mesh


# ─────────────────────────────────────────────────────────────────
# Backend: TripoSR (subprocess isolation)
# ─────────────────────────────────────────────────────────────────


def reconstruct_with_triposr(
    images: list[Path],
    low_vram: bool = False,
    conda_env: str = "crm",
    mc_resolution: int = 256,
    chunk_size: int = 8192,
    timeout: int = 300,
) -> "trimesh.Trimesh":
    """
    Run TripoSR reconstruction via subprocess isolation.

    TripoSR is a feed-forward 3D reconstruction model that converts
    a single image to a 3D mesh. It does NOT require nvdiffrast,
    making it compatible with modern MSVC compilers.

    Model weights are auto-downloaded from HuggingFace on first run.

    Parameters
    ----------
    images : list[Path]
        List of 4 RGBA image paths [front, back, left, right].
        Only the front view is used (TripoSR is single-image).
    low_vram : bool
        If True, use reduced chunk size and resolution.
    conda_env : str
        Name of the conda environment with TripoSR dependencies.
    mc_resolution : int
        Marching cubes grid resolution (higher = more detail).
    chunk_size : int
        Evaluation chunk size (lower = less VRAM).
    timeout : int
        Maximum seconds to wait for the subprocess.

    Returns
    -------
    trimesh.Trimesh
        Reconstructed mesh.

    Raises
    ------
    RuntimeError
        If TripoSR fails or times out.
    FileNotFoundError
        If the bridge script is not found.
    """
    import trimesh

    # TripoSR uses the front view as primary input
    front_image_path = images[0]

    # Determine output path
    output_mesh_path = front_image_path.parent.parent / "stage2_reconstruct" / "triposr_raw.obj"
    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)

    # Locate bridge script
    bridge_script = Path("scripts/triposr_bridge.py")
    if not bridge_script.exists():
        raise FileNotFoundError(
            f"TripoSR bridge script not found at {bridge_script}. "
            "This file should exist in the scripts/ directory."
        )

    # Build subprocess command
    cmd = _build_conda_command(
        conda_env=conda_env,
        script=bridge_script,
        args=[
            "--input",
            str(front_image_path),
            "--output",
            str(output_mesh_path),
            "--mc-resolution",
            str(mc_resolution),
            "--chunk-size",
            str(chunk_size),
        ],
        low_vram=low_vram,
    )

    logger.info(f"  Launching TripoSR subprocess: {' '.join(cmd[:5])}...")

    # Run subprocess
    mesh_path = _run_bridge_subprocess(cmd, output_mesh_path, timeout, "TripoSR")

    # Load the generated mesh
    mesh = trimesh.load(str(mesh_path), force="mesh")
    logger.info(f"  TripoSR produced mesh: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")

    return mesh


# ─────────────────────────────────────────────────────────────────
# Subprocess utilities
# ─────────────────────────────────────────────────────────────────


def _build_conda_command(
    conda_env: str,
    script: Path,
    args: list[str],
    low_vram: bool = False,
) -> list[str]:
    """
    Build a conda run command for subprocess execution.

    Uses `conda run -n <env>` to execute in the target environment
    without needing to activate it first.

    Parameters
    ----------
    conda_env : str
        Name of the conda environment.
    script : Path
        Path to the Python bridge script.
    args : list[str]
        Arguments to pass to the script.
    low_vram : bool
        If True, add --low-vram flag.

    Returns
    -------
    list[str]
        Command list suitable for subprocess.run().
    """
    if conda_env and conda_env.strip():
        cmd = [
            "conda",
            "run",
            "-n",
            conda_env,
            "--no-capture-output",
            "python",
            str(script),
        ]
    else:
        import sys
        cmd = [sys.executable, str(script)]
    cmd.extend(args)

    if low_vram:
        cmd.append("--low-vram")

    return cmd


def _run_bridge_subprocess(
    cmd: list[str],
    expected_output: Path,
    timeout: int,
    backend_name: str,
) -> Path:
    """
    Execute a bridge subprocess and handle results.

    Parses JSON status messages from stdout, logs progress,
    and validates the output mesh was created.

    Parameters
    ----------
    cmd : list[str]
        Subprocess command to execute.
    expected_output : Path
        Expected path of the output mesh file.
    timeout : int
        Maximum seconds to wait.
    backend_name : str
        Name of the backend (for error messages).

    Returns
    -------
    Path
        Path to the output mesh file.

    Raises
    ------
    RuntimeError
        If the subprocess fails, times out, or produces no output.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd()),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{backend_name} reconstruction timed out after {timeout}s. "
            "Try using --low-vram or increasing reconstruction_timeout in config."
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Could not execute conda command. Ensure conda is installed "
            f"and the '{cmd[3]}' environment exists. "
            f"Run: scripts/setup_crm_env.ps1 (or .sh) to create it."
        )

    # Parse stdout for status messages
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            status = json.loads(line)
            level = status.get("status", "info")
            message = status.get("message", line)

            if level == "error":
                logger.error(f"  [{backend_name}] {message}")
            elif level == "warning":
                logger.warning(f"  [{backend_name}] {message}")
            else:
                logger.info(f"  [{backend_name}] {message}")
        except json.JSONDecodeError:
            # Not JSON — log as plain text
            if line:
                logger.debug(f"  [{backend_name}] {line}")

    # Check for errors
    if result.returncode != 0:
        stderr_snippet = result.stderr[-500:] if result.stderr else "(no stderr)"
        raise RuntimeError(
            f"{backend_name} subprocess exited with code {result.returncode}.\n"
            f"stderr: {stderr_snippet}"
        )

    # Verify output exists
    if not expected_output.exists():
        raise RuntimeError(
            f"{backend_name} completed but output mesh not found at {expected_output}. "
            "Check the bridge script output above for errors."
        )

    return expected_output


# ─────────────────────────────────────────────────────────────────
# Mesh post-processing
# ─────────────────────────────────────────────────────────────────


def postprocess_mesh(
    mesh: "trimesh.Trimesh",
    target_faces: int = 50000,
    smoothing_iterations: int = 3,
    use_pymeshlab: bool = True,
) -> "trimesh.Trimesh":
    """
    Post-process a reconstructed mesh for downstream use.

    Steps:
    1. Remove degenerate faces and unreferenced vertices
    2. Decimate to target face count (if over budget)
    3. Laplacian smoothing to remove reconstruction artifacts
    4. Fix normals (consistent winding order)
    5. Attempt watertight closure (fill holes)
    6. Normalize bounding box (center at origin, scale to unit cube)

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Raw mesh from reconstruction.
    target_faces : int
        Maximum face count after decimation.
    smoothing_iterations : int
        Number of Laplacian smoothing passes.
    use_pymeshlab : bool
        If True, use PyMeshLab for higher-quality decimation and hole filling.

    Returns
    -------
    trimesh.Trimesh
        Cleaned mesh ready for semantic partitioning.
    """
    import trimesh

    logger.info(f"  Input: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")

    # 0. Remove degenerate geometry
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()

    # 1. Decimate if over budget
    if len(mesh.faces) > target_faces:
        logger.info(f"  Decimating from {len(mesh.faces)} to {target_faces} faces")
        mesh = _decimate_mesh(mesh, target_faces, use_pymeshlab)

    # 2. Taubin smoothing (feature-preserving, avoids shrinking/melting details)
    if smoothing_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smoothing_iterations)

    # 3. Fix normals
    mesh.fix_normals()

    # 4. Fill holes (attempt watertight)
    mesh = _fill_holes(mesh, use_pymeshlab)

    # 5. Normalize bounding box
    mesh = _normalize_bounding_box(mesh)

    logger.info(f"  Output: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices")
    logger.info(f"  Watertight: {mesh.is_watertight}")

    return mesh


def _decimate_mesh(
    mesh: "trimesh.Trimesh",
    target_faces: int,
    use_pymeshlab: bool = True,
) -> "trimesh.Trimesh":
    """
    Decimate mesh to target face count.

    Uses PyMeshLab's quadric edge collapse when available (higher quality
    than trimesh's simplify_quadric_decimation). Falls back to trimesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.
    target_faces : int
        Target face count.
    use_pymeshlab : bool
        Whether to try PyMeshLab first.

    Returns
    -------
    trimesh.Trimesh
        Decimated mesh.
    """
    import trimesh

    if use_pymeshlab:
        try:
            import pymeshlab
            import numpy as np

            has_colors = hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None
            if has_colors:
                v_colors = (mesh.visual.vertex_colors[:, :4].astype(np.float32) / 255.0)
                m = pymeshlab.Mesh(
                    vertex_matrix=mesh.vertices,
                    face_matrix=mesh.faces,
                    v_color_matrix=v_colors,
                )
            else:
                m = pymeshlab.Mesh(
                    vertex_matrix=mesh.vertices,
                    face_matrix=mesh.faces,
                )

            ms = pymeshlab.MeshSet()
            ms.add_mesh(m)
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target_faces,
                qualitythr=0.3,
                preserveboundary=True,
                preservenormal=True,
                optimalplacement=True,
            )
            result = ms.current_mesh()

            if has_colors and result.has_vertex_color():
                new_v_colors = (result.vertex_color_matrix() * 255.0).astype(np.uint8)
            else:
                new_v_colors = None

            mesh = trimesh.Trimesh(
                vertices=result.vertex_matrix(),
                faces=result.face_matrix(),
                vertex_colors=new_v_colors,
                process=True,
            )
            logger.info(f"  Decimated via PyMeshLab: {len(mesh.faces)} faces (preserved vertex colors)")
            return mesh
        except ImportError:
            logger.warning("  PyMeshLab not installed — falling back to trimesh decimation")
        except Exception as e:
            logger.warning(f"  PyMeshLab decimation failed ({e}) — falling back to trimesh")

    # Fallback: trimesh decimation
    # trimesh's simplify_quadric_decimation expects face_count as int
    # but some versions use target_reduction (fraction). Try both.
    try:
        mesh = mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        # Newer trimesh/fast_simplification uses target_reduction (0-1 fraction)
        current_faces = len(mesh.faces)
        if current_faces > 0:
            reduction = 1.0 - (target_faces / current_faces)
            reduction = max(0.01, min(0.99, reduction))
            try:
                import fast_simplification

                verts_out, faces_out = fast_simplification.simplify(
                    mesh.vertices, mesh.faces, target_reduction=reduction
                )
                mesh = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=True)
            except ImportError:
                logger.warning("  fast_simplification not available — skipping decimation")
    logger.info(f"  Decimated via trimesh: {len(mesh.faces)} faces")
    return mesh


def _fill_holes(
    mesh: "trimesh.Trimesh",
    use_pymeshlab: bool = True,
) -> "trimesh.Trimesh":
    """
    Fill holes in the mesh to achieve watertightness.

    Uses PyMeshLab for more robust hole-filling when available.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (may have holes).
    use_pymeshlab : bool
        Whether to try PyMeshLab first.

    Returns
    -------
    trimesh.Trimesh
        Mesh with holes filled.
    """
    import trimesh

    if use_pymeshlab and not mesh.is_watertight:
        try:
            import pymeshlab

            ms = pymeshlab.MeshSet()
            ms.add_mesh(
                pymeshlab.Mesh(
                    vertex_matrix=mesh.vertices,
                    face_matrix=mesh.faces,
                )
            )
            ms.meshing_close_holes(maxholesize=100)
            result = ms.current_mesh()
            mesh = trimesh.Trimesh(
                vertices=result.vertex_matrix(),
                faces=result.face_matrix(),
                process=True,
            )
            logger.info("  Hole filling via PyMeshLab complete")
            return mesh
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"  PyMeshLab hole filling failed ({e}) — using trimesh")

    # Fallback: trimesh hole filling
    mesh.fill_holes()
    return mesh


def _normalize_bounding_box(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Center mesh at origin and scale to fit in unit cube.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh.

    Returns
    -------
    trimesh.Trimesh
        Normalized mesh.
    """
    # Center at origin
    centroid = mesh.centroid
    mesh.vertices -= centroid

    # Scale to unit cube
    extents = mesh.extents
    max_extent = max(extents)
    if max_extent > 0:
        mesh.vertices /= max_extent

    return mesh


def validate_mesh(mesh: "trimesh.Trimesh") -> dict:
    """
    Compute quality metrics for a mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to validate.

    Returns
    -------
    dict
        Quality metrics including face_count, vertex_count, is_watertight,
        euler_number, and bounding_box.
    """
    return {
        "face_count": len(mesh.faces),
        "vertex_count": len(mesh.vertices),
        "is_watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "bounding_box": mesh.bounds.tolist(),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
    }
