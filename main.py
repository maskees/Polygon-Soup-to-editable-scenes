"""
From Polygon Soup to Editable Scenes
=====================================
Dual-Input Pipeline for Compositional 3D Asset Reconstruction

Usage:
    python main.py --input data/input/subject_001 --output data/output/subject_001
    python main.py --input data/input/subject_001 --stages 0,1,2 --backend crm
    python main.py --input data/input/subject_001 --low-vram --up-axis y
    python main.py --input data/input/subject_001 --dry-run
    python main.py --input data/input/subject_001 --verbose
"""

import time

import click
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

console = Console()


@click.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to directory containing 4 orthogonal images (front, back, left, right)",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="data/output",
    help="Output directory for final USD and intermediate files",
)
@click.option(
    "--stages", "-s",
    type=str,
    default="0,1,2,3,4",
    help="Comma-separated stage indices to run (e.g., '0,1,2')",
)
@click.option(
    "--backend", "-b",
    type=click.Choice(["crm", "unique3d"]),
    default="crm",
    help="3D reconstruction backend",
)
@click.option(
    "--up-axis",
    type=click.Choice(["y", "z", "both"]),
    default="both",
    help="USD up-axis variant to export",
)
@click.option(
    "--low-vram",
    is_flag=True,
    default=False,
    help="Enable low-VRAM mode (6GB GPU)",
)
@click.option(
    "--config", "-c",
    type=click.Path(),
    default="configs/default.yaml",
    help="Path to YAML configuration file",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Skip stages whose outputs already exist",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable debug-level logging",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate config and print plan without executing stages",
)
def main(input, output, stages, backend, up_axis, low_vram, config, skip_existing, verbose, dry_run):
    """Run the full 3D reconstruction and decomposition pipeline."""
    from src.utils.logging_utils import setup_logging
    setup_logging(verbose=verbose)

    from src.config import load_config
    from src.stage0_ingest import run_ingestion
    from src.stage1_segment import run_segmentation
    from src.stage2_reconstruct import run_reconstruction
    from src.stage3_partition import run_partition
    from src.stage4_usd import run_usd_export
    from src.utils.gpu_utils import log_gpu_status, clear_gpu_cache

    # Load configuration
    cfg = load_config(config, low_vram=low_vram)
    stage_list = [int(s.strip()) for s in stages.split(",")]
    input_dir = Path(input)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage registry
    stage_runners = {
        0: ("Image Ingestion", run_ingestion),
        1: ("2D Segmentation (SAM 2)", run_segmentation),
        2: ("3D Reconstruction", run_reconstruction),
        3: ("Semantic Partitioning (SAMPart3D)", run_partition),
        4: ("USD Export", run_usd_export),
    }

    # Print header
    console.rule("[bold blue]Polygon Soup → Editable Scenes Pipeline[/]")
    console.print(f"  Input:      {input_dir}")
    console.print(f"  Output:     {output_dir}")
    console.print(f"  Backend:    {backend}")
    console.print(f"  Stages:     {stage_list}")
    console.print(f"  Low VRAM:   {low_vram}")
    console.print(f"  Verbose:    {verbose}")
    console.print(f"  Config:     {config}")
    console.print(f"  Skip exist: {skip_existing}")
    console.print()
    log_gpu_status()

    # Validate stage list
    for idx in stage_list:
        if idx not in stage_runners:
            console.print(f"[bold red]Unknown stage index: {idx}[/]")
            console.print(f"  Valid stages: {list(stage_runners.keys())}")
            raise click.Abort()

    # Dry run mode — print plan and exit
    if dry_run:
        console.print()
        console.rule("[bold yellow]DRY RUN — Plan[/]")
        for idx in stage_list:
            name, _ = stage_runners[idx]
            console.print(f"  [cyan]Stage {idx}[/]: {name}")
        console.print()
        console.print("[yellow]Configuration validated. No stages executed.[/]")
        console.print(f"  Config file: {config}")
        console.print(f"  Image size: {cfg.target_image_size}px")
        console.print(f"  Target faces: {cfg.target_face_count}")
        console.print(f"  Float16: {cfg.use_float16}")
        console.print(f"  Device: {cfg.device}")
        return

    # Pipeline context — each stage reads from and writes to this dict
    context = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "backend": backend,
        "up_axis": up_axis,
        "cfg": cfg,
        "skip_existing": skip_existing,
    }

    # Execute stages
    pipeline_start = time.time()
    stage_timings = {}
    completed_stages = []
    failed_stage = None

    with Progress() as progress:
        task = progress.add_task("[green]Running pipeline...", total=len(stage_list))
        for idx in stage_list:
            name, runner = stage_runners[idx]
            console.print(f"\n[bold cyan]━━━ Stage {idx}: {name} ━━━[/]")

            stage_start = time.time()
            try:
                context = runner(context)
                elapsed = time.time() - stage_start
                stage_timings[idx] = elapsed
                completed_stages.append(idx)
                console.print(f"[green]  ✓ Stage {idx} complete ({elapsed:.1f}s)[/]")
            except Exception as e:
                elapsed = time.time() - stage_start
                stage_timings[idx] = elapsed
                failed_stage = idx
                console.print(f"[bold red]  ✗ Stage {idx} failed after {elapsed:.1f}s: {e}[/]")

                if verbose:
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")

                # Report what was completed before failure
                if completed_stages:
                    console.print(f"\n[yellow]  Completed stages before failure: {completed_stages}[/]")
                    console.print(f"  To resume, run with: --stages {','.join(str(s) for s in stage_list if s >= idx)}")

                raise

            # Free GPU memory between stages
            clear_gpu_cache()
            progress.advance(task)

    # Summary
    pipeline_elapsed = time.time() - pipeline_start
    console.print()
    console.rule("[bold green]Pipeline Complete[/]")

    # Timing summary table
    timing_table = Table(title="Stage Timing Summary")
    timing_table.add_column("Stage", style="cyan")
    timing_table.add_column("Name", style="white")
    timing_table.add_column("Time", style="green", justify="right")

    for idx in stage_list:
        name, _ = stage_runners[idx]
        elapsed = stage_timings.get(idx, 0)
        status = "✓" if idx in completed_stages else "✗"
        timing_table.add_row(f"{status} {idx}", name, f"{elapsed:.1f}s")

    timing_table.add_row("", "[bold]Total[/]", f"[bold]{pipeline_elapsed:.1f}s[/]")
    console.print(timing_table)

    console.print(f"\n  Output directory: {output_dir}")

    if "usd_files" in context:
        for usd_path in context["usd_files"]:
            console.print(f"  USD file: {usd_path}")

    if "sub_meshes" in context:
        console.print(f"  Sub-meshes: {len(context['sub_meshes'])} parts")


if __name__ == "__main__":
    main()
