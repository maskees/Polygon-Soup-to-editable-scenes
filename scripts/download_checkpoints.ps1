<# 
.SYNOPSIS
    Download all model checkpoints for the pipeline.
.DESCRIPTION
    Windows PowerShell script to download SAM 2, CRM, Unique3D, and SAMPart3D
    model checkpoints. Run from the project root directory.
.EXAMPLE
    .\scripts\download_checkpoints.ps1
#>

$ErrorActionPreference = "Stop"
$CHECKPOINT_DIR = "checkpoints"

Write-Host "=== Downloading Model Checkpoints ===" -ForegroundColor Cyan
Write-Host ""

# Ensure checkpoint directories exist
$dirs = @("sam2", "crm", "unique3d", "sampart3d")
foreach ($d in $dirs) {
    $path = Join-Path $CHECKPOINT_DIR $d
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

# ── SAM 2 ──
Write-Host "[1/4] Downloading SAM 2 (ViT-H)..." -ForegroundColor Yellow
$sam2_path = Join-Path $CHECKPOINT_DIR "sam2\sam2_hiera_large.pt"
if (-not (Test-Path $sam2_path)) {
    $sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large.pt"
    try {
        Write-Host "  Downloading from $sam2_url ..."
        Invoke-WebRequest -Uri $sam2_url -OutFile $sam2_path -UseBasicParsing
        Write-Host "  `u{2713} SAM 2 downloaded" -ForegroundColor Green
    } catch {
        Write-Host "  x SAM 2 download failed: $_" -ForegroundColor Red
        Write-Host "  Manual download: $sam2_url" -ForegroundColor DarkYellow
    }
} else {
    Write-Host "  `u{2713} SAM 2 already exists" -ForegroundColor Green
}

# ── CRM ──
Write-Host "[2/4] Downloading CRM checkpoints..." -ForegroundColor Yellow
$crm_model_dir = Join-Path $CHECKPOINT_DIR "crm\model"
if (-not (Test-Path $crm_model_dir)) {
    try {
        & huggingface-cli download Zhengyi/CRM --local-dir (Join-Path $CHECKPOINT_DIR "crm")
        Write-Host "  `u{2713} CRM downloaded" -ForegroundColor Green
    } catch {
        Write-Host "  ! CRM download failed — install huggingface-cli or download manually" -ForegroundColor DarkYellow
        Write-Host "    pip install huggingface-hub" -ForegroundColor DarkGray
        Write-Host "    huggingface-cli download Zhengyi/CRM --local-dir checkpoints\crm" -ForegroundColor DarkGray
    }
} else {
    Write-Host "  `u{2713} CRM already exists" -ForegroundColor Green
}

# ── Unique3D ──
Write-Host "[3/4] Downloading Unique3D checkpoints..." -ForegroundColor Yellow
$unique3d_model_dir = Join-Path $CHECKPOINT_DIR "unique3d\model"
if (-not (Test-Path $unique3d_model_dir)) {
    try {
        & huggingface-cli download aiuni/Unique3D --local-dir (Join-Path $CHECKPOINT_DIR "unique3d")
        Write-Host "  `u{2713} Unique3D downloaded" -ForegroundColor Green
    } catch {
        Write-Host "  ! Unique3D download failed — install huggingface-cli or download manually" -ForegroundColor DarkYellow
        Write-Host "    huggingface-cli download aiuni/Unique3D --local-dir checkpoints\unique3d" -ForegroundColor DarkGray
    }
} else {
    Write-Host "  `u{2713} Unique3D already exists" -ForegroundColor Green
}

# ── SAMPart3D ──
Write-Host "[4/4] SAMPart3D checkpoints..." -ForegroundColor Yellow
$sampart3d_model_dir = Join-Path $CHECKPOINT_DIR "sampart3d\model"
if (-not (Test-Path $sampart3d_model_dir)) {
    Write-Host "  ! SAMPart3D checkpoints must be downloaded manually." -ForegroundColor DarkYellow
    Write-Host "    Check: external\SAMPart3D\README.md for download instructions" -ForegroundColor DarkGray
    Write-Host "    Place files in: $CHECKPOINT_DIR\sampart3d\" -ForegroundColor DarkGray
} else {
    Write-Host "  `u{2713} SAMPart3D already exists" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Download Complete ===" -ForegroundColor Cyan
Write-Host "Verify with: python -c `"from pxr import Usd; import torch; print('OK')`""
