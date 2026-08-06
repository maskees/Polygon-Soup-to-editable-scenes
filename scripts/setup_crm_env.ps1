<#
.SYNOPSIS
    Set up the isolated CRM conda environment.
.DESCRIPTION
    CRM requires nvdiffrast (NVIDIA differentiable rendering) which must be
    compiled from source. This script creates a conda environment with
    PyTorch 2.6 + CUDA 12.4 (matching the system CUDA toolkit) and installs
    all CRM dependencies including nvdiffrast.

    Note: xformers is optional — CRM falls back to vanilla attention if unavailable.

    Run this ONCE before using the CRM backend.
.EXAMPLE
    .\scripts\setup_crm_env.ps1
.NOTES
    Prerequisites:
    - Conda (Miniconda or Anaconda) installed and on PATH
    - NVIDIA GPU with CUDA 12.x drivers
    - Visual Studio Build Tools with "Desktop development with C++" workload
    - CUDA Toolkit 12.x installed (nvcc must be on PATH)
    - ~10GB disk space for dependencies
#>

$ErrorActionPreference = "Stop"
$ENV_NAME = "crm"
$CRM_DIR = "external/CRM"

Write-Host "=== Setting Up CRM Environment ===" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Clone CRM repo if not present ──
if (-not (Test-Path $CRM_DIR)) {
    Write-Host "[1/6] Cloning CRM repository..." -ForegroundColor Yellow
    git clone https://github.com/thu-ml/CRM.git $CRM_DIR
    Write-Host "  ✓ CRM cloned" -ForegroundColor Green
} else {
    Write-Host "[1/6] CRM repo already exists at $CRM_DIR" -ForegroundColor Green
}

# ── Step 2: Create conda environment ──
Write-Host "[2/6] Creating conda environment '$ENV_NAME' (Python 3.9)..." -ForegroundColor Yellow

# Check if env already exists
$envExists = conda env list 2>&1 | Select-String -Pattern "^$ENV_NAME\s"
if ($envExists) {
    Write-Host "  Environment '$ENV_NAME' already exists. Updating..." -ForegroundColor DarkYellow
} else {
    conda create -n $ENV_NAME python=3.9 -y
    Write-Host "  ✓ Environment created" -ForegroundColor Green
}

# ── Step 3: Install PyTorch 2.6 + CUDA 12.4 ──
# Using PyTorch 2.6 with CUDA 12.4 to match the system's CUDA toolkit (12.1+).
# This avoids the CUDA version mismatch that occurs when nvdiffrast's C++
# extensions detect the system nvcc version vs PyTorch's compiled CUDA version.
Write-Host "[3/6] Installing PyTorch 2.6 + CUDA 12.4..." -ForegroundColor Yellow
conda run -n $ENV_NAME --no-capture-output pip install `
    torch==2.6.0 `
    torchvision==0.21.0 `
    torchaudio==2.6.0 `
    --index-url https://download.pytorch.org/whl/cu124

Write-Host "  ✓ PyTorch installed" -ForegroundColor Green

# ── Step 4: Install nvdiffrast (compiled from source) ──
# nvdiffrast requires compilation via nvcc. We pass -allow-unsupported-compiler
# because the system's MSVC version (19.51) is newer than what CUDA 12.x
# officially supports. This is the standard workaround and works reliably.
Write-Host "[4/6] Installing nvdiffrast (compiling from source)..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes..." -ForegroundColor DarkGray

$env:NVCC_FLAGS = "-allow-unsupported-compiler"
$env:TORCH_CUDA_ARCH_LIST = "8.9"  # RTX 4060

conda run -n $ENV_NAME --no-capture-output pip install ninja
conda run -n $ENV_NAME --no-capture-output pip install `
    "git+https://github.com/NVlabs/nvdiffrast" --no-build-isolation

Write-Host "  ✓ nvdiffrast installed" -ForegroundColor Green

# ── Step 5: Install CRM dependencies ──
Write-Host "[5/6] Installing CRM dependencies..." -ForegroundColor Yellow

# CRM's own requirements (minus torch which is already installed)
if (Test-Path "$CRM_DIR/requirements.txt") {
    conda run -n $ENV_NAME --no-capture-output pip install -r "$CRM_DIR/requirements.txt"
}

# Additional dependencies for our bridge script
conda run -n $ENV_NAME --no-capture-output pip install `
    huggingface-hub `
    rembg `
    Pillow `
    numpy `
    omegaconf `
    kiui `
    trimesh `
    xatlas `
    pymeshlab

Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# ── Step 6: Verify installation ──
Write-Host "[6/6] Verifying installation..." -ForegroundColor Yellow

$verifyScript = @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
try:
    import nvdiffrast.torch as dr
    print(f'Nvdiffrast: OK')
except ImportError as e:
    print(f'Nvdiffrast: FAILED - {e}')
try:
    import xformers
    print(f'xformers: OK')
except ImportError:
    print('xformers: NOT FOUND (optional - will use vanilla attention)')
print('CRM environment ready!')
"@

conda run -n $ENV_NAME --no-capture-output python -c $verifyScript

Write-Host ""
Write-Host "=== CRM Environment Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Download CRM checkpoints:" -ForegroundColor DarkGray
Write-Host "     huggingface-cli download Zhengyi/CRM --local-dir checkpoints/crm" -ForegroundColor DarkGray
Write-Host "  2. Run the pipeline with CRM backend:" -ForegroundColor DarkGray
Write-Host "     python main.py -i data/input/subject_001 --backend crm" -ForegroundColor DarkGray
