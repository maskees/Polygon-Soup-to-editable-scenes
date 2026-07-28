<#
.SYNOPSIS
    Set up the isolated CRM conda environment.
.DESCRIPTION
    CRM requires PyTorch 1.13 + CUDA 11.7, which is incompatible with the
    main project environment (PyTorch 2.3 + CUDA 12.1). This script creates
    a separate conda environment named "crm" with all CRM dependencies.

    Run this ONCE before using the CRM backend.
.EXAMPLE
    .\scripts\setup_crm_env.ps1
.NOTES
    Prerequisites:
    - Conda (Miniconda or Anaconda) installed and on PATH
    - NVIDIA GPU with CUDA 11.7+ drivers
    - ~10GB disk space for dependencies
#>

$ErrorActionPreference = "Stop"
$ENV_NAME = "crm"
$CRM_DIR = "external/CRM"

Write-Host "=== Setting Up CRM Environment ===" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Clone CRM repo if not present ──
if (-not (Test-Path $CRM_DIR)) {
    Write-Host "[1/5] Cloning CRM repository..." -ForegroundColor Yellow
    git clone https://github.com/thu-ml/CRM.git $CRM_DIR
    Write-Host "  ✓ CRM cloned" -ForegroundColor Green
} else {
    Write-Host "[1/5] CRM repo already exists at $CRM_DIR" -ForegroundColor Green
}

# ── Step 2: Create conda environment ──
Write-Host "[2/5] Creating conda environment '$ENV_NAME' (Python 3.9)..." -ForegroundColor Yellow

# Check if env already exists
$envExists = conda env list 2>&1 | Select-String -Pattern "^$ENV_NAME\s"
if ($envExists) {
    Write-Host "  Environment '$ENV_NAME' already exists. Updating..." -ForegroundColor DarkYellow
} else {
    conda create -n $ENV_NAME python=3.9 -y
    Write-Host "  ✓ Environment created" -ForegroundColor Green
}

# ── Step 3: Install PyTorch 1.13 + CUDA 11.7 ──
Write-Host "[3/5] Installing PyTorch 1.13 + CUDA 11.7..." -ForegroundColor Yellow
conda run -n $ENV_NAME --no-capture-output pip install `
    torch==1.13.0+cu117 `
    torchvision==0.14.0+cu117 `
    torchaudio==0.13.0 `
    --extra-index-url https://download.pytorch.org/whl/cu117

Write-Host "  ✓ PyTorch installed" -ForegroundColor Green

# ── Step 4: Install CRM dependencies ──
Write-Host "[4/5] Installing CRM dependencies..." -ForegroundColor Yellow

# Core dependencies
conda run -n $ENV_NAME --no-capture-output pip install `
    torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

# Kaolin (NVIDIA 3D library)
conda run -n $ENV_NAME --no-capture-output pip install `
    kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu117.html

# Nvdiffrast
conda run -n $ENV_NAME --no-capture-output pip install `
    "git+https://github.com/NVlabs/nvdiffrast"

# xformers
conda run -n $ENV_NAME --no-capture-output pip install ninja
conda run -n $ENV_NAME --no-capture-output pip install `
    -v -U "git+https://github.com/facebookresearch/xformers.git@main#egg=xformers"

# CRM's own requirements
if (Test-Path "$CRM_DIR/requirements.txt") {
    conda run -n $ENV_NAME --no-capture-output pip install -r "$CRM_DIR/requirements.txt"
}

# Additional dependencies for our bridge script
conda run -n $ENV_NAME --no-capture-output pip install `
    huggingface-hub `
    rembg `
    Pillow `
    numpy `
    omegaconf

Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# ── Step 5: Verify installation ──
Write-Host "[5/5] Verifying installation..." -ForegroundColor Yellow

$verifyScript = @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
try:
    import kaolin
    print(f'Kaolin: OK')
except ImportError:
    print('Kaolin: NOT FOUND')
try:
    import nvdiffrast
    print(f'Nvdiffrast: OK')
except ImportError:
    print('Nvdiffrast: NOT FOUND')
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
