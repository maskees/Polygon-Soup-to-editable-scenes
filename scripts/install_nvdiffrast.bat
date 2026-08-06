@echo off
REM Install nvdiffrast in the CRM conda environment
REM This script sets the necessary environment variables for compilation

set NVCC_APPEND_FLAGS=-allow-unsupported-compiler
set TORCH_CUDA_ARCH_LIST=8.9
set MAX_JOBS=4

echo === Environment ===
echo NVCC_APPEND_FLAGS=%NVCC_APPEND_FLAGS%
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%

echo.
echo === Installing nvdiffrast ===
pip install ninja
pip install "git+https://github.com/NVlabs/nvdiffrast" --no-build-isolation

echo.
echo === Verifying ===
python -c "import nvdiffrast.torch as dr; print('nvdiffrast OK')"
