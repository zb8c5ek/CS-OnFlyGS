# Local Installation Guide (Windows + RTX 5090)

## Prerequisites

- **GPU**: NVIDIA RTX 5090 (Blackwell, sm_120)
- **CUDA Toolkit**: 12.8+
- **Python**: 3.11
- **VS Build Tools**: 2022 (CUDA 12.8 does **not** support VS 2025)

Download VS 2022 Build Tools: https://aka.ms/vs/17/release/vs_BuildTools.exe  
Install with **"Desktop development with C++"** workload.

## 1. Install PyTorch + Build Dependencies

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install hatchling setuptools wheel ninja
```

## 2. Patch PyTorch Headers

PyTorch 2.10 has two header bugs on Windows that break CUDA extension compilation.

Set the torch include path first (adjust if your environment path differs):

```powershell
$TORCH_INC = "$(python -c 'import torch; print(torch.utils.cpp_extension.include_paths()[0])')"
# Or explicitly: $TORCH_INC = "D:\MICROMAMBA\envs\sfm3r\Lib\site-packages\torch\include"
```

**Bug 1** — `CUDACachingAllocator.h`: Windows `#define small char` corrupts a parameter name.

```powershell
(Get-Content "$TORCH_INC\c10\cuda\CUDACachingAllocator.h" -Raw) `
  -replace 'StreamSegmentSize\(cudaStream_t s, bool small, size_t sz\)\s*\r?\n\s*: stream\(s\), is_small_pool\(small\)', `
  "StreamSegmentSize(cudaStream_t s, bool is_small, size_t sz)`n      : stream(s), is_small_pool(is_small)" `
  | Set-Content "$TORCH_INC\c10\cuda\CUDACachingAllocator.h" -NoNewline
```

**Bug 2** — `compiled_autograd.h`: `std` namespace ambiguity when compiling CUDA extensions.

```powershell
(Get-Content "$TORCH_INC\torch\csrc\dynamo\compiled_autograd.h" -Raw) `
  -replace '#if defined\(_WIN32\) && \(defined\(USE_CUDA\) \|\| defined\(USE_ROCM\)\)', `
  '#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM) || defined(__CUDACC__))' `
  | Set-Content "$TORCH_INC\torch\csrc\dynamo\compiled_autograd.h" -NoNewline
```

> **Note**: These patches are lost when PyTorch is reinstalled.

## 3. Activate VS 2022 & Build

```powershell
# Activate VS 2022 compiler environment
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd.exe /c "`"$vsPath`" && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}

# Verify
where.exe cl  # should show MSVC 14.4x path

# Build
$env:DISTUTILS_USE_SDK=1
pip install -r requirements.txt --no-build-isolation
```

## Code Changes for sm_120 Support

The following `setup.py` files were modified to support RTX 5090:

- `submodules/diff-gaussian-rasterization/setup.py` — added `-allow-unsupported-compiler` to nvcc args
- `fused_ssim/setup.py` — added `-allow-unsupported-compiler` to nvcc args + `compute_120/sm_120` to fallback architectures (integrated from submodule into codebase)
- `submodules/simple-knn/setup.py` — added `-allow-unsupported-compiler` to nvcc args
