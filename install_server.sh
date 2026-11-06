#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_TAG="${CUDA_TAG:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
"$PYTHON_BIN" -c 'import sys; assert (3,10)<=sys.version_info<(3,14), "Use Python 3.10–3.13; Python 3.11 or 3.12 is recommended for this pinned server environment"'
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is unavailable. Verify NVIDIA drivers / GPU allocation first.'; exit 1; }
nvidia-smi
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > requirements.server.lock.txt
.venv/bin/python scripts/download_weights.py
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "Installed PyTorch cannot access CUDA"; x=torch.ones(16,16,device="cuda"); print("CUDA matrix test:",(x@x).sum().item()); print("GPUs:",[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
echo 'Installation complete. Run ./run_server.sh check, then ./run_server.sh start.'
