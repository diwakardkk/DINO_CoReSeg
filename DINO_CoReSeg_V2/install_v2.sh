#!/usr/bin/env bash
set -euo pipefail
cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPU allocation available to you}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert (3,10)<=sys.version_info<(3,14), "Use Python 3.10–3.13; 3.11/3.12 preferred"'
command -v nvidia-smi >/dev/null
nvidia-smi
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "torch==${TORCH_VERSION:-2.11.0}" "torchvision==${TORCHVISION_VERSION:-0.26.0}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG:-cu126}"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > requirements.server.lock.txt
.venv/bin/python scripts/download_weights.py
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print("PyTorch",torch.__version__,"CUDA",torch.version.cuda,"visible",torch.cuda.device_count()); x=torch.ones(16,16,device="cuda:0"); print("CUDA kernel check:",(x@x).sum().item())'
echo 'V2 environment installed independently. Next: CUDA_VISIBLE_DEVICES=5 ./run_v2.sh check'
