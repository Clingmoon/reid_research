#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
VMAMBA_DIR="${VMAMBA_DIR:-$ROOT_DIR/VMamba}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$ROOT_DIR/config/climb-vit-market.yml}"
ENABLE_SELECTIVE_SCAN_BUILD="${ENABLE_SELECTIVE_SCAN_BUILD:-0}"

log() { printf '[env-init] %s\n' "$*"; }

run_clone() {
  local cmd="$1"
  if bash -ilc 'type proxy >/dev/null 2>&1'; then
    bash -ilc "proxy ${cmd}"
  else
    bash -ilc "${cmd}"
  fi
}

ensure_tool() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required tool: $1" >&2; exit 1; }
}

ensure_tool uv
ensure_tool git
ensure_tool bash

log "Creating virtual environment: $VENV_DIR (python $PYTHON_VERSION)"
uv venv --clear --python "$PYTHON_VERSION" "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

log "Installing build basics"
uv pip install packaging wheel setuptools==69.5.1 ninja

log "Installing PyTorch (cu118)"
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.1.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.1.1+cu118

log "Installing project requirements with local compatibility filter"
# Current requirements include packages that cannot build with local CUDA toolkit 10.1.
# Keep environment trainable by skipping those two build-time CUDA extension packages.
rg -v '^(causal-conv1d|mamba-ssm|fsspec|torch==|torchaudio==|torchvision==)' "$ROOT_DIR/requirements.txt" > /tmp/climb.requirements.compat.txt
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  -r /tmp/climb.requirements.compat.txt

if [ ! -d "$VMAMBA_DIR/.git" ]; then
  log "Cloning VMamba into $VMAMBA_DIR"
  run_clone "git clone https://github.com/MzeroMiko/VMamba.git '$VMAMBA_DIR'"
else
  log "VMamba already exists: $VMAMBA_DIR"
fi

log "Installing VMamba requirements"
uv pip install --index-strategy unsafe-best-match -r "$VMAMBA_DIR/requirements.txt"

if [ "$ENABLE_SELECTIVE_SCAN_BUILD" = "1" ]; then
  log "Building selective_scan (optional)"
  (
    cd "$VMAMBA_DIR/kernels/selective_scan"
    uv pip install --no-build-isolation .
  )
else
  log "Skipping selective_scan build (set ENABLE_SELECTIVE_SCAN_BUILD=1 to force build)"
fi

log "Verifying runtime imports"
python - <<'PY'
import torch
import timm
import yacs
import transformers
import mamba.mamba_ssm
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('import_ok', 'mamba.mamba_ssm')
PY

log "Environment initialization completed"
log "Activate env: source $VENV_DIR/bin/activate"
log "Start training with real data example:"
log "CUDA_VISIBLE_DEVICES=0 python train_climb.py --config_file ./config/climb-vit-market.yml DATASETS.ROOT_DIR \"('/path/to/reid_root')\" OUTPUT_DIR \"'./logs'\""
