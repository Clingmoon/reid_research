#!/usr/bin/env bash
# Project-local CUDA toolkit switch (does not modify system driver/global config).
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "[CLIMB-ReID] CUDA_HOME=$CUDA_HOME"
nvcc -V | tail -n 1
