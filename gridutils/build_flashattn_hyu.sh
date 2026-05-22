#!/bin/bash
# build_flashattn_hyu.sh — build flash-attn from source on a GPU worker.
# Submitted via build_flashattn_hyu.sub. Expects torch already installed in venv.
set -e

export CUDA=cu128
export TORCH_CUDA_ARCH_LIST="8.9"
export UV_LINK_MODE=copy

USER=hyu
UV=/lbne/u/${USER}/.local/bin/uv

echo "Sourcing uv environment!"
source /gpfs01/lbne/users/fm/${USER}/uvenv/bin/activate

echo "Installing build deps..."
${UV} pip install wheel setuptools psutil

echo "Building flash_attn from source (this can take ~1.5h)..."
MAX_JOBS=1 ${UV} pip install flash-attn --no-build-isolation --no-cache-dir

echo "Done!"
