#!/bin/bash
# build_env_hyu.sh — base install (no GPU required at build time)
# User-specific copy of build_env.sh for hyu.
set -e

export CUDA=cu128
export TORCH_REL="2.10.0"
export TORCH_RELL="2.10"
export WARPCONV_REL="1.7.8"
export TORCH_CUDA_ARCH_LIST="8.9"
export UV_LINK_MODE=copy

USER=hyu
WARP_WHEEL="https://github.com/NVlabs/WarpConvNet/releases/download/v${WARPCONV_REL}/warpconvnet-${WARPCONV_REL}+torch${TORCH_RELL}${CUDA}-cp311-cp311-linux_x86_64.whl"
UV=/lbne/u/${USER}/.local/bin/uv

echo "Sourcing uv environment!"
source /gpfs01/lbne/users/fm/${USER}/uvenv/bin/activate

echo "Installing pytorch..."
${UV} pip install torch==${TORCH_REL} torchvision --index-url https://download.pytorch.org/whl/${CUDA}
${UV} pip install build ninja

echo "Installing torch-scatter..."
${UV} pip install torch-scatter --find-links https://data.pyg.org/whl/torch-${TORCH_REL}+${CUDA}.html

echo "Installing warpconvnet..."
${UV} pip install ${WARP_WHEEL}

echo "Installing other run dependencies..."
${UV} pip install fire h5py warp-lang

echo "Done!"
