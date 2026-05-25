#!/usr/bin/env bash
# Build llama.cpp with Vulkan backend for Strix Halo (RADV GFX1151).
# Independent of /srv/aurora-ai/llama.cpp/ (ROCm production build).
#
# Prereqs (apt):
#   vulkan-tools libvulkan-dev glslang-tools glslang-dev
#   spirv-headers spirv-tools glslc
#   mesa-vulkan-drivers (Mesa 25.x+ for RADV STRIX_HALO support)
set -euo pipefail

COMMIT=${1:-b9296}
DEST=${2:-/srv/aurora-ai/llama.cpp-vulkan}

rm -rf "$DEST"
git clone --depth 1 --branch "$COMMIT" https://github.com/ggml-org/llama.cpp "$DEST"
cd "$DEST"

cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON \
  -DGGML_NATIVE=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build --parallel "$(nproc)" --target llama-server llama-bench llama-cli

echo
echo "Built. Verify Vulkan sees the GPU:"
echo "  $DEST/build/bin/llama-bench --list-devices"
