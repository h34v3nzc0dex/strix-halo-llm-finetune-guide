#!/usr/bin/env bash
# Build two llama.cpp variants from the SAME source tree, with our
# production CMake flag set (the one CLAUDE.md / the strix-halo guide
# documents) — identical except for GGML_HIP_ROCWMMA_FATTN.
#
# Tests the claim (from cezq + kant12 on r/StrixHalo, plus strixhalo.wiki):
#   "rocwmma FA is slower than runtime FA on gfx1151 Strix Halo"
#
# Source: commit 1acee6bf8 (the exact ggerganov/llama.cpp commit that
# lemonade-sdk b1276 builds — gives us a consistent reference point with
# our prior lemonade tests).
#
# All other production flags held constant:
#   - GGML_HIP_GRAPHS=ON, GGML_HIP_MMQ_MFMA=ON, GGML_HIP_NO_VMM=ON
#   - --gcc-install-dir applied to C/C++/HIP (Ubuntu 24.04 toolchain
#     workaround; identical in both builds, cancels in the A/B).
set -euo pipefail
cd "$(dirname "$0")/llama.cpp"

build_one() {
    local fattn="$1"
    local out_dir="build-fattn-${fattn,,}"
    echo "════════════════════════════════════════════"
    echo "  Configuring  GGML_HIP_ROCWMMA_FATTN=${fattn}"
    echo "════════════════════════════════════════════"
    rm -rf "$out_dir"
    local GCC_DIR_FLAG="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
    cmake -B "$out_dir" -G Ninja \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_C_FLAGS="$GCC_DIR_FLAG" \
        -DCMAKE_CXX_FLAGS="-I/opt/rocm/include $GCC_DIR_FLAG" \
        -DCMAKE_HIP_FLAGS="$GCC_DIR_FLAG" \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMDGPU_TARGETS=gfx1151 \
        -DGPU_TARGETS=gfx1151 \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_HIP=ON \
        -DGGML_HIP_ROCWMMA_FATTN="${fattn}" \
        -DGGML_HIP_GRAPHS=ON \
        -DGGML_HIP_MMQ_MFMA=ON \
        -DGGML_HIP_NO_VMM=ON \
        -DGGML_CUDA_FORCE_CUBLAS=OFF \
        -DGGML_RPC=ON \
        -DGGML_NATIVE=OFF \
        -DCMAKE_SYSTEM_NAME=Linux
    echo "════════════════════════════════════════════"
    echo "  Building     GGML_HIP_ROCWMMA_FATTN=${fattn}"
    echo "════════════════════════════════════════════"
    cmake --build "$out_dir" -j "$(nproc)" --target llama-bench
}

build_one ON
build_one OFF
echo
echo "DONE — binaries:"
ls -la build-fattn-on/bin/llama-bench build-fattn-off/bin/llama-bench
