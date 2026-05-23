#!/usr/bin/env bash
# Build two llama.cpp variants from the SAME source tree, with identical
# flags except for GGML_OPENMP, to isolate the effect of PR #88's single
# proposed change (`-DGGML_OPENMP=OFF` -> `-DGGML_OPENMP=ON`).
#
# Flag set is a verbatim copy of lemonade-sdk/llamacpp-rocm's gfx1151
# cmake invocation in .github/workflows/build-llamacpp-rocm.yml @ main.
# The only addition is `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13`
# in CMAKE_HIP_FLAGS, applied to BOTH builds (workaround for ROCm 7.1
# clang-20 picking gcc-14's runtime dir on Ubuntu 24.04 — runtime-neutral).
set -euo pipefail

cd "$(dirname "$0")/llama.cpp"

build_one() {
    local omp="$1"
    local out_dir="build-omp-${omp,,}"
    echo "════════════════════════════════════════════"
    echo "  Configuring  GGML_OPENMP=${omp}"
    echo "════════════════════════════════════════════"
    rm -rf "$out_dir"
    # --gcc-install-dir is needed on Ubuntu 24.04 (gcc-14 default lacks the
    # libstdc++ path the ROCm clang searches). Lemonade CI runs on Ubuntu
    # 22.04 where gcc-13 is default and resolves implicitly. Applied to all
    # three compilation languages, identically across both builds → cancels
    # in the A/B.
    local GCC_DIR_FLAG="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
    cmake -B "$out_dir" -G Ninja \
        -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
        -DCMAKE_C_FLAGS="$GCC_DIR_FLAG" \
        -DCMAKE_CXX_FLAGS="-I/opt/rocm/include $GCC_DIR_FLAG" \
        -DCMAKE_HIP_FLAGS="$GCC_DIR_FLAG" \
        -DCMAKE_CROSSCOMPILING=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGPU_TARGETS="gfx1151" \
        -DBUILD_SHARED_LIBS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_HIP=ON \
        -DGGML_OPENMP="${omp}" \
        -DGGML_CUDA_FORCE_CUBLAS=OFF \
        -DGGML_RPC=ON \
        -DGGML_HIP_ROCWMMA_FATTN=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_STATIC=OFF \
        -DCMAKE_SYSTEM_NAME=Linux
    echo "════════════════════════════════════════════"
    echo "  Building     GGML_OPENMP=${omp}"
    echo "════════════════════════════════════════════"
    cmake --build "$out_dir" -j "$(nproc)" --target llama-bench
}

build_one OFF
build_one ON
echo
echo "DONE — binaries:"
ls -la build-omp-off/bin/llama-bench build-omp-on/bin/llama-bench
