#!/usr/bin/env bash
# Bench both PR #88 variants identically — matches author's exact
# bench command, only swapping the binary between OpenMP-OFF (baseline,
# lemonade main) and OpenMP-ON (PR #88).
#
# Isolated HF cache so the 55 GB download stays in this dir.
# Nightly HSA overlay so we sidestep the ROCm 7.1.0 libhsa null-ptr
# bug on gfx1151 (orthogonal to the OpenMP question we're testing).
set -uo pipefail

TEST_DIR=/srv/aurora-ai/lemonade-pr88-test
NIGHTLY_LIB=/srv/aurora-ai/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib
ROCM_LIB=/opt/rocm/lib
export HF_HOME="$TEST_DIR/.hf-cache"
mkdir -p "$HF_HOME"

bench_one() {
    local variant="$1"
    local bin="$TEST_DIR/llama.cpp/build-omp-${variant,,}/bin/llama-bench"
    local bin_dir="$(dirname "$bin")"
    local out="$TEST_DIR/bench-omp-${variant,,}.log"

    echo "════════════════════════════════════════════════════════════"
    echo "  Bench  GGML_OPENMP=${variant}    ->  $out"
    echo "════════════════════════════════════════════════════════════"

    GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
    LD_LIBRARY_PATH="$NIGHTLY_LIB:$ROCM_LIB:$bin_dir" \
    "$bin" \
        -hf unsloth/Qwen3-Coder-Next-GGUF:UD-Q5_K_XL \
        -fa 0,1 \
        --mmap 0,1 \
        -ngl 99 \
        2>&1 | tee "$out"
    echo
}

# OFF first so the model download (cached in $HF_HOME) is a one-time cost.
bench_one OFF
bench_one ON
echo "DONE — logs:"
ls -la "$TEST_DIR"/bench-omp-*.log
