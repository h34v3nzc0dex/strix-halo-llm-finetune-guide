#!/usr/bin/env bash
# Bench ROCWMMA_FATTN ON vs OFF on real gfx1151 — both binaries built
# from the same source (1acee6bf8), only differing on the FATTN flag.
#
# Two models:
#   Qwen3.5-27B Q8_0     — dense (our reference workload, matches strix-halo
#                          guide existing benchmarks)
#   Qwen3.6-35B-A3B Q4_K_M — MoE, A=3B active params (daily-driver shape
#                            kant12 + Potential-Leg-639 use)
#
# Bench matrix per (binary, model):
#   Run 1 (short prompts):  pp512 + tg128  @  -fa 0,1
#   Run 2 (long context):   pp2048 + tg128 @  -fa 0,1  -d 0,4196,8392
#
# -fa 0 is the "no FA dispatched" baseline — both binaries should match
# (sanity check). -fa 1 is where the FATTN flag actually changes the
# dispatched kernel; the OFF vs ON delta here is what we're measuring.
#
# Nightly HSA overlay sidesteps the ROCm 7.1.0 libhsa null-ptr bug on
# gfx1151 (orthogonal to FATTN).
set -uo pipefail

TEST_DIR=/srv/aurora-ai/rocwmma-fattn-test
NIGHTLY_LIB=/srv/aurora-ai/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib
ROCM_LIB=/opt/rocm/lib

declare -A MODELS=(
    [qwen35-27b-q8]="/srv/aurora-ai/aurora-effects-qwen35-27b/aurora-effects-v8-q8_0.gguf"
    [qwen36-35b-a3b-q4]="/srv/aurora-ai/qwen36-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
)

bench_combo() {
    local variant="$1"
    local model_key="$2"
    local model_path="$3"
    local bin="$TEST_DIR/llama.cpp/build-fattn-${variant,,}/bin/llama-bench"
    local bin_dir="$(dirname "$bin")"
    local out="$TEST_DIR/bench-${model_key}-fattn-${variant,,}.log"

    echo "════════════════════════════════════════════════════════════"
    echo "  ${model_key}  FATTN=${variant}    ->  $(basename $out)"
    echo "════════════════════════════════════════════════════════════"

    # Run 1: short prompts, fa 0/1
    LD_LIBRARY_PATH="$NIGHTLY_LIB:$ROCM_LIB:$bin_dir" \
    "$bin" -m "$model_path" -p 512 -n 128 -r 3 --mmap 0 -ngl 99 -fa 0,1 2>&1 | tee "$out"
    echo "" >> "$out"
    # Run 2: long context, fa 0/1, depth sweep
    LD_LIBRARY_PATH="$NIGHTLY_LIB:$ROCM_LIB:$bin_dir" \
    "$bin" -m "$model_path" -p 2048 -n 128 -r 3 --mmap 0 -ngl 99 -fa 0,1 -d 0,4196,8392 2>&1 | tee -a "$out"
    echo
}

for model_key in qwen35-27b-q8 qwen36-35b-a3b-q4; do
    bench_combo ON  "$model_key" "${MODELS[$model_key]}"
    bench_combo OFF "$model_key" "${MODELS[$model_key]}"
done

echo "DONE — logs:"
ls -la "$TEST_DIR"/bench-*-fattn-*.log
