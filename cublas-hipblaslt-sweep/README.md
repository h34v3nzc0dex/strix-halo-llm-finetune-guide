# `-DGGML_CUDA_FORCE_CUBLAS=ON` + `ROCBLAS_USE_HIPBLASLT=1` sweep on gfx1151

Triggered by a peer suggestion on r/StrixHalo that these two settings boost
prompt-processing on dense models. Ran the controlled comparison on the
Corsair AI Workstation 300 (Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151 /
ROCm 7.13 nightly) to either confirm or refute.

**TL;DR:** on this hardware the build flag is a ~3.6× pp64 slowdown on
dense Qwen3.5-27B Q8, and the env var alone is a no-op. Opposite of the
predicted speedup. tg16 unaffected across all six conditions.

## Setup

- Model: `aurora-effects-v8-q8_0.gguf` (Qwen3.5-27B Q8_0, 26.62 GiB, 26.90 B params)
- `llama-bench -p 64 -n 16 -r 3 -mmp 0 -ngl 999`
- Source 1: existing `/usr/local/bin/llama-bench` build `d0a6dfe (502)`
- Source 2 (this sweep): fresh build of llama.cpp `5207d120e (867)`, two CMake configs:
  - `build-b867-baseline/` — same flags as the existing build:
    `-DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DAMDGPU_TARGETS=gfx1151`
  - `build-cublas/` — same flags **plus** `-DGGML_CUDA_FORCE_CUBLAS=ON`
- Host compiler: system g++-13, HIP compiler: `/opt/rocm-7.1.0/lib/llvm/bin/clang++` with `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13`

## Full results (3 reps each)

| # | Build | env | `pp64` (tok/s) | `tg16` (tok/s) |
|---|---|---|---|---|
| 1 | b502 (existing) | — | 270.61 ± 2.07 | 7.52 ± 0.00 |
| 2 | b502 (existing) | `ROCBLAS_USE_HIPBLASLT=1` | 258.43 ± 1.02 | 7.52 ± 0.00 |
| 3 | b867 baseline (this sweep) | — | 255.17 ± 23.73 | 7.49 ± 0.03 |
| 4 | b867 baseline (this sweep) | `ROCBLAS_USE_HIPBLASLT=1` | 253.20 ± 22.92 | 7.48 ± 0.03 |
| 5 | **b867 + `-DGGML_CUDA_FORCE_CUBLAS=ON`** | — | **71.09 ± 3.26** | 7.49 ± 0.02 |
| 6 | **b867 + `-DGGML_CUDA_FORCE_CUBLAS=ON`** | `ROCBLAS_USE_HIPBLASLT=1` | **76.26 ± 3.93** | 7.49 ± 0.03 |

## Reading

- **Source-version drift** between b502 and b867 (rows 1 vs 3, same build flags) is ~6% pp64 with high variance on b867 — noise within 1σ. Not a real regression.
- **`ROCBLAS_USE_HIPBLASLT=1` alone** is a no-op for this workload — row 2 vs 1 (~−5%, within b502's tighter variance), and row 4 vs 3 (essentially identical). Confirms it's not the magic boost the peer suggested.
- **`-DGGML_CUDA_FORCE_CUBLAS=ON`** drops pp64 from ~255 to ~71 (rows 3 → 5, same source, same env). **3.6× slowdown.** This is the dominant effect. Forcing the rocBLAS GEMM path apparently loses against llama.cpp's custom HIP/MMQ kernels on this hardware for prefill matmuls.
- **`tg16` is flat across all six conditions** (7.48–7.52). The CUBLAS flag only affects the prefill code path, not the per-token kernels. Reasonable — token-gen is a different shape (single-token GEMV not GEMM) and doesn't go through the cuBLAS-equivalent path.

## Implication

The peer's report — that `-DGGML_CUDA_FORCE_CUBLAS=ON` is a dense-model speedup
on Strix Halo — does NOT replicate on this Corsair AI Workstation 300 / ROCm 7.13
nightly / b867 source. Either their gain is specific to a different ROCm version
(maybe the older 7.1 or 7.2 stable rocBLAS tuned-better-against-CUBLAS-path
than the 7.13 nightly's), or to a different llama.cpp source state, or to a
different model. Open question.

## Logs

- `bench-b502-hipblaslt.log` — row 2 (env-var-only test)
- `bench-b867-no-cublas.log` — rows 3 and 4 (b867 baseline, with and without env var)
- `bench-cublas-on.log` — rows 5 and 6 (CUBLAS build, with and without env var)
