# `GGML_HIP_ROCWMMA_FATTN` — ON vs OFF on gfx1151 (Strix Halo / Radeon 8060S)

## TL;DR

**Turn it OFF.** On gfx1151 at any non-trivial prompt or context depth, the rocwmma flash-attention implementation is dramatically slower than llama.cpp's runtime FA. At a typical 8k context the gap reaches **~2.4× on prefill** for both dense and MoE models. Decode (TG) is flat — TG is memory-bandwidth-bound, the FA implementation doesn't matter. PSA from the [strix-halo guide](../README.md): **build with `-DGGML_HIP_ROCWMMA_FATTN=OFF`**. The strixhalo.wiki [ROCWMMA recommendation](https://strixhalo.wiki/AI/llamacpp-with-ROCm#rocwmma) is correct; this is the supporting hardware evidence.

## Test rig

- Radeon 8060S / gfx1151 / Ryzen AI MAX+ 395 / 128 GiB unified
- Ubuntu 24.04, ROCm 7.1.0 stable + nightly `libhsa-runtime64.so.1` overlay
- llama.cpp commit `1acee6bf8` (the exact commit lemonade `b1276` ships — matches our other tests for consistency)

## Methodology

Two binaries built from the same source tree with our **production** CMake flag set (the one the strix-halo guide documents in Step 6), identical except for `-DGGML_HIP_ROCWMMA_FATTN`:

```
-DGGML_HIP=ON  -DGGML_HIP_GRAPHS=ON  -DGGML_HIP_MMQ_MFMA=ON  -DGGML_HIP_NO_VMM=ON
-DAMDGPU_TARGETS=gfx1151
-DCMAKE_HIP_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13   # Ubuntu 24.04 toolchain workaround
-DGGML_HIP_ROCWMMA_FATTN=ON      ← only this flag differs
-DGGML_HIP_ROCWMMA_FATTN=OFF     ←
```

CMakeCache differential verified — `GGML_HIP_ROCWMMA_FATTN:BOOL=ON` vs `OFF` in the two builds. Lib sizes differ (71.7 MB vs 72.6 MB; the runtime FA kernels are slightly larger than the rocwmma ones). Single differing factor.

Both models benched with the same `llama-bench` matrix:

```bash
./llama-bench -m <gguf> -p 512 -n 128 -r 3 --mmap 0 -ngl 99 -fa 0,1
./llama-bench -m <gguf> -p 2048 -n 128 -r 3 --mmap 0 -ngl 99 -fa 0,1 -d 0,4196,8392
```

- `-fa 0` (FA disabled) is the sanity check — should be identical between the two binaries because no FA dispatch happens. It is.
- `-fa 1` is where the FATTN build flag actually changes the dispatched kernel.

## Results

### Sanity check — `-fa 0` (no FA dispatched) rows match within noise

Both binaries return effectively identical numbers when FA is disabled at runtime. Confirms the build flag is doing what we think (changes which FA kernel is dispatched, not anything else).

| shape | FATTN=ON, -fa 0 | FATTN=OFF, -fa 0 |
|---|---|---|
| qwen3.5-27B Q8 pp2048 d=0 | 325.47 ± 0.70 | 325.75 ± 0.43 |
| qwen3.5-27B Q8 pp2048 d=8392 | 261.75 ± 0.18 | 258.38 ± 1.69 |
| qwen3.6-A3B Q4 pp2048 d=0 | 988.33 ± 7.57 | 997.11 ± 8.12 |
| qwen3.6-A3B Q4 pp2048 d=8392 | 750.44 ± 1.65 | 753.86 ± 0.68 |

### Qwen3.5-27B Q8 dense — `-fa 1` ON vs OFF

| shape | FATTN=ON | FATTN=OFF | Δ |
|---|---|---|---|
| pp512 d=0 | 313.54 ± 3.20 | 317.24 ± 1.88 | +1.2% |
| pp2048 d=0 | 283.90 ± 0.13 | **331.86 ± 0.06** | **+16.9%** |
| pp2048 d=4196 | 167.61 ± 0.04 | **306.83 ± 0.45** | **+83.0%** |
| pp2048 d=8392 | 117.08 ± 0.09 | **282.52 ± 0.52** | **+141.3%** |
| tg128 d=0 | 7.64 ± 0.00 | 7.65 ± 0.00 | flat |
| tg128 d=4196 | 7.47 ± 0.01 | 7.58 ± 0.01 | +1.5% |
| tg128 d=8392 | 7.30 ± 0.00 | 7.49 ± 0.00 | +2.6% |

### Qwen3.6-35B-A3B Q4 MoE — `-fa 1` ON vs OFF

| shape | FATTN=ON | FATTN=OFF | Δ |
|---|---|---|---|
| pp512 d=0 | 934.00 ± 7.88 | 1014.32 ± 9.65 | +8.6% |
| pp2048 d=0 | 813.71 ± 2.37 | **983.86 ± 2.91** | **+20.9%** |
| pp2048 d=4196 | 467.28 ± 0.38 | **881.86 ± 0.64** | **+88.7%** |
| pp2048 d=8392 | 332.32 ± 0.72 | **815.70 ± 0.57** | **+145.4%** |
| tg128 d=0 | 49.43 ± 0.08 | 49.70 ± 0.09 | +0.5% |
| tg128 d=4196 | 46.85 ± 0.22 | 48.64 ± 0.24 | +3.8% |
| tg128 d=8392 | 44.08 ± 0.01 | 46.73 ± 0.02 | +6.0% |

## Interpretation

Two architectures, same pattern:

- **TG (decode) is flat** — token generation is memory-bandwidth-bound on this hardware (the GPU pool is the unified system RAM), so the FA kernel choice doesn't move the needle.
- **PP (prefill) is dramatically slower with rocwmma**, and the gap **widens monotonically with context depth**:
  - d=0 (no context): rocwmma is 1.2-21% slower
  - d=4196: rocwmma is 83-89% slower
  - d=8392: rocwmma is 141-145% slower

In other words, every doubling of context roughly doubles the runtime FA advantage. By 8k context, runtime FA prefills at ~2.4× the speed of rocwmma FA — on dense Qwen3.5 and on MoE Qwen3.6-A3B alike.

The hypothesis: rocwmma's FA implementation on gfx1151 either falls back to a non-WMMA path at long sequences, or the WMMA dispatch pattern has bad scaling characteristics for the RDNA 3.5 wave32 layout. The runtime FA path is simpler but apparently better-tuned for this hardware shape. (We didn't profile the actual kernel paths; the numbers alone are conclusive on the practical question.)

## What changes in the guide

Step 6 of the [main README](../README.md) previously recommended `-DGGML_HIP_ROCWMMA_FATTN=ON` as the official AMD pick for RDNA 3.5. The recommendation is reversed based on this data. For any workload using non-trivial context (eval, batched inference, long-prompt RAG, multi-turn chat), `OFF` is strictly faster. For tiny prompts (pp512 d=0) the delta is small; OFF is still the safe pick.

## Files

| File | What it is |
|---|---|
| `build.sh` | Builds both variants from `1acee6bf8` with the strix-halo guide's production CMake flag set, only `GGML_HIP_ROCWMMA_FATTN` differs |
| `bench.sh` | Runs the `llama-bench` matrix above against both binaries and both models |
| `bench-qwen35-27b-q8-fattn-on.log`    | raw llama-bench output, dense Q8, FATTN=ON |
| `bench-qwen35-27b-q8-fattn-off.log`   | raw, dense Q8, FATTN=OFF |
| `bench-qwen36-35b-a3b-q4-fattn-on.log`| raw, MoE A3B Q4, FATTN=ON |
| `bench-qwen36-35b-a3b-q4-fattn-off.log` | raw, MoE A3B Q4, FATTN=OFF |
