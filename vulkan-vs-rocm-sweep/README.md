# ROCm vs Vulkan on gfx1151 — backend choice depends on precision

> **Long-form writeup:** [ROCm vs Vulkan on AMD Strix Halo: when each wins, and why it inverts at the precision boundary](../articles/2026-05-rocm-vs-vulkan-strix-halo-precision-inversion.md) — same data, more context, written for someone landing here cold.

## TL;DR

On Strix Halo (Radeon 8060S / gfx1151 / RADV STRIX_HALO via Mesa 25.2.8), tested on Qwen3.6-35B-A3B:

- **Quantized (Q4_K_M, UD)** — Vulkan wins decode by ~22% (60.4 vs 49.6 tok/s tg128), prefill ~7% ROCm-favored
- **Full precision (BF16)** — **ROCm wins decode by ~100%** (21.5 vs 10.7 tok/s tg128), prefill ~60% ROCm-favored

The reason is visible right in Vulkan's own capability report:

```
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1151) (radv) | uma: 1 | fp16: 1 | bf16: 0 | ...
                                                                    ^^^^^^^
                                                                no native BF16
```

`bf16: 0` — RADV STRIX_HALO has no native BF16 GEMM path on Mesa 25.2.8; the backend falls back to slower kernels (FP32 emulation most likely). ROCm/HIP via `libggml-hip.so` has BF16 wired through CDNA/RDNA-aware HIP kernels and crushes Vulkan on anything BF16-bound.

**Practical guidance:**

| Workload | Backend |
|---|---|
| Quantized inference (Q4 / Q5 / Q6 / Q8) | **Vulkan** (Mesa RADV) |
| BF16 inference | **ROCm/HIP** (llama.cpp built with `-DGGML_HIP=ON`) |
| Training (always BF16/FP32) | **ROCm/HIP** (only path with the PyTorch nightly stack) |
| Mixed | Whichever your hot path is |

This isn't a one-way win for either backend — it's precision-dependent.

## Test rig

- Radeon 8060S / gfx1151 / Ryzen AI MAX+ 395 / 128 GiB unified
- Ubuntu 24.04, kernel 6.19.14 mainline
- ROCm 7.1.0 system + nightly `libhsa-runtime64.so.1` overlay (for the HIP path)
- Mesa 25.2.8 (RADV STRIX_HALO support built-in; no PPA needed)
- llama.cpp commit `a497476` (b9296) for both backend builds — identical source

## Methodology

Two llama.cpp binaries built from the same source commit:

- **ROCm/HIP** — `/srv/aurora-ai/llama.cpp/` — production build, flags per the [main guide Step 6](../README.md#step-6--llamacpp-hip-build-for-inference): `GGML_HIP=ON`, `ROCWMMA_FATTN=OFF`, `MMQ_MFMA=ON`, `NO_VMM=ON`, `GRAPHS=ON`, gfx1151
- **Vulkan** — separate build at `/srv/aurora-ai/llama.cpp-vulkan/` — `GGML_VULKAN=ON`, no HIP. Build recipe in [`build-vulkan.sh`](build-vulkan.sh)

Same `llama-bench` shapes against both:
```
-p 512 -n 128 -r 3 --mmap 0 -ngl 99 -fa 0,1
-p 2048 -n 128 -r 3 --mmap 0 -ngl 99 -fa 1 -d 0,4196,8392
```

ROCm path uses nightly HSA overlay (`LD_LIBRARY_PATH=$NIGHTLY_LIB:/opt/rocm-7.1.0/lib:…`) — required to dodge the ROCm 7.1.0 `libhsa-runtime64.so` null-ptr bug on gfx1151. Vulkan needs no overlay; Mesa is the driver.

## Results — Qwen3.6-35B-A3B Q4_K_M (UD)

The Q4 ROCm rows come from [`../rocwmma-fattn-sweep/bench-qwen36-35b-a3b-q4-fattn-off.log`](../rocwmma-fattn-sweep/bench-qwen36-35b-a3b-q4-fattn-off.log) (yesterday's sweep). Vulkan rows in this dir.

| shape | ROCm/HIP | Vulkan | Δ |
|---|---|---|---|
| pp512 fa=0 | 1023.75 ± 8.66 | 953.79 ± 17.18 | -6.8% (Vulkan) |
| pp512 fa=1 | 1014.32 ± 9.65 | 942.18 ± 4.92 | -7.1% (Vulkan) |
| tg128 fa=1 d=0 | 49.58 ± 0.09 | **60.39 ± 0.22** | **+21.8% (Vulkan)** |
| tg128 fa=1 d=4196 | 48.64 ± 0.24 | **58.24 ± 0.09** | **+19.7% (Vulkan)** |
| tg128 fa=1 d=8392 | 46.73 ± 0.02 | **57.13 ± 0.09** | **+22.3% (Vulkan)** |
| pp2048 fa=1 d=0 | 983.86 ± 2.91 | 921.04 ± 2.97 | -6.4% (Vulkan) |
| pp2048 fa=1 d=8392 | 815.70 ± 0.57 | 835.33 ± 2.28 | +2.4% (Vulkan crosses over) |

**Q4 pattern:** Vulkan beats ROCm by ~22% on decode (consistent across depths), ROCm beats Vulkan by ~7% on prefill at d=0, gap closes by d=8392 (Vulkan slightly ahead by then).

## Results — Qwen3.6-35B-A3B BF16

Both backends, this dir's logs. Model is 66 GB on disk, loaded fully on the 128 GB unified pool.

| shape | ROCm/HIP | Vulkan | Δ |
|---|---|---|---|
| pp512 fa=0 | **479.55 ± 1.69** | 306.85 ± 1.00 | **+56.3% (ROCm)** |
| pp512 fa=1 | **484.01 ± 4.40** | 305.21 ± 0.87 | **+58.6% (ROCm)** |
| tg128 fa=1 d=0 | **23.71 ± 0.01** | 10.73 ± 0.00 | **+121% (ROCm)** ← over 2× |
| tg128 fa=1 d=4196 | **23.39 ± 0.01** | 10.68 ± 0.01 | **+119% (ROCm)** |
| tg128 fa=1 d=8392 | **23.09 ± 0.00** | 10.64 ± 0.00 | **+117% (ROCm)** |
| pp2048 fa=1 d=0 | **474.69 ± 2.76** | 307.44 ± 1.28 | **+54.4% (ROCm)** |
| pp2048 fa=1 d=8392 | **440.02 ± 0.67** | 289.64 ± 0.84 | **+51.9% (ROCm)** |

**BF16 pattern:** ROCm wins everything by 50–120%. Decode 2× faster, prefill ~55% faster.

## Why the inversion — `bf16: 0` on RADV

Capability reported at `llama-bench --list-devices` (full in [`vulkan-capability-extract.txt`](vulkan-capability-extract.txt)):

```
fp16: 1 | bf16: 0 | int dot: 0 | matrix cores: KHR_coopmat
```

RADV STRIX_HALO supports FP16 natively, but NOT BF16. The `ggml-vulkan` backend has BF16 emit paths but they go through general-purpose shader cores instead of native matrix units. ROCm/HIP via the `libggml-hip.so` backend goes through HIP's BF16 matmul kernels which DO dispatch to native hardware.

Note: this is a Mesa-side gap, not a llama.cpp gap. Once RADV adds BF16 to its STRIX_HALO capability (cooperative matrix BF16 extensions are a known work-in-progress), the gap will likely close. Until then, BF16 = ROCm on this hardware.

## Files

| File | What it is |
|---|---|
| `build-vulkan.sh` | Recipe for building llama.cpp with the Vulkan backend |
| `vulkan-capability-extract.txt` | RADV capability report showing `bf16: 0` |
| `bench-qwen36-a3b-q4-vulkan-shape1.log` | Q4_K_M, Vulkan, pp512+tg128, fa 0/1 |
| `bench-qwen36-a3b-q4-vulkan-shape2.log` | Q4_K_M, Vulkan, pp2048+tg128 depth sweep |
| `bench-qwen36-a3b-bf16-rocm-shape1.log` | BF16, ROCm, pp512+tg128, fa 0/1 |
| `bench-qwen36-a3b-bf16-rocm-shape2.log` | BF16, ROCm, pp2048+tg128 depth sweep |
| `bench-qwen36-a3b-bf16-vulkan-shape1.log` | BF16, Vulkan, pp512+tg128, fa 0/1 |
| `bench-qwen36-a3b-bf16-vulkan-shape2.log` | BF16, Vulkan, pp2048+tg128 depth sweep |

Q4_K_M ROCm logs are cross-referenced from `../rocwmma-fattn-sweep/bench-qwen36-35b-a3b-q4-fattn-off.log` to avoid duplication.

## Reproducing

```bash
# 1. Build Vulkan binary
./build-vulkan.sh b9296

# 2. Get the model (66 GB BF16, or any GGUF you want to compare)
hf download unsloth/Qwen3.6-35B-A3B-GGUF --include "BF16/*.gguf" --local-dir /path/to/model

# 3. Bench Vulkan
/srv/aurora-ai/llama.cpp-vulkan/build/bin/llama-bench \
  -m /path/to/model/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf \
  -p 512 -n 128 -r 3 --mmap 0 -ngl 99 -fa 0,1

# 4. Bench ROCm (don't forget the nightly HSA overlay)
NIGHTLY_LIB=/path/to/_rocm_sdk_core/lib
LD_LIBRARY_PATH=$NIGHTLY_LIB:/opt/rocm-7.1.0/lib:/srv/aurora-ai/llama.cpp/build/bin \
/srv/aurora-ai/llama.cpp/build/bin/llama-bench \
  -m /path/to/model/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf \
  -p 512 -n 128 -r 3 --mmap 0 -ngl 99 -fa 0,1
```

Credit: thanks to [u/Disastrous-Cat-7016](https://reddit.com/r/StrixHalo) (whose [bench.ciru.ai](https://bench.ciru.ai) dashboard is the Vulkan canonical reference for Strix Halo) for prompting the Q4 cross-check, and [u/Potential-Leg-639](https://reddit.com/r/StrixHalo) for the "what pipeline?" question that surfaced the backend distinction in the first place.
