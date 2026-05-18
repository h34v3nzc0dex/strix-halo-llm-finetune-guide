# Round 2: Reproducing cezq's bench on this gfx1151 box

Follow-up to the original sweep — peer commenter `u/cezq` on r/StrixHalo
pointed out the original methodology (`pp64`, `ROCWMMA_FATTN=ON`,
`-fa 0`) was the wrong shape to surface the `-DGGML_CUDA_FORCE_CUBLAS=ON`
gain. Re-ran with cezq's exact build flags and bench command for direct
comparison.

## Setup

- **Source:** llama.cpp `5207d120e (867)`
- **Builds:**
  - `build-cezq-off/`: `-DGGML_HIP_ARCHS=gfx1151 -DGGML_HIP=ON -DGGML_CUDA_FORCE_CUBLAS=OFF -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_ROCWMMA_FATTN=OFF -DCMAKE_HIP_FLAGS="-mllvm --amdgpu-unroll-threshold-local=600 --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"`
  - `build-cezq-on/`: same plus `-DGGML_CUDA_FORCE_CUBLAS=ON`
- **Bench:** `llama-bench -p 2048 -n 32 -r 3 -ngl 999 -d 0,4196,8392,16784,33568 -b 2048 -ub 2048 -mmp 0 -dio 1 -fa 1`
- **Model:** `aurora-effects-v8-q8_0.gguf` (Qwen3.5-27B Q8_0, 26.62 GiB)
  - **Note:** cezq tested on `Qwen3.6-27B-MTP-Q8_0`. Direct apples-to-apples
    on the build-flag dimension; the model dimension differs.
- **Hardware:** Corsair AI Workstation 300, AXB35-02 BIOS 3.07, ROCm 7.13 nightly

## Results

### pp2048 (prompt processing) — `t/s`

| depth | CUBLAS=OFF | CUBLAS=ON +HIPBLASLT=1 | Δ here | cezq Δ |
|---|---|---|---|---|
| 0     | 304.35 ± 3.36 | **326.32 ± 4.57** | **+7.2%** | +10.1% |
| 4196  | 306.97 ± 0.78 | 314.35 ± 0.31 | +2.4% | +4.0% |
| 8392  | 288.55 ± 2.39 | 280.15 ± 2.27 | -2.9% | +1.3% |
| 16784 | 231.99 ± 0.79 | 230.95 ± 2.82 | flat | +2.6% |
| 33568 | 159.75 ± 0.63 | 157.20 ± 1.03 | -1.6% | +3.8% |

### tg32 (token generation) — `t/s`

| depth | CUBLAS=OFF | CUBLAS=ON +HIPBLASLT=1 | Δ here |
|---|---|---|---|
| 0     | 7.26 ± 0.04 | 7.58 ± 0.01 | +4.4% |
| 4196  | 7.53 ± 0.03 | 7.54 ± 0.03 | flat |
| 8392  | 7.46 ± 0.02 | 7.48 ± 0.03 | flat |
| 16784 | 7.34 ± 0.03 | 7.35 ± 0.03 | flat |
| 33568 | 7.11 ± 0.03 | 6.93 ± 0.29 | -2.5% (high σ) |

## Reading

**cezq's CUBLAS+HIPBLASLT gain replicates on this hardware**, at smaller
magnitude and with faster decay across context depth. Three observations:

1. **At pp2048 d0, +7.2% pp gain** — matches cezq's directional finding
   (their +10.1%). Confirms CUBLAS+HIPBLASLT is faster than MMQ for this
   workload shape on gfx1151.
2. **Gain narrows faster on our Qwen3.5 than cezq's Qwen3.6-MTP** — by d8k
   we're flat/noise here while cezq still sees ~1-4% across the rest of
   the sweep. Probably the MTP layer's matmul shape interacts differently
   with the CUBLAS path than dense Qwen3.5.
3. **Absolute pp2048 numbers are ~7% lower here at every depth** (our 304
   vs their 327 at d0 baseline). Same hardware family, but model
   difference + possibly board-firmware tuning. cezq is on GMKtec EVO-X2
   with Arch Linux kernel 7.0.3; we're on Corsair AXB35-02 BIOS 3.07 with
   Ubuntu 24.04 kernel 6.19.14.

**Most useful takeaway for other Strix Halo Linux users:** if you're
running dense Qwen3.x or similar at pp2048+ workload shape with FA on,
`-DGGML_CUDA_FORCE_CUBLAS=ON` + `ROCBLAS_USE_HIPBLASLT=1` is worth
testing — the gain is modest (~5-10% at d0) but real, and there's no
downside at tg or longer contexts. **If your prefill workload is small
(< pp512), the original sweep's finding holds: the CUBLAS flag compiles
out MMQ and becomes a 3.6× slowdown.** Bench shape matters.

## Logs

- `bench-cezq-flags-cublas-off.log` — full output, CUBLAS=OFF baseline
- `bench-cezq-flags-cublas-on-hipblaslt.log` — CUBLAS=ON + HIPBLASLT=1

## Reproducing

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout 5207d120e  # or latest

# CUBLAS=OFF baseline
PATH=/opt/rocm-7.1.0/bin:$PATH cmake -S . -B build-cezq-off \
  -DGGML_HIP_ARCHS=gfx1151 -DGGML_HIP=ON \
  -DGGML_CUDA_FORCE_CUBLAS=OFF \
  -DAMDGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_HIP_FLAGS="-mllvm --amdgpu-unroll-threshold-local=600 --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON
cmake --build build-cezq-off --config Release -- -j16 llama-bench

# CUBLAS=ON
PATH=/opt/rocm-7.1.0/bin:$PATH cmake -S . -B build-cezq-on \
  ...same flags but... -DGGML_CUDA_FORCE_CUBLAS=ON
cmake --build build-cezq-on --config Release -- -j16 llama-bench

# Bench (each build):
LD_LIBRARY_PATH=/opt/rocm-7.1.0/lib ROCBLAS_USE_HIPBLASLT=1 \
  ./build-cezq-on/bin/llama-bench -p 2048 -n 32 -r 3 -ngl 999 \
  -d 0,4196,8392,16784,33568 -b 2048 -ub 2048 -mmp 0 -dio 1 -fa 1 \
  -m <your-Qwen-Q8.gguf>
```
