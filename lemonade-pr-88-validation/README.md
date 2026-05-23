# lemonade-sdk/llamacpp-rocm PR #88 — Independent gfx1151 replication

PR under test: [lemonade-sdk/llamacpp-rocm#88](https://github.com/lemonade-sdk/llamacpp-rocm/pull/88) — *"Enable LTO to all targets, and apply gfx1151 specific configurations"*

The PR's actual diff is a single line: `-DGGML_OPENMP=OFF` → `-DGGML_OPENMP=ON` in the gfx1151 cmake configure step of the CI workflow. The title is broader than the change.

**Author's claim:** "llama-bench shows 15%-20% gain in token generation on gfx1151 when applying these specific llamacpp build configs."

**Our finding:** On this gfx1151 box, OpenMP=ON delivers **essentially zero gain on `tg128`** (0.0% to +1.1% across the full `fa×mmap` matrix, all inside noise). Small effect on `pp512` (+1.1% to +5.3% on clean rows, +11% on one row but with σ=112 — junk).

## Test rig

- Radeon 8060S / gfx1151 / Ryzen AI MAX+ 395 / 128 GiB unified
- Ubuntu 24.04 / kernel 6.19.14 mainline
- ROCm 7.1.0 system + nightly `libhsa-runtime64.so.1` overlay (sidesteps the known ROCm 7.1.0 HSA null-ptr bug on gfx1151 — orthogonal to the OpenMP question)

Differences vs author's setup (per their PR body): they report `Total VRAM: 64042 MiB` (BIOS UMA configured for a 64 GiB GPU pool), build on ROCm 7.13 nightly (per lemonade b1276 release notes). We have the full 128 GiB pool and built on ROCm 7.1.0 stable. These are absolute-baseline differences; the meaningful comparison is the OFF→ON delta on each box.

## Methodology

Both binaries built from the same source commit — `1acee6bf8`, the exact llama.cpp commit lemonade b1276 ships — with identical CMake flags (verbatim copy of lemonade's gfx1151 invocation from `.github/workflows/build-llamacpp-rocm.yml`) except for `-DGGML_OPENMP=OFF/ON`. See `build.sh`.

One Ubuntu 24.04 host-toolchain addition applied identically to both: `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13` (ROCm 7.1 clang-20 picks gcc-14's runtime dir on stock 24.04 which lacks libstdc++; lemonade's CI on Ubuntu 22.04 doesn't need it). Runtime-neutral, identical in both builds → cancels in the A/B.

Verified via `ldd`: `build-omp-off/bin/llama-bench` does NOT link libomp; `build-omp-on/bin/llama-bench` links `libomp.so` from `/opt/rocm-7.1.0/lib/llvm/lib/`. Single differing factor.

Bench command is verbatim from the PR body, run via `bench.sh`:

```bash
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
LD_LIBRARY_PATH=$NIGHTLY_HSA:/opt/rocm/lib:./ \
./llama-bench -hf unsloth/Qwen3-Coder-Next-GGUF:UD-Q5_K_XL -fa 0,1 --mmap 0,1 -ngl 99
```

## Results

| fa | mmap | test  | OMP=OFF (tok/s)      | OMP=ON (tok/s)       | Δ      |
|----|------|-------|----------------------|----------------------|--------|
|  0 |   0  | pp512 | 548.50 ± 7.53        | **577.66 ± 4.42**    | +5.3%  |
|  0 |   0  | tg128 |  35.64 ± 0.13        |  35.84 ± 0.03        | +0.6%  |
|  1 |   0  | pp512 | 559.53 ± 5.97        | 565.62 ± 6.89        | +1.1%  |
|  1 |   0  | tg128 |  35.85 ± 0.04        |  35.87 ± 0.02        | +0.1%  |
|  0 |   1  | pp512 | 507.00 ± **112.23**  | 562.67 ± 6.32        | +11.0% *(OFF row σ is junk)* |
|  0 |   1  | tg128 |  36.10 ± 0.10        |  36.49 ± 0.02        | +1.1%  |
|  1 |   1  | pp512 | 553.21 ± 5.98        | 563.85 ± 8.78        | +1.9%  |
|  1 |   1  | tg128 |  36.16 ± 0.03        |  35.94 ± 1.50        | -0.6%  |

| | Author's tg128 (reported) | Our tg128 |
|---|---|---|
| OFF mean | ~32.9 | ~35.9 |
| ON mean  | ~38.7 | ~36.0 |
| Δ        | **+17.6%** | **+0.5%** |

## Interpretation

The OpenMP=ON build links `libomp.so` and uses OpenMP for ggml's CPU-side thread pool. The headline `tg128` claim (+15-20%) implies a CPU-side dispatch bottleneck — but on a 80B-A3B MoE with all 99 layers GPU-offloaded, TG should be GPU memory-bandwidth-bound. If TG is GPU-bound on this box, OpenMP threading has nothing to optimize and the effect goes to ~0. That's what we see: TG essentially flat.

Why the author sees a large effect we don't is the interesting question. Candidates:
- **ROCm version** — author on 7.13 nightly, us on 7.1.0 stable; the nightly's HIP runtime may dispatch GPU work differently and surface a CPU-side bottleneck we don't have
- **BIOS UMA cap** — author at 64 GiB, us at 128 GiB unified; on a smaller GPU pool the allocator may behave differently for a 55 GiB model
- **CPU core count / topology** — OpenMP's win scales with available threads; we're 16c/32t Ryzen AI MAX+ 395, author's CPU is unspecified

None of these make OpenMP=ON *wrong* — the flag is broadly safe and the upstream llama.cpp default — but the +15-20% gain isn't a universal expectation across gfx1151 hardware.

## Files

| File | What it is |
|---|---|
| `build.sh` | Builds both variants from `1acee6bf8` with lemonade's exact gfx1151 CMake flags + Ubuntu 24.04 toolchain workaround, only differing on `GGML_OPENMP` |
| `bench.sh` | Runs the author's exact bench command against each binary with isolated HF cache + nightly HSA overlay |
| `bench-omp-off.log` | llama-bench raw output, GGML_OPENMP=OFF |
| `bench-omp-on.log`  | llama-bench raw output, GGML_OPENMP=ON |

## Reproducing

```bash
# Source commit identical to lemonade b1276
git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp
git checkout 1acee6bf8 && cd ..

./build.sh   # produces build-omp-off/bin/llama-bench + build-omp-on/bin/llama-bench
./bench.sh   # downloads the 55 GiB model into ./.hf-cache/ on first run; benches both
```
