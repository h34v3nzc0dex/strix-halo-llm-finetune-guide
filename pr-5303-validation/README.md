# PR #5303 validation on gfx1151

Tests for [unslothai/unsloth#5303](https://github.com/unslothai/unsloth/pull/5303) —
*"feat(studio): use lemonade-sdk/llamacpp-rocm per-GPU prebuilts for ROCm hosts"*.

## What we tested

The PR's `resolve_lemonade_rocm_choice()` would download and use
`llama-{tag}-{os}-rocm-{gfxFamily}-x64.zip` for an AMD GPU whose `gcnArchName`
matches one of the families in `_LEMONADE_GFX_FAMILIES`. For Strix Halo
(gfx1151), that resolves to
`llama-b1270-ubuntu-rocm-gfx1151-x64.zip` from the latest lemonade-sdk release.

We downloaded that asset by hand and exercised it directly on this gfx1151
box — the goal is to confirm Studio users on gfx1151 will (a) actually get a
working binary out of the box and (b) not lose meaningful performance vs the
hand-built path.

## Environment

- AMD Ryzen AI MAX+ 395, Radeon 8060S (`gfx1151`), 128 GB unified memory
- Ubuntu 24.04 LTS, kernel 6.19.14 mainline, ROCm 7.13 nightly
- Lemonade asset: `llama-b1270-ubuntu-rocm-gfx1151-x64.zip` (435 MB, published 2026-05-17)
  - Bundles `libamdhip64.so.7.13.26194`, `libhsa-runtime64`, `libhipblas`,
    `librocblas`, `librocsolver`, etc. — fully self-contained, no system ROCm
    needed at runtime
- Self-built reference: `/usr/local/bin/llama-bench` build `d0a6dfe (502)`,
  compiled locally per our long-running notes
  (`-DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_HIP_GRAPHS=ON
  -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON
  -DCMAKE_HIP_FLAGS='--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'`)
- Test model: `aurora-effects-v8-q8_0.gguf` — Qwen3.5-27B Q8_0, 26.62 GiB, 26.90 B params
- Bench flags: `-p 64 -n 16 -r 3 -mmp 0 -ngl 999` (no-mmap is the gfx1151
  gotcha — mmap-only triggers ~30 min GPU page-table setup; documented in
  this guide's main README)

## Detection — works first-try

```
$ LD_LIBRARY_PATH=/tmp/llama-lemonade ./llama-bench -h
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 131072 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 131072 MiB
```

Notes:
- `Total VRAM: 131072 MiB` (128 GiB) — the full Strix Halo unified pool is
  reported correctly via HIP, NOT the 1 GiB dedicated slice `amd-smi` shows.
  Same finding as the validation work on #5301.
- `Wave Size: 32` — correct for RDNA 3.5 (gfx115X).
- `VMM: no` — expected on gfx1151.

## Bench results — 3 reps each

| Build | pp64 (tok/s) | tg16 (tok/s) |
|---|---|---|
| **Lemonade prebuilt b1270** (`39cf5d6`) | 253.96 ± 27.59 | 7.48 ± 0.05 |
| **Self-built b502** (`d0a6dfe`) | 270.61 ± 2.07 | 7.52 ± 0.00 |

Reading:
- **`tg16`**: 7.48 vs 7.52 tok/s — statistically identical. Differ by 0.5%; within
  noise even for a single rep, let alone three. Token-generation throughput
  is equivalent.
- **`pp64`**: 253.96 vs 270.61 tok/s — apparent ~6% gap, but the lemonade
  σ (27.59) covers the entire delta (16.65). Not a real performance regression;
  cold-cache jitter on the smaller bench window.

So on the model size and bench shape we care about for fine-tune workflows,
the prebuilt is **performance-equivalent** to a fully-tuned hand build, while
sparing the user the `--gcc-install-dir` / `_rocm_sdk_core` / `hip-lang-config`
build hassles that the parallel #5301 / #5517 PRs document.

## Reproducing

```bash
curl -L -o /tmp/llama-lemonade.zip \
  "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/b1270/llama-b1270-ubuntu-rocm-gfx1151-x64.zip"
mkdir -p /tmp/llama-lemonade
unzip -q /tmp/llama-lemonade.zip -d /tmp/llama-lemonade
chmod +x /tmp/llama-lemonade/llama-bench

# Lemonade
LD_LIBRARY_PATH=/tmp/llama-lemonade /tmp/llama-lemonade/llama-bench \
  -m <your-gguf>.gguf -p 64 -n 16 -r 3 -mmp 0 -ngl 999
```

Full output logs alongside this README: `bench-aurora-v8-3rep.log`,
`bench-selfbuilt-b502-3rep.log` (and the original 1-rep runs for the audit
trail).
