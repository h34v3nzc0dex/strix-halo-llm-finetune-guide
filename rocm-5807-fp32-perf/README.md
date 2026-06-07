# ROCm #5807 — fp32 matmul slow, fp16/bf16 fine on gfx1151

Independent reproduction of [ROCm/ROCm#5807](https://github.com/ROCm/ROCm/issues/5807)
on a **different distro + ROCm version** than the reporter, plus a nightly-vs-GA
channel comparison.

## Stacks tested
- Hardware: Ryzen AI MAX+ 395 / Radeon 8060S / **gfx1151**, Ubuntu 24.04, kernel 7.0.9
- **Nightly**: torch 2.11.0+rocm7.13.0a (`rocm.nightlies.amd.com/v2-staging/gfx1151/`)
- **GA channel**: torch 2.11.0+rocm7.13.0 (`repo.amd.com/rocm/whl/gfx1151/`)
- Reporter's stack for contrast: Fedora 43, torch 2.11.0a0+rocm7.11

## Repro
```
python repro_fp32_matmul.py      # 4096^3 torch.matmul, per-dtype TFLOPS
```

## Result (`output.txt`)
4096³ matmul:

| dtype | nightly (rocm7.13.0a) | GA (rocm7.13.0) |
|---|---|---|
| fp16 | 33.6 | 33.7 |
| bf16 | 29.5 | 29.5 |
| **fp32** | **3.1** | **3.0** |

## Takeaway
~**11× fp32 penalty** vs fp16, and it holds **across distros (Fedora→Ubuntu),
ROCm versions (7.11→7.13), and both the nightly and GA wheel channels** — so this
is a gfx1151 characteristic, not a one-off config. Looks like fp32 isn't hitting
the WMMA/packed path that fp16/bf16 use. Happy to run more shapes.
