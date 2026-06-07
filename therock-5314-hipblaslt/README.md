# TheRock #5314 — hipBLASLt vs rocBLAS matmul on gfx1151

Datapoint for [ROCm/TheRock#5314](https://github.com/ROCm/TheRock/issues/5314):
matmul throughput on gfx1151 with the hipBLASLt path on vs off, swept across sizes.

## Stack
- Ryzen AI MAX+ 395 / Radeon 8060S / **gfx1151**, Ubuntu 24.04, kernel 7.0.9, ROCm 7.13
- PyTorch 2.11.0+rocm7.13.0a
- **Config note:** we boot `iommu=pt` (the issue thread ran `amd_iommu=off`) — worth
  keeping in mind if IOMMU mode turns out to matter for the regression.

## Repro
```
TORCH_BLAS_PREFER_HIPBLASLT=0 python repro_hipblaslt.py   # rocBLAS path
TORCH_BLAS_PREFER_HIPBLASLT=1 python repro_hipblaslt.py   # hipBLASLt path
```

## Result (`output.txt`) — TFLOPS

| size | rocBLAS fp16 | hipBLASLt fp16 | rocBLAS bf16 | hipBLASLt bf16 |
|---|---|---|---|---|
| 2048³ | 27.9 | **37.2** (+33%) | 31.0 | **36.4** (+17%) |
| 4096³ | 32.1 | **35.2** (+10%) | 29.7 | **36.7** (+24%) |
| 8192³ | 24.0 | **32.2** (+34%) | 24.7 | **32.7** (+32%) |

## Takeaway
On this box **hipBLASLt is faster across the board** — ~10–34% uplift, largest at
8192³ where rocBLAS sags to ~24 TFLOPS while hipBLASLt holds ~32. No correctness
issues either way. If the regression in the issue is config-dependent, our `iommu=pt`
result is a clean "hipBLASLt-on is the win here" counterpoint to compare against.
