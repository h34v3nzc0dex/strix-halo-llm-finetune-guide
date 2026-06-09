# TheRock #5314 — hipBLASLt vs rocBLAS matmul on gfx1151 (by training layout)

Datapoint for [ROCm/TheRock#5314](https://github.com/ROCm/TheRock/issues/5314):
matmul throughput on gfx1151, hipBLASLt path on vs off.

> **Correction (per @woct0rdho on the issue):** the first version of this swept only
> the plain `a @ b` (NN) layout. Training is dominated by other layouts, so this now
> benches the **three GEMMs of a Linear-layer training step** — and the answer turns
> out to be layout-dependent, not a global "hipBLASLt wins."

## Stack
- Ryzen AI MAX+ 395 / Radeon 8060S / **gfx1151**, Ubuntu 24.04, kernel 7.0.9, ROCm 7.13
- PyTorch 2.11.0+rocm7.13.0a
- **Config note:** we boot `iommu=pt` (the issue thread ran `amd_iommu=off`).

## Repro
```
TORCH_BLAS_PREFER_HIPBLASLT=0 python repro_hipblaslt.py   # rocBLAS
TORCH_BLAS_PREFER_HIPBLASLT=1 python repro_hipblaslt.py   # hipBLASLt
```
Layouts (all n×n, so each isolates the transpose config):
| layout | op | training role |
|---|---|---|
| NT | `y = x @ W.T` | forward |
| NN | `dX = dY @ W` | grad wrt input (dgrad) |
| TN | `dW = dY.T @ x` | grad wrt weight (wgrad) |

## Result (`output.txt`) — TFLOPS, fp16 (bf16 within ~1)

**NT — forward**
| size | rocBLAS | hipBLASLt |
|---|---|---|
| 2048³ | 36.1 | 36.9 |
| 4096³ | 21.8 | **32.4** (+49%) |
| 8192³ | 24.9 | **34.5** (+39%) |

**NN — dgrad**
| size | rocBLAS | hipBLASLt |
|---|---|---|
| 2048³ | 27.5 | **34.3** |
| 4096³ | 32.8 | **35.5** |
| 8192³ | 23.7 | **30.6** |

**TN — wgrad**
| size | rocBLAS | hipBLASLt |
|---|---|---|
| 2048³ | **35.6** | 24.2 |
| 4096³ | **39.9** | 16.4 (−59%) |
| 8192³ | **30.9** | 8.4 (−73%) |

(TN collapse reproduces run-to-run; fp16≈bf16 exactly on the hipBLASLt TN path,
which looks like a non-tensor-core fallback.)

## Takeaway
On gfx1151 it's **layout-dependent**:
- **Forward (NT)** and **dgrad (NN)** → hipBLASLt clearly faster (up to +49%).
- **wgrad (TN)** → **rocBLAS wins hard** — hipBLASLt drops to ~8.4 TFLOPS at 8192³
  (~3.7× slower), looks like it's missing a tuned TN kernel and falling back.

So a blanket `TORCH_BLAS_PREFER_HIPBLASLT=1` is *not* a clean win for a training step —
the weight-gradient GEMM regresses badly. Per-op backend selection would be the ideal,
or hipBLASLt needs a tuned kernel for the slow layout on gfx1151.
Thanks to @woct0rdho for the steer toward the training layouts — that's what surfaced this.

## Label correction (2026-06-08, per @woct0rdho)
The section headers above use **PyTorch-op** names; the row-major→BLAS mapping is not 1:1.
Verified against the hipBLASLt bench log (`HIPBLASLT_LOG_MASK=32`) — the actual hipBLASLt
labels are:

| training op | PyTorch | **hipBLASLt (logged)** | speed |
|---|---|---|---|
| forward `y=x·Wᵀ` | `a @ b.T` | **TN** | fast |
| dgrad `dX=dY·W` | `a @ b` | **NN** | fast |
| wgrad `dW=dYᵀ·x` | `aᵀ @ b` | **NT** | slow |

So the regressing GEMM is **wgrad = hipBLASLt NT** (the "wgrad (TN)" table above is that op;
its true hipBLASLt label is **NT**). This matches @woct0rdho's "TN and NN fast, NT/TT slow"
exactly. Numbers unchanged — only the naming is corrected.
