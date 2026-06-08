# llama.cpp #21284 — gfx1151 MMQ prefill tuning: independent A/B

Independent confirmation of [ggml-org/llama.cpp#21284](https://github.com/ggml-org/llama.cpp/issues/21284)
(@pedapudi's report that gfx1151's MMQ defaults leave substantial prefill perf on the table).
We A/B'd his MMQ tiling change on a Strix Halo box.

## What was changed
@pedapudi's patch ([gist](https://gist.github.com/pedapudi/183f337e687630a43eacb293e157c9bd),
mirrored here as `pedapudi-gfx1151.patch`) sets RDNA3.5-specific MMQ tiling:
`mmq_x 128→48`, `mmq_y 128→64`, `nwarps 8→4`. We applied **just the headline mmq.cuh
tiling change** (host + device, gated on `GGML_CUDA_CC_IS_RDNA3_5` / `RDNA3_5`) as guarded
early-returns — the RDNA3-guarded intrinsic swaps in the gist's other files are no-ops on
gfx1151 anyway (only `RDNA3_5` is defined for `__gfx1151__`, not `RDNA3`).

## Method
- Same commit (`a497476`) built **twice**, identical flags, only the MMQ change differing.
- Flags: `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DGGML_HIP_ROCWMMA_FATTN=OFF -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON`
- `llama-bench -p 512,2048 -n 0 -ngl 999 -fa 0 -r 3`
- HW: Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151, Ubuntu 24.04, kernel 7.0.9, ROCm 7.13.
  (Nightly `libhsa-runtime64.so.1` overlay — ROCm 7.1.0 stock HSA segfaults on gfx1151 at model load.)

## Result (prefill t/s, mean of 3)

| model | arch | pp512 stock → patched | pp2048 stock → patched |
|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | **MoE** (3B active) | 1071.6 → **1367.6  (+27.6%)** | 1016.9 → **1317.2  (+29.5%)** |
| Qwen3.5-27B Q4_K_M | **dense** | 333.5 → 311.5  (**−6.6%**) | 322.5 → 309.8  (**−3.9%**) |

## Takeaway
The MMQ tiling change is a **large prefill win on sparse MoE** (+27–30% on A3B — even above
@pedapudi's ~20% on the 122B MoE) but a **small regression on the dense 27B** (−4 to −7%).
So the smaller tiles / lower VGPR pressure clearly help the small per-expert matmuls of a sparse
MoE, while the dense path still prefers the wider 128 tiles. Worth gating the tuning on MoE-ness
(or making it tunable) rather than applying it unconditionally for RDNA3.5.

(Couldn't bench the 122B Q6_K we have — at 95 GB it wouldn't GPU-offload within headroom on
128 GB unified memory and fell back to CPU, so we used Q4 MoE, which also matches the issue's quant.)
