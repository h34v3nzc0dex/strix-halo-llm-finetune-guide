# FLA #913 — GatedDeltaNet bwd "misaligned address": gfx1151 cross-arch control

Cross-architecture control for [fla-org/flash-linear-attention#913](https://github.com/fla-org/flash-linear-attention/issues/913),
where `chunk_gated_delta_rule` backward throws a CUDA *misaligned address* on NVIDIA
Blackwell. We run this exact kernel in production as the core of a Qwen3.5-27B
GatedDeltaNet LoRA fine-tune on AMD Strix Halo, so it's a useful "does the same
Triton path break on a different vendor?" datapoint.

## Stack
- Ryzen AI MAX+ 395 / Radeon 8060S / **gfx1151** (RDNA 3.5), 128 GB unified memory
- Ubuntu 24.04, kernel 7.0.9 mainline, ROCm 7.13
- PyTorch 2.11.0+rocm7.13.0a, Triton 3.6.0, FLA (production editable install)

## Repro
```
python repro_gdn_bwd.py
```
Runs `chunk_gated_delta_rule` fwd+bwd at our production shapes —
`B=1, T=8192, H=16, K=V=128`, bf16, `use_qk_l2norm_in_kernel=True` — 3× for stability.

## Result (`output.txt`)
3/3 runs complete clean, fwd + bwd, **no misaligned address**. Gradients return at
full shape `(1, 8192, 16, 128)` for q/k/v.

## Takeaway
The same Triton GDN code path is healthy on RDNA 3.5 / gfx1151. That points at #913
being **Blackwell-specific** (matches the issue's own narrowing) rather than a
kernel-logic bug. Happy to run any specific shape/dtype combo here as a cross-arch control.
