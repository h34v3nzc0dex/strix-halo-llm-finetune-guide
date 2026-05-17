"""
Round 2: production-scale shape to force the autotune to pick larger configs
(num_warps in {8, 16}) that our patches cap to <= 4.
"""
import os, sys, shutil, faulthandler

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
sys.path.insert(0, "/tmp/fla-vanilla")

# Clear cache so autotune runs from scratch and exercises all configs
_triton_cache = os.path.expanduser("~/.triton/cache")
if os.path.isdir(_triton_cache):
    shutil.rmtree(_triton_cache)
    print(f"cleared Triton cache")

faulthandler.enable()

import torch, triton
print(f"torch {torch.__version__}  triton {triton.__version__}  hip {torch.version.hip}")

import fla
print(f"fla.__file__: {fla.__file__} (override OK: {'/tmp/fla-vanilla' in (fla.__file__ or '')})")

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

# Match production training scale: max_seq_len=8192 from CLAUDE.md (Qwen3.5-27B v8 training).
# GDN typical: head_dim 128, num_v_heads 16. Pick something realistic per layer.
for label, (B, T, H, HV, K, V) in [
    ("Qwen3.5-27B-ish T=8192 K=128 V=128", (1, 8192, 16, 16, 128, 128)),
    ("longer chunks T=4096 K=256",         (1, 4096, 8,  8,  256, 256)),
    ("batched T=2048",                     (2, 2048, 16, 16, 128, 128)),
]:
    print(f"\n--- {label}: B={B} T={T} H={H} HV={HV} K={K} V={V} ---")
    sys.stdout.flush()
    device, dtype = "cuda", torch.bfloat16
    q = (torch.randn(B, T, H,  K, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
    k = (torch.randn(B, T, H,  K, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
    v = (torch.randn(B, T, HV, V, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
    g = (torch.randn(B, T, HV, device=device, dtype=torch.float32) * -0.5).clamp(-5, 0)
    beta = torch.randn(B, T, HV, device=device, dtype=dtype).sigmoid()
    out, _ = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
    print(f"  FORWARD OK: out.shape={tuple(out.shape)}")
    loss = out.float().sum()
    loss.backward()
    has_nan = q.grad.isnan().any().item() or k.grad.isnan().any().item() or v.grad.isnan().any().item()
    print(f"  BACKWARD OK: grads have NaN: q={q.grad.isnan().any().item()}  k={k.grad.isnan().any().item()}  v={v.grad.isnan().any().item()}")
    del q, k, v, g, beta, out, loss
    torch.cuda.empty_cache()

print(f"\n--- ALL LARGE-SHAPE PROBES SUCCEEDED ---")
