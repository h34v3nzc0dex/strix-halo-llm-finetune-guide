"""
PR #5434 validation probe: does vanilla flash-linear-attention 0.5.0 from PyPI
work on gfx1151 (Strix Halo) without the patches we keep in /srv/aurora-ai/fla-patched?

If this script SIGSEGVs or raises a Triton autotune error, PR #5434's pip-install-from-PyPI
strategy is broken for Strix Halo users until upstream FLA / Triton lands the fixes.
If it completes cleanly, our patches may be obsolete and the PR is safe on gfx1151.
"""
import os, sys, shutil, faulthandler

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# Override patched editable install with vanilla PyPI install.
# (sys.path.insert([0]) wins because /tmp/fla-vanilla/fla/__init__.py exists,
# so the namespace-pkg merge can't fall through to the editable path.)
sys.path.insert(0, "/tmp/fla-vanilla")

# Cold Triton cache — autotune num_warps issues bite on cold compile.
_triton_cache = os.path.expanduser("~/.triton/cache")
if os.path.isdir(_triton_cache):
    shutil.rmtree(_triton_cache)
    print(f"cleared Triton cache: {_triton_cache}")

faulthandler.enable()

import torch
print(f"\n--- env ---")
print(f"torch: {torch.__version__}  hip: {torch.version.hip}")
print(f"device 0: {torch.cuda.get_device_name(0)}")
import triton
print(f"triton: {triton.__version__}")

print(f"\n--- fla provenance ---")
import fla
print(f"fla.__file__: {fla.__file__}")
import importlib.metadata as md
print(f"fla version (PyPI shim): {md.version('flash-linear-attention')}")
print(f"fla-core version:        {md.version('fla-core')}")
assert "/tmp/fla-vanilla" in (fla.__file__ or ""), "override failed; would be testing patched FLA"

print(f"\n--- import gated_delta_rule kernel (the Qwen3.5 GDN entry) ---")
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
print(f"chunk_gated_delta_rule:           {chunk_gated_delta_rule.__module__}")
print(f"fused_recurrent_gated_delta_rule: {fused_recurrent_gated_delta_rule.__module__}")

print(f"\n--- chunk_gated_delta_rule probe (bf16, B=1 T=256 H=4 K=64 V=64) ---")
device, dtype = "cuda", torch.bfloat16
B, T, H, HV, K, V_dim = 1, 256, 4, 4, 64, 64
q = torch.randn(B, T, H,  K,     device=device, dtype=dtype, requires_grad=True)
k = torch.randn(B, T, H,  K,     device=device, dtype=dtype, requires_grad=True)
v = torch.randn(B, T, HV, V_dim, device=device, dtype=dtype, requires_grad=True)
g = torch.randn(B, T, HV,        device=device, dtype=torch.float32) * -0.5
beta = torch.randn(B, T, HV,     device=device, dtype=dtype).sigmoid()
print(f"shapes: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} g={tuple(g.shape)} beta={tuple(beta.shape)}")
print("calling chunk_gated_delta_rule (this is where autotune fires)...")
sys.stdout.flush()
out, final_state = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
print(f"FORWARD OK: out.shape={tuple(out.shape)} dtype={out.dtype}")
print(f"FORWARD OK: final_state.shape={tuple(final_state.shape) if final_state is not None else None}")
print(f"first 4 values: {out.flatten()[:4].tolist()}")

print(f"\n--- backward probe (GDN gradient) ---")
loss = out.sum()
loss.backward()
print(f"BACKWARD OK: q.grad.shape={tuple(q.grad.shape)}  q.grad has NaN: {q.grad.isnan().any().item()}")

print(f"\n--- fused_recurrent variant probe (smaller; tests step-mode) ---")
qs = torch.randn(B, T, H,  K,     device=device, dtype=dtype)
ks = torch.randn(B, T, H,  K,     device=device, dtype=dtype)
vs = torch.randn(B, T, HV, V_dim, device=device, dtype=dtype)
gs = torch.randn(B, T, HV,        device=device, dtype=torch.float32) * -0.5
bs = torch.randn(B, T, HV,        device=device, dtype=dtype).sigmoid()
out2, _ = fused_recurrent_gated_delta_rule(qs, ks, vs, gs, bs)
print(f"fused_recurrent OK: out.shape={tuple(out2.shape)}")

print(f"\n--- ALL PROBES SUCCEEDED — vanilla 0.5.0 works on gfx1151 ---")
