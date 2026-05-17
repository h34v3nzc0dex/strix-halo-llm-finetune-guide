"""
Final ground-truth check: with both vanilla FLA 0.5.0 AND tilelang 0.1.8 installed
(the state PR #5434 leaves a Strix Halo box in), does chunk_gated_delta_rule.backward
still work, or does FLA try to dispatch to TileLang and crash?

This is the soft claim in the draft comment — turning it into a hard claim.
"""
import os, sys, shutil, faulthandler

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

# Order matters: FLA first, then tilelang dir on path. Both must be importable
# so TileLangBackend.is_available() returns True.
sys.path.insert(0, "/tmp/tilelang-test")
sys.path.insert(0, "/tmp/fla-vanilla")

_triton_cache = os.path.expanduser("~/.triton/cache")
if os.path.isdir(_triton_cache):
    shutil.rmtree(_triton_cache)
    print("cleared Triton cache")

faulthandler.enable()

import torch, triton
print(f"torch {torch.__version__}  triton {triton.__version__}  hip {torch.version.hip}")

import fla
print(f"fla.__file__: {fla.__file__}")

import tilelang
print(f"tilelang.__file__: {tilelang.__file__}")

# Confirm FLA *sees* tilelang as available
from fla.ops.common.backends.tilelang import TileLangBackend
print(f"\nTileLangBackend.is_available(): {TileLangBackend.is_available()}")
print("(this is what FLA's dispatcher checks before considering the TileLang path)")

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

print("\n--- Qwen3.5-27B-ish: B=1 T=8192 H=16 K=128 V=128, bf16, fwd+bwd ---")
device, dtype = "cuda", torch.bfloat16
B, T, H, K, V = 1, 8192, 16, 128, 128
q = (torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
k = (torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
v = (torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1).detach().requires_grad_(True)
g = (torch.randn(B, T, H, device=device, dtype=torch.float32) * -0.5).clamp(-5, 0)
beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid()
print(f"calling fwd...")
sys.stdout.flush()
out, _ = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
print(f"FWD OK: out.shape={tuple(out.shape)}")
print(f"calling bwd (this is where chunk_bwd_dqkwg dispatch fires)...")
sys.stdout.flush()
loss = out.float().sum()
loss.backward()
print(f"BWD OK: q.grad has NaN: {q.grad.isnan().any().item()}")

print(f"\n--- SUCCESS: FLA + tilelang both importable, GDN bwd on gfx1151 ran clean ---")
print(f"--- Either FLA's dispatcher skipped TileLang (Hopper-only gate) or it fell back gracefully ---")
