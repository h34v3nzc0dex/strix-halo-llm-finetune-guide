"""
Test causal_conv1d_fn + causal_conv1d_update on gfx1151.
CLAUDE.md says these SIGSEGV; verify against the just-built 1.6.2.post1.
PR #5434's fast-path gate requires both functions to be importable AND working.
"""
import os, sys, faulthandler

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
sys.path.insert(0, "/tmp/causal-conv1d-test")
faulthandler.enable()

import torch
print(f"torch {torch.__version__}  hip {torch.version.hip}  device 0: {torch.cuda.get_device_name(0)}")
import causal_conv1d
print(f"causal_conv1d: {causal_conv1d.__file__}")
import importlib.metadata as md
print(f"version: {md.version('causal-conv1d')}")

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
print(f"causal_conv1d_fn:     {causal_conv1d_fn.__module__}")
print(f"causal_conv1d_update: {causal_conv1d_update.__module__}")

# Probe 1: forward over a sequence (training-style)
print("\n--- causal_conv1d_fn(): batch=1 dim=512 seqlen=8192 kernel=4 bf16 ---")
B, D, T, K = 1, 512, 8192, 4
device, dtype = "cuda", torch.bfloat16
x = torch.randn(B, D, T, device=device, dtype=dtype).contiguous()
w = torch.randn(D, K,    device=device, dtype=dtype).contiguous()
b = torch.randn(D,       device=device, dtype=dtype).contiguous()
print(f"shapes: x={tuple(x.shape)} w={tuple(w.shape)} b={tuple(b.shape)}")
sys.stdout.flush()
out = causal_conv1d_fn(x, w, b, activation="silu")
print(f"FORWARD OK: out.shape={tuple(out.shape)} dtype={out.dtype}")
print(f"first 4 values: {out.flatten()[:4].tolist()}")

# Probe 2: step / update mode (inference-style)
print("\n--- causal_conv1d_update(): batch=1 dim=512 kernel=4 bf16 ---")
x_step  = torch.randn(B, D,     device=device, dtype=dtype).contiguous()
state   = torch.randn(B, D, K-1, device=device, dtype=dtype).contiguous()
out_step = causal_conv1d_update(x_step, state, w, b, activation="silu")
print(f"UPDATE OK: out.shape={tuple(out_step.shape)} dtype={out_step.dtype}")

print("\n--- ALL CAUSAL_CONV1D PROBES SUCCEEDED ---")
