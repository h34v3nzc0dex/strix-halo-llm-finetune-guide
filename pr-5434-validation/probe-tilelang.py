"""
Test tilelang import + basic dispatch on gfx1151.
PR #5434 installs apache-tvm-ffi==0.1.9 + tilelang==0.1.8 for the Qwen3.5 family
to enable the FLA tilelang backend (cited benchmark: 1.43x faster step on B200).
Question: does tilelang even load on gfx1151? Does its backend selection work?
"""
import os, sys, faulthandler

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
sys.path.insert(0, "/tmp/tilelang-test")
faulthandler.enable()

import torch
print(f"torch {torch.__version__}  hip {torch.version.hip}  device 0: {torch.cuda.get_device_name(0)}")

print("\n--- import apache-tvm-ffi ---")
try:
    import tvm_ffi
    print(f"tvm_ffi:    {tvm_ffi.__file__}")
    import importlib.metadata as md
    print(f"version:    {md.version('apache-tvm-ffi')}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    sys.exit(2)

print("\n--- import tilelang ---")
try:
    import tilelang
    print(f"tilelang:   {tilelang.__file__}")
    print(f"version:    {md.version('tilelang')}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    sys.exit(2)

print("\n--- tilelang.target.determine_target() ---")
try:
    from tilelang.utils.target import determine_target
    target = determine_target()
    print(f"determined target: {target}")
except Exception as e:
    print(f"determine_target() FAILED: {type(e).__name__}: {e}")

print("\n--- tilelang HIP / ROCm backend availability ---")
# Probe a basic dispatch — does tilelang even compile a tiny kernel for gfx1151?
try:
    import tilelang.language as T

    @T.prim_func
    def add_kernel(
        A: T.Tensor((128,), "float32"),
        B: T.Tensor((128,), "float32"),
        C: T.Tensor((128,), "float32"),
    ):
        with T.Kernel(1, threads=128) as bx:
            tid = T.get_thread_binding(0)
            C[tid] = A[tid] + B[tid]

    print("trying to JIT-compile a tiny add kernel for the current device...")
    sys.stdout.flush()
    jit_kernel = tilelang.compile(add_kernel)
    print(f"COMPILE OK: {jit_kernel}")

    a = torch.ones(128, device="cuda", dtype=torch.float32)
    b = torch.full((128,), 2.0, device="cuda", dtype=torch.float32)
    c = torch.zeros(128, device="cuda", dtype=torch.float32)
    jit_kernel(a, b, c)
    print(f"RUN OK: c[:4] = {c[:4].tolist()}  (expecting [3, 3, 3, 3])")
except Exception as e:
    print(f"tilelang dispatch FAILED: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc(limit=10)
