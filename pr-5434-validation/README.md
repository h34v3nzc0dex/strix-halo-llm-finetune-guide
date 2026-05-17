# PR #5434 validation on gfx1151

Reproductions for the comment at <https://github.com/unslothai/unsloth/pull/5434> —
validation of *"studio: install flash-linear-attention and tilelang for Qwen3.5
family"* on Strix Halo (`gfx1151`, Radeon 8060S, 128 GB unified memory).

## Hardware / software

- AMD Ryzen AI MAX+ 395, Radeon 8060S (`gfx1151`), 128 GB unified
- Ubuntu 24.04, kernel 6.19.14 mainline
- ROCm 7.13 nightly (native `gfx1151`)
- `torch==2.11.0+rocm7.13.0a20260506` / `triton==3.6.0` / HIP `7.13.26176`
- `unsloth_zoo==2026.5.1`, `flash-linear-attention==0.5.0`, `causal-conv1d==1.6.2.post1`,
  `tilelang==0.1.8`, `apache-tvm-ffi==0.1.9`

## Scripts (run with the production venv, override-loading vanilla PyPI installs from `/tmp`)

| Script | Question it answers | Result |
|---|---|---|
| `probe-fla-gdn.py` | Does vanilla FLA 0.5.0 GDN run on gfx1151 at a small shape? | ✅ |
| `probe-fla-gdn-large.py` | …at production scale (T=8192 H=16 K=128 + variants)? | ✅ all three shapes fwd+bwd |
| `probe-causal-conv1d.py` | Does causal-conv1d 1.6.2.post1 run on gfx1151? | ✅ runtime; build needs `HIPCC_COMPILE_FLAGS_APPEND='--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'` |
| `probe-tilelang.py` | Does tilelang 0.1.8 load + compile a trivial kernel on HIP? | ⚠️ imports + targets `hip` but `tilelang.compile()` raises `ValueError: Cannot find global function target.build.tilelang_hip_without_compile` |
| `probe-fla-with-tilelang.py` | With both FLA and tilelang installed (PR #5434's state), does Qwen3.5 GDN backward work? | ❌ **crashes** with `tvm.error.InternalError: Unsupported target for gemm: hip … -mcpu=gfx1151` — running with `FLA_TILELANG=0` makes it succeed |

Each script enables `faulthandler` pre-import and runs subprocess-isolated so SIGSEGVs would leave a clear trace.

## Logs

`probe-*.log` next to each script — full stdout/stderr from the run that produced the
result table above. Captured 2026-05-16.

## How to re-run

```bash
# Install isolated vanilla copies (NOT into your production venv)
mkdir /tmp/fla-vanilla /tmp/causal-conv1d-test /tmp/tilelang-test
pip install --target /tmp/fla-vanilla       --no-deps "flash-linear-attention" "fla-core"

HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13" \
  pip install --target /tmp/causal-conv1d-test --no-deps --no-build-isolation causal-conv1d

pip install --target /tmp/tilelang-test     --no-deps "apache-tvm-ffi==0.1.9" "tilelang==0.1.8" cloudpickle
sudo apt-get install -y libz3-4
sudo ln -sf /usr/lib/x86_64-linux-gnu/libz3.so.4 /usr/lib/x86_64-linux-gnu/libz3.so

# Run probes
PYTORCH_HIP_ALLOC_CONF=expandable_segments:True TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  python probe-fla-gdn-large.py
# ...etc per script
```
