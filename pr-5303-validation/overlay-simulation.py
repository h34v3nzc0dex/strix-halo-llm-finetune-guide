#!/usr/bin/env python3
# Simulates studio/install_llama_prebuilt.py::install_from_archives' runtime
# overlay for a lemonade ROCm bundle: copies only the files matching
# runtime_patterns_for_choice("linux-rocm") into a staging dir, then ldd's
# llama-server to see which transitive deps are left unresolved.
#
# Usage:  overlay-simulation.py <path-to-extracted-lemonade-bundle>
#
# Compares three pattern sets:
#   1. the verbatim list at PR #5303 head c41e20db
#   2. that list + libLLVM.so* + libclang-cpp.so*   (minimal fix)
#   3. a single robust  lib*.so*  glob               (recommended fix)
import os, sys, fnmatch, shutil, subprocess, tempfile

src = sys.argv[1] if len(sys.argv) > 1 else "./extracted"
if not os.path.isdir(src):
    sys.exit(f"not a directory: {src}  (pass the extracted bundle path)")

# verbatim linux-rocm pattern list from install_llama_prebuilt.py @ c41e20db
C41E20DB = [
    "llama-server", "llama-quantize",
    "libllama-common.so*", "libllama.so*", "libggml.so*", "libggml-base.so*",
    "libmtmd.so*", "libggml-cpu*.so*", "libggml-cuda.so*", "libggml-hip.so*",
    "libggml-rpc.so*",
    "libamdhip64.so*", "libhsa-runtime64.so*", "libhipblas.so*", "libhipblaslt.so*",
    "librocblas.so*", "librocsolver.so*", "librocsparse.so*", "librocrand.so*",
    "libMIOpen.so*", "libmagma.so*",
    "libamd_comgr.so*", "librocm_kpack.so*", "librocm_sysdeps_*.so*",
]

def stage_and_ldd(patterns):
    dst = tempfile.mkdtemp(prefix="overlay-")
    for f in os.listdir(src):
        if any(fnmatch.fnmatch(f, p) for p in patterns):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    os.chmod(os.path.join(dst, "llama-server"), 0o755)
    r = subprocess.run(["ldd", "llama-server"], cwd=dst, text=True,
                       capture_output=True, env={**os.environ, "LD_LIBRARY_PATH": dst})
    return [l.strip() for l in r.stdout.splitlines() if "not found" in l]

for label, pats in [
    ("c41e20db patterns (verbatim)        ", C41E20DB),
    ("+ libLLVM.so* + libclang-cpp.so*    ", C41E20DB + ["libLLVM.so*", "libclang-cpp.so*"]),
    ("robust  lib*.so*  glob              ", ["llama-server", "llama-quantize", "lib*.so*"]),
]:
    missing = stage_and_ldd(pats)
    print(f"{label} -> llama-server unresolved: {len(missing)}")
    for m in missing:
        print(f"    {m}")
