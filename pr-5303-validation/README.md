# PR #5303 validation on gfx1151 (Radeon 8060S / Strix Halo)

PR under test: [unslothai/unsloth#5303](https://github.com/unslothai/unsloth/pull/5303) — *feat(studio): use lemonade-sdk/llamacpp-rocm per-GPU prebuilts for ROCm hosts*

Three test passes documented here:

1. **2026-05-18 — manual asset-resolution + bench parity** at HEAD `39cf5d6`. Asset URL pattern works for gfx1151; bundle binaries run cleanly when `LD_LIBRARY_PATH` is set manually; bench parity vs hand-built reference. [Original PR comment](https://github.com/unslothai/unsloth/pull/5303#issuecomment-4485112930).
2. **2026-05-19 — full `./install.sh --local` end-to-end** at HEAD `76fe0912`. Install **fails** — runtime patterns miss the `librocm_kpack` / `librocm_sysdeps_*` family. 3-line patch proposed. [Follow-up PR comment](https://github.com/unslothai/unsloth/pull/5303#issuecomment-4487275873).
3. **2026-05-21 — re-test at `c41e20db`** after the patch from Pass 2 was adopted. That family resolves; preflight now fails one layer deeper on `libLLVM` / `libclang-cpp`. Recommend dropping the allowlist for `lib*.so*`. See Pass 3 below.

## Test rig (both passes)

- Corsair AI Workstation 300 — Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151) / 128 GiB unified
- Ubuntu 24.04 / kernel 6.19.14 mainline
- ROCm 7.1.0 system + ROCm 7.13 nightly in training venv (Studio install creates its own isolated venv at `~/.unsloth/studio/unsloth_studio`)
- Production venv at `/srv/aurora-ai/venv/` NOT touched in either pass
- Test model: `aurora-effects-v8-q8_0.gguf` — Qwen3.5-27B Q8_0, 26.62 GiB, 26.90 B params
- Bench flags: `-p 64 -n 16 -r 3 -mmp 0 -ngl 999` (no-mmap is the gfx1151 gotcha — mmap-only triggers ~30 min GPU page-table setup; documented in this guide's main README)
- Self-built reference (Pass 1): `/usr/local/bin/llama-bench` build `d0a6dfe (502)` with `-DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_NO_VMM=ON -DCMAKE_HIP_FLAGS='--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'`

---

## Pass 1 — 2026-05-18 — manual asset resolution + bench parity

The PR's `resolve_lemonade_rocm_choice()` would download and use `llama-{tag}-{os}-rocm-{gfxFamily}-x64.zip` for an AMD GPU whose `gcnArchName` matches one of the families in `_LEMONADE_GFX_FAMILIES`. For Strix Halo (gfx1151) that resolves to `llama-b1270-ubuntu-rocm-gfx1151-x64.zip` from the latest lemonade-sdk release.

Pulled the asset by hand, exercised it directly:

```
$ LD_LIBRARY_PATH=/tmp/llama-lemonade ./llama-bench -h
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 131072 MiB):
  Device 0: Radeon 8060S Graphics, gfx1151 (0x1151), VMM: no, Wave Size: 32, VRAM: 131072 MiB
```

- `Total VRAM: 131072 MiB` (128 GiB) — full Strix Halo unified pool reported correctly via HIP, NOT the 1 GiB dedicated slice `amd-smi` shows.
- `Wave Size: 32` — correct for RDNA 3.5 (gfx115X).
- `VMM: no` — expected on gfx1151.

Bench, 3 reps each:

| Build | `pp64` (tok/s) | `tg16` (tok/s) |
|---|---|---|
| Lemonade prebuilt b1270 (`39cf5d6`) | 253.96 ± 27.59 | 7.48 ± 0.05 |
| Self-built b502 (`d0a6dfe`) | 270.61 ± 2.07 | 7.52 ± 0.00 |

- **`tg16`**: 7.48 vs 7.52 tok/s — statistically identical, within noise.
- **`pp64`**: 253.96 vs 270.61 — apparent ~6% gap, but the lemonade σ (27.59) covers the entire delta (16.65). Cold-cache jitter, not a real regression.

Performance-equivalent to a fully-tuned hand build at the asset-resolution level.

Raw logs in this directory: `bench-aurora-v8-3rep.log`, `bench-selfbuilt-b502-3rep.log` (and the original 1-rep runs `bench-aurora-v8.log`, `bench-selfbuilt-b867.log` for the audit trail).

---

## Pass 2 — 2026-05-19 — full install.sh end-to-end at HEAD `76fe0912`

`./install.sh --local` against the same hardware **failed**.

**Lemonade prebuilt path** (which `748d59d4` was supposed to make end-to-end-working) fails preflight on extracted binary. Planner tried `b9204` first, walked back to `b9190` — both surfaced the same missing-libs profile:

```
preflight failed: llama-server: missing=
  librocm_kpack.so.0,
  librocm_sysdeps_elf.so.1,
  librocm_sysdeps_drm.so.2,
  librocm_sysdeps_drm_amdgpu.so.1,
  librocm_sysdeps_numa.so.1
```

**Source-build fallback** then dies in cmake (`'cstdlib' not found` on stock Ubuntu 24.04 — the same `--gcc-install-dir` issue that PR #5301's `bbf004c` fixes for `studio/setup.sh`; that fix isn't on this branch yet, blocked on #5301 merging).

### Root cause

`studio/install_llama_prebuilt.py::runtime_patterns_for_choice` for `linux-rocm` after `748d59d4` adds the obvious top-level HIP libs (`libamdhip64`, `libhsa-runtime64`, `libhipblas`, `librocblas`, etc.) but **does not match** the lemonade-bundled `librocm_kpack.so*` or the `librocm_sysdeps_*.so*` family.

`bundle-zip-listing.txt` shows those libs are present in the lemonade zip. `ldd-libamdhip64.txt` shows they are direct `NEEDED` entries of `libamdhip64.so.7` — not optional dlopens. So when `install_from_archives` walks `runtime_patterns_for_choice` to decide which files to overlay into staging, those families are skipped, and the preflight `LD_LIBRARY_PATH` walk fails on `libamdhip64`'s first transitive resolve.

### Fix

Append to the `linux-rocm` branch of `runtime_patterns_for_choice` at `install_llama_prebuilt.py:4169`:

```python
"librocm_kpack.so*",
"librocm_sysdeps_*.so*",
"libamd_comgr.so*",
```

`libamd_comgr.so*` is reached transitively via `librocm_sysdeps_elf.so.1`; including it keeps the closure self-contained against future bundle changes.

### Validation of the fix

Manually staged the FULL bundle (i.e. emulated what `install_from_archives` would do if the missing globs were added) into `~/.unsloth/llama.cpp.fix-test/`:

```bash
unzip llama-b1272-ubuntu-rocm-gfx1151-x64.zip -d ~/.unsloth/llama.cpp.fix-test/
chmod +x ~/.unsloth/llama.cpp.fix-test/llama-*
```

`ldd ./llama-server` against the fully-staged bundle (see `ldd-llama-server.txt`) — **0 lines with `not found`**, all transitive deps resolve to bundle-local libs.

`./llama-server --version` returns clean:

```
version: 1 (45b455e)
built with Clang 23.0.0 for Linux
```

### Bench parity vs Pass 1

Same model + same shape as the 2026-05-18 baseline:

| Test pass | Bundle | `pp64` (tok/s) | `tg16` (tok/s) |
|---|---|---|---|
| Pass 1 (manual `LD_LIBRARY_PATH`) | lemonade b1270 | 253.96 ± 27.59 | 7.48 ± 0.05 |
| **Pass 2 (fixed-pattern manual full stage)** | lemonade b1272 | **253.25 ± 27.28** | **7.48 ± 0.04** |

Statistically identical. Full output in `bench-after-manual-full-stage.log`.

So the only blocker between current `76fe0912` and a working install on gfx1151 is the patterns gap — the lemonade bundle itself runs at performance parity with a hand-tuned source build.

---

## Pass 3 — 2026-05-21 — re-test at `c41e20db` after the patterns fix

Leo adopted the 3-line patch from Pass 2 (`libamd_comgr.so*`, `librocm_kpack.so*`, `librocm_sysdeps_*.so*`). Re-ran `./install.sh --local` against `c41e20db`.

That family resolves now — but the prebuilt preflight fails one transitive layer deeper:

```
prebuilt fallback reason: linux extracted binary preflight failed:
llama-server: missing=libclang-cpp.so.23.0git,libLLVM.so.23.0git
```

`libamd_comgr.so.3` — which `c41e20db` correctly started overlaying — is itself built on LLVM and lists `libLLVM.so.23.0git` + `libclang-cpp.so.23.0git` as direct `NEEDED` entries (see `ldd-libamd_comgr.txt`). Both are in the bundle (130 MB + 89 MB); neither is matched by `runtime_patterns_for_choice`. Overlaying comgr without its LLVM backing just moves the gap down a level.

`overlay-simulation.py` copies only the files matching the verbatim `c41e20db` pattern list into a staging dir, then `ldd`s `llama-server` (`overlay-simulation-output.txt`):

```
c41e20db patterns (verbatim)      -> llama-server unresolved: 2  (libclang-cpp, libLLVM)
+ libLLVM.so* + libclang-cpp.so*  -> llama-server unresolved: 0
robust  lib*.so*  glob            -> llama-server unresolved: 0
```

### Recommended fix — drop the AMD allowlist for `linux-rocm`

This is the second transitive-dep layer (first `librocm_*`, now `libLLVM`/`libclang-cpp`). The lemonade bundle is self-contained by design — every `.so` in it is needed by something in the bundle — so an explicit allowlist will keep missing layers. Overlay `lib*.so*` instead:

```python
if choice.install_kind == "linux-rocm":
    patterns = ["llama-server", "llama-quantize", "lib*.so*"]
```

Safe for the upstream ggml-org tarball path too: that tarball links system `/opt/rocm` and ships only `libllama` / `libggml*`, so `lib*.so*` matches the same set there as the current explicit list.

---

## Pass 4 — 2026-05-26 — re-test at `9d9eeb27` after `lib*.so*` adopted

Leo adopted the broad-glob suggestion in commit `9d9eeb27` (`fix: use broad lib*.so* glob for linux-rocm runtime overlay`). Full `./install.sh --local --verbose` against that head.

### Result — clean lemonade install, end-to-end

Key lines from `install-9d9eeb27-verbose.log`:

```
[llama-prebuilt] AMD GPU 'gfx1151' (gfx1151) -- trying lemonade-sdk ROCm prebuilt llama-b1280-ubuntu-rocm-gfx1151-x64.zip
[llama-prebuilt] selected llama-b1280-ubuntu-rocm-gfx1151-x64.zip (lemonade) from published release b9334 for Linux x86_64
[llama-prebuilt] downloading llama-b1280-ubuntu-rocm-gfx1151-x64.zip from lemonade release (443.8 MiB)
[llama-prebuilt] staged prebuilt validation succeeded for llama-b1280-ubuntu-rocm-gfx1151-x64.zip
[llama-prebuilt] activated install tree confirmed at /home/auroraai/.unsloth/llama.cpp
  llama.cpp      prebuilt installed and validated
```

No preflight failure, no source-build fallback. The lemonade bundle is overlaid onto the upstream source tree and activated as the production install.

### Verification on the installed tree

- **80 `.so` files** present in `/home/auroraai/.unsloth/llama.cpp/build/bin/` (up from ~10 in the upstream-only path), including all previously-missing transitive deps: `libLLVM.so.23.0git`, `libclang-cpp.so.23.0git`, `libamd_comgr.so.3`, `librocm_kpack.so*`, `librocm_sysdeps_*`, `librocblas.so*`, `librocsolver.so.0`, `libroctx64.so.4`.
- **`ldd llama-server`: 0 missing libs**, every NEEDED entry resolves into `/home/auroraai/.unsloth/llama.cpp/build/bin/` (full output in `ldd-llama-server-9d9eeb27.txt`).
- **`llama-server --version`**: `version: 1 (35a74c8), built with Clang 23.0.0 for Linux` — runs cleanly.

### Resolved sequence across all four passes

| Pass | HEAD | Result | What changed |
|---|---|---|---|
| 1 | `39cf5d6` | manual staging works | asset resolution path correct |
| 2 | `76fe0912` | install fails | missing `librocm_kpack`, `librocm_sysdeps_*` |
| 3 | `c41e20db` | install fails one layer deeper | missing `libLLVM`, `libclang-cpp` |
| 4 | `9d9eeb27` | **install succeeds end-to-end** | `lib*.so*` glob covers all transitive deps |
| 5 | `d4d447c0` | install succeeds; summary line distinguishes source from binary origin | `binary_repo` + `binary_release_tag` recorded in `UNSLOTH_PREBUILT_INFO.json`; `setup.sh` summary shows `installed release: unslothai/llama.cpp@b9334 + lemonade@b1280` |

## Reproducing

```bash
# Pass 1 — manual asset resolution
curl -L -o /tmp/llama-lemonade.zip \
  "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/b1270/llama-b1270-ubuntu-rocm-gfx1151-x64.zip"
mkdir -p /tmp/llama-lemonade
unzip -q /tmp/llama-lemonade.zip -d /tmp/llama-lemonade
chmod +x /tmp/llama-lemonade/llama-bench
LD_LIBRARY_PATH=/tmp/llama-lemonade /tmp/llama-lemonade/llama-bench \
  -m <your-gguf>.gguf -p 64 -n 16 -r 3 -mmp 0 -ngl 999

# Pass 2 — install.sh against the PR branch
git clone --branch feature/lemonade-rocm-prebuilts \
  https://github.com/LeoBorcherding/unsloth.git
cd unsloth
./install.sh --local 2>&1 | tee install.log
# (will fail; see install-failure-full.log for what to expect)
```

## Files in this directory

| File | What it is |
|---|---|
| **Pass 1 (2026-05-18)** | |
| `bench-aurora-v8-3rep.log` | Lemonade b1270 prebuilt, 3 reps |
| `bench-aurora-v8.log` | Lemonade b1270 prebuilt, original 1-rep |
| `bench-selfbuilt-b502-3rep.log` | Self-built b502 reference, 3 reps |
| `bench-selfbuilt-b867.log` | Self-built b867 reference |
| **Pass 2 (2026-05-19)** | |
| `install-failure-full.log` | Full `./install.sh --local` output against `76fe0912` — both lemonade preflight failures and source-build fallback cmake failure |
| `bundle-zip-listing.txt` | `unzip -l llama-b1272-ubuntu-rocm-gfx1151-x64.zip` — proves the libs are in the bundle |
| `ldd-libamdhip64.txt` | Direct `NEEDED` entries of `libamdhip64.so.7` — proves the libs are runtime deps |
| `ldd-llama-server.txt` | Full transitive resolution from `llama-server` when all bundle libs are present — proves the fix works |
| `bench-after-manual-full-stage.log` | `llama-bench` raw output with fully-staged bundle on Qwen3.5-27B Q8_0 |
| **Pass 3 (2026-05-21)** | |
| `install-c41e20db.log` | `./install.sh --local` at `c41e20db` — preflight fails on `libLLVM` + `libclang-cpp` |
| `ldd-libamd_comgr.txt` | Direct `NEEDED` entries of `libamd_comgr.so` — shows it links libLLVM and libclang-cpp |
| `overlay-simulation.py` | Reproducible script: simulates `runtime_patterns_for_choice` overlays to find the minimal/sufficient pattern set |
| `overlay-simulation-output.txt` | Output: c41e20db patterns leave 2 unresolved, `+libLLVM/libclang-cpp` resolves all, broader `lib*.so*` resolves all |
| **Pass 4 (2026-05-26)** | |
| `install-9d9eeb27.log` | Non-verbose `./install.sh --local` at `9d9eeb27` — clean success |
| `install-9d9eeb27-verbose.log` | Verbose run with full `[llama-prebuilt]` trace showing lemonade b1280 selection + validation + activation |
| `ldd-llama-server-9d9eeb27.txt` | Full transitive resolution from activated `llama-server` — 0 missing |
| **Pass 5 (2026-05-26 late)** | |
| `install-d4d447c0-verbose.log` | Verbose run after the summary-line surfacing fix; same successful flow, new "+ lemonade@b1280" suffix on the install-complete line |
| `UNSLOTH_PREBUILT_INFO-d4d447c0.json` | Snapshot of the install metadata showing the new `binary_repo` + `binary_release_tag` fields populated for a lemonade install |
