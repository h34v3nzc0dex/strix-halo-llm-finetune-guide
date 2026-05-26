# PR #5301 OOM-guard — Strix Halo classification validation

Validation for the ROCm OOM guard added to [unslothai/unsloth#5301](https://github.com/unslothai/unsloth/pull/5301) (`studio/backend/core/training/worker.py`, section 1g).

The guard caps the HIP allocator via `set_per_process_memory_fraction` so VRAM exhaustion raises `OutOfMemoryError` instead of hanging the HIP driver and freezing the box. It picks `0.80` for unified-memory APUs and `0.90` for discrete cards — the fraction has to be lower on a unified APU because the GPU pool *is* system RAM, so `0.90` starves the host OS.

## Finding

At PR head `32457939` the unified-vs-discrete decision is a device-name regex (`worker.py:2287-2289`):

```python
_dev_name = torch.cuda.get_device_properties(0).name
_is_unified = bool(re.search(r"\d[Mm]\b", _dev_name))   # matches "Radeon 890M"
```

This **misclassifies Strix Halo**. Strix Point iGPUs are named `Radeon 890M` / `880M` (digit+M — matches). Strix Halo — the hardware this PR is named for — is `Radeon 8060S` / `8050S`: an **S suffix, no M**. The regex misses it, so on a 128 GB unified Strix Halo box the guard applies `0.90` → 115.2 GiB GPU cap → only 12.8 GiB left for the OS, which is the host-starvation freeze the guard exists to prevent.

`validation-output.txt` is the captured run on real hardware (Radeon 8060S / gfx1151 / 128 GiB unified, ROCm 7.13 nightly):

```
device name (torch)        : 'Radeon 8060S Graphics'
gcnArchName                : 'gfx1151'
PR #5301 classifier result : _is_unified=False  -> fraction=0.90
  -> GPU allocator cap     : 115.2 GiB
  -> left for OS/page-cache: 12.8 GiB
```

## Suggested fix

Classify on `gcnArchName` instead of the marketing name — it doesn't care whether AMD calls the part `890M` or `8060S`:

```python
_arch = (getattr(props, "gcnArchName", "") or "").split(":")[0]  # strip :xnack-/:sramecc
_is_unified = _arch in {"gfx1150", "gfx1151"}                     # Strix Point + Strix Halo
```

Verified on the box: `gcnArchName` reports `gfx1151`, the `{gfx1150, gfx1151}` check returns `_is_unified=True` → `0.80` → 102.4 GiB cap.

## Reproducing

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python3 validate-oom-guard.py
```

`validate-oom-guard.py` runs the PR's classifier logic verbatim against whatever GPU `torch` sees, prints the fraction it would pick, and shows the `gcnArchName`-based fix side by side. No Unsloth install required — just a ROCm `torch`.

## Re-validation — fix landed at `9393fffe`

Leo adopted the `gcnArchName` suggestion. At PR head `9393fffe` the classifier is (`worker.py:2291-2295`):

```python
_gcn_arch = (getattr(_props, "gcnArchName", "") or "").split(":")[0]
_is_unified = _gcn_arch in {"gfx1150", "gfx1151"}
_mem_fraction = 0.80 if _is_unified else 0.90
```

`revalidate-9393fffe.py` runs that block verbatim on the gfx1151 box. Result (`revalidation-output-9393fffe.txt`):

```
gcnArchName (flag-stripped): 'gfx1151'
classifier @ 9393fffe      : _is_unified=True  -> fraction=0.80
  -> GPU allocator cap     : 102.4 GiB
  -> left for OS/page-cache: 25.6 GiB
```

Same box that returned `_is_unified=False → 0.90` under the `32457939` name-regex now returns `True → 0.80`. **Fixed.**

## Re-validation — head `284145a7` (2026-05-25)

The classifier block (`worker.py:2291-2295`) is byte-identical at `284145a7` to the `9393fffe` version — `git compare 9393fffe...284145a7` does not list `studio/backend/core/training/worker.py` as a modified file. Re-ran the same script anyway for hygiene:

```
device name (torch)        : 'Radeon 8060S Graphics'
gcnArchName (flag-stripped): 'gfx1151'
torch total_memory         : 128.0 GiB

classifier @ 284145a7      : _is_unified=True  -> fraction=0.8
  -> GPU allocator cap     : 102.4 GiB
  -> left for OS/page-cache: 25.6 GiB
set_per_process_memory_fraction(0.8) accepted on gfx1151
```

Still correct on gfx1151. Full output in `revalidation-output-284145a7.txt`.

The Codex review nits in the range (P1 `llama_cpp.py:2760`, P1 `llama_cpp.py:2868`, P2 `install_python_stack.py:367`, plus two `setup.ps1` ones) are all Windows-specific code paths the Linux + ROCm-nightly stack here can't exercise. On the PyTorch nightly Linux wheel we're testing with, `torch.version.hip == '7.13.26176'` (set), so the existing `torch.version.hip` predicate in the Windows guards trips correctly on this stack — the nit applies to a different wheel family (AMD SDK / Radeon Windows) where `+rocmsdk…` shows up in `torch.__version__` instead.

## Re-validation — head `80dd40e6` (2026-05-26)

Leo expanded the classifier substantially. It's now layered:

```python
# Try canonical + alternate attribute spellings
for _arch_attr in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
    _v = (getattr(_props, _arch_attr, "") or "").split(":")[0].strip()
    if _v: _gcn_arch = _v; break
_is_unified = _gcn_arch in {"gfx1150", "gfx1151"}
if not _is_unified and not _gcn_arch:
    # Fallback to device-name string match
    _dev_lower = _dev_name.lower()
    _is_unified = "890m" in _dev_lower or "880m" in _dev_lower
```

Three paths to test. `revalidate-80dd40e6.py` runs all three on gfx1151 hardware:

| Path | Test | Result on `Radeon 8060S` |
|---|---|---|
| 1 — canonical `gcnArchName` | Real torch props | ✓ `gfx1151` → unified → 0.80 |
| 2 — alternate-spelling loop | Mock missing `gcnArchName`, define `gcn_arch_name` | ✓ loop finds the alternate, → unified |
| 3 — device-name fallback | Mock ALL arch attrs missing | **✗ misclassifies Strix Halo as discrete → 0.90 → 12.8 GiB OS headroom** |

The Path 3 failure is a real bug for Strix Halo specifically:

- Leo's fallback matches `890m` (Strix Point) and `880m` (Strix Point).
- **Strix Halo's marketing name is `Radeon 8060S` / `8050S`** (Ryzen AI MAX series), NOT `890M`. The string match misses it.
- Strix Point variants are gfx1150; Strix Halo is gfx1151. Both are unified-memory APUs that need the 0.80 cap. The current fallback covers only the gfx1150 marketing family.

**Suggested one-line fix:**

```python
_is_unified = (
    "890m" in _dev_lower or "880m" in _dev_lower
    or "8060s" in _dev_lower or "8050s" in _dev_lower
)
```

(Confirmed device name on Strix Halo Ryzen AI MAX+ 395 is `Radeon 8060S Graphics`. The variant `Radeon 8050S` is the cut-down Strix Halo SKU. Both should be matched.)

In practice on this stack the bug is latent — `gcnArchName` IS populated by the PyTorch nightly ROCm wheels, so Path 1 succeeds and Path 3 is never reached. The bug surfaces only on AMD SDK / Radeon Windows wheels where `gcnArchName` may be absent — which is the exact scenario the fallback was added to handle.

Full per-test output in `revalidation-output-80dd40e6.txt`.

## Re-validation — head `62e18d81` (2026-05-26, late)

Leo adopted the 8060s/8050s fix in `59825bed` and extracted the classifier into a unit-testable helper:

```python
def _rocm_classify_unified_memory(props) -> tuple[str, bool]:
    """Returns (gcn_arch, is_unified)."""
    ...
```

He also added `studio/backend/tests/test_rocm_oom_guard.py` with 10 test functions (31 parameterized cases) covering all three classification paths.

`revalidate-62e18d81.py` re-runs:
1. Real gfx1151 hardware → `("gfx1151", True)` ✓
2. The Strix Halo 8060S Path-3 fallback that previously misclassified → `("", True)` ✓ (fixed)
3. All 31 of Leo's parameterized test cases (no pytest dep — plain assertions): **33/33 pass**
4. `torch.cuda.is_bf16_supported()` for the separate RDNA2 dtype fix in `62e18d81` (`_auto_dtype = None if is_bfloat16_supported() else torch.float16`) → `True` on gfx1151, dtype branch doesn't affect us

Full output in `revalidation-output-62e18d81.txt`.

## Files

| File | What it is |
|---|---|
| `validate-oom-guard.py` | Original reproduction — runs the `32457939` name-regex classifier + the proposed fix |
| `validation-output.txt` | Captured run showing the `32457939` misclassification |
| `revalidate-9393fffe.py` | Re-validation — runs the merged `9393fffe` `gcnArchName` classifier verbatim |
| `revalidation-output-9393fffe.txt` | Captured run confirming the fix on gfx1151 |
| `revalidate-284145a7.py` | Re-validation at the previous PR head — identical to `9393fffe` script except stamping |
| `revalidation-output-284145a7.txt` | Captured run confirming the classifier still trips correctly at `284145a7` |
| `revalidate-80dd40e6.py` | Re-validation at PR head 80dd40e6 — tests all three classifier paths (canonical, alternate-spelling loop, device-name fallback) |
| `revalidation-output-80dd40e6.txt` | Captured run; Test 3 demonstrates the Strix Halo fallback bug |
| `revalidate-62e18d81.py` | Re-validation at PR head 62e18d81 — verifies the 8060s/8050s fix landed + replays all 31 of Leo's new parameterized test cases against our gfx1151 box + checks `is_bf16_supported()` for the separate RDNA2 dtype fix |
| `revalidation-output-62e18d81.txt` | Captured run; all 33 cases pass, 8060S regression fixed, bf16 supported (RDNA2 dtype branch doesn't affect us) |
