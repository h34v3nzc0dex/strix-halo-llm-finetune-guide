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

## Files

| File | What it is |
|---|---|
| `validate-oom-guard.py` | Original reproduction — runs the `32457939` name-regex classifier + the proposed fix |
| `validation-output.txt` | Captured run showing the `32457939` misclassification |
| `revalidate-9393fffe.py` | Re-validation — runs the merged `9393fffe` `gcnArchName` classifier verbatim |
| `revalidation-output-9393fffe.txt` | Captured run confirming the fix on gfx1151 |
