#!/usr/bin/env python3
# Re-validation of the PR #5301 OOM-guard classifier at head 80dd40e6.
# Leo expanded the classifier to a layered approach:
#   Path 1: try `gcnArchName` and alternate attribute spellings → match gfx1150/gfx1151
#   Path 2: (no separate path — this is just the loop over attribute spellings)
#   Path 3: if NO arch attr populated → device-name string fallback for 890m / 880m
#
# This script tests all three on real gfx1151 hardware:
#   1) Canonical path — runs the classifier verbatim, expects unified=True
#   2) Alternate-spelling path — mocks gcnArchName missing, expects loop to find
#      one of the alternate spellings if present (likely none on this stack)
#   3) Device-name fallback — mocks ALL arch attributes missing, exposes whether
#      the 890m/880m string match works on Strix Halo's "Radeon 8060S Graphics"
#
# All three are run side-by-side against Leo's exact code block.
import os, types
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
import torch

# ── verbatim from worker.py:2314-2342 @ 80dd40e6 ─────────────────────
def classify_unified(props_like):
    """Runs Leo's classifier block against any props-like object."""
    _dev_name = props_like.name
    _gcn_arch = ""
    for _arch_attr in (
        "gcnArchName",
        "gcn_arch_name",
        "arch_name",
        "gfx_arch_name",
    ):
        _v = (getattr(props_like, _arch_attr, "") or "").split(":")[0].strip()
        if _v:
            _gcn_arch = _v
            break
    _is_unified = _gcn_arch in {"gfx1150", "gfx1151"}
    fallback_fired = False
    if not _is_unified and not _gcn_arch:
        _dev_lower = _dev_name.lower()
        _is_unified = "890m" in _dev_lower or "880m" in _dev_lower
        fallback_fired = True
    _mem_fraction = 0.80 if _is_unified else 0.90
    return _is_unified, _mem_fraction, _gcn_arch, _dev_name, fallback_fired
# ─────────────────────────────────────────────────────────────────────

real_props = torch.cuda.get_device_properties(0)
total_gib = torch.cuda.mem_get_info()[1] / 1024**3

print("═" * 70)
print(f"Hardware: {real_props.name}, gcnArchName='{getattr(real_props, 'gcnArchName', None)}', total_mem={total_gib:.1f} GiB")
print("═" * 70)

# Test 1: canonical path — pass real props
print("\n[Test 1] Canonical path (real torch props, gcnArchName populated)")
unified, frac, arch, name, fb = classify_unified(real_props)
print(f"  _gcn_arch         : {arch!r}")
print(f"  _is_unified       : {unified}")
print(f"  fallback fired?   : {fb}")
print(f"  fraction          : {frac}")
print(f"  GPU cap           : {total_gib * frac:.1f} GiB")
print(f"  OS headroom       : {total_gib * (1-frac):.1f} GiB")
print(f"  RESULT            : {'✓ correct (Strix Halo → unified → 0.80)' if unified and frac == 0.80 else '✗ WRONG'}")

# Test 2: alternate-spelling path — clear gcnArchName, see if alternates exist
print("\n[Test 2] Alternate-spelling path (mock gcnArchName missing)")
print(f"  attributes available on real props that match the alternate names:")
for attr in ("gcn_arch_name", "arch_name", "gfx_arch_name"):
    present = hasattr(real_props, attr)
    val = getattr(real_props, attr, None) if present else None
    print(f"    {attr}: present={present}, value={val!r}")

class Mock1(types.SimpleNamespace):
    pass
mock1 = Mock1(name=real_props.name)
# Don't define gcnArchName, but DO define gcn_arch_name to test second attr in loop
mock1.gcn_arch_name = getattr(real_props, "gcnArchName", "")
unified, frac, arch, name, fb = classify_unified(mock1)
print(f"  → with gcn_arch_name='{mock1.gcn_arch_name}' set:")
print(f"    _gcn_arch       : {arch!r}")
print(f"    fallback fired? : {fb}")
print(f"    RESULT          : {'✓ alternate-spelling loop works' if unified and not fb else '✗ FAILED'}")

# Test 3: device-name fallback — clear ALL arch attributes, force fallback
print("\n[Test 3] Device-name fallback path (mock ALL arch attrs missing)")
class Mock2(types.SimpleNamespace):
    pass
mock2 = Mock2(name=real_props.name)  # only `name` attribute defined
unified, frac, arch, name, fb = classify_unified(mock2)
print(f"  device name passed to fallback : {name!r}")
print(f"  _gcn_arch                      : {arch!r} (empty → fallback engaged)")
print(f"  fallback fired?                : {fb}")
print(f"  _is_unified                    : {unified}")
print(f"  fraction                       : {frac}")
print(f"  GPU cap                        : {total_gib * frac:.1f} GiB")
print(f"  OS headroom                    : {total_gib * (1-frac):.1f} GiB")
if unified:
    print(f"  RESULT                         : ✓ fallback recognized Strix Halo")
else:
    print(f"  RESULT                         : ✗ FALLBACK MISCLASSIFIES STRIX HALO AS DISCRETE")
    print(f"    Why: device name {name!r} does not contain '890m' or '880m'.")
    print(f"    890M / 880M are Strix POINT marketing names (gfx1150).")
    print(f"    Strix HALO (gfx1151) ships as Radeon 8060S / 8050S.")
    print(f"    Fix needed: extend fallback to also match '8060s' and '8050s'.")
    print(f"    Impact: 0.90 cap = {total_gib * 0.90:.1f} GiB → {total_gib * 0.10:.1f} GiB OS headroom (will freeze a 128 GB box).")

# Test 4: prove the fix works for Strix Point (890M) just to confirm Leo's intent is right
print("\n[Test 4] Sanity check — Strix Point (890M) hits the existing fallback correctly")
class MockSP(types.SimpleNamespace):
    pass
mock_sp = MockSP(name="AMD Radeon 890M Graphics")
unified, frac, arch, name, fb = classify_unified(mock_sp)
print(f"  device name        : {name!r}")
print(f"  fallback fired?    : {fb}")
print(f"  _is_unified        : {unified}")
print(f"  RESULT             : {'✓ Strix Point fallback works' if unified else '✗ broken for Strix Point too'}")
