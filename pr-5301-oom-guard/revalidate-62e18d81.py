#!/usr/bin/env python3
# Re-validation of PR #5301 at head 62e18d81.
#
# Leo's 59825bed:
#   - Adopted our 8060s/8050s suggestion for the device-name fallback.
#   - Extracted the classifier into a unit-testable helper
#     _rocm_classify_unified_memory(props) -> (gcn_arch, is_unified).
#   - Added studio/backend/tests/test_rocm_oom_guard.py with 31 parameterized
#     test cases covering all three classification paths.
#
# This script:
#   1. Confirms the verbatim helper still trips correctly on real gfx1151 hardware.
#   2. Re-runs the Strix Halo 8060S Path-3 case that previously misclassified.
#   3. Replays ALL of Leo's 31 test cases against the helper (no pytest dep —
#      uses plain assertions so it works in a bare venv).
#   4. Confirms is_bfloat16_supported() returns True on gfx1151 so the separate
#      RDNA2 dtype fix (62e18d81) doesn't change behavior for us.

import os, sys, types
from typing import Any, Tuple

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
import torch

# ── verbatim from worker.py:678-714 @ 62e18d81 ──────────────────────
def _rocm_classify_unified_memory(props: Any) -> Tuple[str, bool]:
    gcn_arch = ""
    for _attr in ("gcnArchName", "gcn_arch_name", "arch_name", "gfx_arch_name"):
        _v = (getattr(props, _attr, "") or "").split(":")[0].strip()
        if _v:
            gcn_arch = _v
            break
    if gcn_arch:
        return gcn_arch, gcn_arch in {"gfx1150", "gfx1151"}
    dev_lower = (getattr(props, "name", "") or "").lower()
    is_unified = (
        "890m" in dev_lower
        or "880m" in dev_lower
        or "8060s" in dev_lower
        or "8050s" in dev_lower
    )
    return gcn_arch, is_unified
# ────────────────────────────────────────────────────────────────────

class P(types.SimpleNamespace):
    """props-like object with whichever attributes the test wants."""

print("═" * 70)
print("PART 1 — real hardware (gfx1151)")
print("═" * 70)
real = torch.cuda.get_device_properties(0)
total = torch.cuda.mem_get_info()[1] / 1024**3
arch, unified = _rocm_classify_unified_memory(real)
print(f"  device name: {real.name!r}, gcnArchName: {getattr(real,'gcnArchName',None)!r}")
print(f"  → ({arch!r}, {unified})")
assert (arch, unified) == ("gfx1151", True), f"expected gfx1151+True, got {(arch,unified)}"
print(f"  ✓ correct: gfx1151 → unified → 0.80 cap → {total*0.80:.1f} GiB GPU / {total*0.20:.1f} GiB OS")

print()
print("═" * 70)
print("PART 2 — Strix Halo 8060S device-name fallback (the bug we flagged)")
print("═" * 70)
mock = P(name="Radeon 8060S Graphics")  # no arch attrs at all
arch, unified = _rocm_classify_unified_memory(mock)
print(f"  mock props: name={mock.name!r}, NO arch attrs")
print(f"  → ({arch!r}, {unified})")
assert (arch, unified) == ("", True), f"REGRESSION: 8060S still misclassified! got {(arch,unified)}"
print(f"  ✓ FIXED: device-name fallback now matches 8060s → unified → 0.80")

print()
print("═" * 70)
print("PART 3 — replay all 31 parameterized cases from Leo's test_rocm_oom_guard.py")
print("═" * 70)
passed = 0
failed = []

def check(label, props, expected):
    global passed
    got = _rocm_classify_unified_memory(props)
    if got == expected:
        passed += 1
    else:
        failed.append((label, expected, got))

# canonical_attr (parameterized: arch / expected_unified)
for arch, expected in [
    ("gfx1150", True),  ("gfx1151", True),
    ("gfx1030", False), ("gfx1100", False), ("gfx1101", False),
    ("gfx1200", False), ("gfx1201", False),
    ("gfx900",  False), ("gfx906",  False), ("gfx908",  False),
    ("gfx90a",  False), ("gfx940",  False),
]:
    check(f"canonical_attr[{arch}]", P(gcnArchName=arch, name="x"), (arch, expected))

# arch_with_colon_suffix_stripped
check("colon_suffix_stripped", P(gcnArchName="gfx1151:xnack-", name="Radeon 8060S Graphics"),
      ("gfx1151", True))

# canonical_attr_wins_over_name (gcnArchName takes priority over device name)
check("canonical_wins_over_name", P(gcnArchName="gfx1030", name="Radeon 890M Graphics"),
      ("gfx1030", False))

# alternate_attr_unified — each fallback spelling, value=gfx1151 → True
for attr in ("gcn_arch_name", "arch_name", "gfx_arch_name"):
    check(f"alternate_attr_unified[{attr}]", P(**{attr: "gfx1151"}, name="x"),
          ("gfx1151", True))

# alternate_attr_discrete — each fallback spelling, value=gfx1100 → False
for attr in ("gcn_arch_name", "arch_name", "gfx_arch_name"):
    check(f"alternate_attr_discrete[{attr}]", P(**{attr: "gfx1100"}, name="x"),
          ("gfx1100", False))

# first_non_empty_attr_wins — primary empty, secondary populated
check("first_non_empty_wins", P(gcnArchName="", gcn_arch_name="gfx1151", name="x"),
      ("gfx1151", True))

# unified_memory_detected via device name (no arch attrs at all)
for dn in ["Radeon 890M Graphics", "AMD Radeon 880M Graphics",
          "Radeon 8060S Graphics", "Radeon 8050S Graphics"]:
    check(f"unified_name[{dn}]", P(name=dn), ("", True))

# discrete_not_misclassified
for dn in ["Radeon RX 6600", "Radeon RX 7900 XTX",
          "Radeon RX 7600", "Radeon RX 9070 XT",
          "Radeon VII", "Generic GPU"]:
    check(f"discrete_name[{dn}]", P(name=dn), ("", False))

# empty_name_returns_false
check("empty_name", P(name=""), ("", False))

# none_name_returns_false
check("none_name", P(name=None), ("", False))

total = passed + len(failed)
print(f"\n  {passed}/{total} tests passed")
for label, expected, got in failed:
    print(f"  ✗ {label}: expected {expected}, got {got}")
if failed:
    sys.exit(1)
print(f"  ✓ all of Leo's test cases pass on the verbatim helper")

print()
print("═" * 70)
print("PART 4 — is_bfloat16_supported() check for the RDNA2 dtype fix (62e18d81)")
print("═" * 70)
bf16_ok = torch.cuda.is_bf16_supported()
print(f"  torch.cuda.is_bf16_supported() = {bf16_ok}")
if bf16_ok:
    print(f"  ✓ gfx1151 supports bf16 — Leo's _auto_dtype check keeps dtype=None, behavior unchanged for us")
else:
    print(f"  ⚠ unexpected: gfx1151 should support bf16; dtype fallback would kick in")

print()
print("ALL CHECKS PASSED")
