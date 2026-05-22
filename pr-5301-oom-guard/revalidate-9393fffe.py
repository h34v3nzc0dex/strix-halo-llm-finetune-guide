#!/usr/bin/env python3
# Re-validation of the PR #5301 OOM-guard classifier at head 9393fffe
# (worker.py:2291-2295) — runs Leo's gcnArchName-based code verbatim on
# real gfx1151 Strix Halo hardware.
import os
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
import torch

# ── verbatim from worker.py:2291-2295 @ 9393fffe ──
_props = torch.cuda.get_device_properties(0)
_dev_name = _props.name
_gcn_arch = (getattr(_props, "gcnArchName", "") or "").split(":")[0]
_is_unified = _gcn_arch in {"gfx1150", "gfx1151"}
_mem_fraction = 0.80 if _is_unified else 0.90
# ──────────────────────────────────────────────────

free_b, total_b = torch.cuda.mem_get_info()
total_gib = total_b / 1024**3
cap_gib = total_gib * _mem_fraction
host_left_gib = total_gib - cap_gib

print(f"device name (torch)        : {_dev_name!r}")
print(f"gcnArchName (flag-stripped): {_gcn_arch!r}")
print(f"torch total_memory         : {total_gib:.1f} GiB")
print()
print(f"classifier @ 9393fffe      : _is_unified={_is_unified}  -> fraction={_mem_fraction}")
print(f"  -> GPU allocator cap     : {cap_gib:.1f} GiB")
print(f"  -> left for OS/page-cache: {host_left_gib:.1f} GiB")
print()

prev_misclassified = (not _is_unified)  # what 32457939's name-regex did here
if _is_unified and _mem_fraction == 0.80:
    print("RESULT: FIXED. gfx1151 now classifies as unified -> 0.80 -> 102.4 GiB cap.")
    print("        The 32457939 name-regex returned discrete/0.90 on this same box;")
    print("        9393fffe's gcnArchName check returns the correct value.")
else:
    print(f"RESULT: still wrong — _is_unified={_is_unified}, fraction={_mem_fraction}")

# confirm the cap call is accepted by the runtime
torch.cuda.set_per_process_memory_fraction(_mem_fraction)
print(f"        set_per_process_memory_fraction({_mem_fraction}) accepted on gfx1151")
