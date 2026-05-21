#!/usr/bin/env python3
# Faithful reproduction of the PR #5301 OOM-guard classifier (worker.py:2278-2298
# at head 32457939) run on real gfx1151 Strix Halo hardware.
import os, re
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
import torch

props = torch.cuda.get_device_properties(0)

# ── verbatim from worker.py:2287-2289 ──
_dev_name = props.name
_is_unified = bool(re.search(r"\d[Mm]\b", _dev_name))
_mem_fraction = 0.80 if _is_unified else 0.90
# ───────────────────────────────────────

free_b, total_b = torch.cuda.mem_get_info()
total_gib = total_b / 1024**3
cap_gib = total_gib * _mem_fraction
host_left_gib = total_gib - cap_gib

print(f"device name (torch)        : {_dev_name!r}")
print(f"gcnArchName                : {getattr(props, 'gcnArchName', 'n/a')!r}")
print(f"torch total_memory         : {total_gib:.1f} GiB  (unified pool — = system RAM)")
print()
print(f"PR #5301 classifier result : _is_unified={_is_unified}  -> fraction={_mem_fraction}")
print(f"  -> GPU allocator cap     : {cap_gib:.1f} GiB")
print(f"  -> left for OS/page-cache: {host_left_gib:.1f} GiB")
print()
if not _is_unified:
    print("VERDICT: MISCLASSIFIED. Radeon 8060S is a Strix Halo unified APU but the")
    print("         digit+M regex misses the 'S' suffix, so the guard applies 0.90.")
    print(f"        0.90 of a {total_gib:.0f} GiB unified pool leaves only {host_left_gib:.1f} GiB for the")
    print("         entire OS — the host-starvation freeze the guard exists to prevent.")
else:
    print("VERDICT: classified unified — OK")
print()

# ── proposed fix: classify on gcnArchName ──
arch = getattr(props, "gcnArchName", "") or ""
arch_base = arch.split(":")[0]   # strip xnack/sram-ecc feature flags
fix_unified = arch_base in {"gfx1150", "gfx1151"}
fix_fraction = 0.80 if fix_unified else 0.90
print(f"proposed gcnArchName fix   : arch_base={arch_base!r} -> _is_unified={fix_unified} -> fraction={fix_fraction}")
print(f"  -> GPU allocator cap     : {total_gib*fix_fraction:.1f} GiB   (correct)")

# confirm the cap call itself is accepted on this hardware
torch.cuda.set_per_process_memory_fraction(fix_fraction)
print(f"  -> set_per_process_memory_fraction({fix_fraction}) accepted on gfx1151")
