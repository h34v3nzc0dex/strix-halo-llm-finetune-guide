#!/usr/bin/env python3
"""Re-apply the gfx1151 patches to a fresh flash-linear-attention checkout.

Run this:
  - After a fresh `git clone` of fla-org/flash-linear-attention
  - After every `git pull` upstream
  - After bumping FLA to a new release

Two patches are applied in-place:

  1. Cap `num_warps` at 4 across all .py files in fla/. RDNA 3 / 3.5 cores
     can't run num_warps > 4 reliably under Triton 3.6 — kernels assert
     or produce garbage. See triton-lang/triton#5609.

  2. Filter `NUM_WARPS_AUTOTUNE = [...]` lists so only values <=4 remain.
     Same root cause as #1.

  3. Replace fla/ops/utils/cumsum.py with a pure-PyTorch wrapper. The
     Triton cumsum kernel hits a tl.cumsum + tl.sum codegen bug on
     gfx1151. PyTorch's torch.cumsum is fast enough that the perf hit
     is negligible. See triton-lang/triton#3017.

Patch 3 needs a backup of the patched cumsum.py — keep one alongside
this script as cumsum-pytorch.py and pass --cumsum-backup if you want
that step performed.

After running, clear the Triton autotune cache:
  rm -rf ~/.triton/cache

And re-install fla in your training venv:
  pip install -e /path/to/fla-patched
"""
import argparse
import pathlib
import re
import shutil
import sys

NUM_WARPS_RE = re.compile(r"num_warps\s*=\s*(\d+)")
LIST_RE = re.compile(r"\[\s*([0-9 ,]+)\s*\]")


def cap_num_warps(text: str) -> tuple[str, int]:
    n = 0

    def repl(m: re.Match) -> str:
        nonlocal n
        v = int(m.group(1))
        if v > 4:
            n += 1
            return "num_warps=4"
        return m.group(0)

    return NUM_WARPS_RE.sub(repl, text), n


def cap_warps_autotune_lists(text: str) -> tuple[str, int]:
    n = 0

    def repl_assign(m: re.Match) -> str:
        nonlocal n
        rhs = m.group(2)

        def repl_list(lm: re.Match) -> str:
            nums = [int(x) for x in re.findall(r"\d+", lm.group(1))]
            kept = [v for v in nums if v <= 4]
            return "[" + ", ".join(str(v) for v in kept) + "]"

        new_rhs = LIST_RE.sub(repl_list, rhs)
        if new_rhs != rhs:
            n += 1
        return m.group(1) + new_rhs

    pat = re.compile(r"(NUM_WARPS_AUTOTUNE\s*=\s*)(.+)$", re.MULTILINE)
    return pat.sub(repl_assign, text), n


def main():
    parser = argparse.ArgumentParser(description="Re-apply gfx1151 patches to flash-linear-attention")
    parser.add_argument("--fla-root", required=True,
                        help="Path to FLA checkout (contains fla/ subdir)")
    parser.add_argument("--cumsum-backup", default=None,
                        help="Path to a known-good patched cumsum.py to copy in. "
                             "Skip the cumsum patch if not provided.")
    args = parser.parse_args()

    fla_pkg = pathlib.Path(args.fla_root) / "fla"
    if not fla_pkg.is_dir():
        print(f"ERROR: {fla_pkg} doesn't exist", file=sys.stderr)
        sys.exit(1)

    total_files = 0
    files_changed = 0
    total_nw = 0
    total_lists = 0

    for p in sorted(fla_pkg.rglob("*.py")):
        total_files += 1
        text = p.read_text()
        new, c1 = cap_num_warps(text)
        new, c2 = cap_warps_autotune_lists(new)
        if new != text:
            files_changed += 1
            total_nw += c1
            total_lists += c2
            p.write_text(new)

    print(f"Scanned {total_files} .py files, modified {files_changed}")
    print(f"  num_warps caps: {total_nw}")
    print(f"  NUM_WARPS_AUTOTUNE list filters: {total_lists}")

    if args.cumsum_backup:
        backup = pathlib.Path(args.cumsum_backup)
        target = fla_pkg / "ops" / "utils" / "cumsum.py"
        if not backup.is_file():
            print(f"ERROR: backup {backup} doesn't exist", file=sys.stderr)
            sys.exit(1)
        shutil.copy(backup, target)
        print(f"Copied patched cumsum.py: {backup} -> {target}")
    else:
        print("Skipped cumsum.py replacement (no --cumsum-backup provided).")
        print("  WARNING: without that patch, Triton's tl.cumsum codegen will")
        print("           crash on gfx1151. See triton-lang/triton#3017.")


if __name__ == "__main__":
    main()
