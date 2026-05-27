#!/usr/bin/env python3
"""Generate the paper's figures from the raw bench logs.

Outputs three PDFs in ./figures/:
  fig1-vulkan-vs-rocm-q4.pdf       — Q4_K_M tg128 across depths, Vulkan vs ROCm
  fig2-vulkan-vs-rocm-bf16.pdf     — BF16 tg128 across depths, Vulkan vs ROCm  (the inversion)
  fig3-rocwmma-fattn-onoff.pdf     — ROCWMMA_FATTN ON vs OFF on dense + MoE Q4

Bench log format: pipe-separated rows, columns include backend, ngl, fa, mmap, test, t/s.
Depth-sweep runs use `-d 0,4196,8392` so the `test` column has rows like `tg128@d4196`.
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SWEEPS = {
    "vrocm": ROOT / "vulkan-vs-rocm-sweep",
    "rocwmma": ROOT / "rocwmma-fattn-sweep",
}
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# matplotlib styling — academic / restrained
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "figure.dpi": 150,
})

ROCM_COLOR = "#dc2626"     # red
VULKAN_COLOR = "#2563eb"   # blue
FATTN_OFF_COLOR = "#16a34a"  # green
FATTN_ON_COLOR  = "#a16207"  # amber

# ── parse helpers ─────────────────────────────────────────────────

def parse_log(path: Path) -> list[dict]:
    """Return list of {backend, fa, mmap, test, depth, tps, sigma} from a bench log."""
    rows = []
    if not path.exists():
        return rows
    text = path.read_text(errors="ignore")
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "------" in line or "model" in line.lower():
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 8:
            continue
        try:
            backend = cells[3]
            ngl = cells[4]; fa = cells[5]; mmap = cells[6]
            test = cells[7]
            tps_cell = cells[8]
            m = re.match(r"([\d.]+)\s*±\s*([\d.]+)", tps_cell)
            if not m: continue
            tps = float(m.group(1)); sigma = float(m.group(2))
            depth = 0
            dm = re.search(r"@\s*d(\d+)", test)
            if dm: depth = int(dm.group(1))
            test_kind = re.split(r"\s*@", test)[0].strip()
            rows.append(dict(
                backend=backend, fa=int(fa), mmap=int(mmap),
                test=test_kind, depth=depth, tps=tps, sigma=sigma,
            ))
        except (ValueError, IndexError):
            continue
    return rows

def pick(rows, **filters):
    """Filter rows by kv pairs, return first match or None."""
    for r in rows:
        if all(r.get(k) == v for k, v in filters.items()):
            return r
    return None

# ── load all the sweep data we need ───────────────────────────────

q4_rocm     = parse_log(SWEEPS["rocwmma"] / "bench-qwen36-35b-a3b-q4-fattn-off.log")
q4_vulk_s1  = parse_log(SWEEPS["vrocm"]   / "bench-qwen36-a3b-q4-vulkan-shape1.log")
q4_vulk_s2  = parse_log(SWEEPS["vrocm"]   / "bench-qwen36-a3b-q4-vulkan-shape2.log")
bf16_rocm_s1 = parse_log(SWEEPS["vrocm"]  / "bench-qwen36-a3b-bf16-rocm-shape1.log")
bf16_rocm_s2 = parse_log(SWEEPS["vrocm"]  / "bench-qwen36-a3b-bf16-rocm-shape2.log")
bf16_vulk_s1 = parse_log(SWEEPS["vrocm"]  / "bench-qwen36-a3b-bf16-vulkan-shape1.log")
bf16_vulk_s2 = parse_log(SWEEPS["vrocm"]  / "bench-qwen36-a3b-bf16-vulkan-shape2.log")

q35_off = parse_log(SWEEPS["rocwmma"] / "bench-qwen35-27b-q8-fattn-off.log")
q35_on  = parse_log(SWEEPS["rocwmma"] / "bench-qwen35-27b-q8-fattn-on.log")
q36_off = parse_log(SWEEPS["rocwmma"] / "bench-qwen36-35b-a3b-q4-fattn-off.log")
q36_on  = parse_log(SWEEPS["rocwmma"] / "bench-qwen36-35b-a3b-q4-fattn-on.log")

# ── figure 1 — Q4_K_M tg128 across depths ─────────────────────────

def tg_at(rows, depth):
    """Find tg128 at the given depth, fa=1."""
    r = pick(rows, test="tg128", depth=depth, fa=1)
    return (r["tps"], r["sigma"]) if r else (None, None)

depths = [0, 4196, 8392]
rocm_q4 = [tg_at(q4_rocm, d) for d in depths]
vulk_q4 = [tg_at(q4_vulk_s1 + q4_vulk_s2, d) for d in depths]

fig, ax = plt.subplots(figsize=(5.5, 3.2))
x = list(range(len(depths)))
w = 0.36
ax.bar([i - w/2 for i in x], [r[0] for r in rocm_q4], w,
       yerr=[r[1] for r in rocm_q4], capsize=3, color=ROCM_COLOR, label="ROCm / HIP")
ax.bar([i + w/2 for i in x], [r[0] for r in vulk_q4], w,
       yerr=[r[1] for r in vulk_q4], capsize=3, color=VULKAN_COLOR, label="Vulkan / RADV")
ax.set_xticks(x); ax.set_xticklabels([f"d={d}" for d in depths])
ax.set_xlabel("Context depth (tokens)")
ax.set_ylabel("Decode throughput (tok/s)")
ax.set_title("Qwen3.6-35B-A3B Q4_K_M, tg128, fa=1\nVulkan wins by 19-22% across depths")
ax.legend(loc="lower left", frameon=False)
for i, (a, b) in enumerate(zip(rocm_q4, vulk_q4)):
    if a[0] and b[0]:
        delta = (b[0] - a[0]) / a[0] * 100
        ax.text(i, max(a[0], b[0]) + 4, f"+{delta:.1f}%", ha="center", fontsize=8, color=VULKAN_COLOR)
plt.tight_layout()
plt.savefig(OUT / "fig1-vulkan-vs-rocm-q4.pdf")
plt.close()

# ── figure 2 — BF16 tg128 across depths (the inversion) ───────────

rocm_bf = [tg_at(bf16_rocm_s1 + bf16_rocm_s2, d) for d in depths]
vulk_bf = [tg_at(bf16_vulk_s1 + bf16_vulk_s2, d) for d in depths]

fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar([i - w/2 for i in x], [r[0] for r in rocm_bf], w,
       yerr=[r[1] for r in rocm_bf], capsize=3, color=ROCM_COLOR, label="ROCm / HIP")
ax.bar([i + w/2 for i in x], [r[0] for r in vulk_bf], w,
       yerr=[r[1] for r in vulk_bf], capsize=3, color=VULKAN_COLOR, label="Vulkan / RADV")
ax.set_xticks(x); ax.set_xticklabels([f"d={d}" for d in depths])
ax.set_xlabel("Context depth (tokens)")
ax.set_ylabel("Decode throughput (tok/s)")
ax.set_title("Qwen3.6-35B-A3B BF16, tg128, fa=1\nROCm wins by 117-121% (over 2x)")
ax.legend(loc="upper right", frameon=False)
for i, (a, b) in enumerate(zip(rocm_bf, vulk_bf)):
    if a[0] and b[0]:
        delta = (a[0] - b[0]) / b[0] * 100
        ax.text(i, max(a[0], b[0]) + 0.7, f"+{delta:.0f}%", ha="center", fontsize=8, color=ROCM_COLOR)
plt.tight_layout()
plt.savefig(OUT / "fig2-vulkan-vs-rocm-bf16.pdf")
plt.close()

# ── figure 3 — ROCWMMA_FATTN ON vs OFF, prefill + tg, two models ──

def pp_tg_at(rows, depth, test):
    r = pick(rows, test=test, depth=depth, fa=1)
    return (r["tps"], r["sigma"]) if r else (None, None)

# Use pp2048 at depth sweep (the dramatic effect appears at deeper context)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
configs = [
    ("Qwen3.5-27B Q8 (dense)", q35_off, q35_on),
    ("Qwen3.6-35B-A3B Q4 (MoE)", q36_off, q36_on),
]
for ax, (label, off_rows, on_rows) in zip(axes, configs):
    depths_local = [0, 4196, 8392]
    pp_off = [pp_tg_at(off_rows, d, "pp2048") for d in depths_local]
    pp_on  = [pp_tg_at(on_rows,  d, "pp2048") for d in depths_local]
    x2 = list(range(len(depths_local)))
    ax.bar([i - w/2 for i in x2], [r[0] for r in pp_off], w,
           yerr=[r[1] for r in pp_off], capsize=3, color=FATTN_OFF_COLOR, label="FATTN=OFF")
    ax.bar([i + w/2 for i in x2], [r[0] for r in pp_on], w,
           yerr=[r[1] for r in pp_on], capsize=3, color=FATTN_ON_COLOR, label="FATTN=ON")
    ax.set_xticks(x2); ax.set_xticklabels([f"d={d}" for d in depths_local])
    ax.set_xlabel("Context depth (tokens)")
    ax.set_ylabel("Prefill throughput pp2048 (tok/s)")
    ax.set_title(label)
    ax.legend(loc="upper right", frameon=False)
    for i, (a, b) in enumerate(zip(pp_off, pp_on)):
        if a[0] and b[0]:
            delta = (a[0] - b[0]) / b[0] * 100
            ax.text(i, max(a[0], b[0]) + 18, f"OFF +{delta:.0f}%", ha="center", fontsize=8, color=FATTN_OFF_COLOR)
fig.suptitle("ROCWMMA_FATTN=OFF outperforms ON on gfx1151 (against AMD docs)", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "fig3-rocwmma-fattn-onoff.pdf", bbox_inches="tight")
plt.close()

print(f"wrote 3 figures to {OUT}/")
for f in sorted(OUT.glob("*.pdf")):
    print(f"  {f.name}  ({f.stat().st_size} bytes)")
