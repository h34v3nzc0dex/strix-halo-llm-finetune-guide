# Fine-Tuning 27B+ LLMs on AMD Strix Halo — A Home Enthusiast's Guide

A reproducible recipe for fine-tuning Qwen3.5-27B (or larger) hybrid LLMs on a single AMD Strix Halo APU (Ryzen AI MAX+ 395, Radeon 8060S, gfx1151) with 128 GB of unified memory — including the patches, system tuning, and out-of-process evaluation orchestrator that make multi-day training runs survivable on consumer hardware.

> **Status:** Tested on a Corsair AI Workstation 300 (Sixunited AXB35-02 board) running Ubuntu 24.04 LTS, mainline kernel 6.19.14, ROCm 7.13 nightly. The same recipe should work on Framework Desktop, GMKtec EVO-X2, FEVM FA-EX9, Bosgame M5 — any AXB35-02 / Strix Halo system.

---

## TL;DR

If you tried to fine-tune a ≥27B model on a Strix Halo box and ran into:

- `Configured ROCm binary not found at libbitsandbytes_rocm83.so`
- Triton kernels asserting on `num_warps > 4`
- Trainer eval OOMing or dying with `page allocation failure: order:0` mid-eval
- Memory-watchdog SIGKILL during eval
- TRL crashing in `create_model_card` with `PackageNotFoundError: trl`
- Linux mainline `.deb` kernels failing to install with `run-parts: missing operand`
- `/srv` perms randomly regressing to `0750` after `apt upgrade`

…this guide solves every one of those. It's the writeup of about a week of iteration on a real production fine-tune. We don't claim novelty on any individual piece — but the *combination* on this hardware isn't documented anywhere else we could find.

---

## Who this is for

You have a Strix Halo / gfx1151 workstation with 128 GB unified memory. You want to fine-tune a 7B–32B parameter LLM (or larger MoE in the 100B class) locally. You're comfortable with Linux, bash, git, Python, and the HuggingFace stack. You don't have a cloud GPU budget. You're willing to patch a few open-source projects and accept multi-day training times.

---

## Hardware

| Component | What we tested with | Substitutes |
|---|---|---|
| APU | AMD Ryzen AI MAX+ 395, Radeon 8060S (gfx1151) | Any Ryzen AI MAX 300 series — 385, 390, 395 |
| Board | Sixunited AXB35-02 (BIOS AXB35-02 v3.07) | Same board ships in Corsair AI Workstation 300, Framework Desktop, GMKtec EVO-X2, Bosgame M5, FEVM FA-EX9 |
| Memory | 128 GB LPDDR5X-8000 (unified) | The 64 GB or 96 GB SKUs work but cap your model size |
| Storage | 1 TB+ NVMe | Plan for ≥200 GB free for the venv + models |
| BIOS UMA | **1 GB** (minimum). Let GTT auto-size to the rest dynamically | Don't pin VRAM higher — it just shrinks the unified pool |

---

## The stack we'll build

| Layer | Version | Source | Why this version |
|---|---|---|---|
| Linux kernel | **6.19.14 mainline** | Ubuntu kernel.ubuntu.com | KFD driver fixes for gfx1151; older kernels hit fence/dma_buf sync bugs |
| ROCm system | **7.1.0** | Radeon repo (`repo.radeon.com/rocm/apt/7.1`) | `rocm-cmake`, `hipcc`, `hipBLAS` etc. for builds |
| ROCm Python wheels | **7.13 nightly** | `https://rocm.nightlies.amd.com/v2-staging/gfx1151/` | Native gfx1151 — no `HSA_OVERRIDE_GFX_VERSION` needed |
| PyTorch | **2.11.0+rocm7.13.0a*** | gfx1151 nightly index | bf16 LoRA + AOTriton SDPA work natively |
| flash-linear-attention | **0.5.1 from source, patched** | github.com/fla-org/flash-linear-attention | GatedDeltaNet (Qwen3.5) needs Triton kernels |
| bitsandbytes | **0.50.0.dev0 built from source for gfx1151** | github.com/bitsandbytes-foundation/bitsandbytes | PyPI wheels ship zero ROCm binaries |
| llama.cpp | b867+ rebuilt with `--gcc-install-dir` flag | github.com/ggerganov/llama.cpp | For inference of fine-tuned + base models |
| transformers / trl / peft | 5.4 / 0.29.1 / 0.18.1 | PyPI | Stable for our patterns |

---

## The big-picture architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  train_orchestrator.sh   (long-running)                │
│                                                                     │
│  read latest checkpoint step  ──►  if history < step, run eval first│
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐    target = next save_steps boundary              │
│  │ run_segment  │    spawns python3 train_qwen3_32b.py --max-steps N│
│  └──────┬───────┘                                                   │
│         │ exit 0 at max_steps                                       │
│         ▼                                                           │
│  wait_gpu_release  (pgrep + VRAM<5GB + gpu-defrag-mem)           │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────┐    spawns python3 eval_checkpoint.py              │
│  │  run_eval    │    --adapter checkpoint-N --history *.jsonl       │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  parse latest 2 history entries  ──►  Δ vs prior  ──►  Telegram ✅  │
│         │                                                           │
│         ▼                                                           │
│  loop until step >= total_steps  ──►  Telegram 🎉                   │
└─────────────────────────────────────────────────────────────────────┘
```

The orchestrator is bash. The training and eval scripts are Python. They never coexist in the same process — that's the whole point. Training holds the GPU until it exits cleanly at a `max_steps` boundary, GPU memory fully releases, then a fresh Python process loads from the just-saved checkpoint and runs eval. This sidesteps the in-process eval failure modes that bite on unified memory APUs.

---

## Quick start (for the impatient)

```bash
# 1. Install kernel 6.19+
sudo apt install -y linux-headers-generic
# Mainline kernels: download from kernel.ubuntu.com/mainline/v6.19.14/amd64
# Apply scripts/fix-kernel-run-parts.py to the .debs before installing
# (see docs/02-kernel.md for details)

# 2. Set up sysctl + THP
sudo cp configs/90-strix-halo-vm-tuning.conf /etc/sysctl.d/
sudo sysctl --system
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo defer  | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# 3. Add /srv perm watchdog (CRITICAL — prevents random crashes mid-train)
sudo cp configs/srv-perms-watch.cron /etc/cron.d/srv-perms-watch

# 4. Add defrag helper + sudoers (replace <user> with your Linux username)
sudo cp scripts/gpu-defrag-mem /usr/local/bin/
sudo chmod +x /usr/local/bin/gpu-defrag-mem
sed "s/<user>/$(whoami)/" configs/gpu-defrag-mem.sudoers \
    | sudo tee /etc/sudoers.d/gpu-defrag-mem > /dev/null
sudo chmod 0440 /etc/sudoers.d/gpu-defrag-mem

# 5. GRUB — add ttm.* + transparent_hugepage to kernel cmdline
# (see configs/grub-cmdline.example), then sudo update-grub && reboot

# 6. Set up venv + nightly PyTorch
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate
pip install --pre \
  "torch==2.11.0+rocm7.13.0a20260506" \
  "torchvision==0.26.0+rocm7.13.0a20260506" \
  "torchaudio==2.11.0+rocm7.13.0a20260506" \
  "triton==3.6.0+rocm7.13.0a20260506" \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151/ \
  --extra-index-url https://pypi.org/simple/

# 7. Build flash-linear-attention from patched source (see §FLA below)
# 8. Build bitsandbytes from source for ROCm gfx1151 (see §bnb below)
# 9. Set up Telegram alerts (optional — see scripts/tg_alert.sh)

# 10. Run the orchestrator
nohup ./scripts/train_orchestrator.sh \
    --total-steps 448 \
    --output-dir /path/to/your/output \
    --lora-r 128 --lora-alpha 256 \
    > orchestrator.log 2>&1 &
```

If any of those steps don't make sense, keep reading.

---

## Step 1 — Kernel 6.19.14 (mainline)

Recent gfx1151 KFD driver fixes are in mainline kernels only. Distros lag. Use Ubuntu's mainline build.

### Install

Download the four `.deb` files from `https://kernel.ubuntu.com/mainline/v6.19.14/amd64/`:

```
linux-headers-6.19.14-061914_*_all.deb
linux-headers-6.19.14-061914-generic_*_amd64.deb
linux-image-unsigned-6.19.14-061914-generic_*_amd64.deb
linux-modules-6.19.14-061914-generic_*_amd64.deb
```

### Fix the `run-parts` bug (CRITICAL)

Mainline kernel `.deb`s have a **double-dir `run-parts` bug** that breaks `dpkg -i` on Ubuntu 24.04+. The maintainer scripts call:

```bash
run-parts --report --exit-on-error --arg=$version \
    --arg=$image_path /etc/kernel/postinst.d /usr/share/kernel/postinst.d
```

`run-parts` only accepts ONE directory. Multi-dir form errors out, dpkg leaves the package half-configured. The fix script in this repo (`scripts/fix-kernel-run-parts.py`) rewrites these to:

```bash
if [ -d /etc/kernel/postinst.d ]; then
    run-parts ... /etc/kernel/postinst.d
fi
if [ -d /usr/share/kernel/postinst.d ]; then
    run-parts ... /usr/share/kernel/postinst.d
fi
```

The `if/fi` form (NOT `&&`) matters — using `[ -d ] && cmd` propagates exit-1 from a missing `/usr/share/kernel/X.d` out of the heredoc-generated trigger script and half-configures the package anyway.

```bash
# Repack the affected .debs:
mkdir -p extracted
for f in linux-image*.deb linux-modules*.deb linux-headers-*-generic_*amd64.deb; do
    name=$(basename "$f" .deb)
    mkdir -p "extracted/$name"
    dpkg-deb -R "$f" "extracted/$name"
done
python3 scripts/fix-kernel-run-parts.py \
    extracted/linux-image*/DEBIAN/{preinst,postinst,prerm,postrm} \
    extracted/linux-modules*/DEBIAN/postinst \
    extracted/linux-headers-*-generic_*/DEBIAN/postinst
for d in extracted/*; do dpkg-deb --build "$d" "$(basename "$d")-fixed.deb"; done

# Install:
sudo dpkg -i linux-headers-*-all.deb *-fixed.deb
sudo update-grub && sudo reboot
```

### Boot params

After reboot, edit `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt pcie_aspm.policy=performance amdgpu.runpm=0 ttm.pages_limit=33554432 ttm.page_pool_size=33554432 transparent_hugepage=always numa_balancing=disable"
```

Then `sudo update-grub && sudo reboot`.

**Note:** `transparent_hugepage=always` doesn't always stick on Ubuntu — something in early boot resets it to `madvise`. Add to `/etc/rc.local`:

```bash
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo defer  > /sys/kernel/mm/transparent_hugepage/defrag
```

### Verify

```bash
uname -r                                              # 6.19.14-061914-generic
cat /proc/cmdline | tr ' ' '\n' | grep -E "ttm|hugepage"
sudo dmesg | grep "GTT memory ready"                  # should show 131072M
ls /sys/class/drm/card0/device/mem_info_vram_used     # must exist
```

---

## Step 2 — System tuning

Per AMD's MI300A optimization guide (the only AMD-blessed unified-memory APU tuning doc), **proactive page compaction is mandatory** for unified-memory APUs. Without it, the GPU's TTM allocator hits page-allocation failures during heavy bursts (mid-training eval, model load) even when 90% of system RAM is free, because the page allocator's free-list is fragmented.

> **About the `configs/` directory:** every file under `configs/` is **provided by this repo** — none of them exist on a fresh Ubuntu install. You're adding them. Read each one before copying it into place; they're short and well-commented. The `<user>` placeholder in the sudoers file must be replaced with your actual username before install (the section below shows how).

### sysctl — `90-strix-halo-vm-tuning.conf`

This file goes into `/etc/sysctl.d/` as a *new drop-in*. Linux's sysctl loader processes everything in `/etc/sysctl.d/` in lexical order at boot and on `sysctl --system`. The `90-` prefix means "load near the end so I override earlier defaults" — your existing `/etc/sysctl.conf` and other drop-ins aren't touched.

```bash
sudo cp configs/90-strix-halo-vm-tuning.conf /etc/sysctl.d/
sudo sysctl --system
sysctl vm.compaction_proactiveness   # should print 20
```

The two key settings (the file has more — open it and read the comments):

```
vm.compaction_proactiveness = 20
vm.compact_unevictable_allowed = 1
```

### Transparent huge pages

THP=always doesn't always stick from the GRUB cmdline (Ubuntu 24.04+ has something in early boot that resets it to `madvise`). Set live AND add to `/etc/rc.local` for persistence:

```bash
echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo defer  | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# Persist across reboots:
sudo tee -a /etc/rc.local > /dev/null <<'EOF'
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo defer  > /sys/kernel/mm/transparent_hugepage/defrag
EOF
sudo chmod +x /etc/rc.local
```

### Defrag helper — `gpu-defrag-mem` + sudoers

`scripts/gpu-defrag-mem` is a tiny shell script that runs `compact_memory + drop_caches`. The training orchestrator calls it before each eval (and you can invoke it manually any time) to give the GPU's TTM allocator contiguous free pages on a unified-memory pool.

The sudoers drop-in lets your training user run `gpu-defrag-mem` with `sudo -n` (no password). **Replace `<user>` with your actual Linux username** before installing:

```bash
# Install the script
sudo cp scripts/gpu-defrag-mem /usr/local/bin/
sudo chmod +x /usr/local/bin/gpu-defrag-mem

# Edit the sudoers file: replace <user> with your actual username
# (e.g. paul, daisy, whatever `whoami` returns)
sed "s/<user>/$(whoami)/" configs/gpu-defrag-mem.sudoers \
    | sudo tee /etc/sudoers.d/gpu-defrag-mem > /dev/null
sudo chmod 0440 /etc/sudoers.d/gpu-defrag-mem
sudo visudo -c -f /etc/sudoers.d/gpu-defrag-mem  # validate

# Test (should run without prompting for password):
sudo -n /usr/local/bin/gpu-defrag-mem && echo OK
```

If `visudo -c` reports a syntax error, the placeholder substitution failed — re-check the username doesn't contain shell-special characters.

### `/srv` perm watchdog — `srv-perms-watch.cron` (this one bit us hard)

Some apt postinst scripts (we suspect systemd / dpkg / snapd updates) silently chmod `/srv` to `0750`, which breaks every non-root process needing to traverse to anything under `/srv/*`. We hit this **mid-segment** during a 9-hour training run — the trainer crashed in `create_model_card → importlib.metadata.version("trl")` because the metadata path lookup couldn't traverse `/srv`. We lost the entire segment.

```bash
sudo cp configs/srv-perms-watch.cron /etc/cron.d/srv-perms-watch
sudo chmod 0644 /etc/cron.d/srv-perms-watch
# The cron now restores /srv to 755 every minute. Idempotent.
```

You won't hit this on every system, and it's not specific to fine-tuning — it's a defensive fix for an apt postinst regression. If you're storing your venv, training output, or any long-running process state under `/srv/`, install this. It's three bytes of cron and saves you from a 9-hour-loss class of bug.

---

## Step 3 — PyTorch nightly + ROCm

```bash
python3 -m venv /path/to/venv
source /path/to/venv/bin/activate

# Install PyTorch 2.11 + Triton 3.6 from the gfx1151 nightly index.
# Pick a date when all four packages are available — 20260506 here.
pip install --pre \
  "torch==2.11.0+rocm7.13.0a20260506" \
  "torchvision==0.26.0+rocm7.13.0a20260506" \
  "torchaudio==2.11.0+rocm7.13.0a20260506" \
  "triton==3.6.0+rocm7.13.0a20260506" \
  "rocm==7.13.0a20260506" \
  "rocm-sdk-core==7.13.0a20260506" \
  "rocm-sdk-libraries-gfx1151==7.13.0a20260506" \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151/ \
  --extra-index-url https://pypi.org/simple/

# Verify
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python -c "
import torch
print('torch:', torch.__version__)
print('hip:', torch.version.hip)
print('arch:', torch.cuda.get_arch_list())
x = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
y = x @ x.T
torch.cuda.synchronize()
print('bf16 matmul OK')
"
```

Two non-obvious points:

1. **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` MUST be set before `import torch`.** AOTriton on gfx1151 is gated behind this flag. Without it SDPA falls back to a 19× slower path.
2. **Don't set `HSA_OVERRIDE_GFX_VERSION`.** It's a habit from earlier ROCm 6 setups and it actively *breaks* native gfx1151 kernels. If your `~/.bashrc` has it, remove it.

---

## Step 4 — flash-linear-attention (patched source)

Hybrid models with linear-attention layers (Qwen3.5, Mamba-3, GatedDeltaNet variants) need FLA's Triton kernels. The PyPI wheel works on H100 but **crashes on gfx1151** for two reasons:

1. **`num_warps > 4` triggers Triton assertion failures on RDNA 3 / 3.5.** Upstream Triton bug ([triton#5609](https://github.com/triton-lang/triton/issues/5609)).
2. **`tl.cumsum + tl.sum` interaction has a codegen bug** that hits gfx1151 (and apparently also H100 in some configs). [triton#3017](https://github.com/triton-lang/triton/issues/3017).

The fix:

```bash
# Clone
git clone https://github.com/fla-org/flash-linear-attention /path/to/fla-patched
cd /path/to/fla-patched

# Apply the patches via the script in this repo
python3 /path/to/strix-halo-llm-finetune-guide/scripts/fla_repatch.py \
    --fla-root /path/to/fla-patched

# Replace cumsum.py with a PyTorch wrapper. We keep our patched copy at
# /path/to/fla-cumsum-patched.py — see docs/04-fla-patches.md for
# the contents of that file (or grab it from a previous patched checkout).

# Clear stale autotune cache
rm -rf ~/.triton/cache
find . -name __pycache__ -exec rm -rf {} +

# Install editable
pip install -e .
```

Re-run `fla_repatch.py` after every `git pull`. It's idempotent.

---

## Step 5 — bitsandbytes from source for ROCm

**The PyPI bnb wheel ships zero ROCm binaries.** It only has CPU + CUDA `.so` files. If you try `optim="paged_adamw_8bit"` you'll get:

```
RuntimeError: Configured ROCm binary not found at libbitsandbytes_rocm83.so
```

Build from source:

```bash
# Required apt packages
sudo apt install -y hiprand-dev rocrand-dev hipcub-dev rocprim-dev rocthrust-dev

# Clone
git clone https://github.com/bitsandbytes-foundation/bitsandbytes /path/to/bnb-rocm
cd /path/to/bnb-rocm

# Configure with ROCm 7.1.0 toolchain + gcc-13 for clang's libstdc++ lookup
PATH=/opt/rocm-7.1.0/bin:$PATH \
cmake -G Ninja \
  -DCOMPUTE_BACKEND=hip \
  -DBNB_ROCM_ARCH="gfx1151" \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCM_VERSION=83 \
  -DCMAKE_HIP_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13" \
  -S . -B build

# Build
PATH=/opt/rocm-7.1.0/bin:$PATH cmake --build build --config Release

# Symlink: bnb's runtime version detection expects libbitsandbytes_rocm713.so
# on PyTorch 2.10/2.11 + HIP 7.13, but the build produced rocm83.so.
cd bitsandbytes
ln -sf libbitsandbytes_rocm83.so libbitsandbytes_rocm713.so
cd ..

# Install editable (replaces PyPI bnb)
pip uninstall -y bitsandbytes
pip install -e .
```

### CRITICAL gotcha — the namespace package shadow

If a previous setup left `/path/to/venv/lib/python3.12/site-packages/bitsandbytes/libbitsandbytes_rocm82.so` lying around (a symlink to a non-existent file from an older bnb install), Python treats that directory as a **namespace package** — and silently shadows your editable install. Symptom: `import bitsandbytes; print(bitsandbytes.__file__)` returns `None`, no `.optim` attribute. Cure:

```bash
rm -rf /path/to/venv/lib/python3.12/site-packages/bitsandbytes
# Then re-test:
python -c "import bitsandbytes; print(bitsandbytes.__file__)"
# Should resolve to /path/to/bnb-rocm/bitsandbytes/__init__.py
```

### Verify

```python
import torch
import bitsandbytes
assert bitsandbytes.__file__ is not None
from bitsandbytes.optim import PagedAdamW8bit
p = torch.nn.Parameter(torch.randn(64, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True))
opt = PagedAdamW8bit([p], lr=1e-4)
(p*p).sum().backward()
opt.step()
torch.cuda.synchronize()
print("PagedAdamW8bit step succeeded")
```

---

## Step 6 — llama.cpp HIP build (for inference)

If you want to run the resulting fine-tune via `llama-server`, build llama.cpp with the `--gcc-install-dir` flag (without it, ROCm 7.1.0's clang-20 can't find `<cmath>`):

```bash
cd /path/to/llama.cpp
PATH=/opt/rocm-7.1.0/bin:$PATH \
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_MMQ_MFMA=ON \
  -DGGML_HIP_NO_VMM=ON \
  -DAMDGPU_TARGETS=gfx1151 \
  -DCMAKE_HIP_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
PATH=/opt/rocm-7.1.0/bin:$PATH cmake --build build --parallel $(nproc)
```

`GGML_HIP_GRAPHS=ON` is now upstream default (b867+) but explicitly enabling doesn't hurt.

---

## Step 7 — The eval problem

In-process Trainer eval **does not work** at 27B + 8192 seq length on Strix Halo. We have three documented failure modes from real runs:

1. **`page allocation failure: order:0`** in `ttm_pool_alloc_page`. The unified-memory page allocator's free-list is fragmented — the GPU can't get a single contiguous 4 KB page despite 90%+ system RAM free. Killed by amdgpu kernel module.
2. **`memory-watchdog SIGKILL`** when eval pushes free RAM under the watchdog threshold (8 GB on our box). Eager attention matrix on 24 heads × 8192² float32 ≈ 25 GB on top of the 60 GB training state — ~50 GB consumed in ~23 seconds during the eval batches.
3. **`importlib.metadata.PackageNotFoundError: trl`** in TRL's `_save_checkpoint → create_model_card`. This one was caused by `/srv` perms regressing mid-segment, breaking the venv's metadata path traversal.

The system-tuning stack from §2 fixes (1) and (3) — but **(2) is structural**. Eval and training simply cannot coexist in the same process at 128 GB unified memory + 27B models.

The solution is to **move eval out-of-process**.

---

## Step 8 — The orchestrator

`scripts/train_orchestrator.sh` drives training as a sequence of segments aligned to `save_steps=50` boundaries. After each segment:

1. Trainer reaches `max_steps` cleanly, writes checkpoint, exits → process dies → GPU memory fully releases.
2. `wait_gpu_release()` confirms `pgrep` empty + VRAM-used < 5 GB + runs the defrag helper.
3. `eval_checkpoint.py` spawns as a **fresh process**, loads base model + adapter from the just-saved checkpoint, runs eval over a 50-sample subset, appends one line to `eval_history.jsonl`.
4. Orchestrator parses last 2 history entries, computes Δ, sends Telegram with success/comparison or warning.
5. Loop until total_steps reached.

**Resume-safe.** Killing the orchestrator and restarting picks up from the latest checkpoint and runs any missed eval first.

**Argument summary** (full list in `scripts/train_orchestrator.sh`):

```
--total-steps     448            # final step count
--save-steps      50             # MUST match the training script's save_steps
--output-dir      /path/to/out   # where checkpoints land
--eval-data       /path/to/eval.jsonl
--history         /path/to/eval_history.jsonl
--lora-r          128
--lora-alpha      256
--epochs          2
--grad-accum      4
--base-model      Qwen/Qwen3.5-27B
```

**Launch under nohup so it survives session close:**

```bash
cd /path/to/workspace
nohup ./scripts/train_orchestrator.sh \
    --total-steps 448 \
    --output-dir /path/to/output \
    --lora-r 128 --lora-alpha 256 \
    > orchestrator.log 2>&1 &
```

### How the alignment math works

If you resume from `checkpoint-87` (e.g., a pre-eval-save callback wrote it at a non-aligned step), the orchestrator computes:

```
target = ((step / save_steps) + 1) * save_steps
       = ((87 / 50) + 1) * 50
       = 100
```

So segment 1 trains 87→100 (13 steps), trainer's auto-save fires at step 100, segment exits. Subsequent segments are full 50-step blocks (100→150, 150→200, …).

---

## Step 9 — Telegram alerts (optional but nice)

`scripts/tg_alert.sh` is a 50-line bash helper that sends HTML messages to a Telegram bot. Set up:

1. Talk to `@BotFather` on Telegram, create a bot, save the token.
2. Get your numeric chat ID from `@userinfobot`.
3. Store credentials:

```bash
sudo mkdir -p /etc/strix-halo
sudo tee /etc/strix-halo/telegram.env > /dev/null <<EOF
TELEGRAM_BOT_TOKEN=<your-token>
TELEGRAM_CHAT_ID=<your-chat-id>
EOF
sudo chown root:<user> /etc/strix-halo/telegram.env
sudo chmod 0640 /etc/strix-halo/telegram.env
```

4. Test:

```bash
./scripts/tg_alert.sh "<b>Test</b> — Strix Halo guide setup OK"
# Should appear in your Telegram chat.
```

The orchestrator sends:

- 🚀 startup notice with current step
- ✅ per-segment success with eval_loss + Δ + perplexity + segment runtime + ETA
- ❌ segment failure with exit code + last 30 lines of log (HTML-escaped)
- ⚠️ eval failure (non-fatal — training continues)
- 🛑 SIGINT/SIGTERM (Ctrl-C handler with the latest checkpoint step)
- 🎉 final completion with total runtime

---

## Verified results

This guide was developed on a real production fine-tune. Excerpt from `eval_history.jsonl`:

```json
{"step":87,"eval_loss":0.1324,"eval_perplexity":1.1416,"eval_token_accuracy":0.9646,"n_samples":48,"timestamp":"2026-05-07T23:42:02Z"}
{"step":100,"eval_loss":0.1312,"eval_perplexity":1.1402,"eval_token_accuracy":0.9645,"n_samples":48,"timestamp":"2026-05-08T03:46:07Z"}
{"step":150,"eval_loss":<filled in as runs land>,...}
```

Target: 448 steps total. Step time: ~11 min. Total wall-clock: ~4 days. GPU temp range during training: 60–72 °C with `power_dpm_force_performance_level=auto`. Peak GPU memory: ~80 GB reserved during training, ~73 GB during eval.

---

## What's still unsolved

We're not done; this guide is a snapshot, not a victory lap.

- **Eval still takes ~5–10 min per checkpoint.** The base model has to reload each time. Could be amortized with a long-running eval daemon that holds the model warm.
- **`svm_range_restore_work` thrash** during heavy GPU bursts is an open AMD bug ([ROCm#5952](https://github.com/ROCm/ROCm/issues/5952)). The Oct 2025 patch on amd-gfx covers only the MADV_FREE deadlock, not the CPU-hog-during-attention case. We work around it by exiting the training process between segments.
- **Why `/srv` perms regress** is still unknown. We have a cron watchdog as defense in depth, but the actual postinst script doing the chmod hasn't been pinned down. If you find it, file a bug.
- **TRL `create_model_card` is fragile.** It calls `importlib.metadata.version("trl")` which traverses sys.path and silently fails if any `.dist-info` dir is unreachable. A more defensive trl would catch this.
- **PyTorch 2.11 nightly is unstable by definition.** Pin a specific date that worked for you.

---

## Project layout

```
strix-halo-llm-finetune-guide/
├── README.md                              # this file
├── LICENSE                                # MIT
├── scripts/
│   ├── train_orchestrator.sh              # segment orchestrator (bash)
│   ├── eval_checkpoint.py                 # standalone out-of-process eval
│   ├── tg_alert.sh                        # Telegram alert helper
│   ├── gpu-defrag-mem                     # compact_memory + drop_caches wrapper
│   ├── fix-kernel-run-parts.py            # mainline kernel .deb fixer
│   └── fla_repatch.py                     # FLA num_warps + cumsum patcher
├── configs/
│   ├── 90-strix-halo-vm-tuning.conf       # → /etc/sysctl.d/
│   ├── gpu-defrag-mem.sudoers             # → /etc/sudoers.d/gpu-defrag-mem
│   ├── srv-perms-watch.cron               # → /etc/cron.d/srv-perms-watch
│   └── grub-cmdline.example               # → edits to /etc/default/grub
└── docs/
    └── (deep dives — to be expanded)
```

---

## Credits

Built and tested on a Corsair AI Workstation 300 by **Paul Durkin** ([@h34v3nzc0dex](https://github.com/h34v3nzc0dex)) with the assistance of Claude (Anthropic). Every patch in this repo was discovered by hitting a real failure on a real run and digging until we understood the root cause.

The community resources that got us most of the way:

- [AMD Strix Halo system optimization (official)](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html)
- [AMD MI300A system optimization (official)](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300a.html) — our north-star tuning doc
- [Strix Halo Wiki](https://strixhalo.wiki) — cross-OEM firmware and kernel-param notes
- [Framework community fine-tuning thread](https://community.frame.work/t/finetuning-llms-on-strix-halo-full-lora-and-qlora-on-gemma-3-qwen-3-and-gpt-oss-20b/76986)
- [kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) — vLLM-focused, but the kernel parameter notes were a useful sanity check

If this guide helps you, [open an issue](https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide/issues) with what worked, what didn't, and what hardware you're on. We'll fold it back in.

## License

MIT — see `LICENSE`.
