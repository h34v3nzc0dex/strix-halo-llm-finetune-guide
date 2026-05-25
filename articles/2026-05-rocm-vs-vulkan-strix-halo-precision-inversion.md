# ROCm vs Vulkan on AMD Strix Halo: when each wins, and why it inverts at the precision boundary

I rebuilt llama.cpp with the Vulkan backend last week to settle a benchmark dispute on r/StrixHalo, and ended up with a finding that wasn't on my radar: on AMD Strix Halo, ROCm and Vulkan beat each other depending entirely on whether the model is quantized or full precision. And not by a small margin — for full-precision BF16 inference, ROCm decodes Qwen3.6-35B-A3B over twice as fast as Vulkan. For the exact same model at Q4_K_M, Vulkan beats ROCm by about 22%. Same hardware, same source commit, same bench command — only the model's precision differs.

The two numbers on tg128: 23.7 vs 10.7 tok/s for BF16 (ROCm wins), 60.4 vs 49.6 tok/s for Q4 (Vulkan wins). Below I'll show all the depths, both prefill and decode, and the one capability-line in Vulkan's startup log that explains the whole thing.

## How this came up

Strix Halo (AMD's Ryzen AI MAX+ 395 with the Radeon 8060S iGPU and 128 GB of unified memory) has unusual upside for solo AI developers: enough memory to load 27B–35B class models without quantization, at $2,400-ish for the whole workstation versus $8K+ for the equivalent NVIDIA card. I've been daily-driving one for six months — fine-tuning Qwen3.5-27B with bf16 LoRA, serving Qwen3.6-A3B for chat, the works — and the entire guide for getting this hardware production-ready is at <https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide>.

The catch on Strix Halo is the software stack. Two paths exist for GPU inference:

- **ROCm/HIP** — AMD's official Compute stack. llama.cpp builds with `-DGGML_HIP=ON`, links against `libamdhip64`. This is also the only path that works for training (PyTorch nightly + ROCm 7.13).
- **Vulkan via Mesa RADV** — uses the open-source graphics driver. llama.cpp builds with `-DGGML_VULKAN=ON`, links against `libvulkan`. Mesa 25.x added explicit RADV STRIX_HALO support; recent kernels expose the compute features.

For most of my time on this hardware I've been ROCm-only (training forces the choice). When [u/Disastrous-Cat-7016 on r/StrixHalo](https://reddit.com/r/StrixHalo) — whose [bench.ciru.ai](https://bench.ciru.ai) dashboard is the canonical Strix Halo Vulkan reference — posted numbers in the 59–74 tok/s range for Qwen3.6-35B-A3B at Q4 (versus my ~50 tok/s), the gap was too big to be config. It had to be the backend.

So I built a Vulkan llama.cpp at the same source commit and ran both, head-to-head. Then I figured "while I have the Vulkan binary up, let me also throw BF16 at it." The Q4 numbers matched their dashboard within noise. The BF16 numbers were the surprise.

## Methodology — same source, same hardware, same shape

Both binaries were built from `ggerganov/llama.cpp` commit `a497476` (release tag `b9296`). One cmake configure with `-DGGML_HIP=ON` (everything else per [Step 6 of the strix-halo guide](https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide#step-6--llamacpp-hip-build-for-inference) — `MMQ_MFMA=ON`, `NO_VMM=ON`, `GRAPHS=ON`, `ROCWMMA_FATTN=OFF`, `AMDGPU_TARGETS=gfx1151`). Another with `-DGGML_VULKAN=ON`. No HIP libraries in the Vulkan build, no Vulkan headers needed in the HIP build. The reproducible build recipes for both are at <https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide/tree/main/vulkan-vs-rocm-sweep> (the `build-vulkan.sh` script + the guide's Step 6 cmake invocation).

Model: `unsloth/Qwen3.6-35B-A3B-GGUF` in both Q4_K_M (UD variant, 20.6 GB on disk) and BF16 (66 GB on disk). Same exact GGUF file given to both backends.

Bench command was identical for both runs:

```bash
llama-bench -m <gguf> -p 512 -n 128 -r 3 --mmap 0 -ngl 99 -fa 0,1
llama-bench -m <gguf> -p 2048 -n 128 -r 3 --mmap 0 -ngl 99 -fa 1 -d 0,4196,8392
```

For the ROCm path, the launch needed `LD_LIBRARY_PATH=<nightly HSA>:/opt/rocm-7.1.0/lib:…` — Mesa Vulkan needs no overlay (Mesa is the driver). The nightly HSA overlay is a known workaround for a ROCm 7.1.0 `libhsa-runtime64.so` null-pointer bug on gfx1151; see [the guide's Troubleshooting section](https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide#troubleshooting).

Hardware: Corsair AI Workstation 300 — Ryzen AI MAX+ 395, Radeon 8060S, 128 GB unified memory, BIOS UMA set to 1 GB (kernel auto-sizes GTT to the full 128 GB). Ubuntu 24.04 LTS, kernel 6.19.14 mainline, Mesa 25.2.8.

## Results — Qwen3.6-35B-A3B Q4_K_M

Quantized inference. This is the workload most Strix Halo users actually run for chat/coding/RAG.

| shape | ROCm/HIP | Vulkan | Winner |
|---|---|---|---|
| pp512 fa=0 | 1023.75 ± 8.66 | 953.79 ± 17.18 | ROCm (+7.3%) |
| pp512 fa=1 | 1014.32 ± 9.65 | 942.18 ± 4.92 | ROCm (+7.7%) |
| tg128 fa=1 d=0 | 49.58 ± 0.09 | **60.39 ± 0.22** | **Vulkan (+21.8%)** |
| tg128 fa=1 d=4196 | 48.64 ± 0.24 | **58.24 ± 0.09** | **Vulkan (+19.7%)** |
| tg128 fa=1 d=8392 | 46.73 ± 0.02 | **57.13 ± 0.09** | **Vulkan (+22.3%)** |
| pp2048 fa=1 d=0 | 983.86 ± 2.91 | 921.04 ± 2.97 | ROCm (+6.8%) |
| pp2048 fa=1 d=8392 | 815.70 ± 0.57 | 835.33 ± 2.28 | Vulkan (+2.4%) |

Decode favors Vulkan by about 22%, consistent across context depths. ROCm beats Vulkan by ~7% on prefill at shallow context, but the gap closes by ~8k tokens. For a chat workload (decode-bound, modest prompt sizes), Vulkan wins outright. For a heavy-prefill RAG or eval workload at d=0–4k, it's mixed. Bench.ciru.ai's tuned Vulkan tg128 of ~61 tok/s for the same model class matches our 60.39 within margin of error.

## Results — Qwen3.6-35B-A3B BF16

Same model, no quantization. 66 GB on disk, ~70 GB resident at runtime. Comfortable on the 128 GB unified pool, no swapping.

| shape | ROCm/HIP | Vulkan | Winner |
|---|---|---|---|
| pp512 fa=0 | **479.55 ± 1.69** | 306.85 ± 1.00 | **ROCm (+56.3%)** |
| pp512 fa=1 | **484.01 ± 4.40** | 305.21 ± 0.87 | **ROCm (+58.6%)** |
| tg128 fa=1 d=0 | **23.71 ± 0.01** | 10.73 ± 0.00 | **ROCm (+121%)** |
| tg128 fa=1 d=4196 | **23.39 ± 0.01** | 10.68 ± 0.01 | **ROCm (+119%)** |
| tg128 fa=1 d=8392 | **23.09 ± 0.00** | 10.64 ± 0.00 | **ROCm (+117%)** |
| pp2048 fa=1 d=0 | **474.69 ± 2.76** | 307.44 ± 1.28 | **ROCm (+54.4%)** |
| pp2048 fa=1 d=8392 | **440.02 ± 0.67** | 289.64 ± 0.84 | **ROCm (+51.9%)** |

ROCm wins everything on BF16, by 50–120%. Decode is over 2× faster on ROCm; prefill is ~55% faster. The opposite of the Q4 picture, on the same hardware, with the same source commit, with the only difference being the quantization level of the model file.

## The smoking gun — `bf16: 0`

This isn't subtle. When llama.cpp with the Vulkan backend starts, it logs the capabilities the Vulkan driver exposes for the device:

```
ggml_vulkan: 0 = AMD Radeon Graphics (RADV GFX1151) (radv) | uma: 1 | fp16: 1 | bf16: 0 | warp size: 64 | shared memory: 65536 | int dot: 0 | matrix cores: KHR_coopmat
```

`fp16: 1` — Vulkan can dispatch FP16 ops to the GPU's cooperative matrix units (the gfx1151 RDNA 3.5 equivalent of tensor cores).

`bf16: 0` — Vulkan cannot dispatch BF16 ops to those units. So when the model is BF16, the ggml-vulkan backend falls back to general-purpose shader code (FP32 emulation, most likely) to handle the matmuls. Same hardware that does FP16 matmul at full speed grinds through BF16 in software.

ROCm/HIP doesn't have this gap. The `libggml-hip.so` backend goes through HIP's BF16-capable matmul kernels, which dispatch to the same matrix units that RADV is missing the BF16 codepath for. So on the BF16 path, the two backends are doing fundamentally different work on the same silicon.

This is a Mesa-side gap, not a llama.cpp gap. BF16 cooperative-matrix extensions for RADV STRIX_HALO are work-in-progress (the [VK_KHR_cooperative_matrix2 spec](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_cooperative_matrix2.html) covers them and implementations are landing across drivers throughout 2026), so the gap will close. Until then: BF16 = ROCm on this hardware.

## What this means in practice

If you're using a Strix Halo box for inference, the backend choice should be a function of your workload type, not a one-time decision:

| Workload | Backend | Why |
|---|---|---|
| Quantized chat / code generation (Q4–Q8) | **Vulkan** | ~22% decode advantage on quants; small prefill cost |
| Full-precision inference (BF16) | **ROCm/HIP** | ~2× decode advantage, ~60% prefill advantage |
| Mixed quant + BF16 in the same workflow | Whichever your hot path is | Run two `llama-server` instances on different ports if needed |
| Fine-tuning + training | **ROCm/HIP** (no choice) | Only path with the PyTorch nightly stack; training is BF16 anyway |
| Inference + occasional fine-tuning | **ROCm/HIP** | Pin the inference side to ROCm too; avoid swapping driver stacks |

For my workflow specifically — fine-tuning Qwen3.5-27B + serving the result + running a few BF16 reference models alongside — ROCm pin is the right call because training forces it. If I were inference-only with quantized models, I'd be on Vulkan.

## Open questions

A few things I haven't dug into yet but that someone might:

- **When does Mesa add BF16 cooperative matrix for RADV STRIX_HALO?** Tracking it via Mesa MRs is the right place to look. Once it lands, the Vulkan BF16 numbers should jump and the inversion may disappear.
- **Does the same inversion hold on other Strix Halo models?** I've tested Qwen3.6-35B-A3B (a MoE). Dense models like Qwen3.5-27B should show the same pattern — same underlying matmul ops — but I haven't run the explicit A/B.
- **Does the inversion hold on RDNA 3 / RDNA 4 discrete cards too?** RX 7900 / RX 9060 XT use different Vulkan capability profiles. The Strix Halo capability gap is specific to RADV's STRIX_HALO codepath.
- **Multi-quant interactions** — Q6 / Q8 / IQ4 sit somewhere on the quant-precision spectrum. The big-picture answer is probably "Vulkan wins all quantized variants, ROCm wins BF16/FP16," but the exact magnitudes would be interesting.

I'll keep the [`vulkan-vs-rocm-sweep/`](https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide/tree/main/vulkan-vs-rocm-sweep) directory in the guide updated as we test more variants.

## Reproducing this

Full bench logs, the Vulkan build script, and the capability extract: <https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide/tree/main/vulkan-vs-rocm-sweep>. ROCm build instructions are in [the guide's Step 6](https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide#step-6--llamacpp-hip-build-for-inference).

If you're on Strix Halo and want to verify: build both binaries from the same commit, download the same `unsloth/Qwen3.6-35B-A3B-GGUF` files (Q4 and BF16), and run the bench commands at the top of the Methodology section. Should take about 90 minutes end-to-end including the 66 GB BF16 download.

## Credits

Thanks to [u/Disastrous-Cat-7016](https://reddit.com/r/StrixHalo) — their [bench.ciru.ai](https://bench.ciru.ai) is the canonical Vulkan Strix Halo benchmark dashboard, and the conversation thread that started this whole comparison is at <https://reddit.com/r/StrixHalo/comments/1tlv8g7/>. Thanks also to [u/Potential-Leg-639](https://reddit.com/r/StrixHalo) and [u/kant12](https://reddit.com/r/StrixHalo) for ongoing exchanges that have pushed the guide forward over the past month.

Article corrections or counter-data: open an issue on [the guide repo](https://github.com/h34v3nzc0dex/strix-halo-llm-finetune-guide/issues). Strix Halo OEM-spread on these numbers is real (Corsair / EVO-X2 / Framework all behave slightly differently on the kernel and memory side), so more data points = better picture for everyone.
