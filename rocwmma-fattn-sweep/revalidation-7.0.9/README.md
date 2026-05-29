# ROCWMMA_FATTN sweep — re-validation on kernel 7.0.9 (2026-05-29)

The original sweep (parent dir) was measured on **mainline kernel 6.19.14**. After upgrading the rig to **mainline 7.0.9** (6.19 hit EOL 2026-04-22), we re-ran the *identical* sweep to confirm the finding still holds and the kernel move didn't shift throughput.

**Method:** same `llama-bench` binaries, same source commit (`1acee6b`), same matrix, same nightly-HSA overlay — **only the kernel differed** (6.19.14 → 7.0.9). So any delta is attributable to the kernel alone.

## Result — the ~2.4× "OFF wins on prefill" finding holds; throughput is kernel-neutral

Headline row (`pp2048 @ d8392, fa=1`, rocwmma-ON vs runtime-OFF):

| Model | metric | 6.19.14 | 7.0.9 |
|---|---|---|---|
| Qwen3.5-27B Q8 | ON / OFF t/s | 117.08 / 282.52 | 120.93 / 284.06 |
| Qwen3.5-27B Q8 | **OFF/ON** | **2.41×** | **2.35×** |
| Qwen3.6-35B-A3B Q4 | ON / OFF t/s | 332.32 / 815.70 | 334.52 / 809.03 |
| Qwen3.6-35B-A3B Q4 | **OFF/ON** | **2.45×** | **2.42×** |

Differences are within run-to-run noise (~1–2% on absolute t/s). Decode (TG) flat on both kernels, as before. **Conclusion: the 6.19.14→7.0.9 kernel upgrade is throughput-neutral for this benchmark; the posted numbers stand.**

Raw logs in this directory are the full 7.0.9 matrix (pp512/tg128 + pp2048 depth-sweep 0/4196/8392 at `-fa 0,1`). The 6.19.14 baseline logs are in the parent directory.
