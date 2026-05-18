# Qwen3.6 Q4 bench on gfx1151

Follow-up to a r/StrixHalo comment thread — peer commenter reported
~45 t/s on Qwen3.6-35B-A3B Q4 and ~20 t/s on Qwen3.6-27B-MTP Q?, and
suggested we'd see "really bad" numbers on our setup. Ran the comparison
to find out.

## Setup

- Corsair AI Workstation 300, Ryzen AI MAX+ 395, Radeon 8060S (gfx1151)
- Ubuntu 24.04, kernel 6.19.14 mainline, ROCm 7.13 nightly
- 128 GB unified LPDDR5X
- Models downloaded fresh from unsloth's HF repos
- `llama-bench`, fa on, no-mmap, 999 layers on GPU
- 3 reps per condition

### Build notes

- **35B-A3B**: ran fine on our self-built b867 (`5207d120e`)
- **27B-MTP**: **failed to load on b867** with `missing tensor 'blk.64.ssm_conv1d.weight'` — the MTP layer's SSM tensors aren't recognized by b867. Loaded fine on **lemonade-sdk b1270 prebuilt** (build `39cf5d6`). MTP support in llama.cpp is newer than b867.

## Results

### Qwen3.6-35B-A3B-UD Q4_K_M (raw inference, b867)

| test  | t/s             |
|-------|-----------------|
| pp64  | 522.47 ± 1.74   |
| tg64  | **50.23 ± 0.29** |

User reported: ~45 t/s. We hit **~50 t/s on tg here** — same hardware-class. A3B = 3B active params per token, hence the headline-grabbing speed.

### Qwen3.6-27B-MTP Q4_K_M (raw inference, lemonade b1270)

| test  | t/s             |
|-------|-----------------|
| pp64  | 240.26 ± 22.23  |
| tg64  | **12.00 ± 0.05** |
| pp512 | 333.95 ± 7.08   |
| tg128 | **12.05 ± 0.03** |

User reported: ~20 t/s. We hit **~12 t/s on raw inference** without MTP speculative decoding. User's ~20 t/s implies the MTP speculative path gives ~1.67× speedup, which lines up with typical MTP gains in the literature.

### MTP speculative-decoding test — attempted, failed

```
llama-cli -m Qwen3.6-27B-MTP-Q4_K_M.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -p "..." -n 200 -ngl 999 --no-mmap -fa on
```

Process pegged a CPU core at 99% with **0% GPU usage**, hung for 17+ minutes before we killed it. The model loaded but the `draft-mtp` speculative path doesn't seem to use the GPU on the lemonade b1270 build for gfx1151. Either a flag-combo gap or the GPU-offload path for draft-mtp isn't wired up. Worth a separate issue against lemonade or llama.cpp.

## Reading

The original critique was that our previous Qwen3.5-27B Q8 dense numbers (~7.5 t/s tg) were "really bad". Looking at it through these new data points:

| Change | speedup |
|---|---|
| Qwen3.5-27B Q8 → Qwen3.5-27B Q4 (quant only) | ~1.6× (half the per-token memory bandwidth at decode) |
| 27B Q4 dense → 35B-A3B Q4 MoE (3B active) | ~4× (cuts compute per token) |
| 27B-MTP Q4 dense → 27B-MTP Q4 + speculation | ~1.67× (MTP draft head + verify) |

So the original gap was a workload-shape difference, not hardware. User's stack picks the right axes for raw throughput: Q4 quant + MoE A3B + MTP speculation where applicable. Apples-to-apples on 35B-A3B Q4, our hardware is ~12% faster.

## Logs

- `bench-qwen36-35b-a3b-raw-pp64-tg64-b867.log` — 35B-A3B numbers (b867)
- `bench-qwen36-27b-mtp-raw-pp64-tg64-lemonade-b1270.log` — 27B-MTP short bench (lemonade)
- `bench-qwen36-27b-mtp-raw-pp512-tg128-lemonade-b1270.log` — 27B-MTP longer bench (lemonade)
- `bench-qwen36-27b-mtp-FAILED-on-b867-missing-tensor.log` — the b867 load failure (for the record)
