# bitsandbytes #1842 — gfx1151 4-bit support (build + datapoint)

For [bitsandbytes-foundation/bitsandbytes#1842](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1842):
confirmation that a from-source build runs 4-bit on gfx1151, plus the packaging gap
that bites stock-PyPI users on Strix Halo.

## Stack
- Ryzen AI MAX+ 395 / Radeon 8060S / **gfx1151**, Ubuntu 24.04, kernel 7.0.9, ROCm 7.13
- PyTorch 2.11.0+rocm7.13.0a
- bitsandbytes **0.50.0.dev0 built from source** for gfx1151
  (`-DROCM_VERSION=83 -DBNB_ROCM_ARCH=gfx1151`, with a `rocm83→rocm713` shim)

## The packaging gap
Stock PyPI `bitsandbytes` (0.49.2 at time of writing) ships ROCm binaries only up to
`libbitsandbytes_rocm72.so`, but a ROCm 7.13 / gfx1151 stack asks for
`libbitsandbytes_rocm83.so` → import/load fails with no gfx1151 binary present.
You currently have to build from source. That's the practical blocker for Strix Halo
users following an otherwise-working `unsloth[amd]` / QLoRA path.

## Repro
```
python repro_bnb_4bit.py    # nf4 quantize + timed dequantize, 8192x8192
```

## Result (`output.txt`)
```
bnb: 0.50.0.dev0 (built-from-source gfx1151) | torch 2.11.0+rocm7.13.0a | Radeon 8060S
  4bit dequantize 8192x8192: 0.89 ms/iter, ~152 GB/s effective
```

## Takeaway
4-bit (nf4) quantize + dequantize works correctly on gfx1151 once bnb is built from
source — ~152 GB/s effective on the dequant. The fix users actually need is a
**published gfx1151 / rocm8x wheel** so this works without a manual build.
