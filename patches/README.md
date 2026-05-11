# Patches

Out-of-tree fixes the guide depends on. Each patch is `git`-style (apply with
`patch -p1` from the target project's root).

## `convert_lora_to_gguf-qwen35-vhead-reorder.patch`

**Targets:** `llama.cpp/convert_lora_to_gguf.py` (any recent build; tested against b867).

**What it fixes:** Qwen3.5 LoRA → GGUF conversion crashes with
`NotImplementedError: can't reshape the row size trivially` in
`LazyTensor.reshape`. The upstream Qwen3.5 model class calls
`_LinearAttentionVReorderBase._reorder_v_heads(dim=1, ...)` on the SSM
`out_proj` LoRA tensor — a reshape→permute→reshape sequence that
`LazyTensor`'s shape primitive doesn't support, because the conceptual
`(out, in)` shape would need to grow to 4D (`(out, num_k_heads, num_v_per_k,
head_v_dim)`) and `LazyTensor` encodes only `(*B.shape[:-1], A.shape[-1])` —
one in-dim.

**How the patch works:** Override `_reorder_v_heads` on the dynamic
`LoraModel` class. The reshape→permute→reshape is mathematically equivalent
to a column permutation along one axis. Compute the permutation directly and
apply it via `tensor.index_select(...)` on either `lora_B` (when `dim`
maps to rows = out-dim) or `lora_A`'s last dim (when `dim` maps to cols =
in-dim). No `LazyTensor` shape-primitive changes needed — bypasses the
limitation entirely.

**Numerical verification:** Synthetic tensors with `K=4, V=2, H=3` on both
dense and LoRA-factored inputs produce bit-identical results between the
parent's reshape→permute→reshape and the patch's `index_select`. Run the
verification snippet from the test we used:

```python
import torch

def parent(T, dim, K, V, H):
    s = list(T.shape); ns = s[:dim] + [K, V, H] + s[dim+1:]
    T = T.reshape(*ns); p = list(range(len(ns))); p[dim], p[dim+1] = p[dim+1], p[dim]
    return T.permute(*p).contiguous().reshape(*s)

def patched(T, dim, K, V, H):
    N = K*V*H; perm = torch.empty(N, dtype=torch.long)
    for new_p in range(N):
        v = new_p // (K*H); k = (new_p % (K*H)) // H; h = new_p % H
        perm[new_p] = k*V*H + v*H + h
    return T.index_select(dim, perm)

# 2D case, dim=0
T = torch.randn(24, 5);   assert torch.equal(parent(T,0,4,2,3), patched(T,0,4,2,3))
# 2D case, dim=1
T = torch.randn(5, 24);   assert torch.equal(parent(T,1,4,2,3), patched(T,1,4,2,3))
# LoRA-factored dim=1 (columns of conceptual W = columns of A)
r=4; A=torch.randn(r,24); B=torch.randn(5,r)
assert torch.allclose(parent(B@A, 1, 4,2,3),  B @ patched(A, -1, 4,2,3))
# LoRA-factored dim=0 (rows of conceptual W = rows of B)
A=torch.randn(r,5); B=torch.randn(24,r)
assert torch.allclose(parent(B@A, 0, 4,2,3),  patched(B, 0, 4,2,3) @ A)
print("all good")
```

**Apply:**

```bash
cd /path/to/llama.cpp
patch -p1 < /path/to/this/repo/patches/convert_lora_to_gguf-qwen35-vhead-reorder.patch
# verify it took:
grep -n "_reorder_v_heads" convert_lora_to_gguf.py
# should now find a staticmethod override inside the LoraModel class
```

**Use:**

```bash
cd /path/to/llama.cpp
python3 convert_lora_to_gguf.py /path/to/checkpoint-N \
    --outfile /path/to/checkpoint-N/gguf/lora-f16.gguf \
    --outtype f16
```

A correct Qwen3.5-27B LoRA at `r=128, alpha=256` produces ~992 tensors / ~1.87 GiB f16 GGUF in ~8 s on this hardware.

**Worth upstreaming.** This is a real gap in `ggml-org/llama.cpp` — Qwen3.5 LoRA conversion is broken without it. If you submit a PR, link this README so the verification snippet travels with the change.
