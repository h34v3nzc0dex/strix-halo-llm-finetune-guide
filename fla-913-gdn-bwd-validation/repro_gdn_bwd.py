#!/usr/bin/env python3
# Cross-arch control for fla-org/flash-linear-attention#913 (GDN bwd "misaligned
# address" reported on NVIDIA Blackwell). Runs chunk_gated_delta_rule fwd+bwd at
# our production Qwen3.5-27B LoRA shapes on AMD gfx1151. Expected: completes clean.
import os; os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL","1")
import torch, triton
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
print("torch:", torch.__version__, "| triton:", triton.__version__, "| gpu:", torch.cuda.get_device_name(0))
B,H,T,K,V = 1,16,8192,128,128; dt=torch.bfloat16; dev="cuda"
q=torch.randn(B,T,H,K,device=dev,dtype=dt,requires_grad=True)
k=torch.randn(B,T,H,K,device=dev,dtype=dt,requires_grad=True)
v=torch.randn(B,T,H,V,device=dev,dtype=dt,requires_grad=True)
g=torch.rand(B,T,H,device=dev,dtype=torch.float32,requires_grad=True)
beta=torch.rand(B,T,H,device=dev,dtype=dt,requires_grad=True)
o,_=chunk_gated_delta_rule(q,k,v,g,beta,use_qk_l2norm_in_kernel=True)
o.sum().backward(); torch.cuda.synchronize()
print("RESULT: fwd+bwd OK, no misaligned address. grad shapes",
      tuple(q.grad.shape), tuple(k.grad.shape), tuple(v.grad.shape))
