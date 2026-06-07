#!/usr/bin/env python3
# Reproduction of ROCm/ROCm#5807 (fp32 slow / fp16 fine on gfx1151) on a DIFFERENT
# stack than the reporter (they: Fedora 43 / torch 2.11.0a0+rocm7.11; here: Ubuntu
# 24.04 / kernel 7.0.9 / ROCm 7.13 / torch 2.11.0+rocm7.13). 4096^3 matmul TFLOPS.
import os; os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL","1")
import torch, time
def tflops(dtype,n=4096,it=50,warm=10):
    a=torch.randn(n,n,device='cuda',dtype=dtype); b=torch.randn(n,n,device='cuda',dtype=dtype)
    for _ in range(warm): (a@b)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(it): c=a@b
    torch.cuda.synchronize()
    return (2*n**3)/((time.time()-t)/it)/1e12
print("torch:", torch.__version__, "| gpu:", torch.cuda.get_device_name(0))
for dt in (torch.float16, torch.bfloat16, torch.float32):
    print(f"  {str(dt).split('.')[-1]:8s}: {tflops(dt):.1f} TFLOPS")
