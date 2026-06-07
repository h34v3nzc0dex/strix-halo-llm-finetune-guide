import os; os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL","1")
import torch, time, bitsandbytes as bnb
import bitsandbytes.functional as F
print("bnb:",bnb.__version__,"(built-from-source gfx1151) | torch",torch.__version__,"|",torch.cuda.get_device_name(0))
# 4-bit quant + dequant bandwidth on a 70B-ish proj shape
n=8192
w=torch.randn(n,n,device='cuda',dtype=torch.bfloat16)
qw,state=F.quantize_4bit(w, quant_type='nf4')
for _ in range(5): F.dequantize_4bit(qw,state)
torch.cuda.synchronize();t=time.time();it=50
for _ in range(it): dw=F.dequantize_4bit(qw,state)
torch.cuda.synchronize();dt=(time.time()-t)/it
gb=w.numel()*2/1e9  # bf16 output bytes moved
print(f"  4bit dequantize {n}x{n}: {dt*1e3:.2f} ms/iter, ~{gb/dt:.0f} GB/s effective")
