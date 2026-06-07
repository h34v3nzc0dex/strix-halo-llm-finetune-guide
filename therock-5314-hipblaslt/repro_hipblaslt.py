import os; os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL","1")
import torch, time, os as _o
def tf(n,dt,it=50):
    a=torch.randn(n,n,device='cuda',dtype=dt);b=torch.randn(n,n,device='cuda',dtype=dt)
    for _ in range(10):(a@b)
    torch.cuda.synchronize();t=time.time()
    for _ in range(it):c=a@b
    torch.cuda.synchronize();return (2*n**3)/((time.time()-t)/it)/1e12
print("HIPBLASLT=",_o.environ.get("TORCH_BLAS_PREFER_HIPBLASLT","unset"),"| torch",torch.__version__)
for n in (2048,4096,8192):
    print(f"  {n}^3  fp16 {tf(n,torch.float16):.1f}  bf16 {tf(n,torch.bfloat16):.1f} TFLOPS")
