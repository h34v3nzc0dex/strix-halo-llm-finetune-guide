import os; os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL","1")
import torch, time, os as _o

# Bench the three GEMM transpose layouts that actually show up in a Linear-layer
# training step (per @woct0rdho on ROCm/TheRock#5314):
#   NT : y = x @ W.T            -> forward
#   NN : dX = dY @ W            -> grad wrt input
#   TN : dW = dY.T @ x          -> grad wrt weight
# All n x n, square, so each isolates the layout difference.
def gemm(layout, a, b):
    if layout == "NN": return a @ b
    if layout == "NT": return a @ b.t()
    if layout == "TN": return a.t() @ b
def tf(layout, n, dt, it=50):
    a = torch.randn(n, n, device='cuda', dtype=dt)
    b = torch.randn(n, n, device='cuda', dtype=dt)
    for _ in range(10): gemm(layout, a, b)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(it): c = gemm(layout, a, b)
    torch.cuda.synchronize()
    return (2*n**3)/((time.time()-t)/it)/1e12

print("HIPBLASLT=", _o.environ.get("TORCH_BLAS_PREFER_HIPBLASLT","unset"), "| torch", torch.__version__)
for layout in ("NT", "NN", "TN"):
    role = {"NT":"fwd  y=x@W.T", "NN":"dgrad dX=dY@W", "TN":"wgrad dW=dY.T@x"}[layout]
    print(f"  [{layout}] {role}")
    for n in (2048, 4096, 8192):
        print(f"      {n}^3  fp16 {tf(layout,n,torch.float16):5.1f}  bf16 {tf(layout,n,torch.bfloat16):5.1f} TFLOPS")
