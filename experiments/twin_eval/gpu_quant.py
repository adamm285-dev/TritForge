"""GPU twins of the harness quantizers, built to reproduce the CPU results bit for bit.

ls_optimal_torch mirrors TritForge's numpy 'ls_optimal' block quantizer (fix/review-wave1 @ aec7981,
quantize.py lines 147-160) step by step:
  - sort |w| per 128-block, descending (exact: sorting does no arithmetic)
  - running sum added strictly left to right in float32, as np.cumsum does
    (torch.cumsum on GPU is a parallel scan that rounds in a different order)
  - objective c*c/k in float32, argmax taking the first maximum (numpy and torch agree on ties)
  - trits: |w| >= the k-th largest |w| (ties at the threshold included, as in numpy)
  - scale c_k / k: numpy promotes float32 / int64 to float64 and rounds straight to FP16;
    torch would round through float32 first, so this one small step is done by numpy itself
int4_torch is the harness's plain 4-bit RTN (group 128, FP16 scale absmax/7, levels -8..7), with the one
division done tensor-by-tensor so CUDA rounds it exactly like the CPU.
Checked by gpu_quant_check.py (real FP8 experts, one whole layer, and full-mantissa synthetic weights).
"""
import numpy as np
import torch


def ls_optimal_torch(W2, bs=128):
    """(rows, in) float32 on any device -> reconstruction trits * FP16 scale, same device."""
    r, i = W2.shape
    b = W2.reshape(r, i // bs, bs)
    a = b.abs()
    s = torch.sort(a, dim=-1, descending=True).values
    c = s.clone()
    for j in range(1, bs):
        c[..., j] += c[..., j - 1]
    ks = torch.arange(1, bs + 1, device=W2.device, dtype=torch.float32)
    k = torch.argmax(c * c / ks, dim=-1, keepdim=True) + 1
    thr = torch.gather(s, -1, k - 1)
    c_k = torch.gather(c, -1, k - 1).cpu().numpy()
    alpha = (c_k / k.cpu().numpy()).astype(np.float16).astype(np.float32)
    t = torch.where(a >= thr, torch.sign(b), torch.zeros_like(b))
    return (t * torch.from_numpy(alpha).to(W2.device)).reshape(r, i)


def int4_torch(W, gs=128):
    """(c, out, in) float32 -> plain symmetric 4-bit RTN reconstruction; all-zero groups (dead neurons) -> 0."""
    c, o, i = W.shape
    g = W.reshape(c * o, i // gs, gs)
    m = g.abs().amax(-1, keepdim=True)
    # divide by a tensor, not the Python number 7.0: CUDA turns division by a scalar into multiplication by
    # its reciprocal, which rounds differently from the CPU's exact division (55% of quotients 1 ulp off)
    s = (m / torch.full_like(m, 7.0)).half().float()
    q = torch.where(s > 0, (g / s.clamp_min(1e-30)).round().clamp(-8, 7), torch.zeros_like(g))
    return (q * s).reshape(c, o, i)
