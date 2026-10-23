"""Is the GPU quantizer identical to the CPU one? Real expert matrices of Qwen3.6-35B-A3B-FP8, exact FP8 dequant.
ls_optimal: TritForge numpy (fix @ aec7981) on CPU vs gpu_quant.ls_optimal_torch on CUDA.
int4: the same torch code on CPU vs CUDA. Pass = every reconstructed weight equal, bit for bit."""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tritforge.quantize import absmax_ternary_quantize  # noqa: E402
from gpu_quant import int4_torch, ls_optimal_torch  # noqa: E402

MODEL = Path(r"C:\AI_Models\Qwen3.6-35B-A3B-FP8")
torch.set_grad_enabled(False)
where = {}
for f in sorted(MODEL.glob("*.safetensors")):
    with safe_open(str(f), "pt") as h:
        for k in h.keys():
            where[k] = f


def raw(name):
    with safe_open(str(where[name]), "pt") as h:
        return h.get_tensor(name)


def get(name):
    """Same exact FP8 block dequant as ternary_arms.get."""
    t = raw(name)
    if t.dtype != torch.float8_e4m3fn:
        return t.float()
    s = raw(name + "_scale_inv").float()
    r, c = t.shape
    br, bc = math.ceil(r / s.shape[0]), math.ceil(c / s.shape[1])
    return t.float() * s.repeat_interleave(br, 0)[:r].repeat_interleave(bc, 1)[:, :c]


def cpu_ls(W2):
    q = absmax_ternary_quantize(W2.numpy(), scaling_mode="block", scale_method="ls_optimal", block_size=128)
    sc = q.scales.astype(np.float16).astype(np.float32)
    r, i = W2.shape
    return torch.from_numpy((q.trits.astype(np.float32).reshape(r, i // 128, 128) * sc[..., None]).reshape(r, i))


P = "model.language_model.layers."
cases = [(l, e, p) for l in (0, 3, 20, 39) for e in (0, 77, 255) for p in ("gate_proj", "up_proj", "down_proj")]
total = {"ls": [0, 0], "int4": [0, 0]}
t0 = time.time()
for l, e, p in cases:
    W = get(f"{P}{l}.mlp.experts.{e}.{p}.weight").contiguous()
    a, b = cpu_ls(W), ls_optimal_torch(W.cuda()).cpu()
    c, d = int4_torch(W[None]), int4_torch(W[None].cuda()).cpu()
    n_ls, n_q = int((a != b).sum()), int((c != d).sum())
    total["ls"][0] += n_ls; total["ls"][1] += W.numel()
    total["int4"][0] += n_q; total["int4"][1] += W.numel()
    print(f"L{l:02d} e{e:03d} {p:9s} {tuple(W.shape)} | ls_optimal mismatches {n_ls} | int4 mismatches {n_q}", flush=True)

# full-mantissa float32 weights (what a rotated or rescaled arm would feed in): here the summation order
# changes the running sums in most positions, so this case has the power to catch a non-sequential sum
g = torch.Generator().manual_seed(0)
R = torch.randn(8192, 2048, generator=g) * 0.02
n_ls = int((cpu_ls(R) != ls_optimal_torch(R.cuda()).cpu()).sum())
n_q = int((int4_torch(R[None]) != int4_torch(R[None].cuda()).cpu()).sum())
total["ls"][0] += n_ls; total["ls"][1] += R.numel()
total["int4"][0] += n_q; total["int4"][1] += R.numel()
print(f"synthetic gaussian {tuple(R.shape)} | ls_optimal mismatches {n_ls} | int4 mismatches {n_q}", flush=True)

# one whole layer, every expert and projection, in the harness's 32-expert stacks
for p in ("gate_proj", "up_proj", "down_proj"):
    n = cnt = 0
    for s0 in range(0, 256, 32):
        Ws = torch.stack([get(f"{P}20.mlp.experts.{e}.{p}.weight") for e in range(s0, s0 + 32)])
        a = cpu_ls(Ws.reshape(-1, Ws.shape[-1])).reshape(Ws.shape)
        b = ls_optimal_torch(Ws.cuda().reshape(-1, Ws.shape[-1])).cpu().reshape(Ws.shape)
        n += int((a != b).sum()); cnt += Ws.numel()
        del Ws, a, b
    total["ls"][0] += n; total["ls"][1] += cnt
    print(f"L20 all 256 experts {p:9s} | ls_optimal mismatches {n} of {cnt:,}", flush=True)

for k, (bad, tot) in total.items():
    print(f"{k}: {bad} of {tot:,} weights differ -> {'BIT-EXACT' if bad == 0 else 'NOT EXACT'}")
print(f"{time.time() - t0:.0f}s")
