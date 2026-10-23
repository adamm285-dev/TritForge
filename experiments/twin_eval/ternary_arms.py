"""First real ternary-vs-original measurement, on Qwen3.6-35B-A3B-FP8 (scaled-down twin of the 2.4T).

Official `transformers` Qwen3.5-MoE modules, run layer by layer in float32 (CPU, or GPU with TA_DEVICE=cuda:
TF32 off, one layer on the card at a time, original expert weights kept in system RAM).
Three hidden-state streams go through every layer, each keeping its own history:
  control : the unchanged FP8 model (dequantized exactly, block 128x128)
  arm1    : routed experts replaced by TritForge's 'optimal' heuristic ternary reconstruction (today's baseline)
  arm2    : routed experts replaced by 'ls_optimal' exact per-block optimum (C-8 fix)
  int4    : routed experts replaced by plain 4-bit round-to-nearest, symmetric, FP16 scale per 128 weights
            (~4.125 bits/weight; as simple as the ternary arms, for a same-pipeline comparison)
Everything else (attention, linear attention, router, shared expert, norms, embeddings, LM head) is untouched.
Quantizers come from the frozen export of fix/review-wave1 @ aec7981.

Usage: python ternary_arms.py [tokens_per_seq] [n_seqs]
"""
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from safetensors import safe_open

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tritforge.quantize import absmax_ternary_quantize  # noqa: E402
from gpu_quant import int4_torch, ls_optimal_torch  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig  # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeTextRotaryEmbedding,
)

MODEL = Path(r"C:\AI_Models\Qwen3.6-35B-A3B-FP8")
SEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 512
NSEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAX_LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else None  # debug: run only the first N layers
TEXT = os.environ.get("TA_TEXT", "default")  # "unseen": documents written 2026-10-22/23, after the model's training
# arms: arm1 = heuristic ternary, arm2 = exact-optimal ternary, int4 = plain 4-bit,
# mix_kurt / mix_err = arm2 with the top MIX_FRAC of each layer's expert matrices in 4-bit,
#   picked by outlier-heaviness (kurtosis) or by largest ternary squared error
DEFAULT_ARMS = "arm1,arm2,int4"
ARMS = tuple(os.environ.get("TA_ARMS", DEFAULT_ARMS).split(","))
MIX_FRAC = 0.10
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT = RESULTS_DIR / (f"ternary_arms_{SEQ}x{NSEQ}{'' if TEXT == 'default' else '_' + TEXT}"
                     f"{'' if ','.join(ARMS) == DEFAULT_ARMS else '_' + '-'.join(ARMS)}"
                     f"{'' if MAX_LAYERS is None else f'_debug{MAX_LAYERS}L'}"
                     f"{'' if os.environ.get('TA_DEVICE', 'cpu') == 'cpu' else '_gpu'}.json")
torch.set_grad_enabled(False)
DEV = torch.device(os.environ.get("TA_DEVICE", "cpu"))
if DEV.type == "cuda":  # keep full FP32 everywhere: no TF32 shortcuts in matmuls or the DeltaNet conv
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
t_start = time.time()

# ------------------------------------------------------------------ tensor map from shard headers
where = {}
handles = {}
for f in sorted(MODEL.glob("*.safetensors")):
    with safe_open(str(f), "pt") as h:
        for k in h.keys():
            where[k] = f


def raw(name):
    f = where[name]
    if f not in handles:
        # one open shard at a time: on Windows an open handle keeps the whole shard (copy-on-write mapping) in the
        # commit charge, and keeping every shard open exhausted the commit limit around layer 32
        handles.clear()
        handles[f] = safe_open(str(f), "pt")
    return handles[f].get_tensor(name)


def get(name):
    """Exact float32 value of a checkpoint tensor (FP8 block-dequantized when needed)."""
    t = raw(name)
    if t.dtype == torch.float8_e4m3fn:
        s = raw(name + "_scale_inv").float()
        r, c = t.shape
        br, bc = math.ceil(r / s.shape[0]), math.ceil(c / s.shape[1])
        s = s.repeat_interleave(br, 0)[:r].repeat_interleave(bc, 1)[:, :c]
        return t.float() * s
    return t.float()


layer_key = next(k for k in where if re.search(r"layers\.0\.", k) and "visual" not in k and "mtp" not in k)
P = layer_key[: layer_key.index("layers.0.")]          # e.g. "model.language_model."
embed_name = next(k for k in where if k.endswith("embed_tokens.weight") and "visual" not in k and "mtp" not in k)
norm_name = P + "norm.weight"
head_name = next(k for k in where if k.endswith("lm_head.weight"))
print(f"tensors {len(where)} | layer prefix '{P}' | embed '{embed_name}' | head '{head_name}'", flush=True)

root = json.loads((MODEL / "config.json").read_text())
cfg = Qwen3_5MoeTextConfig(**root.get("text_config", root))
cfg._attn_implementation = "eager"
L, E = cfg.num_hidden_layers, cfg.num_experts

# ------------------------------------------------------------------ evaluation text (local, mixed prose / markdown / code)
tok = AutoTokenizer.from_pretrained(str(MODEL))
if TEXT == "unseen":
    text_dir = Path(os.environ.get("TA_TEXT_DIR", r"C:\AI_Models\twin_eval\unseen_text"))
    sums_file = HERE / "SHA256SUMS.txt"
    if sums_file.exists():
        import hashlib
        expected_sums = {}
        for line in sums_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            expected_sums[Path(parts[1].lstrip("*")).name] = parts[0]
        for name in ("ADVERSARIAL_REVIEW_2026-10-23.md", "ADVERSARIAL_REVIEW_WAVE2_2026-10-23.md",
                     "2026-10-23_review_wave1.md", "tritslack_spec.md"):
            fp = text_dir / name
            if not fp.exists():
                raise FileNotFoundError(f"Required eval text {fp} not found in {text_dir}")
            actual_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
            if name in expected_sums and actual_sha != expected_sums[name]:
                raise ValueError(f"SHA-256 mismatch for {fp}: expected {expected_sums[name]}, got {actual_sha}")
    sources = [text_dir / n for n in ("ADVERSARIAL_REVIEW_2026-10-23.md", "ADVERSARIAL_REVIEW_WAVE2_2026-10-23.md",
                                      "2026-10-23_review_wave1.md", "tritslack_spec.md")]
else:
    sources = [MODEL / "README.md", Path(r"C:\Python313\LICENSE.txt"),
               Path(r"C:\Python313\Lib\json\decoder.py"), Path(r"C:\Python313\Lib\textwrap.py")]
seqs = []
for p in sources:
    ids = tok(p.read_text(encoding="utf-8", errors="replace"))["input_ids"]
    if len(ids) >= SEQ:
        seqs.append(ids[:SEQ])
    if len(seqs) == NSEQ:
        break
ids = torch.tensor(seqs)
B, T = ids.shape
print(f"eval tokens: {B} x {T} from {[p.name for p in sources[:B]]}", flush=True)

emb = get(embed_name)
h0 = emb[ids].to(DEV)
del emb
_t, _q = 1.6 + 16 / 128, 4 + 16 / 128
BITS = {"arm1": _t, "arm2": _t, "int4": _q,
        "mix_kurt": (1 - MIX_FRAC) * _t + MIX_FRAC * _q, "mix_err": (1 - MIX_FRAC) * _t + MIX_FRAC * _q}
streams = {name: h0.clone() for name in ("control",) + ARMS}

pos = torch.arange(T, device=DEV).view(1, 1, -1).expand(4, B, -1)
text_pos, pos3 = pos[0], pos[1:]
rope = Qwen3_5MoeTextRotaryEmbedding(config=cfg).to(DEV)
pe = rope(h0, pos3)
masks = {
    "full_attention": create_causal_mask(config=cfg, inputs_embeds=h0, attention_mask=None, past_key_values=None,
                                         position_ids=text_pos, allow_is_causal_skip=False),
    "linear_attention": create_recurrent_attention_mask(config=cfg, inputs_embeds=h0, attention_mask=None),
}

# ------------------------------------------------------------------ helpers
def recon(W, name):
    """Reconstruction of a small stack of expert matrices (c, out, in); block/group 128 along `in`, row-wise.
    On GPU, arm2 and int4 use the bit-exact twins in gpu_quant.py; arm1 always uses TritForge's numpy quantizer."""
    c, o, i = W.shape
    if name == "int4":
        return int4_torch(W.to(DEV))
    if name == "arm2" and DEV.type == "cuda":
        return ls_optimal_torch(W.to(DEV).reshape(-1, i)).reshape(c, o, i)
    method = "optimal" if name == "arm1" else "ls_optimal"
    q = absmax_ternary_quantize(W.cpu().reshape(-1, i).contiguous().numpy(), scaling_mode="block",
                                scale_method=method, block_size=128)
    sc16 = q.scales.astype(np.float16).astype(np.float32)  # FP16 scales, exactly as the .trit container stores them
    return torch.from_numpy((q.trits.astype(np.float32).reshape(-1, i // 128, 128) * sc16[..., None])
                            .reshape(c, o, i)).to(DEV)


def fill_arm(layer, gu0, dn0, name, chunk=int(os.environ.get("TA_CHUNK", "32"))):
    """Write the arm's expert weights straight into the layer, `chunk` experts at a time (bounded memory).
    Mixed arms: exact-optimal ternary everywhere, then the top MIX_FRAC of the layer's (expert, projection)
    matrices re-done in 4-bit, chosen by kurtosis (mix_kurt) or by ternary squared error (mix_err).
    Returns per-projection weight relative error (and, for mixed arms, how many matrices of each got 4-bit)."""
    gu, dn = layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj
    I = gu0.shape[1] // 2
    base = "arm2" if name.startswith("mix_") else name
    views = {"gate": (gu0[:, :I], gu[:, :I]), "up": (gu0[:, I:], gu[:, I:]), "down": (dn0, dn)}
    sq_err = {p: torch.zeros(gu0.shape[0], device=DEV) for p in views}
    kurt = {p: torch.zeros(gu0.shape[0], device=DEV) for p in views}
    den = {p: 0.0 for p in views}
    for s in range(0, gu0.shape[0], chunk):
        for part, (src_all, dst_all) in views.items():
            src = src_all[s:s + chunk].to(DEV)
            r = recon(src, base)
            dst_all[s:s + chunk].copy_(r)
            sq_err[part][s:s + chunk] = ((r - src) ** 2).sum(dim=(1, 2))
            den[part] += float((src.double() ** 2).sum())
            if name == "mix_kurt":
                flat = src.reshape(src.shape[0], -1)
                z = (flat - flat.mean(1, keepdim=True)) / flat.std(1, keepdim=True).clamp_min(1e-30)
                kurt[part][s:s + chunk] = (z ** 4).mean(1)
            del r
    picked = {p: 0 for p in views}
    if name.startswith("mix_"):
        key = kurt if name == "mix_kurt" else sq_err
        flat_scores = torch.cat([key[p] for p in views])           # order: gate experts, up experts, down experts
        n_pick = int(round(MIX_FRAC * flat_scores.numel()))
        for j in torch.topk(flat_scores, n_pick).indices.tolist():
            part = list(views)[j // gu0.shape[0]]
            e = j % gu0.shape[0]
            src_all, dst_all = views[part]
            src1 = src_all[e:e + 1].to(DEV)
            r = recon(src1, "int4")
            dst_all[e:e + 1].copy_(r)
            sq_err[part][e] = float(((r - src1) ** 2).sum())
            picked[part] += 1
    out = {p: math.sqrt(float(sq_err[p].sum()) / den[p]) for p in views}
    if name.startswith("mix_"):
        out["int4_matrices"] = picked
    return out


def load_layer(i):
    with torch.device("meta"):
        layer = Qwen3_5MoeDecoderLayer(cfg, i)
    layer = layer.to_empty(device=DEV).float()
    sd, loaded = layer.state_dict(), set()
    pre = f"{P}layers.{i}."
    for k in sd:
        if k in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj") and pre + k not in where:
            # fill expert by expert, straight into the module (no full-size temporary list/stack/cat)
            dst = sd[k]
            for e in range(E):
                if k == "mlp.experts.down_proj":
                    dst[e].copy_(get(f"{pre}mlp.experts.{e}.down_proj.weight"))
                else:
                    half = dst.shape[1] // 2
                    dst[e, :half].copy_(get(f"{pre}mlp.experts.{e}.gate_proj.weight"))
                    dst[e, half:].copy_(get(f"{pre}mlp.experts.{e}.up_proj.weight"))
            loaded.add(k)
            continue
        val = get(pre + k)
        if tuple(val.shape) != tuple(sd[k].shape):
            raise ValueError(f"layer {i} {k}: checkpoint {tuple(val.shape)} vs module {tuple(sd[k].shape)}")
        sd[k].copy_(val)
        loaded.add(k)
    missing = set(sd) - loaded
    if missing:
        raise ValueError(f"layer {i}: params not loaded {missing}")
    return layer.eval()


routing = {}


def hook(_m, _inp, out):
    routing["idx"] = out[2].clone()


# ------------------------------------------------------------------ layer loop
per_layer, start = [], 0
CKPT = OUT.with_suffix(".ckpt.pt")
if CKPT.exists():  # resume after an interruption
    ck = torch.load(CKPT, weights_only=False)
    if torch.equal(ck["ids"], ids) and tuple(ck["arms"]) == ARMS:
        streams = {k: v.to(DEV) for k, v in ck["streams"].items()}
        per_layer, start = ck["per_layer"], ck["next_layer"]
        print(f"resuming from layer {start}", flush=True)
for i in range(start, L if MAX_LAYERS is None else MAX_LAYERS):
    t0 = time.time()
    layer = load_layer(i)
    layer.mlp.gate.register_forward_hook(hook)
    # originals stay in system RAM; only the layer being computed lives on the device
    gu0, dn0 = layer.mlp.experts.gate_up_proj.detach().cpu().clone(), layer.mlp.experts.down_proj.detach().cpu().clone()
    I = gu0.shape[1] // 2
    rec = {"layer": i, "type": cfg.layer_types[i]}
    idx = {}
    for name in ("control",) + ARMS:
        if name == "control":
            layer.mlp.experts.gate_up_proj.copy_(gu0)
            layer.mlp.experts.down_proj.copy_(dn0)
        else:
            rec[f"{name}_weight_rel_err"] = fill_arm(layer, gu0, dn0, name)
        streams[name] = layer(streams[name], position_embeddings=pe, attention_mask=masks[cfg.layer_types[i]],
                              position_ids=text_pos, past_key_values=None)
        idx[name] = routing["idx"]
    c = streams["control"]
    for name in ARMS:
        a = streams[name]
        overlap = np.mean([len(set(x.tolist()) & set(y.tolist())) / x.numel() for x, y in zip(idx["control"], idx[name])])
        rec[name] = {"hidden_rel_err": float((a - c).norm() / c.norm()),
                     "hidden_cos": float((a * c).sum() / (a.norm() * c.norm())),
                     "routing_overlap_vs_control": float(overlap)}
    rec["seconds"] = round(time.time() - t0, 1)
    per_layer.append(rec)
    print(f"L{i:02d} {rec['type'][:6]:6s} | " + " | ".join(
        f"{a} hid err {rec[a]['hidden_rel_err']:.3f} route {rec[a]['routing_overlap_vs_control']:.2f}" for a in ARMS)
        + f" | {rec['seconds']}s | private {psutil.Process().memory_info().private / 2**30:.1f} GB", flush=True)
    del layer, gu0, dn0
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    tmp = CKPT.with_suffix(".tmp")
    torch.save({"ids": ids, "arms": list(ARMS), "streams": {k: v.cpu() for k, v in streams.items()},
                "per_layer": per_layer, "next_layer": i + 1}, tmp)
    tmp.replace(CKPT)

# ------------------------------------------------------------------ final norm + LM head, metrics per chunk
norm = Qwen3_5MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(DEV)
norm.weight.data = get(norm_name).to(DEV)
head = get(head_name).to(DEV)
hid = {k: norm(v) for k, v in streams.items()}
targets = ids[:, 1:].reshape(-1).to(DEV)
nll = {k: 0.0 for k in hid}
agree = {a: 0 for a in ARMS}
kl = {a: 0.0 for a in ARMS}
n = 0
flat = {k: v[:, :-1].reshape(-1, cfg.hidden_size) for k, v in hid.items()}
for s in range(0, targets.numel(), 256):
    tgt = targets[s:s + 256]
    lp = {k: torch.log_softmax(v[s:s + 256] @ head.T, dim=-1) for k, v in flat.items()}
    for k in lp:
        nll[k] += float(-lp[k].gather(1, tgt[:, None]).sum())
    top_c = lp["control"].argmax(-1)
    pc = lp["control"].exp()
    for k in ARMS:
        agree[k] += int((lp[k].argmax(-1) == top_c).sum())
        kl[k] += float((pc * (lp["control"] - lp[k])).sum())
    n += tgt.numel()

ppl = {k: math.exp(v / n) for k, v in nll.items()}
summary = {
    "model": "Qwen/Qwen3.6-35B-A3B-FP8", "eval_tokens": n, "seq": SEQ, "nseq": NSEQ, "device": str(DEV),
    "perplexity": ppl,
    "routed_expert_bits_per_weight": BITS,
    "perplexity_ratio_vs_control": {k: ppl[k] / ppl["control"] for k in ARMS},
    "top1_agreement_with_control": {k: agree[k] / n for k in agree},
    "mean_kl_control_to_arm_nats": {k: kl[k] / n for k in kl},
    "minutes": round((time.time() - t_start) / 60, 1),
}
json.dump({"summary": summary, "per_layer": per_layer}, open(OUT, "w"), indent=1)
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=1))
