"""Does the GPU harness reproduce the CPU results? Same text, same arms, same quantizers (bit-exact)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
if len(sys.argv) >= 3:
    cpu_path = Path(sys.argv[1])
    gpu_path = Path(sys.argv[2])
else:
    cpu_path = HERE / "results" / "reference" / "ternary_arms_512x4_unseen.json"
    gpu_path = HERE / "results" / "reference" / "ternary_arms_512x4_unseen_arm2-int4_gpu.json"

cpu = json.loads(cpu_path.read_text())
gpu = json.loads(gpu_path.read_text())
cs, gs = cpu["summary"], gpu["summary"]
print(f"{'metric':38s} {'CPU':>14s} {'GPU':>14s} {'rel diff':>10s}")
rows = [("perplexity control", cs["perplexity"]["control"], gs["perplexity"]["control"])]
for a in ("arm2", "int4"):
    rows += [(f"perplexity {a}", cs["perplexity"][a], gs["perplexity"][a]),
             (f"ppl ratio {a}", cs["perplexity_ratio_vs_control"][a], gs["perplexity_ratio_vs_control"][a]),
             (f"top-1 agreement {a}", cs["top1_agreement_with_control"][a], gs["top1_agreement_with_control"][a]),
             (f"KL {a}", cs["mean_kl_control_to_arm_nats"][a], gs["mean_kl_control_to_arm_nats"][a])]
for name, c, g in rows:
    print(f"{name:38s} {c:14.8f} {g:14.8f} {abs(g - c) / abs(c):10.2e}")
print(f"{'minutes':38s} {cs['minutes']:14} {gs['minutes']:14}")

worst = {}
for lc, lg in zip(cpu["per_layer"], gpu["per_layer"]):
    for a in ("arm2", "int4"):
        for k in ("hidden_rel_err", "routing_overlap_vs_control"):
            d = abs(lg[a][k] - lc[a][k])
            if d > worst.get((a, k), (0, None))[0]:
                worst[(a, k)] = (d, lc["layer"], lc[a][k], lg[a][k])
        for p in ("gate", "up", "down"):
            d = abs(lg[f"{a}_weight_rel_err"][p] - lc[f"{a}_weight_rel_err"][p]) / lc[f"{a}_weight_rel_err"][p]
            if d > worst.get((a, "weight_rel_err"), (0, None))[0]:
                worst[(a, "weight_rel_err")] = (d, lc["layer"], lc[f"{a}_weight_rel_err"][p], lg[f"{a}_weight_rel_err"][p])
print("\nlargest per-layer difference (abs; weight error is relative):")
for (a, k), (d, layer, c, g) in sorted(worst.items()):
    print(f"  {a:5s} {k:28s} {d:.2e} at L{layer:02d}  (CPU {c:.6f}, GPU {g:.6f})")
print("\nseconds per layer: CPU", round(sum(l["seconds"] for l in cpu["per_layer"]) / len(cpu["per_layer"]), 1),
      "| GPU", round(sum(l["seconds"] for l in gpu["per_layer"]) / len(gpu["per_layer"]), 1))
