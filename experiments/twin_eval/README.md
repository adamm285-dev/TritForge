# End-to-End Twin Measurement: `Qwen3.6-35B-A3B-FP8`

## Overview
This directory contains the reproducible layer-streaming evaluation harness comparing **TritForge** calibration-free ternary quantization arms against an unquantized `FP8` control model and a symmetric 4-bit round-to-nearest (`int4` RTN) baseline.

### Target Model
- **Control Model:** `Qwen/Qwen3.6-35B-A3B-FP8` (default path: `C:\AI_Models\Qwen3.6-35B-A3B-FP8`).
- **Architecture:** Hybrid `Qwen3.5-MoE` decoder stack (40 layers, 3:1 ratio of Gated-DeltaNet linear attention to Grouped-Query Attention, zero-centered RMSNorms, and 256 routed SwiGLU experts per layer with top-8 active per token).

### Measured Arms
1. **`control`:** Unchanged `FP8` source model (dequantized exactly per `128 x 128` block in IEEE `float32`).
2. **`arm1`:** Routed experts replaced with TritForge heuristic ternary reconstruction (`optimal`).
3. **`arm2`:** Routed experts replaced with exact per-block least-squares optimal ternary reconstruction (`ls_optimal`).
4. **`int4`:** Routed experts replaced with plain symmetric 4-bit round-to-nearest (RTN) quantization, group size 128, `FP16` scale (`4.125 bits/weight`).
5. **`mix_kurt` / `mix_err`:** Exact-optimal ternary (`arm2`) with the top 10% outlier-heavy expert matrices per layer retained in 4-bit, selected by kurtosis (`mix_kurt`) or ternary squared error (`mix_err`) (`1.965 bits/weight`).

All non-routed components (attention, linear attention, router gates, shared experts, layer norms, embeddings, and language-modeling head) remain untouched across every arm.

---

## Summary of Verified Results (`4 x 512` Unseen Tokens)

| Quantization Arm | Routed Expert Bits/Weight | Perplexity (PPL) ↓ | PPL Ratio vs. Control ↓ | Top-1 Token Agreement ↑ | Mean KL Divergence (nats) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Control (`FP8` exact dequant)** | `8.000` | 11.9111 | `1.0000x` | 100.0% | `0.0000` |
| **`arm1` (Heuristic ternary `optimal`)** | `1.725` | 16.7450 | `1.4058x` | 68.8% | `0.5351` |
| **`arm2` (Closed-form LS ternary `ls_optimal`, CPU)** | `1.725` | 15.7743 | `1.3243x` | 70.5% | `0.4592` |
| **`arm2` (Closed-form LS ternary `ls_optimal`, GPU)** | `1.725` | 15.7742 | `1.3243x` | 70.5% | `0.4592` |
| **`mix_kurt` (`arm2` + top 10% `int4` by kurtosis)** | `1.965` | 14.4784 | `1.2155x` | 74.4% | `0.3514` |
| **`mix_err` (`arm2` + top 10% `int4` by squared error)** | `1.965` | 14.2451 | `1.1959x` | 74.2% | `0.3363` |
| **`int4` (Plain symmetric 4-bit RTN `g128`, CPU)** | `4.125` | 12.0733 | `1.0136x` | 91.7% | `0.0352` |
| **`int4` (Plain symmetric 4-bit RTN `g128`, GPU)** | `4.125` | 12.0737 | `1.0136x` | 91.7% | `0.0352` |

---

## Environment and Dependencies
Experiments use the pinned dependencies in [`requirements.txt`](requirements.txt) (default virtual environment path: `C:\AI_Models\venv-gpu`):
- `torch==2.13.0+cu130`
- `transformers==5.15.0`
- `safetensors==0.8.0`
- `tokenizers==0.22.2`
- `numpy==2.5.2`
- `psutil==7.2.2`

---

## Execution Commands

### 1. Bit-Exactness Sanity Check
Validates that the CUDA quantizers (`ls_optimal_torch`, `int4_torch`) match the CPU quantizers bit-for-bit across real expert layers (`859,832,320` weights) and synthetic Gaussian weights:
```powershell
C:\AI_Models\venv-gpu\Scripts\python.exe experiments/twin_eval/gpu_quant_check.py
```

### 2. End-to-End Evaluation Run
Executes the 40-layer forward evaluation on GPU (`TA_DEVICE=cuda`) over the held-out unseen evaluation texts (`TA_TEXT=unseen`, verified against [`SHA256SUMS.txt`](SHA256SUMS.txt)) comparing `arm2` and `int4`:
```powershell
$env:TA_DEVICE="cuda"
$env:TA_TEXT="unseen"
$env:TA_ARMS="arm2,int4"
C:\AI_Models\venv-gpu\Scripts\python.exe experiments/twin_eval/ternary_arms.py 512 4
```
*Checkpoints are saved after each layer to `results/*.ckpt.pt` (automatically gitignored).*

### 3. GPU vs. CPU Reproduction Verification
Compares the reproduced output JSON against the reference run:
```powershell
C:\AI_Models\venv-gpu\Scripts\python.exe experiments/twin_eval/gpu_vs_cpu.py experiments/twin_eval/results/ternary_arms_512x4_unseen_arm2-int4_gpu.json experiments/twin_eval/results/reference/ternary_arms_512x4_unseen_arm2-int4_gpu.json
```
