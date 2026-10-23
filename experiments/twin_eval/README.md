# End-to-End Twin Measurement: Qwen3.6-35B-A3B-FP8

## Overview
This experiment directory provides the first real end-to-end measurement comparing TritForge ternary quantization arms against an unquantized control model and standard int4 RTN baselines.

### Target Model
- **Control Model:** Unchanged FP8 twin `Qwen/Qwen3.6-35B-A3B-FP8` located at `C:\AI_Models\Qwen3.6-35B-A3B-FP8`.
- **Architecture:** Hybrid Qwen3.5-MoE architecture with DeltaNet linear attention, Gated DeltaNet, GQA attention, and 256 routed MoE experts per layer (8 active per token).

### Measured Arms
1. **control:** Unchanged FP8 source model (dequantized exactly block-by-block 128x128 in FP32).
2. **arm1:** Routed experts replaced with TritForge heuristic ternary reconstruction (`optimal`, baseline).
3. **arm2:** Routed experts replaced with exact per-block least-squares optimal ternary reconstruction (`ls_optimal`).
4. **int4:** Routed experts replaced with plain 4-bit round-to-nearest (RTN) symmetric quantization, group size 128, FP16 scale (~4.125 bits/weight).
5. **mix_kurt / mix_err:** Exact-optimal ternary (`arm2`) with the top 10% outlier-heavy expert matrices retained in 4-bit, selected by kurtosis (`mix_kurt`) or ternary squared error (`mix_err`).

All non-routed components (attention, linear attention, router gates, shared experts, layer norms, embeddings, LM head) remain untouched.

### Evaluated Metrics
- **Perplexity (PPL):** Autoregressive next-token prediction perplexity across evaluation tokens.
- **Perplexity Ratio vs. Control:** Ratio of arm perplexity to control perplexity ($PPL_{\text{arm}} / PPL_{\text{control}}$).
- **Top-1 Token Agreement:** Fraction of positions where the arm's argmax next-token prediction matches control.
- **Mean KL Divergence:** Mean Kullback-Leibler divergence from control output distribution to arm output distribution in nats.
- **Layer Telemetry:** Hidden state relative error, cosine similarity, and routing decision overlap vs. control per layer.

---

## Environment & Dependencies
Experiments require Python virtual environment `C:\AI_Models\venv-gpu` pinned in `requirements.txt`:
- `torch==2.13.0+cu130`
- `transformers==5.15.0`
- `safetensors==0.8.0`
- `tokenizers==0.22.2`
- `numpy==2.5.2`
- `psutil==7.2.2`

---

## Execution Commands

### 1. Bit-Exactness Sanity Check
Validates that GPU quantizer CUDA implementations (`ls_optimal_torch`, `int4_torch`) match CPU quantizers bit-for-bit across real expert layers and synthetic Gaussian weights:
```powershell
C:\AI_Models\venv-gpu\Scripts\python.exe experiments/twin_eval/gpu_quant_check.py
```
*Expected runtime:* ~10–15 seconds.

### 2. End-to-End Evaluation Run
Executes the layer-by-layer forward evaluation on GPU (`TA_DEVICE=cuda`) with unseen calibration texts (`TA_TEXT=unseen`) comparing `arm2` and `int4`:
```powershell
$env:TA_DEVICE="cuda"
$env:TA_TEXT="unseen"
$env:TA_ARMS="arm2,int4"
C:\AI_Models\venv-gpu\Scripts\python.exe experiments/twin_eval/ternary_arms.py 512 4
```
*Expected runtime:* ~15–20 minutes on an NVIDIA GPU (e.g. RTX 4090 / A100). Checkpoints are saved after each layer to `results/*.ckpt.pt` (automatically gitignored).

### 3. GPU vs. CPU / Reproduction Verification
Compares reproduced results against the reference run:
```powershell
C:\AI_Models\venv-gpu\Scripts\python.exe experiments/twin_eval/gpu_vs_cpu.py experiments/twin_eval/results/ternary_arms_512x4_unseen_arm2-int4_gpu.json experiments/twin_eval/results/reference/ternary_arms_512x4_unseen_arm2-int4_gpu.json
```
