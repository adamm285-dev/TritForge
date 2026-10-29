# TritForge ⚡

<p align="center">
  <b>Empirical Study & Clean-Room Toolkit for 1.58-Bit Ternary Post-Training Quantization (PTQ) on Sparse MoE Models</b>
</p>

<p align="center">
  <a href="https://github.com/adamm285-dev/TritForge/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg" alt="Python Version"></a>
  <a href="#"><img src="https://img.shields.io/badge/Core-Pure%20NumPy-orange.svg" alt="Pure NumPy"></a>
  <a href="experiments/twin_eval/"><img src="https://img.shields.io/badge/Empirical%20Result-Negative%20(70.5%25%20Top--1)-informational.svg" alt="Empirical Result"></a>
</p>

---

## 🧪 Executive Summary & Empirical Verdict (Negative Result)

**TritForge** was built to test a straightforward hypothesis on massive sparse Mixture-of-Experts (MoE) models (`Qwen3.5-MoE` / `Qwen3.8-2.4T-A95B-FP8` and its architectural twin `Qwen3.6-35B-A3B-FP8`):

> **Hypothesis:** Because routed feed-forward experts comprise $>95\%$ of total parameters in ultra-sparse MoEs, can we preserve all **Gatekeeper** weights (Attention, Gated-DeltaNet linear attention, MoE routers, shared experts, RMSNorms, embeddings, and LM head) in high precision while applying **calibration-free 1.58-bit Base-3 ternary PTQ** ($\{-1, 0, +1\}$ with per-128 FP16 block scales, `1.725 bits/weight`) to the routed experts—achieving a **~72.3% physical disk reduction** while retaining $\ge 90\%$ next-token fidelity?

### The Empirical Answer: **No.**

When evaluated end-to-end across all 40 layers of the **`Qwen/Qwen3.6-35B-A3B-FP8`** architectural twin on unseen text ($4 \times 512$ tokens, $N = 2,044$ target predictions, reproduced bit-for-bit on both CPU and GPU in [`experiments/twin_eval/`](experiments/twin_eval/)), **calibration-free ternary PTQ on routed experts fails to meet the $\ge 90\%$ retention threshold**, even after replacing standard threshold heuristics with exact closed-form per-block least-squares optimization (`ls_optimal`) or preserving the top 10% outlier matrices in 4-bit.

By contrast, **plain symmetric 4-bit round-to-nearest (`int4`, group size 128, `4.125 bits/weight`)** through the exact same asymmetric pipeline retains **91.7% top-1 token agreement** and a **1.0136x perplexity ratio** (`12.07` vs. `11.91` control).

We are publishing the full quantization toolkit, `.trit` binary container implementation, and bit-exact evaluation harness so other researchers and practitioners have concrete numbers, failure modes, and reusable infrastructure.

---

## 📊 End-to-End Benchmark Results (`Qwen3.6-35B-A3B-FP8` Twin)

**Evaluation Setup ([`experiments/twin_eval/`](experiments/twin_eval/)):**
* **Control Model:** `Qwen/Qwen3.6-35B-A3B-FP8` (Hybrid Qwen3.5-MoE: 40 layers, 3:1 Gated-DeltaNet to GQA attention ratio, 256 routed experts per layer, top-8 active per token), dequantized exactly per $128 \times 128$ FP8 block in `float32` (`TF32` disabled).
* **Corpus:** 4 sequences $\times$ 512 tokens ($2,048$ tokens total, $2,044$ evaluated targets) of unseen technical prose and code written after the model's training cutoff, locked via [`SHA256SUMS.txt`](experiments/twin_eval/SHA256SUMS.txt).
* **Asymmetric Boundary:** In every arm below, all non-routed parameters (attention, linear attention, router gates, shared experts, norms, embeddings, LM head) are identical to Control; **only the routed expert projections (`gate_proj`, `up_proj`, `down_proj`) are quantized.**

| Quantization Arm | Routed Expert Bits/Weight | Perplexity (PPL) ↓ | PPL Ratio vs. Control ↓ | Top-1 Token Agreement ↑ | Mean KL Divergence (nats) ↓ | Status vs. $\ge 90\%$ Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Control (`FP8` exact dequant)** | `8.000` | **11.9111** | `1.0000x` | **100.0%** | `0.0000` | Reference |
| **`arm1` (Heuristic ternary `optimal`)** | `1.725` | 16.7450 | `1.4058x` | 68.8% | `0.5351` | ❌ **FAIL** |
| **`arm2` (Closed-form LS ternary `ls_optimal`, CPU)** | `1.725` | 15.7743 | `1.3243x` | 70.5% | `0.4592` | ❌ **FAIL** |
| **`arm2` (Closed-form LS ternary `ls_optimal`, GPU)** | `1.725` | 15.7742 | `1.3243x` | 70.5% | `0.4592` | ❌ **FAIL** |
| **`mix_kurt` (`arm2` + top 10% `int4` by kurtosis)** | `1.965` | 14.4784 | `1.2155x` | 74.4% | `0.3514` | ❌ **FAIL** |
| **`mix_err` (`arm2` + top 10% `int4` by squared error)** | `1.965` | 14.2451 | `1.1959x` | 74.2% | `0.3363` | ❌ **FAIL** |
| **`int4` (Plain symmetric 4-bit RTN `g128`, CPU)** | `4.125` | **12.0733** | **`1.0136x`** | **91.7%** | **`0.0352`** | ✅ **PASS** |
| **`int4` (Plain symmetric 4-bit RTN `g128`, GPU)** | `4.125` | **12.0737** | **`1.0136x`** | **91.7%** | **`0.0352`** | ✅ **PASS** |

*All raw JSON trajectories and layer-by-layer telemetry are committed under [`experiments/twin_eval/results/`](experiments/twin_eval/results/).*

---

## 🔍 Why Calibration-Free Ternary PTQ Failed & Key Technical Findings

### 1. Heavy-Tailed `down_proj` Outliers Erase ~10% of Routed Experts Under Heuristic Thresholds
In a census of 368 randomly sampled routed experts across all 92 layers of the 2.4T FP8 source model, most projections have benign weight distributions (median relative Frobenius error `0.443` under heuristic scaling vs. `0.434` under exact optimization). However, **~10% of routed experts exhibit extreme outlier kurtosis ($>20$) in `down_proj`**, causing standard ternary threshold heuristics to suffer **`0.89–0.98` relative reconstruction error**—effectively zeroing out the expert's contribution.

Replacing heuristic scaling with the exact 1D closed-form least-squares block optimizer (`ls_optimal` in [`tritforge/quantize.py`](tritforge/quantize.py), which evaluates the cumulative prefix sums of sorted $|w|$ per 128-element block) reduces median reconstruction error on kurtosis $>20$ experts from **`0.93` down to `0.38`** and brings every sampled expert below `0.60`. End-to-end, `ls_optimal` (`arm2`) cuts the perplexity penalty from `1.406x` (`arm1`) to `1.324x`, **but 1.58-bit ternary still loses too much precision across 40+ stacked MoE layers (`70.5%` top-1 agreement, `0.459` nats KL).**

### 2. Promoting the Worst 10% of Expert Matrices to 4-Bit (`mix_err` / `mix_kurt`) Is Not Enough
To test whether a small mixed-precision budget could rescue ternary PTQ, `mix_kurt` and `mix_err` keep the worst 10% of expert matrices in each layer (ranked by kurtosis or ternary squared error) in 4-bit (`int4` g128) while leaving the remaining 90% in `ls_optimal` ternary (`1.965 bits/weight` overall). While this lowers the perplexity ratio to **`1.196x–1.216x`** and raises top-1 agreement to **`74.2%–74.4%`**, it confirms that ternary quantization error is not confined to a small outlier subset—it accumulates across the bulk of the 3-state expert projections.

### 3. Lossless Discovery: 24% Layer-Averaged Structured Dead-Neuron Sparsity in Source FP8 Weights
During verification of `.trit` block scale rows against the raw FP8 master shards, we found that 38%–56% of block scales in early layers were legitimately zero. Inspecting the source FP8 tensors revealed **structured dead intermediate neurons**: entire SwiGLU channels where row $i$ of `gate_proj`, row $i$ of `up_proj`, and column $i$ of `down_proj` are simultaneously zero (or $\le 1.3 \times 10^{-6}$):
* **Layer-Averaged Dead-Neuron Fraction (all 92 layers):** **24%** of all routed-expert intermediate channels.
* **Early Layers (Layers 0–9):** **54% average** (up to **84%** in Layer 1, and 75%–90% in individual experts).
* **Implication:** Dropping dead intermediate channels **post-quantization** (never pre-quantization, as removing zeros alters block alignment and changes `mean(|w|)` thresholds) yields a **100% mathematically lossless** storage and compute reduction.

### 4. `Qwen3.5-MoE` Architecture Gotchas for Custom Runtimes
* **Zero-Centered RMSNorm (`1 + w`):** `Qwen3.5-MoE` decoder norms compute $y = \hat{x} \cdot (1 + w)$ rather than $y = \hat{x} \cdot w$. Because learned norm weights $w$ center near $-0.6$ in Layer 0, using standard RMSNorm flips the sign of **99.9% of the channels feeding the MoE router**.
* **Hybrid Gated-DeltaNet Attention:** Only 1 in 4 layers uses standard Grouped-Query Attention (GQA); the other 3 in 4 layers use **Gated-DeltaNet linear attention**.
* **Flat Pre-Top-K Router Distributions:** On real Layer-0 token activations, the top-10 routed experts hold only **~32%** of the pre-renormalization 512-way softmax probability mass; reaching 90% cumulative mass requires a median $K = 265$ experts (`10th–90th` percentile: `138–314`). Dynamic-$K$ schemes that threshold raw 512-way softmax mass at 90% will therefore clamp to `max_k` rather than reducing active expert counts.

---

## 🧰 What Works in This Repository

While calibration-free ternary PTQ fell short of our quality target, the engineering infrastructure in this repository is clean-room, dependency-light (core requires only Python $\ge 3.10$ + `numpy`), and tested (`114 passed` offline unit tests):

1. **Bit-Exact CPU & GPU Twin Evaluation Suite ([`experiments/twin_eval/`](experiments/twin_eval/)):**
   * Layer-by-layer streaming evaluation over `Qwen3.5-MoE` models without multi-GPU VRAM requirements (streams one layer at a time to CUDA with `TF32` off).
   * Vectorized PyTorch/CUDA and NumPy implementations of `ls_optimal` ternary and `int4` RTN quantizers verified bit-for-bit identical across `859,832,320` weights (`gpu_quant_check.py`).
2. **Zero-Copy `.safetensors` Binary Slicer & Shard Auditor ([`tritforge/slice.py`](tritforge/slice.py), [`tritforge/verifier.py`](tritforge/verifier.py)):**
   * Reads 8-byte header prefixes across 200+ multi-terabyte `.safetensors` shards in `< 2 seconds` without allocating tensor payloads in RAM.
   * Automatically resolves cross-shard $128 \times 128$ `weight_scale_inv` FP8 block scales and decodes `F8_E4M3` / `BF16` in pure NumPy.
   * Enforces a 64 MB header allocation guard against malformed/oversized headers and filters `.aria2` partial downloads.
3. **Native Base-3 `.trit` Container & Batch Converter ([`tritforge/container.py`](tritforge/container.py), [`tritforge/converter.py`](tritforge/converter.py), [`tritforge/quantize.py`](tritforge/quantize.py)):**
   * Packs 5 ternary trits into each byte ($3^5 = 243 \le 256$, `1.6 bits/trit` + `0.125 bits/weight` FP16 block scales = `1.725 bits/weight`), compressing a 512-expert layer from `~25.3 GB` (FP8 shards) down to `7.01 GB` (`72.3%` physical disk reduction).
   * Crash-safe atomic serialization (`.atomic_tmp` + `os.fsync`), SHA-256 manifest verification, and seek-based single-expert loading.

---

## 📦 Installation & Usage

### Standard Installation (Pure Python + NumPy)
```bash
git clone https://github.com/adamm285-dev/TritForge.git
cd TritForge
pip install .
```

### CLI Commands

```bash
# 1. Verify multi-shard .safetensors header & byte-boundary integrity
tritforge verify-shards -i /path/to/safetensors_shards

# 2. Convert specific layers into Base-3 .trit containers (using ls_optimal scaling)
tritforge convert -i /path/to/shards -o /path/to/trit_output -l 0

# 3. Verify converted .trit container integrity and SHA-256 digests against manifest
tritforge verify-trit -m /path/to/trit_output

# 4. Run theoretical storage & memory footprint audit
tritforge audit

# 5. Run synthetic toy-model mechanics demonstration
tritforge demo --prompt "Hello" --max-new-tokens 16
```

> **Note on Standalone Runtime (`tritforge run` / `tritforge serve`):**  
> Because `Qwen3.5-MoE` uses hybrid Gated-DeltaNet linear attention (3 of every 4 layers) which is not yet implemented in the pure-NumPy engine, `tritforge run` exits with a non-zero status and `/v1/chat/completions` on `tritforge serve` returns `HTTP 501 Not Implemented` rather than returning simulated or toy outputs. For real end-to-end model evaluation on `Qwen3.5-MoE`, use the Hugging Face `transformers` layer-streaming harness in [`experiments/twin_eval/`](experiments/twin_eval/README.md).

---

## 🧪 Running the Test Suite

The test suite runs offline in ~20 seconds with zero external model files required (hardware vault tests on `H:\` are deselected by default via `pyproject.toml`):

```bash
pytest tests -v
```

To reproduce the `Qwen3.6-35B-A3B-FP8` twin evaluation on GPU or CPU, see [`experiments/twin_eval/README.md`](experiments/twin_eval/README.md).

---

## 📄 License

This project is licensed under the **Apache License, Version 2.0**. See the [LICENSE](LICENSE) file for details.
