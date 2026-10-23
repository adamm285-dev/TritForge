# TritSlack Formal Specification & Threat Model
<!-- SPEC-TRITSLACK-001 | TritForge Architecture Reference -->

## 1. Mathematical Formulation
In TritForge 5-trit balanced ternary polynomial packing:
$$\text{byte\_value} = \sum_{i=0}^{4} t_i \cdot 3^i, \quad t_i \in \{0, 1, 2\} \implies \text{byte\_value} \in [0, 242]$$

The polynomial slack space consists of the 13 unmapped integer values:
$$\mathcal{S}_{\text{slack}} = \{243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255\}$$

---

## 2. Canonical Tag Allocation Table

| Byte (Decimal) | Hex | Semantic Meaning | Payload Following Tag |
|---|---|---|---|
| **$0 \dots 242$** | `0x00`–`0xF2` | Standard 5-Trits Packing | None (Self-contained) |
| **$243$** | `0xF3` | Hessian Outlier: FP8 E4M3 scale factor | 1 byte (FP8 scale) |
| **$244$** | `0xF4` | Hessian Outlier: FP16 scale factor | 2 bytes (FP16 scale) |
| **$245$** | `0xF5` | Skip 5 consecutive zeros (1 block) | 0 bytes |
| **$246$** | `0xF6` | Skip 10 consecutive zeros (2 blocks) | 0 bytes |
| **$247$** | `0xF7` | Skip 25 consecutive zeros (5 blocks) | 0 bytes |
| **$248$** | `0xF8` | Skip 50 consecutive zeros (10 blocks) | 0 bytes |
| **$249$** | `0xF9` | Skip 100 consecutive zeros (20 blocks) | 0 bytes |
| **$250$** | `0xFA` | Skip $N$ zeros (Variable Run-Length) | 1 byte ($N \le 255$ blocks) |
| **$251$** | `0xFB` | Skip Row Tail (Zero padding to end of row) | 0 bytes |
| **$252$** | `0xFC` | Reserved: Dynamic Attention GQA Checkpoint | 0 bytes |
| **$253$** | `0xFD` | Reserved: Telemetry Sync Marker | 0 bytes |
| **$254$** | `0xFE` | Subsystem Exception / Safety Intercept | 0 bytes |
| **$255$** | `0xFF` | End-of-Expert (EOE) Alignment Delimiter | 0 bytes |

---

## 3. Second-Order Threat Modeling & Edge Case Hardening

### Threat A: Out-of-Bounds Stream Overrun (Run-Length Desync)
- **Vulnerability**: If tag `250` (Skip $N$ blocks) requests skipping past the end of the current expert's allocation, a rogue or corrupted weight stream could cause an out-of-bounds memory read into an adjacent expert or kernel memory.
- **Hardening Invariant**: The decoding loop maintains a strict hardware boundary counter:
  $$\text{remaining\_weights} = D_{\text{expert}} - \text{current\_index}$$
  $$\text{effective\_skip} = \min(N \times 5, \text{remaining\_weights})$$
  Any attempt to skip past $D_{\text{expert}}$ raises an explicit memory-fence exception and logs the anomaly.

### Threat B: Variable-Stride Desynchronization
- **Vulnerability**: Because tags $243$ and $244$ consume variable payloads (1 or 2 trailing bytes), a single corrupted bit could cause the parser to interpret a weight as a tag or vice-versa, permanently throwing off downstream alignment.
- **Hardening Invariant**: Every expert block terminates with tag `255` (`0xFF`). If the parser reaches the expected byte offset without seeing `0xFF`, or if `0xFF` arrives before expected, the thread rejects the block and falls back to uncompressed fallback weights.

### Threat C: SIMD Divergence & Cache Line Poisoning
- **Vulnerability**: High variance in skip lengths across adjacent SIMD lanes can cause branch divergence and cache-line misses on multi-core systems.
- **Hardening Invariant**: All skips are block-aligned to multiples of 5 trits (exact byte boundaries), ensuring alignment with 64-byte CPU cache lines.
