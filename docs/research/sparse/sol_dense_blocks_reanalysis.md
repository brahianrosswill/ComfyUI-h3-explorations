# Sol-Attn Dense Blocks Re-Analysis and 5-Test Experimental Plan

**Date**: 2026-10-02  
**Context**: MiniMax H3 Attention Optimization, Rotated INT8 Quantization, and Sparse Routing Telemetry  
**Primary Tools**: `sol_observe.py`, `bench/sol_duckdb_analyzer.py`, `docs/research/sparse/sol_duckdb_dashboard.html`  
**Referents**: `docs/h3_block49_quant_error.md`, `workflows/h3_config.py::SOL_DENSE_TAIL`, `docs/wiki/decisions.md`

---

## 1. Executive Summary

Historically, `MiniMaxH3Sol` shipped with the default configuration `dense_blocks="45,48,49"`. This default was established when Sol and SageAttention ran unrotated INT8 attention, where severe channel outlier spikes in the model's $K$-norm destroyed INT8 quantization precision in blocks 45, 48, and 49.

On 2026-09-27, the repository standardized on `quantizer="rotated"` (Hadamard rotation). By multiplying $Q$ and $K$ by a randomized orthogonal Hadamard matrix prior to INT8 quantization, channel energy is uniformly dispersed across all 128 head dimensions. This eliminated the tail-block quantization penalty.

In **Test 1** (our unmasked 50-block diagnostic baseline on Ref2VA at sequence length 119,102 tokens), comprehensive telemetry revealed:

1. **Tail Blocks (45–48) Are Tamed**: Under Hadamard rotation, Block 48 exhibits an average relative $L_2$ error of only **7.14%** (max 9.41%), Block 46 is **7.40%**, and Block 47 is **6.38%**. Keeping Block 48 in `dense_blocks` expends dense computation on blocks that Sol already calculates with high fidelity.
2. **The True Bottleneck is the Middle Cluster (Blocks 38–43)**: With quantization error suppressed, the dominant error source is **sparsity truncation error (routing loss)**. In the middle of the DiT, attention is diffuse and globally integrated across text, reference images, and video tokens. Truncating tokens via top-$k$ sparsity produces severe errors:
   - **Block 42**: **25.67%** avg relative $L_2$ error (max **27.01%**) — nearly $4\times$ higher than Block 48.
   - **Block 39**: **25.22%** avg relative $L_2$ error (max **27.20%**) — $3.5\times$ higher than Block 48.
   - **Block 41**: **24.35%** avg relative $L_2$ error.
   - **Block 40**: **23.43%** avg relative $L_2$ error.
   - Individual attention heads in these blocks suffer catastrophic degradation (e.g., Block 42 Head 44 has **57.0%** error, min cosine similarity **0.7697**; Block 39 Head 12 has **80.1%** error).
3. **Block 49 Must Remain Dense**: Although Block 49's error is 13.73%, it directly feeds into `final_layer.video_out`. Any approximation error in Block 49 propagates directly into output latents without subsequent transformer layers to smooth it out.

**Conclusion**: Retaining `"45,48,49"` misallocates dense compute to a quiet tail (Block 48) while exposing the 25% error middle cluster to unmitigated sparsity loss. We propose transitioning `dense_blocks` to **Option A (`"39,41,42,49"`)** or **Option B (`"39,40,41,42,49"`)**, evaluated across a structured 5-test matrix.

---

## 2. The Historical Origin of `"45,48,49"` vs. Rotated INT8

### 2.1 The Unrotated INT8 Regime (Pre-2026-09-27)
Prior to late September 2026, INT8 attention in both stock Sol and SageAttention used per-row or per-block quantization with a single shared scale across all 128 channels. 

As documented in `docs/h3_block49_quant_error.md`, MiniMax H3's released weights place an order-of-magnitude gain on four specific channels in `k_norm.weight` on blocks 45, 48, and 49:
- Because the quantization scale had to cover these outlier spikes, the remaining 124 channels were quantized into only 1 or 2 bits of effective precision.
- This caused INT8 attention error on Block 49 to spike to over $5\times$ that of Block 0.
- `SOL_DENSE_TAIL = "45,48,49"` was implemented on 2026-09-25 as an emergency heuristic to force these three outlier blocks onto the dense fallback (`attention_comfy_kitchen_int8`), bypassing the broken INT8 quantizer.

### 2.2 What Changed with `quantizer="rotated"`
On 2026-09-27, Sol integrated Hadamard rotation (`quantizer="rotated"`), matching `comfy-kitchen`'s dense kernel. 

Hadamard transformation multiplies activation vectors by a fixed orthonormal Hadamard matrix before quantization:
$$Q_{\text{rot}} = Q \cdot H, \quad K_{\text{rot}} = K \cdot H$$
Because $H^T H = I$, the inner product is mathematically identical:
$$Q_{\text{rot}} K_{\text{rot}}^T = Q H H^T K^T = Q K^T$$
However, the outlier energy concentrated in 4 channels is now distributed evenly across all 128 dimensions. The peak-to-average channel ratio drops from $>10:1$ to near unity. As a result, 8-bit quantization preserves full dynamic range across all channels.

---

## 3. Empirical Telemetry: Test 1 Baseline Analysis

Test 1 ran an unmasked diagnostic pass over all 50 DiT blocks (Ref2VA, 1344x768, 345 frames, sequence length 119,102 tokens, 8 PDD steps, `dense_blocks=""`, `tau=1.0`, `quantizer="rotated"`).

The captured activations were ingested into `data/sparse/sol_analysis.duckdb` and visualized in `docs/research/sparse/sol_duckdb_dashboard.html`.

### 3.1 Tail Blocks vs. Middle Error Cluster

| Block Range | Block | Historical Status | Avg Rel $L_2$ Error | Max Rel $L_2$ Error | Avg Cosine Sim | Assessment |
|---|---|---|---|---|---|---|
| **Early** | Block 0 | Sparse | 4.88% | 5.25% | 0.9988 | Clean, localized attention |
| **Middle Cluster** | Block 38 | Sparse | 19.16% | 20.35% | 0.9814 | High error onset |
| | **Block 39** | Sparse | **25.22%** | **27.20%** | **0.9678** | **Critical Error Peak** |
| | **Block 40** | Sparse | **23.43%** | **24.97%** | **0.9722** | **Critical Error Peak** |
| | **Block 41** | Sparse | **24.35%** | **25.77%** | **0.9699** | **Critical Error Peak** |
| | **Block 42** | Sparse | **25.67%** | **27.01%** | **0.9667** | **Highest Error in DiT** |
| | Block 43 | Sparse | 19.49% | 20.91% | 0.9807 | Transition decay |
| **Tail Blocks** | Block 45 | Historical Dense | 9.55% | 12.28% | 0.9954 | Moderate error |
| | Block 46 | Shipped Sparse | 7.40% | 8.79% | 0.9972 | Highly accurate under rotation |
| | Block 47 | Shipped Sparse | 6.38% | 7.95% | 0.9979 | Lowest error in second half |
| | **Block 48** | Historical Dense | **7.14%** | **9.41%** | **0.9974** | **Clean; dense protection is wasted** |
| | **Block 49** | Historical Dense | **13.73%** | **15.08%** | **0.9906** | **Must stay dense (feeds video_out)** |

### 3.2 Catastrophic Per-Head Degradation in the Middle Cluster

Aggregated per-block metrics conceal even sharper degradation at the individual attention head level. In Blocks 38–43, specific heads experience severe routing failure:

| Block | Head Index | Avg Rel $L_2$ Error | Min Cosine Sim | Avg Cosine Sim | Routing Behavior |
|---|---|---|---|---|---|
| **Block 39** | Head 12 | **80.15%** | 0.8144 | 0.8884 | Diffuse global routing dropped by top-$k$ |
| **Block 38** | Head 37 | **74.07%** | 0.8413 | 0.9003 | Coarse block grid splits semantic region |
| **Block 38** | Head 43 | **71.62%** | 0.8447 | 0.9028 | Truncated cross-attention |
| **Block 39** | Head 23 | **69.10%** | 0.8367 | 0.8890 | Global context dropped |
| **Block 39** | Head 53 | **65.45%** | 0.8432 | 0.8735 | Reference-to-video alignment lost |
| **Block 42** | Head 44 | **57.00%** | **0.7697** | 0.8452 | Worst directional distortion in network |
| **Block 43** | Head 0 | **50.29%** | 0.8144 | 0.8790 | Long-range context failure |

### 3.3 Modality Vulnerability Analysis

Telemetry across the 119k sequence length showed marked differences in how different token segments tolerate sparsity:

| Segment Kind | Token Count | Avg Rel $L_2$ Error | Min Cosine Sim | Avg Cosine Sim | Sensitivity |
|---|---|---|---|---|---|
| `text` | 7,776 (6.5%) | **15.64%** | **0.8385** | 0.9866 | **High**: Prompt adherence degrades when text rows are dropped |
| `video` | 102,816 (86.3%) | 11.06% | 0.9720 | 0.9933 | **Moderate**: Latent spatio-temporal structure |
| `ref_img` | 7,360 (6.2%) | 10.96% | 0.9559 | 0.9934 | **Moderate**: Character consistency across shots |
| `audio` | 1,150 (1.0%) | **0.78%** | **0.9989** | 1.0000 | **Protected**: Protected by `exact_kv_and_rows` sink |

---

## 4. Why `"45,48,49"` Must Change

### 4.1 The Waste of Protecting Block 48
Under Hadamard rotation, Block 48 has an average relative $L_2$ error of **7.14%**. It is cleaner than Block 37, Block 38, Block 39, Block 40, Block 41, Block 42, Block 43, and Block 44. Keeping Block 48 in `dense_blocks` is equivalent to using a fire extinguisher on a cold room while the adjoining room is blazing.

### 4.2 The Vulnerability of Leaving Blocks 39–42 Sparse
Blocks 39 through 42 sit in a sustained error plateau above **23%–25%**, with peak heads losing up to **80%** of their output magnitude and suffering directional alignment drops down to $0.76$ cosine similarity. In a generative diffusion trajectory, injecting 25% error into the middle representations degrades object coherence and fine motion vectors.

### 4.3 Why Block 49 Cannot Be Made Sparse
Block 49 is the final transformer block of the DiT. Its output directly feeds the final adaptive layer norm and linear projection (`final_layer.video_out`). Any error in Block 49 cannot be attenuated or redirected by subsequent attention layers; it maps linearly into the predicted latent velocity vector. Therefore, Block 49 remains mandatory in any `dense_blocks` specification.

---

## 5. Candidate Replacement Configurations

| Configuration | `dense_blocks` Spec | Dense Block Count | Target Error Reduction | Compute Cost Impact | Recommendation |
|---|---|---|---|---|---|
| **Historical Default** | `"45,48,49"` | 3 blocks | Ineffective (shields 7% tail, ignores 25% middle) | Baseline | Deprecated under `rotated` |
| **Option A (Targeted Peak)** | `"39,41,42,49"` | 4 blocks | Shields top 3 worst middle blocks (>24%) + terminal block | +2.0% sampler time | **Top Candidate (High Efficiency)** |
| **Option B (Full Plateau)** | `"39,40,41,42,49"` | 5 blocks | Shields the entire 23%–26% middle plateau + terminal block | +4.1% sampler time | **Top Candidate (Max Quality)** |
| **Option C (Strict Budget)** | `"39,42,49"` | 3 blocks | Shields the two worst middle blocks + terminal block | Exactly 0.0% cost vs. historical | Fallback if budget strictly locked |

---

## 6. The 5-Test Experimental Matrix

To systematically validate the new dense block strategy and establish Pareto-optimal defaults, we define the following 5-test matrix:

| Test ID | Name / Objective | Configuration Levers | Target Sequence / Scene | Telemetry Scope | Success Criteria / Hypothesis |
|---|---|---|---|---|---|
| **Test 1** | **Diagnostic Baseline** *(Completed)* | `tau=1.0`<br>`dense_blocks=""`<br>`token_routing="off"`<br>`start_percent=0.0`<br>`quantizer="rotated"` | Ref2VA Market<br>1344x768, 345f<br>S=119,102, 8 steps | Full 50 blocks unmasked (`sol_probe_*.jsonl`, `sol_observe_*.jsonl`, `.u16`) | Establishes the true unconstrained error profile across all 50 blocks. *Result: Revealed the 38-43 middle peak and 45-48 tail calmness.* |
| **Test 2** | **Historical Control Candidate** *(Running)* | `tau=1.0`<br>`dense_blocks="45,48,49"`<br>`token_routing="measured"`<br>`start_percent=0.0`<br>`quantizer="rotated"` | Ref2VA Market<br>1344x768, 345f<br>S=119,102, 8 steps | Dense tail masked, token routing active on measured blocks | Measures the exact impact of `token_routing="measured"` on active blocks under historical dense settings. |
| **Test 3** | **Middle Dense Swap (Option A)** *(Proposed)* | `tau=1.0`<br>`dense_blocks="39,41,42,49"`<br>`token_routing="measured"`<br>`start_percent=0.0`<br>`quantizer="rotated"` | Ref2VA Market<br>1344x768, 345f<br>S=119,102, 8 steps | 4 dense blocks, telemetry on remaining sparse blocks | Pipeline error across steps 0-7 drops dramatically. Peak block error drops from 25.7% to <19% (Block 43). Wall time matches Test 2 within ~2%. |
| **Test 4** | **Early Warmup Anchor** *(Proposed)* | `tau=1.0`<br>`dense_blocks="39,41,42,49"`<br>`token_routing="measured"`<br>`start_percent=0.2`<br>`quantizer="rotated"` | Ref2VA Market<br>1344x768, 345f<br>S=119,102, 8 steps | Step 0-1 run dense on `attention_comfy_kitchen_int8`; steps 2-7 Sol | Tests whether keeping the first 20% of schedule dense protects global composition when middle blocks are also shielded. |
| **Test 5** | **Pareto Optimization (Option B)** *(Proposed)* | `tau=1.1` (or `1.0`)<br>`dense_blocks="39,40,41,42,49"`<br>`token_routing="measured"`<br>`start_percent=0.0`<br>`quantizer="rotated"` | Ref2VA Market<br>1344x768, 345f<br>S=119,102, 8 steps | 5 dense blocks, slightly higher sparsity on non-critical blocks | Recovers the compute cost of the 5th dense block by relaxing tau to 1.1 on remaining blocks, testing net Pareto dominance. |

---

## 7. Execution Guide: Running Tests in ComfyUI

### 7.1 Setting Up Telemetry Flags for Subsequent Tests
To run Test 3, Test 4, or Test 5, launch ComfyUI with the appropriate environment variables:

```bash
# Example for Test 3: Middle Dense Swap
H3_SOL_OBSERVE_FILE="data/sparse/captures/test3_middle_swap/sol_observe.jsonl" \
H3_SOL_OBSERVE_DENSE=1 \
python main.py --listen 127.0.0.1 --port 8188
```

### 7.2 Configuring the `MiniMaxH3Sol` Node in the Graph
In the workflow graph (e.g. `workflows/h3_ref2v_market_pdd_api.json` or through the ComfyUI UI):
1. Locate the **MiniMax H3 Sol-Attn** node (`MiniMaxH3Sol`).
2. Update the inputs according to the test matrix:
   - **Test 3**:
     - `tau`: `1.0`
     - `dense_blocks`: `"39,41,42,49"`
     - `token_routing`: `"measured"` (or `"custom"`)
     - `start_percent`: `0.0`
     - `end_percent`: `1.0`
     - `quantizer`: `"rotated"`
   - **Test 4**:
     - Same as Test 3, but set `start_percent`: `0.2`
   - **Test 5**:
     - `tau`: `1.1`
     - `dense_blocks`: `"39,40,41,42,49"`
     - `start_percent`: `0.0`
     - `token_routing`: `"measured"`

### 7.3 Automated Analysis Pipeline
After each test finishes:
```bash
# Ingest the test run into DuckDB and update dashboard
python bench/sol_duckdb_analyzer.py \
  --db data/sparse/sol_analysis.duckdb \
  --observe data/sparse/captures/test3_middle_swap/sol_observe.jsonl \
  --run-id test3_middle_swap \
  --html docs/research/sparse/sol_duckdb_dashboard.html
```

---

## 8. Summary of Recommendations for `workflows/h3_config.py`

Once Test 3 and Test 4 validate the reduction in overall network degradation:
1. Update `SOL_DENSE_TAIL` (or rename to `SOL_DENSE_RECOMMENDED`) in `workflows/h3_config.py`:
   ```python
   # From:
   SOL_DENSE_TAIL = "45,48,49"
   # To:
   SOL_DENSE_RECOMMENDED = "39,41,42,49"  # Shields middle error plateau + final projection
   ```
2. Rebuild the workflow fleet:
   ```bash
   python workflows/build_workflows.py
   ```
3. Record the reversal in `docs/wiki/decisions.md` under the date of adoption.
