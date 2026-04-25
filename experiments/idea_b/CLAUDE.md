# Idea B: Structured Sparse Attention on AiM

## Goal

Show that structured sparse attention (sliding window, strided+sink, H2O top-K) maps trivially to AiM instruction-level row skips, yielding latency reduction with zero kernel redesign — a property impossible to replicate on GPU without custom CUDA kernels. Quantify latency reduction (from simulator) and accuracy cost (PPL on WikiText-2) across sparsity patterns and context lengths.

## Hypothesis

Sparse attention patterns that are memory-access-hostile on GPU (strided access → cache miss → poor performance) are access-neutral on AiM, because `MAC_ABK` instruction skipping decouples sparsity pattern complexity from memory efficiency. Sliding window W=512 at seqlen=4096 reduces attention GEMV MAC_ABK count by 87.5%, yielding ~40-44% total attention block latency reduction. At seqlen=32k the effect dominates: ~70% block latency reduction. GPU cannot match this without custom kernels.

**Core thesis**: *Sparsity patterns that are memory-access-hostile on GPU are access-neutral on PIM. AiM's instruction-level row granularity decouples sparsity pattern complexity from memory efficiency.*

## Sparsity Patterns

### Pattern A — Sliding Window (baseline, simplest)
```
attend only to last W tokens
active_tokens = range(seqlen - W, seqlen)
MAC_ABK count = W  (vs seqlen dense)
latency reduction on score GEMV = seqlen / W
```

### Pattern B — Strided + Sink (Longformer/StreamingLLM style)
```
active_tokens = {0..n_sinks} ∪ {seqlen-W..seqlen} ∪ {i : i % stride == 0}
```
AiM advantage: strided = sequential skip → no bank conflict, no row activation penalty for skipped rows. GPU strided = non-coalesced memory access = severe penalty.

### Pattern C — H2O Top-K Importance (hardest, most accurate)
```
Phase 1 (score pre-screen): one pass Q×K at coarse 64-token block granularity → pick top-K blocks
Phase 2 (sparse output): score×V with only top-K K/V rows active
```
Phase 1 modeled as reduced-density MAC_ABK pass (1/64 row count of dense).

## Quantifying the Opportunity

At seqlen=4096, Llama2-7B:
- Weight GEMVs (W_Q/K/V/O/W1/W2/W3): seqlen-independent, dominate at short context
- Score GEMV (Q×K) + Output GEMV (score×V): ≈40-50% of total attention latency at seqlen=4096
- At seqlen=32k: score+output GEMV dominates (80%+ of attention latency)

| seqlen | Window W | MAC_ABK reduction | Attention block latency reduction |
|--------|----------|-------------------|------------------------------------|
| 4096   | 512      | 87.5%             | ~40-44%                           |
| 32768  | 512      | 98.4%             | ~70%                              |

## Design Space

| Axis | Values |
|---|---|
| Model | Llama2-7B, Llama2-13B, Llama2-70B |
| seqlen | 512, 1024, 2048, 4096, 8192, 32768 |
| Pattern | dense, sliding, strided+sink, H2O |
| Window W | 64, 128, 256, 512, 1024 |
| Stride (Pattern B) | 16, 32, 64 |
| n_sinks (Pattern B) | 4 (fixed, per StreamingLLM finding) |
| TopK (Pattern C) | 128, 256, 512 tokens |

Two output figures:
1. **Simulator figure**: latency vs window_size × seqlen — pure architectural result, no accuracy needed
2. **Pareto figure**: latency reduction vs PPL degradation per pattern (requires accuracy_eval.py)

## Key Files to Modify

| File | Change |
|---|---|
| `cent_simulation/TransformerBlock.py` | Add `attention_mask` param to `Vector_Matrix_Mul_score_pim_only_trace()` and `Vector_Matrix_Mul_output_pim_only_trace()`; add `compute_attention_mask(seqlen)` method |
| `cent_simulation/GPT.py` | Pass mask to score/output GEMV at lines 147 and 198; add `--sparse-pattern` dispatch before those calls |
| `cent_simulation/utils.py` | Add `--sparse-attention`, `--sparse-pattern`, `--window-size`, `--stride`, `--n-sinks`, `--topk` CLI flags to `get_args()` |
| `cent_simulation/run_sim.py` | Pass sparse flags through to `function_sim.py` subprocess commands in `generate_trace()`; sweep window sizes in `update_csv()` |
| New `experiments/idea_b/accuracy_eval.py` | Load real Llama2 weights via HuggingFace, apply same attention_mask, measure PPL on WikiText-2 for each (pattern, seqlen, W) — outputs `accuracy_vs_sparsity.csv` |

### Core code change sketch

In `TransformerBlock.py`:
```python
def compute_attention_mask(self, seqlen):
    if self.sparse_pattern == "sliding":
        return list(range(max(0, seqlen - self.window_size), seqlen))
    elif self.sparse_pattern == "strided":
        sinks   = list(range(self.n_sinks))
        recent  = list(range(max(0, seqlen - self.window_size), seqlen))
        strided = list(range(0, seqlen, self.stride))
        return sorted(set(sinks + recent + strided))
    elif self.sparse_pattern == "h2o":
        return self.h2o_topk_mask(seqlen)
    return None  # dense

def Vector_Matrix_Mul_score_pim_only_trace(self, row_index, seqlen, timing_key,
                                            attention_mask=None):
    active_tokens = attention_mask if attention_mask is not None else range(seqlen)
    for token_idx in active_tokens:
        # existing MAC_ABK emission, indexed to token_idx row
        ...
```

In `GPT.py:trace_only()`:
```python
mask = self.compute_attention_mask(seqlen)
self.Vector_Matrix_Mul_score_pim_only_trace(
    self.cache_k_row_index, seqlen, "breakdown_sa_score", attention_mask=mask)
# line 198:
self.Vector_Matrix_Mul_output_pim_only_trace(
    self.cache_v_row_index, seqlen, "breakdown_sa_output", attention_mask=mask)
```

## References

**Sparse attention hardware (arch conferences):**
- Wang et al., *"SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning"*, **HPCA 2021** — gold standard hardware sparse attention; cascade token+head pruning. Read first.
- Ham et al., *"A³: Accelerating Attention Mechanisms in Neural Networks with Approximation"*, **HPCA 2020** — approximate attention via score thresholding; foundational.
- Ding et al., *"Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture"*, **MICRO 2021** — hardware-software co-design for sparse attention. *(verify title)*

**Sparsity patterns / algorithms (for accuracy_eval.py):**
- Beltagy et al., *"Longformer: The Long-Document Transformer"*, 2020 — sliding window + strided + global token; reference for Pattern B.
- Zhang et al., *"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"*, **NeurIPS 2023** — dynamic top-K KV eviction by cumulative attention score; reference for Pattern C.
- Xiao et al., *"Efficient Streaming Language Models with Attention Sinks"*, **ICLR 2024** — sink token phenomenon (first 4 tokens disproportionate score); explains why n_sinks=4 is necessary for Pattern B correctness.
- Liu et al., *"Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression"*, **NeurIPS 2023** — important tokens persist across layers; justifies persistent mask approach.

## Session Log

| Date | Action | Finding |
|---|---|---|
| 2026-04-25 | Idea defined | Sparse attention maps to MAC_ABK skip — zero overhead on AiM vs custom kernel on GPU |

## Status

- [x] Define experiment
- [ ] Add CLI flags to `utils.py` and `run_sim.py`
- [ ] Implement `compute_attention_mask()` in `TransformerBlock.py`
- [ ] Add `attention_mask` param to `Vector_Matrix_Mul_score_pim_only_trace()` and output variant
- [ ] Wire mask into `GPT.py:trace_only()` lines 147 and 198
- [ ] Run simulator sweep (seqlen_gap=128, all 3 models, all patterns)
- [ ] Implement `accuracy_eval.py` for PPL measurement
- [ ] Plot latency vs window_size figure
- [ ] Plot Pareto (latency reduction vs PPL) figure
- [ ] Write analysis
