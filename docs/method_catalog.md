# Method Catalog

This document groups the 33 evaluated methods into the design ideas they represent. The goal is not just to list names, but to show which mechanisms were being tested across the study.

## Reference Baseline

### BF16

- `BF16`
- The uncompressed reference baseline
- Stores all keys and values in native BF16 precision
- Used as the reference for fidelity and systems deltas

## Naive Rounding Baselines

These are the simplest low-bit baselines. They tell us what happens if we quantize the cache without adding special handling for outliers, recency, or spatial structure.

### RTN

- `RTN_INT4`
- `RTN_INT2`
- `RTN_K2_V4`
- `RTN_INT4_RECENT2`
- `RTN_INT4_REFRESH`

Core idea:

- Blockwise round-to-nearest quantization
- Scale a block by its max magnitude
- Quantize to a fixed low-bit grid
- Reconstruct approximately on read

Variants:

- `INT4`, `INT2`: uniform 4-bit and 2-bit baselines
- `K2_V4`: lower precision for keys than values
- `RECENT2`: keep the two most recent context blocks in BF16
- `REFRESH`: re-quantize the cache on a refresh schedule

## Asymmetric Key/Value Quantization

### KIVI

- `KIVI_INT4`
- `KIVI_INT2`
- `KIVI_INT4_REFRESH`
- `KIVI_K2_V4`

Core idea:

- Keys and values have different statistical structure
- Quantize keys channel-wise
- Quantize values token-wise
- Use asymmetric quantization with a zero-point

This family tests whether distribution-aware quantization beats naive RTN at the same nominal bit budget.

## Rotation-Based High-Fidelity Quantization

### QuaRot

- `QUAROT_KV_INT4`
- `QUAROT_KV_INT2`
- `QUAROT_KV_INT4_RECENT2`
- `QUAROT_KV_INT4_REFRESH`

Core idea:

- Use a Hadamard rotation before quantization
- Spread outlier energy across channels
- Apply RTN-style quantization in rotated space
- Invert the rotation after dequantization

This family tests whether removing outlier concentration improves low-bit accuracy enough to justify the extra compute.

## Residual / Outlier-Aware Custom Quantizers

### PRQ

- `PRQ_INT4`
- `PRQ_INT2`

Core idea:

- Quantize once
- Quantize the residual error again
- Store both codes and reconstruct by summation

### QAQ

- `QAQ_INT4`
- `QAQ_INT2`

Core idea:

- Treat large outliers separately
- Keep extreme activations in higher precision
- Quantize the in-range bulk asymmetrically

### TPTQ

- `TPTQ_INT2`

Core idea:

- Multi-zone temporal cache treatment
- Different treatment for recent context, older context, and explicit outliers

These methods are the project’s direct attempts to preserve fidelity more intelligently than a single low-bit pass.

## Temporal Heuristics

### Age-Tier

- `AGE_TIER_INT4`
- `AGE_TIER_INT2`

Core idea:

- Split the cache into newer and older regions
- Keep newer context at higher precision
- Compress older context more aggressively

This family tests the hypothesis that recency matters more than distant history.

## FlowCache-Style Policy Adaptations

These methods move beyond pure low-bit quantization and ask a different question: what parts of the cache should be preserved, summarized, skipped, or evicted?

### FlowCache Hybrid

- `FLOWCACHE_HYBRID_INT2`

Core idea:

- Chunkwise age-aware precision
- Higher precision for recent chunks
- More aggressive compression for older chunks
- Layer-aware precision budgets

### FlowCache Adaptive

- `FLOWCACHE_ADAPTIVE_INT2`

Core idea:

- Assign importance scores to older chunks
- Keep more important chunks at higher precision
- Compress less important chunks more aggressively

### FlowCache Prune

- `FLOWCACHE_PRUNE_INT4`
- `FLOWCACHE_PRUNE_INT2`

Core idea:

- Hard-evict low-importance chunks
- Reconstruct pruned chunks as zeros at read time

### FlowCache Soft Prune

- `FLOWCACHE_SOFT_PRUNE_INT4`
- `FLOWCACHE_SOFT_PRUNE_INT2`

Core idea:

- Replace hard zero-eviction with a summary token
- Keep stronger memory savings while reducing the visual damage of hard pruning

### FlowCache Native

- `FLOWCACHE_NATIVE`
- `FLOWCACHE_NATIVE_SOFT_PRUNE_INT4`

Core idea:

- Reuse cached internal features when drift is low
- Reduce recomputation cost
- In the soft-prune variant, combine reuse with low-bit historical summarization

These methods form the practical memory-saving branch of the study.

## Spatial Heuristics

These methods test whether the cache should be quantized differently for foreground and background regions.

### Spatial Mixed Precision

- `SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2`
- `SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4`
- `SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2`
- `SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2`

Core idea:

- Keep foreground tokens at higher precision
- Compress background tokens more aggressively
- Estimate the foreground/background split from spatial motion

This family turned into one of the clearest negative results in the repo: intuitively appealing spatial partitioning did not align well with autoregressive video-generation dependencies.

## Coverage Summary

Method totals by family:

- `BF16`: 1
- `RTN`: 5
- `KIVI`: 4
- `QuaRot`: 4
- `PRQ`: 2
- `QAQ`: 2
- `Age-Tier`: 2
- `TPTQ`: 1
- `FlowCache`: 7
- `Spatial mixed`: 4

Total:

- `33` methods

## How To Read The Method Space

The design questions behind the catalog are:

- Does naive low-bit quantization already work?
- If not, do outlier-aware or residual-aware quantizers fix it?
- Does protecting recent context help more than treating all tokens equally?
- Can chunk importance and pruning outperform pure quantization?
- Does spatial partitioning map cleanly onto autoregressive video generation?

The repo’s results suggest that the most promising practical region comes from cache-policy adaptation, while the most informative high-fidelity research signals come from outlier handling and recency protection.
