# KV Cache Quantization for Self-Forcing Long Video Generation: Final Empirical Study

## Abstract

This report analyzes the final empirical dataset for self-forcing long-video generation, comprising **610 prompt-level observations** and **63 benchmark-level method summaries**. The study spans **33 MovieGen summaries** and **30 StoryEval summaries**, with each method evaluated through runtime, peak VRAM, KV compression, BF16-referenced fidelity, perceptual quality, and terminal temporal drift. Three conclusions are empirically robust. First, the **INT4 FlowCache pruning region** offers the strongest overall memory-quality trade-off: `FLOWCACHE_SOFT_PRUNE_INT4` attains **5.49x** MovieGen KV compression, roughly **39.3%** lower peak VRAM, and only **0.00%** imaging-quality degradation relative to BF16. Second, methods that preserve quality most strongly are not necessarily the best systems choices: `PRQ_INT2`, `PRQ_INT4`, and `QUAROT_KV_INT4` maintain near-BF16 quality, yet they remain too slow, too memory-hungry, or both. Third, the **spatially mixed foreground/background variants constitute clear negative results**, with the weakest variant collapsing to average imaging **0.399** and average terminal drift **0.396** across the two benchmarks. Cross-benchmark agreement is strong—**r = 0.9999** for compression, **r = 0.9996** for runtime, **r = 0.9318** for imaging quality, and **r = 0.9374** for drift—indicating that the principal conclusions are stable rather than benchmark-specific.

## 1. Introduction

Self-forcing video generation extends temporal horizons by feeding generated content back into the model during autoregressive rollout. That mechanism makes long-form generation feasible, but it also amplifies the cost of the attention cache: each new segment lengthens the stored history, increases the memory footprint of keys and values, and makes memory bandwidth a critical systems constraint. KV-cache quantization is therefore attractive because it directly targets the quantity that grows with horizon length.

The central empirical question is not whether a method compresses the cache in isolation, but whether that compression becomes a viable long-video operating point after the full generation process is considered. A useful method must withstand a multi-axis evaluation: it should reduce effective memory pressure, avoid intolerable runtime inflation, preserve perceptual and pixel-level fidelity, and remain stable under temporal rollout. The final empirical dataset allows all of these criteria to be tested jointly.

## 2. Background & Methodology

### 2.1 The KV-cache memory bottleneck in transformer-based video generation

Transformer decoders store a **key** tensor and a **value** tensor for every processed token so that subsequent attention steps can reuse past context instead of recomputing it. In long-video generation, the number of cached tokens grows with the temporal horizon, and the memory required for the KV cache scales approximately with sequence length, number of layers, number of heads, and head dimension. Self-forcing aggravates this pattern because the model repeatedly conditions on its own generated content: early context must remain available while later content is generated, so the cache is both large and long-lived.

This creates two coupled bottlenecks. The first is **capacity**: as the cache grows, peak VRAM becomes the limiting resource. The second is **bandwidth and latency**: even if a configuration fits in memory, reading and writing a large cache at every decoding step can slow generation substantially. Any rigorous quantization study must therefore examine both the nominal compression ratio and the realized end-to-end memory and runtime behavior.

### 2.2 Benchmark surfaces: MovieGen versus StoryEval

The empirical study uses two complementary benchmark surfaces.

- **MovieGen** is a **single-shot prompt benchmark**. Each method generates a standalone 10-second sample for a fixed prompt, and the evaluation emphasizes single-prompt quality, fidelity to the BF16 reference, and direct systems behavior.
- **StoryEval** is a **multi-prompt narrative benchmark**. It stresses whether a method can preserve quality and temporal stability across longer, story-like progression rather than only in a single isolated sample.

This distinction is important. A method can look acceptable in a single-shot setting yet drift under longer narrative rollout. Conversely, a method that remains stable under StoryEval offers stronger evidence that it can survive self-forcing’s temporal feedback loop.

### 2.3 Quantization method families

The methods in this study belong to several conceptually distinct families.

- **RTN (Round-to-Nearest)** applies uniform low-bit quantization to keys and values by partitioning tensors into blocks and rounding each value to the nearest representable level after scale estimation. It is the simplest direct precision-reduction baseline.
- **KIVI** uses **asymmetric key/value quantization**. In the implementation studied here, keys are quantized per channel across sequence blocks, whereas values are quantized per token across channel dimensions. The goal is to exploit the different statistics of keys and values rather than quantizing both identically.
- **QuaRot** reduces quantization difficulty by applying an **orthogonal Hadamard rotation** before quantization, then performing low-bit quantization in the rotated basis and inverse-rotating at read time. The intuition is that rotation redistributes large outliers so that low-bit rounding becomes less damaging.
- **PRQ** is a **progressive residual quantizer**. It first quantizes the original tensor, reconstructs that approximation, then quantizes the residual error in a second stage. Compression therefore comes from a two-stage low-bit representation rather than a single coarse rounding step.
- **QAQ** is an **outlier-aware asymmetric quantizer**. It clips the bulk of the tensor into a low-bit asymmetric code, but it stores extreme values explicitly so that large-magnitude outliers are not irreversibly destroyed.
- **Age-Tier** is a **recency-aware tiered scheme**. Recent tokens remain in higher precision, while older tokens are stored at lower precision. It treats temporal recency itself as the resource-allocation signal.
- **FlowCache** is a broader family that treats the cache as a **retention and reuse problem**, not merely a numeric quantization problem. The hybrid and adaptive variants keep recent chunks at higher precision, allocate older chunks according to temporal chunking, relative importance, and layer budget, and quantize older retained chunks more aggressively. The prune variants go further by evicting the least important old chunks entirely, whereas the soft-prune variants replace evicted chunks with pooled summaries instead of zeros. The native variant uses feature-drift-based reuse to skip or reuse work rather than only compressing tensor values.
- **TPTQ** is a **temporal progressive tiered quantizer**. Recent tokens remain in higher precision, older tokens are assigned a progressive residual representation, and extreme old-key outliers are preserved explicitly.
- **Spatial mixed methods** partition the token space into **foreground and background regions** using temporal variance, then quantize those regions with different quantizers and bit-widths. Compression arises from giving less salient spatial regions a more aggressive representation.

The key methodological point is that these families reduce memory through different mechanisms: direct low-bit rounding, asymmetric quantization, rotated quantization, residual coding, outlier preservation, recency-aware tiering, chunk pruning, soft chunk summarization, or foreground/background specialization. Comparing them within one empirical matrix is therefore informative because it separates genuinely effective mechanisms from mechanisms that only look plausible in principle.

### 2.4 Evaluation protocol and metrics

Each method is assessed along four axes.

1. **Performance:** average runtime per prompt.
2. **Memory footprint:** peak VRAM together with BF16 and compressed KV byte counts, summarized by the compression ratio.
3. **Generation quality:** BF16-referenced PSNR, SSIM, and LPIPS, plus perceptual metrics for background consistency, imaging quality, subject consistency, and aesthetic quality.
4. **Temporal stability:** the final available drift value, reported as the last imaging-quality score in the drift trajectory.

The empirical dataset is prompt-level, but the analysis below aggregates one summary row per method per benchmark by retaining the benchmark-level quality aggregates already present in the empirical data, taking the maximum observed prompt-level peak VRAM as the benchmark-level peak VRAM, and using the benchmark-specific terminal drift value as the temporal endpoint. Three archival MovieGen variants—`KIVI_K2_V4`, `RTN_K2_V4`, and `QUAROT_KV_INT4_REFRESH`—have fewer than ten videos and are therefore interpreted more cautiously than the fully covered rows.

### 2.5 Coverage and benchmark composition

**Table 1. Empirical coverage summary.**

| Benchmark | Prompt rows | Method summaries | Coverage note |
| --- | --- | --- | --- |
| MovieGen | 310 | 33 | 30 methods have 10 videos; KIVI_K2_V4 has 3, RTN_K2_V4 has 5, and QUAROT_KV_INT4_REFRESH has 2 |
| StoryEval | 300 | 30 | All 30 methods have 10 evaluated prompts |

## 3. Results and Empirical Analysis

### 3.1 Systems results: compression, peak VRAM, and runtime

The most important systems lesson is that **nominal KV compression and realized peak-VRAM relief are not equivalent**. `QUAROT_KV_INT4` reports **3.20x** compression on MovieGen and **3.20x** on StoryEval, yet its peak VRAM remains **19.98 GB** and **19.98 GB**, both slightly above BF16. `RTN_INT4_REFRESH` behaves similarly: it compresses the KV representation by roughly **3.20x**, but its peak memory actually rises to **22.64 GB** and **22.64 GB**.

By contrast, the INT4 FlowCache pruning methods convert compression into end-to-end systems relief. `FLOWCACHE_SOFT_PRUNE_INT4` reduces peak VRAM by approximately **39.3%** on MovieGen and **39.0%** on StoryEval while maintaining compression near **5.49x / 5.42x**. `FLOWCACHE_PRUNE_INT4` is similarly strong, with slightly more quality loss but essentially the same memory profile. At the opposite end of the memory axis, `FLOWCACHE_PRUNE_INT2` is the most aggressive memory point, reaching **7.78x** and **7.68x** compression with peak VRAM near **11.11 / 11.14 GB**.

`FLOWCACHE_NATIVE` shows a different phenomenon: it is the **only method that is faster than BF16 on both benchmarks**, improving runtime by **-17.6%** on MovieGen and **-13.7%** on StoryEval. However, its compression ratio is **1.00x**, so it should be understood as a latency-oriented reuse strategy rather than a memory-reduction solution.

**Table 2. Representative MovieGen systems results.**

| Method | Videos | Compression (x) | Peak VRAM (GB) | Runtime / Prompt (s) | Runtime vs BF16 |
| --- | --- | --- | --- | --- | --- |
| BF16 | 10 | 1.00 | 19.28 | 58.6 | +0.0% |
| FLOWCACHE_NATIVE | 10 | 1.00 | 19.31 | 48.3 | -17.6% |
| FLOWCACHE_SOFT_PRUNE_INT4 | 10 | 5.49 | 11.71 | 75.0 | +28.0% |
| FLOWCACHE_PRUNE_INT4 | 10 | 5.50 | 11.71 | 72.2 | +23.3% |
| FLOWCACHE_PRUNE_INT2 | 10 | 7.78 | 11.11 | 69.9 | +19.3% |
| RTN_INT4_REFRESH | 10 | 3.20 | 22.64 | 65.0 | +11.1% |
| QUAROT_KV_INT4 | 10 | 3.20 | 19.98 | 236.6 | +303.9% |
| PRQ_INT2 | 10 | 2.00 | 20.69 | 156.6 | +167.4% |
| PRQ_INT4 | 10 | 1.60 | 20.69 | 160.0 | +173.1% |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | 10 | 3.46 | 14.38 | 224.8 | +283.8% |

**Table 3. Representative StoryEval systems results.**

| Method | Videos | Compression (x) | Peak VRAM (GB) | Runtime / Prompt (s) | Runtime vs BF16 |
| --- | --- | --- | --- | --- | --- |
| BF16 | 10 | 1.00 | 19.28 | 56.8 | +0.0% |
| FLOWCACHE_NATIVE | 10 | 1.00 | 19.31 | 49.0 | -13.7% |
| FLOWCACHE_SOFT_PRUNE_INT4 | 10 | 5.42 | 11.76 | 75.2 | +32.3% |
| FLOWCACHE_PRUNE_INT4 | 10 | 5.43 | 11.75 | 72.4 | +27.5% |
| FLOWCACHE_PRUNE_INT2 | 10 | 7.68 | 11.14 | 70.2 | +23.5% |
| RTN_INT4_REFRESH | 10 | 3.20 | 22.64 | 64.6 | +13.7% |
| QUAROT_KV_INT4 | 10 | 3.20 | 19.98 | 239.6 | +321.7% |
| PRQ_INT2 | 10 | 2.00 | 20.69 | 155.6 | +174.0% |
| PRQ_INT4 | 10 | 1.60 | 20.69 | 158.0 | +178.0% |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | 10 | 3.46 | 14.38 | 224.1 | +294.4% |

### 3.2 Quality results: fidelity, perceptual quality, and drift

If one looks only at quality, the **PRQ family** emerges as the strongest non-BF16 region. `PRQ_INT4` has average imaging quality **0.719** and average terminal drift **0.719** across the two benchmarks, with `PRQ_INT2` essentially tied. The difficulty is that this quality comes with substantial systems cost: `PRQ_INT2` runs at **156.6 s** on MovieGen and **155.6 s** on StoryEval, and it peaks at **20.69 / 20.69 GB**, both above BF16.

The most practically important quality result is therefore the INT4 FlowCache band rather than the raw PRQ optimum. `FLOWCACHE_SOFT_PRUNE_INT4` is especially strong: on MovieGen, its imaging quality is **0.739** versus BF16 at **0.739**, and its terminal drift is **0.738** versus **0.739**. On StoryEval, it records **0.680** imaging quality and **0.679** terminal drift against BF16’s **0.693** and **0.695**. This is the critical academic point: a method need not be the single best quality point to be the best systems choice; it only needs to preserve enough quality while moving memory and runtime in the correct direction.

`QUAROT_KV_INT4` is the strongest counterexample. Its quality is near BF16—within **0.20%** on MovieGen and **0.90%** on StoryEval—yet it is not a strong deployment choice because its runtime remains around **4× BF16** and its peak VRAM is not reduced in practice.

**Table 4. Representative MovieGen quality results.**

| Method | PSNR | SSIM | LPIPS | Background | Imaging | Subject | Aesthetic | Drift Last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | inf | 1.000 | 0.000 | 0.955 | 0.739 | 0.954 | 0.665 | 0.739 |
| FLOWCACHE_NATIVE | 13.25 | 0.412 | 0.451 | 0.952 | 0.738 | 0.961 | 0.661 | 0.737 |
| FLOWCACHE_SOFT_PRUNE_INT4 | 17.67 | 0.544 | 0.297 | 0.948 | 0.739 | 0.946 | 0.655 | 0.738 |
| FLOWCACHE_PRUNE_INT4 | 15.30 | 0.457 | 0.412 | 0.934 | 0.727 | 0.916 | 0.644 | 0.726 |
| FLOWCACHE_PRUNE_INT2 | 15.26 | 0.467 | 0.483 | 0.897 | 0.637 | 0.832 | 0.538 | 0.633 |
| RTN_INT4_REFRESH | 21.45 | 0.693 | 0.178 | 0.947 | 0.736 | 0.941 | 0.649 | 0.735 |
| QUAROT_KV_INT4 | 22.64 | 0.724 | 0.148 | 0.951 | 0.738 | 0.949 | 0.658 | 0.738 |
| PRQ_INT2 | 25.13 | 0.800 | 0.094 | 0.954 | 0.739 | 0.955 | 0.669 | 0.740 |
| PRQ_INT4 | 26.54 | 0.824 | 0.082 | 0.955 | 0.739 | 0.956 | 0.665 | 0.739 |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | 14.06 | 0.433 | 0.570 | 0.811 | 0.399 | 0.665 | 0.399 | 0.394 |

**Table 5. Representative StoryEval quality results.**

| Method | PSNR | SSIM | LPIPS | Background | Imaging | Subject | Aesthetic | Drift Last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | inf | 1.000 | 0.000 | 0.932 | 0.693 | 0.921 | 0.556 | 0.695 |
| FLOWCACHE_NATIVE | 11.95 | 0.451 | 0.508 | 0.926 | 0.681 | 0.889 | 0.550 | 0.682 |
| FLOWCACHE_SOFT_PRUNE_INT4 | inf | 0.518 | 0.416 | 0.909 | 0.680 | 0.900 | 0.549 | 0.679 |
| FLOWCACHE_PRUNE_INT4 | inf | 0.465 | 0.490 | 0.900 | 0.682 | 0.873 | 0.551 | 0.680 |
| FLOWCACHE_PRUNE_INT2 | 14.26 | 0.492 | 0.506 | 0.865 | 0.516 | 0.769 | 0.455 | 0.516 |
| RTN_INT4_REFRESH | 18.55 | 0.654 | 0.252 | 0.923 | 0.678 | 0.914 | 0.541 | 0.679 |
| QUAROT_KV_INT4 | 19.25 | 0.685 | 0.217 | 0.926 | 0.687 | 0.920 | 0.545 | 0.689 |
| PRQ_INT2 | 20.66 | 0.733 | 0.179 | 0.933 | 0.698 | 0.927 | 0.554 | 0.698 |
| PRQ_INT4 | inf | 0.724 | 0.188 | 0.931 | 0.699 | 0.921 | 0.557 | 0.699 |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | 12.97 | 0.444 | 0.599 | 0.808 | 0.400 | 0.624 | 0.346 | 0.398 |

### 3.3 Pareto-optimal trade-offs

To formalize the trade-off, we define a method as **Pareto-optimal** within a benchmark if no competing method achieves both **greater or equal compression** and **greater or equal quality**, with at least one strict improvement. We compute that front twice: once with **imaging quality** as the quality axis and once with **terminal drift**.

The resulting fronts contain two qualitatively different types of points. The first type consists of **extreme compression points**, such as `FLOWCACHE_PRUNE_INT2` and `FLOWCACHE_SOFT_PRUNE_INT2`. These methods survive on the front because they move far on the compression axis, not because their absolute quality is especially attractive. The second type consists of **practical near-BF16 points**, including `FLOWCACHE_SOFT_PRUNE_INT4`, `FLOWCACHE_PRUNE_INT4`, `QUAROT_KV_INT4`, and the PRQ variants, depending on benchmark and quality axis.

This distinction is important for an academic defense. Pareto optimality by itself is not a deployment recommendation; it only states that a point is mathematically nondominated. The most useful recommendation still comes from combining the front with absolute quality and systems judgment. Under that stricter criterion, the INT4 FlowCache region remains the strongest overall operating point.

**Table 6. Pareto-optimal methods under compression-versus-quality trade-offs.**

| Benchmark | Quality axis | Method | Compression (x) | Quality | Runtime / Prompt (s) | Peak VRAM (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| MovieGen | Imaging quality | FLOWCACHE_PRUNE_INT2 | 7.78 | 0.637 | 69.9 | 11.11 |
| MovieGen | Imaging quality | FLOWCACHE_SOFT_PRUNE_INT2 | 6.82 | 0.662 | 76.1 | 11.71 |
| MovieGen | Imaging quality | FLOWCACHE_PRUNE_INT4 | 5.50 | 0.727 | 72.2 | 11.71 |
| MovieGen | Imaging quality | FLOWCACHE_SOFT_PRUNE_INT4 | 5.49 | 0.739 | 75.0 | 11.71 |
| MovieGen | Imaging quality | PRQ_INT2 | 2.00 | 0.739 | 156.6 | 20.69 |
| MovieGen | Terminal drift | FLOWCACHE_PRUNE_INT2 | 7.78 | 0.633 | 69.9 | 11.11 |
| MovieGen | Terminal drift | FLOWCACHE_SOFT_PRUNE_INT2 | 6.82 | 0.658 | 76.1 | 11.71 |
| MovieGen | Terminal drift | FLOWCACHE_PRUNE_INT4 | 5.50 | 0.726 | 72.2 | 11.71 |
| MovieGen | Terminal drift | FLOWCACHE_SOFT_PRUNE_INT4 | 5.49 | 0.738 | 75.0 | 11.71 |
| MovieGen | Terminal drift | PRQ_INT2 | 2.00 | 0.740 | 156.6 | 20.69 |
| StoryEval | Imaging quality | FLOWCACHE_PRUNE_INT2 | 7.68 | 0.516 | 70.2 | 11.14 |
| StoryEval | Imaging quality | FLOWCACHE_SOFT_PRUNE_INT2 | 6.72 | 0.532 | 74.4 | 11.76 |
| StoryEval | Imaging quality | FLOWCACHE_PRUNE_INT4 | 5.43 | 0.682 | 72.4 | 11.75 |
| StoryEval | Imaging quality | QUAROT_KV_INT4 | 3.20 | 0.687 | 239.6 | 19.98 |
| StoryEval | Imaging quality | PRQ_INT2 | 2.00 | 0.698 | 155.6 | 20.69 |
| StoryEval | Imaging quality | PRQ_INT4 | 1.60 | 0.699 | 158.0 | 20.69 |
| StoryEval | Terminal drift | FLOWCACHE_PRUNE_INT2 | 7.68 | 0.516 | 70.2 | 11.14 |
| StoryEval | Terminal drift | FLOWCACHE_SOFT_PRUNE_INT2 | 6.72 | 0.536 | 74.4 | 11.76 |
| StoryEval | Terminal drift | FLOWCACHE_PRUNE_INT4 | 5.43 | 0.680 | 72.4 | 11.75 |
| StoryEval | Terminal drift | QUAROT_KV_INT4 | 3.20 | 0.689 | 239.6 | 19.98 |
| StoryEval | Terminal drift | PRQ_INT2 | 2.00 | 0.698 | 155.6 | 20.69 |
| StoryEval | Terminal drift | PRQ_INT4 | 1.60 | 0.699 | 158.0 | 20.69 |

### 3.4 Cross-benchmark stability

The rankings are notably stable across the two benchmarks. Compression ratio and runtime are almost perfectly correlated at the method level, and even the perceptual metrics remain strongly aligned. That matters because it indicates that the conclusions are structural rather than benchmark-specific: the methods that perform well under a single-shot setting generally remain strong under the narrative setting, and the most severe failures remain severe failures.

**Table 7. Cross-benchmark method-level correlations.**

| Metric | Pearson r |
| --- | --- |
| Compression ratio | 0.9999 |
| Runtime / prompt | 0.9996 |
| Imaging quality | 0.9318 |
| Terminal drift | 0.9374 |
| SSIM | 0.9876 |

This stability is visible qualitatively as well. The same small cluster—`FLOWCACHE_SOFT_PRUNE_INT4`, `FLOWCACHE_PRUNE_INT4`, `RTN_INT4_REFRESH`, `QUAROT_KV_INT4`, and the PRQ variants—occupies the upper-quality region across both benchmarks, while the weakest spatially mixed variants occupy the lower-quality, lower-drift region in both cases.

### 3.5 Failure analysis and negative results

The study contains several decisive negative results. The clearest is `SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2`, which compresses by only **3.46x** on MovieGen while collapsing to average imaging **0.399**, average terminal drift **0.396**, and a MovieGen runtime of **224.8 s**. Simpler alternatives are both faster and better.

The second negative lesson is that **aggressive INT2 compression can be mathematically appealing but empirically weak**. `RTN_INT2` and `QUAROT_KV_INT2` both attain roughly **5.33x** KV compression, but their perceptual and temporal quality degrade sharply, and `QUAROT_KV_INT2` remains extremely slow. These methods are therefore useful as lower-bound references for the compression-quality curve, not as recommended operating points.

**Table 8. Empirically weak or dominated configurations.**

| Method | Avg imaging | Avg drift | MovieGen compression (x) | MovieGen runtime (s) | Interpretation |
| --- | --- | --- | --- | --- | --- |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | 0.399 | 0.396 | 3.46 | 224.8 | Lowest average imaging and drift in the study, while still running far slower than BF16. |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 | 0.416 | 0.413 | 3.68 | 106.6 | Second-worst quality cluster; the compression gain does not justify the quality collapse. |
| SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2 | 0.492 | 0.486 | 3.45 | 110.4 | Substantial quality loss despite only mid-range compression. |
| RTN_INT2 | 0.516 | 0.516 | 5.33 | 87.1 | Aggressive compression point that is dominated by the stronger INT2 FlowCache variants. |
| QUAROT_KV_INT2 | 0.539 | 0.538 | 5.33 | 242.0 | High compression but very slow and lower quality than better 5x-class alternatives. |

## 4. Conclusion

The empirical evidence supports a narrow but strong claim: **the INT4 FlowCache pruning family is the best overall answer to the KV-cache bottleneck in this self-forcing long-video setting**. It is not the absolute highest-quality region, but it is the region that best reconciles memory reduction, acceptable runtime, perceptual quality, and temporal stability.

A second conclusion is that **quality preservation alone is an insufficient selection rule**. `PRQ_INT2`, `PRQ_INT4`, and `QUAROT_KV_INT4` all preserve quality impressively, yet they do not provide the most useful operating point because runtime and peak VRAM remain poor. Conversely, `FLOWCACHE_NATIVE` improves runtime but does not materially reduce memory. The correct selection criterion is therefore joint systems-and-quality performance, not a single metric.

Finally, the negative results are as informative as the positive ones. The spatially mixed foreground/background methods and several aggressive INT2 baselines demonstrate that not all intuitively plausible compression strategies remain stable under self-forcing long-video rollout. The strongest academic conclusion is therefore comparative rather than absolute: among the tested methods, **INT4 FlowCache pruning is the most credible practical operating region, whereas the weakest spatially mixed variants are clearly unsuitable for deployment in this setting**.

## Appendix A. Complete MovieGen tables

**Table A1. Full MovieGen systems matrix.**

| Method | Videos | Compression (x) | Peak VRAM (GB) | Runtime / Prompt (s) | Runtime vs BF16 |
| --- | --- | --- | --- | --- | --- |
| BF16 | 10 | 1.00 | 19.28 | 58.6 | +0.0% |
| FLOWCACHE_NATIVE | 10 | 1.00 | 19.31 | 48.3 | -17.6% |
| FLOWCACHE_NATIVE_SOFT_PRUNE_INT4 | 10 | 5.49 | 11.74 | 63.6 | +8.6% |
| FLOWCACHE_SOFT_PRUNE_INT4 | 10 | 5.49 | 11.71 | 75.0 | +28.0% |
| FLOWCACHE_PRUNE_INT4 | 10 | 5.50 | 11.71 | 72.2 | +23.3% |
| FLOWCACHE_SOFT_PRUNE_INT2 | 10 | 6.82 | 11.71 | 76.1 | +30.0% |
| FLOWCACHE_PRUNE_INT2 | 10 | 7.78 | 11.11 | 69.9 | +19.3% |
| RTN_INT4 | 10 | 3.20 | 19.98 | 86.3 | +47.3% |
| RTN_INT4_REFRESH | 10 | 3.20 | 22.64 | 65.0 | +11.1% |
| QUAROT_KV_INT4 | 10 | 3.20 | 19.98 | 236.6 | +303.9% |
| PRQ_INT2 | 10 | 2.00 | 20.69 | 156.6 | +167.4% |
| PRQ_INT4 | 10 | 1.60 | 20.69 | 160.0 | +173.1% |
| AGE_TIER_INT2 | 10 | 4.41 | 14.38 | 105.3 | +79.7% |
| AGE_TIER_INT4 | 10 | 3.18 | 14.38 | 103.9 | +77.3% |
| FLOWCACHE_ADAPTIVE_INT2 | 10 | 4.27 | 14.38 | 92.6 | +58.2% |
| FLOWCACHE_HYBRID_INT2 | 10 | 4.61 | 14.38 | 82.6 | +41.0% |
| KIVI_INT2 | 10 | 5.31 | 19.99 | 95.5 | +63.0% |
| KIVI_INT4 | 10 | 3.19 | 19.99 | 92.7 | +58.2% |
| KIVI_INT4_REFRESH | 10 | 3.19 | 22.63 | 68.1 | +16.2% |
| KIVI_K2_V4 | 3 | NA | 22.67 | 76.3 | +30.3% |
| QAQ_INT2 | 10 | 5.18 | 14.42 | 109.8 | +87.4% |
| QAQ_INT4 | 10 | 3.14 | 14.42 | 110.0 | +87.8% |
| QUAROT_KV_INT2 | 10 | 5.33 | 19.98 | 242.0 | +313.2% |
| QUAROT_KV_INT4_RECENT2 | 10 | 2.43 | 21.69 | 111.3 | +90.0% |
| QUAROT_KV_INT4_REFRESH | 2 | NA | 22.82 | 97.5 | +66.5% |
| RTN_INT2 | 10 | 5.33 | 19.98 | 87.1 | +48.7% |
| RTN_INT4_RECENT2 | 10 | 2.43 | 21.37 | 68.9 | +17.6% |
| RTN_K2_V4 | 5 | NA | 22.68 | 75.3 | +28.6% |
| SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2 | 10 | 3.45 | 14.38 | 110.4 | +88.4% |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | 10 | 3.46 | 14.38 | 224.8 | +283.8% |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 | 10 | 3.68 | 14.38 | 106.6 | +82.0% |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4 | 10 | 3.18 | 14.38 | 105.5 | +80.0% |
| TPTQ_INT2 | 10 | 2.72 | 19.85 | 167.2 | +185.5% |

**Table A2. Full MovieGen quality matrix.**

| Method | PSNR | SSIM | LPIPS | Background | Imaging | Subject | Aesthetic | Drift Last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | inf | 1.000 | 0.000 | 0.955 | 0.739 | 0.954 | 0.665 | 0.739 |
| FLOWCACHE_NATIVE | 13.25 | 0.412 | 0.451 | 0.952 | 0.738 | 0.961 | 0.661 | 0.737 |
| FLOWCACHE_NATIVE_SOFT_PRUNE_INT4 | 13.26 | 0.411 | 0.475 | 0.948 | 0.726 | 0.945 | 0.647 | 0.724 |
| FLOWCACHE_SOFT_PRUNE_INT4 | 17.67 | 0.544 | 0.297 | 0.948 | 0.739 | 0.946 | 0.655 | 0.738 |
| FLOWCACHE_PRUNE_INT4 | 15.30 | 0.457 | 0.412 | 0.934 | 0.727 | 0.916 | 0.644 | 0.726 |
| FLOWCACHE_SOFT_PRUNE_INT2 | 15.84 | 0.482 | 0.440 | 0.907 | 0.662 | 0.852 | 0.561 | 0.658 |
| FLOWCACHE_PRUNE_INT2 | 15.26 | 0.467 | 0.483 | 0.897 | 0.637 | 0.832 | 0.538 | 0.633 |
| RTN_INT4 | 21.32 | 0.688 | 0.180 | 0.945 | 0.735 | 0.941 | 0.647 | 0.734 |
| RTN_INT4_REFRESH | 21.45 | 0.693 | 0.178 | 0.947 | 0.736 | 0.941 | 0.649 | 0.735 |
| QUAROT_KV_INT4 | 22.64 | 0.724 | 0.148 | 0.951 | 0.738 | 0.949 | 0.658 | 0.738 |
| PRQ_INT2 | 25.13 | 0.800 | 0.094 | 0.954 | 0.739 | 0.955 | 0.669 | 0.740 |
| PRQ_INT4 | 26.54 | 0.824 | 0.082 | 0.955 | 0.739 | 0.956 | 0.665 | 0.739 |
| AGE_TIER_INT2 | 15.18 | 0.457 | 0.470 | 0.864 | 0.578 | 0.794 | 0.510 | 0.573 |
| AGE_TIER_INT4 | 21.32 | 0.688 | 0.180 | 0.945 | 0.735 | 0.940 | 0.647 | 0.734 |
| FLOWCACHE_ADAPTIVE_INT2 | 15.19 | 0.448 | 0.464 | 0.867 | 0.616 | 0.787 | 0.513 | 0.611 |
| FLOWCACHE_HYBRID_INT2 | 15.62 | 0.471 | 0.454 | 0.877 | 0.616 | 0.815 | 0.524 | 0.612 |
| KIVI_INT2 | 11.42 | 0.241 | 0.671 | 0.817 | 0.621 | 0.668 | 0.461 | 0.618 |
| KIVI_INT4 | 13.07 | 0.405 | 0.571 | 0.915 | 0.681 | 0.871 | 0.603 | 0.678 |
| KIVI_INT4_REFRESH | 13.73 | 0.420 | 0.509 | 0.924 | 0.714 | 0.886 | 0.620 | 0.712 |
| KIVI_K2_V4 | 13.03 | 0.374 | 0.578 | 0.894 | 0.623 | 0.873 | 0.604 | 0.619 |
| QAQ_INT2 | 13.34 | 0.365 | 0.530 | 0.868 | 0.620 | 0.789 | 0.538 | 0.618 |
| QAQ_INT4 | 11.97 | 0.262 | 0.647 | 0.822 | 0.589 | 0.686 | 0.498 | 0.586 |
| QUAROT_KV_INT2 | 14.73 | 0.440 | 0.467 | 0.874 | 0.601 | 0.796 | 0.523 | 0.597 |
| QUAROT_KV_INT4_RECENT2 | inf | 0.706 | 0.183 | 0.943 | 0.730 | 0.931 | 0.641 | 0.729 |
| QUAROT_KV_INT4_REFRESH | 19.64 | 0.613 | 0.214 | 0.941 | 0.722 | 0.937 | 0.648 | 0.719 |
| RTN_INT2 | 15.04 | 0.451 | 0.475 | 0.861 | 0.567 | 0.788 | 0.507 | 0.562 |
| RTN_INT4_RECENT2 | 23.69 | 0.732 | 0.148 | 0.949 | 0.736 | 0.944 | 0.647 | 0.735 |
| RTN_K2_V4 | 14.74 | 0.434 | 0.495 | 0.844 | 0.531 | 0.775 | 0.517 | 0.524 |
| SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2 | 13.72 | 0.427 | 0.642 | 0.854 | 0.529 | 0.729 | 0.507 | 0.521 |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | 14.06 | 0.433 | 0.570 | 0.811 | 0.399 | 0.665 | 0.399 | 0.394 |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 | 13.93 | 0.421 | 0.558 | 0.809 | 0.411 | 0.671 | 0.404 | 0.407 |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4 | 18.89 | 0.577 | 0.310 | 0.919 | 0.693 | 0.883 | 0.583 | 0.690 |
| TPTQ_INT2 | 19.91 | 0.627 | 0.240 | 0.942 | 0.724 | 0.926 | 0.629 | 0.722 |

## Appendix B. Complete StoryEval tables

**Table B1. Full StoryEval systems matrix.**

| Method | Videos | Compression (x) | Peak VRAM (GB) | Runtime / Prompt (s) | Runtime vs BF16 |
| --- | --- | --- | --- | --- | --- |
| BF16 | 10 | 1.00 | 19.28 | 56.8 | +0.0% |
| FLOWCACHE_NATIVE | 10 | 1.00 | 19.31 | 49.0 | -13.7% |
| FLOWCACHE_NATIVE_SOFT_PRUNE_INT4 | 10 | 5.42 | 11.78 | 64.2 | +13.1% |
| FLOWCACHE_SOFT_PRUNE_INT4 | 10 | 5.42 | 11.76 | 75.2 | +32.3% |
| FLOWCACHE_PRUNE_INT4 | 10 | 5.43 | 11.75 | 72.4 | +27.5% |
| FLOWCACHE_SOFT_PRUNE_INT2 | 10 | 6.72 | 11.76 | 74.4 | +30.9% |
| FLOWCACHE_PRUNE_INT2 | 10 | 7.68 | 11.14 | 70.2 | +23.5% |
| RTN_INT4 | 10 | 3.20 | 19.98 | 88.8 | +56.3% |
| RTN_INT4_REFRESH | 10 | 3.20 | 22.64 | 64.6 | +13.7% |
| QUAROT_KV_INT4 | 10 | 3.20 | 19.98 | 239.6 | +321.7% |
| PRQ_INT2 | 10 | 2.00 | 20.69 | 155.6 | +174.0% |
| PRQ_INT4 | 10 | 1.60 | 20.69 | 158.0 | +178.0% |
| AGE_TIER_INT2 | 10 | 4.41 | 14.38 | 101.9 | +79.4% |
| AGE_TIER_INT4 | 10 | 3.18 | 14.38 | 102.4 | +80.3% |
| FLOWCACHE_ADAPTIVE_INT2 | 10 | 4.26 | 14.38 | 91.3 | +60.7% |
| FLOWCACHE_HYBRID_INT2 | 10 | 4.59 | 14.38 | 82.2 | +44.6% |
| KIVI_INT2 | 10 | 5.31 | 19.99 | 94.7 | +66.7% |
| KIVI_INT4 | 10 | 3.19 | 19.99 | 93.0 | +63.7% |
| KIVI_INT4_REFRESH | 10 | 3.19 | 22.63 | 66.7 | +17.5% |
| QAQ_INT2 | 10 | 5.19 | 14.42 | 109.9 | +93.4% |
| QAQ_INT4 | 10 | 3.15 | 14.42 | 109.9 | +93.4% |
| QUAROT_KV_INT2 | 10 | 5.33 | 19.98 | 239.0 | +320.8% |
| QUAROT_KV_INT4_RECENT2 | 10 | 2.43 | 21.69 | 112.9 | +98.8% |
| RTN_INT2 | 10 | 5.33 | 19.98 | 86.1 | +51.6% |
| RTN_INT4_RECENT2 | 10 | 2.43 | 21.37 | 68.6 | +20.8% |
| SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2 | 10 | 3.45 | 14.38 | 110.2 | +94.0% |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | 10 | 3.46 | 14.38 | 224.1 | +294.4% |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 | 10 | 3.69 | 14.38 | 106.6 | +87.6% |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4 | 10 | 3.18 | 14.38 | 106.6 | +87.7% |
| TPTQ_INT2 | 10 | 2.72 | 19.77 | 166.6 | +193.2% |

**Table B2. Full StoryEval quality matrix.**

| Method | PSNR | SSIM | LPIPS | Background | Imaging | Subject | Aesthetic | Drift Last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BF16 | inf | 1.000 | 0.000 | 0.932 | 0.693 | 0.921 | 0.556 | 0.695 |
| FLOWCACHE_NATIVE | 11.95 | 0.451 | 0.508 | 0.926 | 0.681 | 0.889 | 0.550 | 0.682 |
| FLOWCACHE_NATIVE_SOFT_PRUNE_INT4 | 11.94 | 0.436 | 0.523 | 0.920 | 0.657 | 0.876 | 0.547 | 0.657 |
| FLOWCACHE_SOFT_PRUNE_INT4 | inf | 0.518 | 0.416 | 0.909 | 0.680 | 0.900 | 0.549 | 0.679 |
| FLOWCACHE_PRUNE_INT4 | inf | 0.465 | 0.490 | 0.900 | 0.682 | 0.873 | 0.551 | 0.680 |
| FLOWCACHE_SOFT_PRUNE_INT2 | 14.18 | 0.497 | 0.495 | 0.876 | 0.532 | 0.794 | 0.471 | 0.536 |
| FLOWCACHE_PRUNE_INT2 | 14.26 | 0.492 | 0.506 | 0.865 | 0.516 | 0.769 | 0.455 | 0.516 |
| RTN_INT4 | 18.66 | 0.661 | 0.245 | 0.923 | 0.674 | 0.912 | 0.539 | 0.675 |
| RTN_INT4_REFRESH | 18.55 | 0.654 | 0.252 | 0.923 | 0.678 | 0.914 | 0.541 | 0.679 |
| QUAROT_KV_INT4 | 19.25 | 0.685 | 0.217 | 0.926 | 0.687 | 0.920 | 0.545 | 0.689 |
| PRQ_INT2 | 20.66 | 0.733 | 0.179 | 0.933 | 0.698 | 0.927 | 0.554 | 0.698 |
| PRQ_INT4 | inf | 0.724 | 0.188 | 0.931 | 0.699 | 0.921 | 0.557 | 0.699 |
| AGE_TIER_INT2 | 13.67 | 0.463 | 0.523 | 0.862 | 0.469 | 0.758 | 0.457 | 0.473 |
| AGE_TIER_INT4 | 18.66 | 0.661 | 0.245 | 0.923 | 0.674 | 0.912 | 0.539 | 0.676 |
| FLOWCACHE_ADAPTIVE_INT2 | inf | 0.451 | 0.528 | 0.852 | 0.498 | 0.736 | 0.443 | 0.496 |
| FLOWCACHE_HYBRID_INT2 | 13.95 | 0.479 | 0.512 | 0.871 | 0.492 | 0.776 | 0.457 | 0.494 |
| KIVI_INT2 | inf | 0.243 | 0.735 | 0.798 | 0.531 | 0.605 | 0.380 | 0.527 |
| KIVI_INT4 | 12.41 | 0.424 | 0.575 | 0.891 | 0.635 | 0.835 | 0.512 | 0.635 |
| KIVI_INT4_REFRESH | 12.49 | 0.414 | 0.569 | 0.881 | 0.645 | 0.829 | 0.500 | 0.641 |
| QAQ_INT2 | 11.79 | 0.324 | 0.635 | 0.839 | 0.579 | 0.712 | 0.461 | 0.585 |
| QAQ_INT4 | 11.03 | 0.238 | 0.719 | 0.808 | 0.567 | 0.630 | 0.430 | 0.571 |
| QUAROT_KV_INT2 | 13.01 | 0.443 | 0.532 | 0.861 | 0.477 | 0.754 | 0.459 | 0.480 |
| QUAROT_KV_INT4_RECENT2 | inf | 0.707 | 0.222 | 0.919 | 0.666 | 0.905 | 0.538 | 0.670 |
| RTN_INT2 | 13.57 | 0.459 | 0.526 | 0.859 | 0.464 | 0.753 | 0.453 | 0.471 |
| RTN_INT4_RECENT2 | inf | 0.721 | 0.187 | 0.924 | 0.680 | 0.914 | 0.545 | 0.684 |
| SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2 | 13.20 | 0.453 | 0.654 | 0.832 | 0.454 | 0.664 | 0.409 | 0.451 |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | 12.97 | 0.444 | 0.599 | 0.808 | 0.400 | 0.624 | 0.346 | 0.398 |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 | 12.64 | 0.430 | 0.587 | 0.810 | 0.421 | 0.636 | 0.352 | 0.419 |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4 | inf | 0.587 | 0.349 | 0.899 | 0.606 | 0.856 | 0.513 | 0.607 |
| TPTQ_INT2 | 17.03 | 0.615 | 0.301 | 0.921 | 0.654 | 0.906 | 0.532 | 0.658 |

## Reproducibility note

All numerical claims in this document are drawn from the same empirical dataset used throughout the experimental evaluation. Values reported as `inf` arise when the fidelity evaluator encounters exact frame equality with the BF16 reference.
