# KV Cache Quantization for Self-Forcing Long Video Generation: Final Empirical Study

## Abstract

This report analyzes KV-cache compression for self-forcing long video generation using a fully populated empirical dataset containing **610 prompt-level observations**, split across **310 MovieGen evaluations** and **300 StoryEval evaluations**. The study covers **33 MovieGen method summaries** and **30 StoryEval method summaries**, and it measures systems behavior, perceptual quality, BF16-referenced fidelity, and temporal drift jointly rather than in isolation.

Four conclusions are robust. First, **naive low-bit quantization is not sufficient**: `RTN_INT4` achieves a nominal **3.200x** KV compression on both benchmarks, yet its peak VRAM remains slightly *higher* than BF16 at **19.983 GB** versus **19.280 GB**. Second, **quality-preserving compression can still fail as a systems method**: `PRQ_INT4` is the strongest compressed method when strict BF16 replication matters, reaching **MovieGen SSIM = 0.824**, **MovieGen PSNR = 26.545**, and **MovieGen LPIPS = 0.082**, but it is also extremely slow at **159.971 s** per MovieGen prompt and **157.960 s** per StoryEval prompt. Third, the **custom FlowCache-inspired soft-prune adaptation** is the strongest memory-relief point in the study, reducing peak VRAM to **11.711 GB** on MovieGen and **11.756 GB** on StoryEval while reaching **5.490x** and **5.424x** KV compression, respectively. Fourth, that same method reveals the study's central scientific paradox: it preserves **VBench imaging quality** almost perfectly on MovieGen (**0.739** versus BF16 **0.739**) while suffering a sharp collapse in **structural fidelity** (**SSIM = 0.544**, **PSNR = 17.673**). In other words, it produces videos that still look plausible and aesthetically coherent, but they are no longer faithful reproductions of the BF16 baseline's exact spatial arrangement.

The main recommendation therefore depends on the operational objective. If the goal is **maximum VRAM relief with acceptable perceptual realism**, the custom FlowCache-inspired soft-prune method is the most attractive operating point. If the goal is **strict replication of the BF16 reference trajectory**, `PRQ_INT4` is the only viable compressed method among the representative boundary points, despite its severe runtime cost.

## 1. Introduction

Self-forcing is attractive for long-horizon video generation because it converts a short-horizon generator into a longer narrative generator by repeatedly feeding generated context back into the model. The benefit is temporal reach; the cost is that the model must repeatedly attend over an ever-growing history of keys and values. As rollout length grows, the attention cache becomes one of the dominant systems constraints.

This creates a research problem that is simultaneously algorithmic and systems-oriented. A compression method may appear promising when judged only by nominal bit width or only by qualitative visual inspection, yet fail once end-to-end peak memory, runtime, structural fidelity, and long-horizon stability are examined together. In long video generation, the question is not merely whether the cache can be compressed. The real question is whether a compressed cache still supports the same generative computation strongly enough to produce useful output under realistic runtime and VRAM budgets.

The present study addresses that question through a benchmarked empirical comparison of KV-cache compression methods for a self-forcing transformer video generator. The report deliberately distinguishes between two quality axes that are often conflated. The first is **visual plausibility**, measured here by **VBench imaging quality**. The second is **strict adherence to the BF16 reference**, measured by **PSNR, SSIM, and LPIPS**. That distinction proves decisive. Several methods generate videos that remain visually plausible in isolation, yet they do not preserve the same object placement, geometry, or scene structure as the uncompressed baseline.

The report proceeds in four steps. Section 2 defines the architecture-level bottleneck, the benchmarks, the quality metrics, and the method families. Section 3 presents benchmark-spanning systems and quality tables for five representative boundary points: BF16, `RTN_INT4`, `PRQ_INT4`, `FLOWCACHE_SOFT_PRUNE_INT4`, and the spatial mixed failure case. Section 3 then develops a dedicated analysis of the **FlowCache paradox**, namely the coexistence of near-baseline VBench realism with sharply degraded fidelity. Section 4 concludes with deployment guidance.

## 2. Background & Methodology

### 2.1 The KV-cache bottleneck in self-forcing transformers

Transformer decoders cache a **key tensor** and a **value tensor** for every processed token so that future attention operations can reuse past context rather than recomputing it from scratch. In video generation, the number of retained tokens grows with temporal horizon, spatial resolution, and model depth. In self-forcing settings, this burden is amplified because previously generated content is fed back into the model as conditioning context for later generation. As a result, the cache is both large and long-lived.

That burden manifests as two distinct bottlenecks.

The first is a **capacity bottleneck**. Keys and values occupy device memory directly, so peak VRAM rises as the temporal horizon lengthens. A method that improves nominal KV compression but does not lower realized peak VRAM has not solved the capacity problem in practice.

The second is a **bandwidth and latency bottleneck**. Even when the cache still fits in memory, every decoding step requires moving large key and value blocks through memory hierarchies. Long-horizon attention therefore becomes increasingly constrained by memory traffic, dequantization overhead, and cache management cost. A method that is compact on paper but expensive to read, write, rotate, or reconstruct can preserve memory while still making generation too slow to be useful.

A rigorous KV-cache study must therefore measure both **what is stored** and **what it costs to use**. This is why the present analysis evaluates compression ratio, peak VRAM, and runtime together.

### 2.2 Benchmark definitions: MovieGen versus StoryEval

The empirical study uses two complementary evaluation surfaces.

**MovieGen** is a **single-shot prompt completion benchmark**. Each prompt is treated as an independent generation task, so the benchmark emphasizes per-prompt systems behavior and isolated sample quality. MovieGen is the most direct setting for asking whether a compressed cache still reproduces the BF16 reference for a single generated video.

**StoryEval** is a **multi-prompt narrative stability benchmark**. Rather than treating each sample as an isolated one-shot generation, StoryEval evaluates whether a method remains stable across a longer narrative progression. It is therefore more sensitive to accumulated temporal degradation, consistency drift, and the kinds of long-range errors that self-forcing can amplify.

This benchmark pairing matters because a method can look acceptable in a single video while still degrading under narrative rollout. MovieGen measures whether a method works at all for a prompt. StoryEval measures whether it remains coherent when the generative process becomes temporally recursive.

### 2.3 Metric definitions: VBench realism versus BF16 fidelity

The study uses two conceptually different quality axes, and this distinction is essential for interpreting the results correctly.

#### VBench imaging quality

**VBench imaging quality** is a perceptual realism metric. It asks whether a generated video looks like a coherent, visually plausible video. High imaging-quality scores indicate that the sample is aesthetically credible and visually well formed when viewed on its own terms.

In practical language, this metric answers the question: **"Does the video look real and coherent?"** It does **not** require the compressed method to reproduce the exact same object geometry, layout, or frame-by-frame structure as the BF16 reference.

#### Fidelity: PSNR, SSIM, and LPIPS

**Fidelity metrics** evaluate structural adherence to the BF16 baseline. They are reference-based metrics rather than no-reference realism metrics.

- **PSNR** penalizes pixel-level reconstruction error. Higher PSNR indicates smaller average deviation from the BF16 reference.
- **SSIM** evaluates structural similarity. Higher SSIM indicates better preservation of scene layout, edges, and local structure relative to the BF16 reference.
- **LPIPS** is a perceptual distance metric computed relative to the BF16 reference. Lower LPIPS indicates that the compressed output remains perceptually closer to the baseline.

In practical language, these metrics answer the question: **"Did the model preserve the same object, layout, and structure as the BF16 baseline, or did it hallucinate a different but still plausible video?"**

This distinction is the foundation of the report's main analytical claim. A method can remain strong on VBench while collapsing on fidelity if it produces **visually pleasing but structurally divergent** samples.

#### Temporal drift

The temporal metric used in the report is **terminal drift**, implemented as the **last available imaging-quality point** in the drift trajectory. It summarizes whether visual quality remains stable at the end of rollout rather than only at the beginning. High drift-last values indicate that the method preserves its own visual quality over time.

### 2.4 Method families and compression mechanisms

The methods evaluated in the broader study belong to several compression families.

- **RTN (Round-to-Nearest)** applies uniform low-bit rounding to keys and values after scale estimation. It is the simplest numerical compression baseline: reduce precision, store fewer bits, and keep the overall cache policy unchanged.
- **KIVI** uses **asymmetric key/value quantization**, exploiting the fact that keys and values have different statistics. Keys are quantized differently from values rather than forcing a single symmetric scheme onto both tensors.
- **QuaRot** uses an **orthogonal rotation** before quantization so that outliers are redistributed across dimensions. The goal is to make low-bit rounding less destructive by smoothing the quantization landscape.
- **PRQ** uses **progressive residual quantization**. It first encodes a coarse low-bit approximation, reconstructs it, and then quantizes the residual error in a second stage. This is a fidelity-oriented design because it explicitly spends extra representation budget on the remaining reconstruction error.
- **Spatial mixed methods** partition the representation into **foreground and background regions** and compress those regions differently. The underlying assumption is that background regions can tolerate more aggressive compression than visually salient foreground content.

These families do not simply differ in bit width. They differ in *what* they preserve: raw numeric precision, tensor asymmetry, rotated basis structure, residual information, or spatial salience.

### 2.5 Our custom FlowCache-inspired adaptation

The official FlowCache framework does not directly support the self-forcing **Wan2.1** architecture used in this study. For that reason, the FlowCache-labeled results in the empirical dataset are not presented here as upstream framework results. Instead, they are presented as an **in-house, custom FlowCache-inspired adaptation** engineered specifically for this unsupported architecture.

That adaptation combines two ideas.

First, it introduces **chunkwise residual reuse**. A lightweight feature representation is computed for the current causal block, compared against the previous denoising step, and reused when the relative feature drift remains sufficiently small. This component targets redundant computation.

Second, it introduces **FlowCache-adapted KV retention policies**, including **soft-pruning**. Rather than treating every historical KV chunk as equally valuable, the method retains recent and important chunks more faithfully while aggressively compressing or replacing older chunks. In the **soft-prune** variant, old chunks are not always kept in full precision and are not always deleted outright; instead, they can be replaced with pooled or summarized representations that preserve coarse context while discarding fine-grained spatial detail.

This design is precisely why the method is strong on memory relief. It also explains why the method can fail on fidelity. If fine-grained historical structure is replaced by a summary rather than preserved exactly, the model may still generate a plausible continuation while losing the exact object placement or scene geometry present in the BF16 reference.

### 2.6 Empirical protocol

The empirical dataset contains **610 prompt-level rows**, comprising **310 MovieGen rows** and **300 StoryEval rows**. These roll up to **33 benchmark-level MovieGen method summaries** and **30 benchmark-level StoryEval method summaries**. The analysis below focuses on five representative boundary points chosen because together they reveal the main systems-quality trade-offs in the full study:

1. **BF16** as the uncompressed reference.
2. **`RTN_INT4`** as the naive low-bit baseline.
3. **`PRQ_INT4`** as the fidelity-preserving but systems-expensive method.
4. **`FLOWCACHE_SOFT_PRUNE_INT4`** as the custom memory-relief method.
5. **Spatial Mixed** as the definitive failure case.

All numeric values reported in the tables below are extracted from the benchmark-level empirical summaries and rounded to three decimals for readability.

## 3. Results & Empirical Analysis

### 3.1 Systems comparison: memory relief is not guaranteed by nominal compression

Table 1 compares the systems footprint of the five representative methods across both benchmarks.

**Table 1. Systems comparison across MovieGen and StoryEval.**

| Method | MovieGen Peak VRAM (GB) | MovieGen Compression (x) | MovieGen Runtime (s) | StoryEval Peak VRAM (GB) | StoryEval Compression (x) | StoryEval Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 19.280 | 1.000 | 58.573 | 19.280 | 1.000 | 56.811 |
| RTN_INT4 | 19.983 | 3.200 | 86.264 | 19.983 | 3.200 | 88.789 |
| PRQ_INT4 | 20.686 | 1.600 | 159.971 | 20.686 | 1.600 | 157.960 |
| FLOWCACHE_SOFT_PRUNE_INT4 | 11.711 | 5.490 | 74.995 | 11.756 | 5.424 | 75.151 |
| Spatial Mixed | 14.376 | 3.461 | 224.823 | 14.376 | 3.461 | 224.055 |

Three findings are immediate.

First, **`RTN_INT4` does not solve the capacity bottleneck**. It reduces the nominal KV representation to **3.200x** compression, but peak VRAM *increases* by roughly **0.703 GB** relative to BF16 on both benchmarks. This is the clearest demonstration that nominal compression and realized peak memory are not interchangeable. The cache may be smaller in representation terms while the full pipeline still incurs dequantization, staging, or management costs that erase the expected VRAM gain.

Second, **`PRQ_INT4` solves fidelity better than systems**. Among the representative compressed methods, it is the most faithful to BF16, but it is also slower than all other boundary points except the catastrophic spatial mixed method. Relative to BF16, MovieGen runtime rises from **58.573 s** to **159.971 s**, and StoryEval runtime rises from **56.811 s** to **157.960 s**. Peak VRAM is also worse than BF16, reaching **20.686 GB** on both benchmarks.

Third, **`FLOWCACHE_SOFT_PRUNE_INT4` is the only representative compressed method that converts compression into large, realized VRAM relief**. On MovieGen it reduces peak VRAM from **19.280 GB** to **11.711 GB**, a **39.3%** reduction, while delivering **5.490x** compression. On StoryEval it reduces peak VRAM from **19.280 GB** to **11.756 GB**, a **39.0%** reduction, with **5.424x** compression. Its runtime penalty is real but moderate compared with PRQ and Spatial Mixed: **74.995 s** on MovieGen and **75.151 s** on StoryEval.

The spatial mixed method demonstrates why aggressive specialization alone is not enough. Although it reaches **14.376 GB** peak VRAM, it also requires approximately **224 s** per prompt on both benchmarks, making it too slow to justify its poor quality profile.

### 3.2 Quality comparison: realism and fidelity are different objectives

Table 2 reports the benchmark-level quality summaries. Here the distinction between **VBench realism** and **BF16 fidelity** becomes decisive.

**Table 2. Quality comparison across MovieGen and StoryEval.**

| Method | MovieGen VBench Imaging | MovieGen SSIM | MovieGen PSNR | MovieGen LPIPS | MovieGen Drift | StoryEval VBench Imaging | StoryEval SSIM | StoryEval PSNR | StoryEval LPIPS | StoryEval Drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.739 | 1.000 | inf | 0.000 | 0.739 | 0.693 | 1.000 | inf | 0.000 | 0.695 |
| RTN_INT4 | 0.735 | 0.688 | 21.320 | 0.180 | 0.734 | 0.674 | 0.661 | 18.662 | 0.245 | 0.675 |
| PRQ_INT4 | 0.739 | 0.824 | 26.545 | 0.082 | 0.739 | 0.699 | 0.724 | inf | 0.188 | 0.699 |
| FLOWCACHE_SOFT_PRUNE_INT4 | 0.739 | 0.544 | 17.673 | 0.297 | 0.738 | 0.680 | 0.518 | inf | 0.416 | 0.679 |
| Spatial Mixed | 0.399 | 0.433 | 14.060 | 0.570 | 0.394 | 0.400 | 0.444 | 12.969 | 0.599 | 0.398 |

Two patterns matter most.

First, **`PRQ_INT4` is the fidelity-preserving reference among the compressed methods**. On MovieGen it reaches **SSIM = 0.824**, **PSNR = 26.545**, and **LPIPS = 0.082**, while retaining **VBench imaging quality = 0.739**, effectively matching BF16 perceptual quality. On StoryEval it again remains the closest compressed method to the BF16 reference, with **SSIM = 0.724**, **LPIPS = 0.188**, **VBench imaging quality = 0.699**, and **terminal drift = 0.699**. If the scientific objective is to replicate the BF16 baseline as closely as possible, `PRQ_INT4` is the only representative compressed method that does so consistently.

Second, **the spatial mixed method fails on every quality axis that matters**. MovieGen imaging quality falls to **0.399**, MovieGen drift to **0.394**, StoryEval imaging quality to **0.400**, and StoryEval drift to **0.398**. Its fidelity metrics are likewise poor. This is not a nuanced trade-off; it is a dominated negative result.

The remaining representative method, `RTN_INT4`, sits between these extremes. It preserves a moderate fraction of BF16 fidelity and perceptual quality, but because it fails to reduce peak VRAM, it does not justify itself as a practical memory solution.

### 3.3 The FlowCache paradox: preserved VBench, collapsed fidelity

The most important analytical result in the study is the discrepancy between the two quality axes for **`FLOWCACHE_SOFT_PRUNE_INT4`**.

On the surface, the method looks excellent. On MovieGen, its **VBench imaging quality is 0.739**, which is effectively identical to BF16's **0.739**. Expressed on a percentage-like 0-100 scale, both values correspond to roughly **73.9**, which is why the method appears visually competitive in dashboards and summary plots. Even its terminal MovieGen drift remains high at **0.738**, only marginally below BF16's **0.739**. On StoryEval, the method remains respectable by the same perceptual criterion, reaching **imaging quality = 0.680** and **terminal drift = 0.679**.

However, the fidelity metrics tell a completely different story. On MovieGen, **SSIM collapses from 1.000 for BF16 to 0.544**, **PSNR falls to 17.673 dB**, and **LPIPS rises to 0.297**. Those are not small degradations. They indicate a large structural departure from the BF16 reference. The same pattern persists on StoryEval, where **SSIM = 0.518** and **LPIPS = 0.416**, far weaker than `PRQ_INT4`.

This is not a contradiction once the metrics are interpreted correctly. VBench imaging quality asks whether the output looks like a plausible video; fidelity asks whether the output is still *the same* video, in a structural sense, as the BF16 reference. A soft-pruned cache can preserve enough high-level context for the model to generate aesthetically coherent motion and broadly plausible scenes, yet still discard the fine-grained historical information needed to maintain exact object placement, geometry, and layout.

The scientific conclusion is therefore clear: **soft-pruning older KV-cache chunks causes the model to lose track of exact spatial constraints, leading it to hallucinate a visually pleasing but structurally divergent video**. The method is strong at preserving realism; it is weak at preserving strict baseline faithfulness.

This is the correct interpretation of the paradox. `FLOWCACHE_SOFT_PRUNE_INT4` should be called **Pareto-optimal only under a realism-centered objective**, where strong VRAM relief and preserved perceptual quality matter more than exact reproduction of the BF16 baseline. If structural hallucination is unacceptable—for example, when exact object continuity or layout preservation is required—then the soft-prune method is not the correct recommendation.

### 3.4 Deployment interpretation: which method is actually preferable?

The empirical ranking depends on what is being optimized.

If the objective is **maximum VRAM relief with acceptable perceptual realism**, the custom FlowCache-inspired soft-prune method is the best boundary point in the representative set. No other representative compressed method combines **approximately 39% lower peak VRAM**, **over 5.4x KV compression**, and near-baseline VBench realism.

If the objective is **strict baseline replication**, the answer changes immediately. In that setting, **`PRQ_INT4` is the only viable compressed method among the representative boundary points**, because it preserves the BF16 reference far more faithfully than the alternatives. The cost is severe: runtime inflates to roughly **160 s** per prompt and peak VRAM rises above BF16 rather than below it. Nevertheless, the method remains the correct choice whenever structural hallucination is not acceptable.

`RTN_INT4` is a useful methodological baseline but not a deployment recommendation. It demonstrates that naive low-bit rounding can preserve moderate quality while still failing to relieve peak VRAM. The spatial mixed method is worse: it is neither fast, nor faithful, nor perceptually strong.

The resulting guidance is therefore conditional rather than absolute.

- **Choose BF16** when memory is not the limiting factor and exact reference behavior is required.
- **Choose `PRQ_INT4`** when compressed operation is necessary but fidelity to the BF16 baseline remains the dominant scientific requirement.
- **Choose `FLOWCACHE_SOFT_PRUNE_INT4`** when VRAM relief is the decisive systems constraint and some structural hallucination is acceptable in exchange for visually plausible output.
- **Do not choose Spatial Mixed** as configured here; it is an empirical failure case.

### 3.5 Negative results and what they teach us

A useful report should not treat negative results as peripheral. In this study, the negative results sharpen the scientific message.

The naive result is that **uniform low-bit rounding is insufficient**. `RTN_INT4` compresses the cache numerically but does not materially improve end-to-end memory. This indicates that capacity relief in self-forcing video generation requires more than just shrinking the stored representation. The cache policy itself matters.

The quality-preserving failure is that **residual coding can protect fidelity without solving systems cost**. `PRQ_INT4` is scientifically valuable because it shows that low-bit compression does not inherently require visual collapse. Yet it is also a warning that fidelity preservation can be purchased at a runtime cost so high that the method becomes impractical.

The definitive failure is the **spatial mixed configuration**. It illustrates that seemingly intuitive importance partitioning can still break both realism and fidelity when the foreground/background decomposition and bit-allocation policy do not align with the actual information needs of long-horizon self-forcing generation.

Together, these failures explain why the custom FlowCache-inspired family emerged as the central engineering direction in the study. It is the only family among the representative methods that directly attacks the true bottleneck: retaining enough high-level temporal context to keep the generator visually coherent while aggressively reducing historical cache burden.

## 4. Conclusion

This study shows that KV-cache quantization for self-forcing long video generation cannot be judged by compression ratio alone. The correct evaluation is inherently multi-objective: capacity, bandwidth, runtime, perceptual realism, reference fidelity, and temporal stability must all be considered together.

The empirical results identify three distinct regimes. `RTN_INT4` is the naive baseline that compresses nominally but does not relieve peak VRAM. `PRQ_INT4` is the fidelity-preserving method that remains scientifically credible when exact BF16 replication matters, but its runtime and memory costs are severe. `FLOWCACHE_SOFT_PRUNE_INT4` is the strongest memory-relief method, achieving over **5.4x** compression and approximately **39%** lower peak VRAM while preserving near-baseline VBench realism.

The central caveat is also the central scientific contribution: the custom FlowCache-inspired soft-prune method preserves *visual plausibility* far better than it preserves *structural fidelity*. It is therefore the right recommendation only when structurally divergent yet visually plausible outputs are acceptable. When exact adherence to the BF16 baseline is required, `PRQ_INT4` remains the correct compressed method despite its systems cost.

In short, the study does not identify a universal best method. It identifies a boundary between two deployment philosophies: **memory relief with controlled hallucination**, versus **faithful replication with heavy runtime overhead**. That trade-off is the real result.
