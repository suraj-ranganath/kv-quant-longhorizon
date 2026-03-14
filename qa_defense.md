# Q&A Defense Document

## 1. Why is the KV cache the correct systems target rather than a secondary optimization detail?

The KV cache is the correct target because its cost grows with temporal context length, and self-forcing deliberately extends that context by rolling generated content forward as new conditioning information. As a result, the cache is both a capacity bottleneck and a bandwidth bottleneck. The empirical results confirm that this is not a theoretical concern: methods that reduce cache cost effectively can lower peak VRAM to roughly **11.1-11.8 GB**, whereas methods that only appear compressed on paper often remain near or above the BF16 memory footprint.

## 2. Why should the audience trust that the MovieGen conclusions carry over to StoryEval?

Because the ranking structure is strongly aligned across the two benchmark surfaces. Compression ratio has **Pearson r = 0.9999**, runtime has **r = 0.9996**, imaging quality has **r = 0.9318**, and terminal drift has **r = 0.9374**. Those correlations indicate that the main method ordering is stable rather than benchmark-specific.

## 3. Why was it academically acceptable to adapt FlowCache principles for an unsupported model, and what exactly was adapted?

It is academically acceptable because the study does **not** claim official FlowCache compatibility or upstream FlowCache results. Instead, it states explicitly that the official framework does not support Self-Forcing Wan2.1 and that the evaluated method is an **in-house FlowCache-inspired adaptation**. The adapted logic is specific and testable: it computes lightweight block features, measures **relative-L1 feature drift** across denoising steps, reuses the cached denoising residual when drift remains below threshold, and combines that reuse policy with **prune and soft-prune KV-cache policies** that retain recent chunks, preserve important old chunks, and replace evicted chunks with pooled summaries when appropriate. The academic claim is therefore narrow and honest: not “official FlowCache works on this model,” but rather “our custom adaptation of FlowCache principles works well in this unsupported setting.”

## 4. What is the strongest empirical operating point if the objective is practical memory relief with acceptable quality retention?

The strongest overall operating point is **FlowCache-Adapted Soft-Prune INT4**, with **FlowCache-Adapted Prune INT4** as a close alternative. On MovieGen, FlowCache-Adapted Soft-Prune INT4 reaches **5.49x** compression, **11.71 GB** peak VRAM, **0.739** imaging quality, and **0.738** terminal drift. On StoryEval, it reaches **5.42x** compression, **11.76 GB** peak VRAM, **0.680** imaging quality, and **0.679** terminal drift. No other method combines that level of memory relief with comparable quality preservation.

## 5. Why is `PRQ_INT4` not the final recommendation even though it achieves the strongest high-fidelity compressed result among the presentation archetypes?

Because the study is a systems study, not a quality-only ranking. `PRQ_INT4` preserves quality extremely well, but it is systems-negative. It runs at **160.0 s** on MovieGen and **158.0 s** on StoryEval, and it peaks at **20.69 / 20.69 GB**, both above BF16. Therefore `PRQ_INT4` is a high-quality reference point, not the best memory-reduction regime.

## 6. Why is `RTN_INT4` an important baseline if it is not the recommended method?

Because it exposes the core systems trap in low-bit KV studies. `RTN_INT4` reports **3.20x** KV compression on both benchmarks, yet peak VRAM remains approximately **19.98 GB**, almost unchanged from BF16. That mismatch demonstrates that stored-KV compression is not automatically equivalent to realized system relief. Without a baseline like `RTN_INT4`, one might overinterpret nominal compression ratios.

## 7. What empirical evidence shows that nominal compression ratio is insufficient as a selection criterion?

The strongest evidence is the mismatch between compression ratio and peak VRAM. `RTN_INT4` reports about **3.20x** compression, yet its peak VRAM is still effectively BF16-class. By contrast, FlowCache-Adapted Soft-Prune INT4 reaches roughly **5.4x** compression and also reduces peak VRAM by about **39%**. This demonstrates that algorithmic compression must be validated by end-to-end memory measurements rather than assumed to translate directly into system savings.

## 8. Why is temporal drift a necessary evaluation axis in self-forcing generation?

Temporal drift is necessary because self-forcing couples later generations to earlier generated content. Small local errors can therefore accumulate rather than remain isolated. The empirical results show that some methods with moderate single-shot quality deteriorate noticeably in terminal drift, while strong methods preserve both immediate quality and late-sequence stability. The weakest spatially mixed variant, for example, collapses to average imaging **0.399** and average drift **0.396**, which makes clear that the problem is temporal, not merely frame-local.

## 9. How should one interpret Pareto-optimal points that still appear weak in absolute quality?

Pareto optimality is a geometric property, not an endorsement. A method may lie on the frontier simply because it compresses extremely aggressively, even if its quality is substantially below the BF16 reference. FlowCache-Adapted Prune INT2 and FlowCache-Adapted Soft-Prune INT2 illustrate this point: they survive on the frontier because they move far on the compression axis, but their StoryEval imaging quality is only **0.516** and **0.532**, respectively. They are informative boundary points, not necessarily the preferred deployment choice.

## 10. Why are the weakest spatially mixed methods described as failures rather than merely weaker alternatives?

Because they fail simultaneously on several axes. `SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2` has the lowest average imaging and drift in the study and still runs at **224.8 s** on MovieGen. The other weak spatially mixed variants occupy the same low-quality region. These methods are not simply marginally worse; they are dominated by simpler strategies that are faster, more stable, or both.

## 11. What limitations remain, and why do they not overturn the study's central conclusion?

The main limitation is uneven coverage for three archival MovieGen variants: `KIVI_K2_V4`, `RTN_K2_V4`, and `QUAROT_KV_INT4_REFRESH` have fewer than ten videos. However, the central conclusion does not depend on those rows. The recommendation for FlowCache-Adapted Soft-Prune INT4 is driven by the fully covered methods and is reinforced independently by both benchmark surfaces. Thus the limitation narrows the interpretation of a small number of archival variants, but it does not undermine the main comparative claim.
