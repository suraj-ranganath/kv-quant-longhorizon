# Final Presentation Content and Speaker Script

## Slide 1

- **Slide Title:** KV Cache Quantization for Self-Forcing Long Video Generation
- **Slide Content (Bullet Points):**
  - Self-forcing makes the attention cache the dominant long-horizon systems bottleneck.
  - The full written study evaluates **33 MovieGen summaries** and **30 StoryEval summaries**.
  - This 10-minute talk focuses on **five representative archetypes**; the full ablation is in the written report.
- **Visuals Required:** **Table 1** and **Table 2** from the report.
- **Speaker Notes (Script):** “Good afternoon. This presentation addresses KV-cache quantization for self-forcing long-video generation. The written report contains the full 33-method MovieGen analysis and 30-method StoryEval analysis. In this oral defense, I will not attempt to enumerate every row. Instead, I will focus on five representative archetypes that capture the decision boundary: the BF16 baseline, a naive low-bit baseline, a high-fidelity but systems-negative method, our custom FlowCache-inspired adaptation, and a definitive failure case.”
- **Dashboard Integration:** No dashboard switch on this slide.

## Slide 2

- **Slide Title:** Why Self-Forcing Turns the KV Cache into the Bottleneck
- **Slide Content (Bullet Points):**
  - Every new causal block must attend to an ever-growing retained history.
  - The bottleneck is both **capacity** and **memory bandwidth**.
  - A useful method must reduce memory without destabilizing long-horizon rollout.
- **Visuals Required:** **Figure 1** from the report.
- **Speaker Notes (Script):** “The background is straightforward but important. Transformer decoders retain keys and values for all prior tokens. In self-forcing generation, previously generated content becomes new conditioning context, so that retained history grows block by block. The consequence is a dual systems bottleneck: peak VRAM rises because more cache must be stored, and runtime rises because larger cache tensors must be read and written repeatedly. This is why the relevant question is not whether a method compresses the cache numerically, but whether it improves end-to-end generation under long-horizon rollout.”
- **Dashboard Integration:** No dashboard switch on this slide.

## Slide 3

- **Slide Title:** Academic-Honesty Note and the Five Archetypes
- **Slide Content (Bullet Points):**
  - The official FlowCache framework does **not** support Self-Forcing Wan2.1.
  - We therefore evaluate an **in-house FlowCache-inspired adaptation**, not the upstream framework.
  - The five archetypes are: `BF16`, `RTN_INT4`, `PRQ_INT4`, `FLOWCACHE_SOFT_PRUNE_INT4`, and the weakest spatially mixed variant.
- **Visuals Required:** **Figure 2** and **Table 2** from the report.
- **Speaker Notes (Script):** “For academic accuracy, I need to make one point explicit. The official FlowCache framework does not support the Self-Forcing Wan2.1 architecture used here. Accordingly, the FlowCache-labeled rows in our study are not presented as upstream FlowCache results. They are our own in-house adaptation of FlowCache principles for an unsupported architecture. Concretely, that adaptation uses relative-L1 feature drift to decide when cached residuals can be reused, and it combines that logic with prune and soft-prune KV-cache policies. With that clarification in place, the rest of the presentation focuses on the five archetypes shown here.”
- **Dashboard Integration:** No dashboard switch on this slide.

## Slide 4

- **Slide Title:** Systems Boundary Points
- **Slide Content (Bullet Points):**
  - `RTN_INT4` compresses the KV representation by **3.20x** but leaves peak VRAM near BF16.
  - `PRQ_INT4` preserves quality but remains **158–160 s** per prompt and **20.69 GB** peak VRAM.
  - Our custom FlowCache-inspired soft-prune adaptation reaches **5.42–5.49x** compression at **11.71–11.76 GB** peak VRAM.
- **Visuals Required:** **Table 3** and **Table 4** from the report.
- **Speaker Notes (Script):** “These systems results establish the core decision boundary. RTN INT4 is the naive baseline. It achieves 3.20x KV compression, but peak VRAM remains essentially unchanged at roughly twenty gigabytes. That is the empirical proof that nominal compression alone is insufficient. PRQ INT4 sits at the opposite extreme: it preserves quality very well, but it is systems-negative, taking roughly one hundred fifty-eight to one hundred sixty seconds per prompt and peaking above BF16 memory. Our custom FlowCache-inspired soft-prune adaptation is the practical compromise. It increases runtime only moderately relative to BF16, but it more than halves effective KV cost and lowers peak VRAM to roughly 11.7 gigabytes on both benchmarks.”
- **Dashboard Integration:** No dashboard switch on this slide.

## Slide 5

- **Slide Title:** Quality and Drift Boundary Points
- **Slide Content (Bullet Points):**
  - `PRQ_INT4` is the strongest high-fidelity reference among the five archetypes.
  - FlowCache-Adapted Soft-Prune INT4 stays near BF16 on imaging quality and terminal drift.
  - The spatially mixed variant is a decisive failure on both immediate quality and long-horizon drift.
- **Visuals Required:** **Table 5** and **Table 6** from the report.
- **Speaker Notes (Script):** “If one ranks the five archetypes by quality alone, PRQ INT4 is strongest among the compressed methods. On MovieGen it matches the BF16 imaging score, and on StoryEval it slightly exceeds the custom adaptation on imaging and drift. But that quality comes at prohibitive cost. The more important point is that our FlowCache-inspired soft-prune adaptation remains very close to BF16 quality while delivering the largest practical memory relief in the study. The spatially mixed failure case makes the contrast explicit: imaging collapses to approximately 0.4, and drift collapses with it. This is not a marginally weaker point. It is a true failure regime.”
- **Dashboard Integration:** No dashboard switch on this slide.

## Slide 6

- **Slide Title:** Recommendation from the Full Ablation Study
- **Slide Content (Bullet Points):**
  - The written report contains the full 33-method and 30-method ablation matrices.
  - The practical recommendation is **FlowCache-Adapted Soft-Prune INT4**.
  - `QUAROT_KV_INT4` and `PRQ_INT4` remain useful quality references, not default systems choices.
- **Visuals Required:** **Table 7** and **Table 8** from the report.
- **Speaker Notes (Script):** “The full written report goes beyond these five boundary points and evaluates the broader ablation study. That larger view does not overturn the recommendation. Instead, it sharpens it. The methods that lie on the compression-versus-quality frontier fall into two groups: practical operating points and academic reference points. FlowCache-Adapted Soft-Prune INT4 remains the strongest practical operating point. QuaRot KV INT4 and PRQ INT4 remain useful as high-fidelity references, but they fail the systems test. The cross-benchmark correlations then show that this conclusion is stable rather than benchmark-specific.”
- **Dashboard Integration:** No dashboard switch on this slide.

## Slide 7

- **Slide Title:** Live Dashboard Defense
- **Slide Content (Bullet Points):**
  - Restrict the dashboard to the same five archetypes used in the talk.
  - Show MovieGen first for single-shot behavior, then StoryEval for narrative stability.
  - Verify that the same ranking holds interactively.
- **Visuals Required:** Live dashboard, with **Table 3**, **Table 4**, **Table 5**, and **Table 6** from the report available as backup.
- **Speaker Notes (Script):** “I will now verify the same argument directly in the dashboard. I will not browse the full ablation interactively, because that would dilute the presentation. Instead, I restrict the interface to the same five archetypes. First I show MovieGen, where the systems-quality structure is especially clean. Then I switch to StoryEval to demonstrate that the same ranking survives in the longer-horizon narrative setting. The key points to watch are RTN INT4 failing to reduce peak VRAM, PRQ INT4 preserving quality but remaining slow and memory-heavy, the custom FlowCache-inspired adaptation occupying the strongest practical trade-off region, and the spatially mixed variant remaining a clear failure.”
- **Dashboard Integration:** **[Minute 6:20]** Switch from the slide deck to the live dashboard. In the sidebar set `Benchmark = moviegen`. Keep all available source groups and run groups selected. In `Methods`, select exactly `BF16`, `RTN_INT4`, `PRQ_INT4`, `FLOWCACHE_SOFT_PRUNE_INT4`, and `SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2`. Stay on the `Overview` tab and highlight the `Unified method table`, `Quality-efficiency tradeoff`, and `Drift summary` panels. State explicitly that `RTN_INT4` reports **3.20x** compression but still sits near **19.98 GB** peak VRAM, that `PRQ_INT4` preserves quality while taking **160.0 s** per prompt, and that `FLOWCACHE_SOFT_PRUNE_INT4` drops to **11.71 GB** while retaining **0.738** MovieGen drift. **[Minute 7:20]** change `Benchmark = storyeval`, keep the same method subset where available, remain on `Overview`, and highlight the `Unified method table` and `Long-Horizon Drift (Imaging Quality)` plot. State explicitly that the same ordering survives: `PRQ_INT4` retains the strongest quality but remains at **20.69 GB** and **158.0 s**, `FLOWCACHE_SOFT_PRUNE_INT4` remains the best practical trade-off at **11.76 GB** and **0.679** drift, and the spatially mixed variant still collapses near **0.398** drift. **[Minute 8:20]** switch back to the slide deck.

## Slide 8

- **Slide Title:** Final Claim
- **Slide Content (Bullet Points):**
  - `BF16` is the quality ceiling but not the memory-efficient operating point.
  - `RTN_INT4` is the naive baseline that fails to lower peak VRAM.
  - `PRQ_INT4` is the high-fidelity method that fails the systems test.
  - FlowCache-Adapted Soft-Prune INT4 is the recommended solution.
  - The weakest spatially mixed method is a definitive failure case.
- **Visuals Required:** **Table 9** from the report.
- **Speaker Notes (Script):** “I will conclude with a precise and defensible claim. BF16 remains the quality ceiling. RTN INT4 shows that naive low-bit quantization does not solve the systems problem. PRQ INT4 shows that high fidelity alone is not a sufficient criterion, because a method can preserve quality and still fail on runtime and memory. Our custom FlowCache-inspired soft-prune adaptation is therefore the recommended operating point for this unsupported self-forcing architecture. And the spatially mixed failure case shows that not every plausible compression strategy survives long-horizon rollout. The full 33-method ablation remains available in the written report, but these five archetypes are sufficient to defend the central result.”
- **Dashboard Integration:** No dashboard switch on this slide.
