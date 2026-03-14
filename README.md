# KV Cache Quantization for Self-Forcing Video Generation

This repository is the research artifact for our empirical study of KV-cache quantization in self-forcing video generation. The core question is simple: as self-forcing pushes a short-horizon model to longer rollouts, which KV-cache compression methods actually help in the full system, and which ones only look promising if you ignore runtime, reconstruction overhead, or temporal drift?

We evaluate 33 quantization and cache-policy variants on MovieGen and StoryEval, measure systems behavior and output quality jointly, and package the results into a reproducible benchmark harness plus a presentation-oriented Streamlit dashboard.

## Why This Repo Exists

Self-forcing extends a short-horizon video model by repeatedly feeding generated output back in as future context. That makes long rollout possible, but it also causes the KV cache to grow with time. The result is the central tension of this project:

- We need enough compression to make longer rollouts feasible on finite hardware.
- We need enough fidelity to avoid drift, structural collapse, or hallucinated scene changes.
- We cannot judge a method from one metric alone.

That is why this repo is organized around a multi-axis empirical study rather than a single benchmark score.

## At A Glance

- `33` method variants evaluated
- `2` benchmarks: `MovieGen` and `StoryEval`
- `5+` quality/system axes tracked jointly: peak VRAM, runtime, compression ratio, perceptual realism, structural fidelity, and drift
- Streamlit dashboard with presentation mode, synchronized videos, Pareto plots, constraint rankings, traces, and prompt-level drilldowns
- Full benchmark harness for generation, evaluation, summarization, backfills, combined dataset construction, and dashboard presentation

## Curated Demo Gallery

The posters below link to short six-method comparison videos for the prompts we used most in presentation:

### MovieGen: candle / flame

[![MovieGen flame comparison](docs/assets/media/moviegen_flame_selected_methods.png)](docs/assets/media/moviegen_flame_selected_methods.mp4)

### MovieGen: coral reef / fish

[![MovieGen fish comparison](docs/assets/media/moviegen_fish_selected_methods.png)](docs/assets/media/moviegen_fish_selected_methods.mp4)

### StoryEval: bear in water

[![StoryEval bear comparison](docs/assets/media/storyeval_bear_selected_methods.png)](docs/assets/media/storyeval_bear_selected_methods.mp4)

Each comparison uses the same six presentation methods:

- `BF16`
- `FLOWCACHE_SOFT_PRUNE_INT4`
- `FLOWCACHE_PRUNE_INT4`
- `RTN_INT4_RECENT2`
- `RTN_INT4_REFRESH`
- `QUAROT_KV_INT4`

The full curated media notes, prompt texts, and dashboard walkthrough live in [docs/results_gallery.md](docs/results_gallery.md).

## Headline Findings

### 1. The problem is multi-objective, not one-dimensional.

A method can compress the KV cache strongly and still fail as a practical systems method if temporary BF16 reconstruction, scratch buffers, or refresh policies erase the memory savings at peak. That happened repeatedly in this study.

### 2. FlowCache-style pruning produced the strongest realized memory wins.

The clearest practical operating region was the FlowCache branch, especially `FLOWCACHE_SOFT_PRUNE_INT4` and `FLOWCACHE_PRUNE_INT4`.

- On MovieGen, `FLOWCACHE_SOFT_PRUNE_INT4` reaches about `5.49x` KV compression with about `11.23 GB` peak VRAM and `0.739` imaging quality.
- `FLOWCACHE_PRUNE_INT4` lands in a very similar systems region, but trades more structural fidelity for slightly simpler behavior.

### 3. Quality-preserving quantization ideas were still valuable even when peak VRAM did not improve.

`QUAROT_KV_INT4`, `RTN_INT4_RECENT2`, and `RTN_INT4_REFRESH` matter because they isolate useful research directions:

- outlier handling and rotation can preserve fidelity better
- recency-aware protection helps more than naive uniform quantization
- cadence and refresh policy matter for quality, even if the current memory integration is imperfect

These are important research outcomes even when the current implementation does not convert them into lower peak VRAM.

### 4. Perceptual realism and structural fidelity can diverge sharply.

One of the central lessons of the repo is the split between:

- perceptual realism: does the output still look plausible?
- structural fidelity: does it still stay close to the BF16 reference video?

The FlowCache-style soft-prune branch is the clearest example of this tension: visually strong outputs can still diverge substantially from the BF16 baseline under SSIM / LPIPS / PSNR.

## Benchmark Design

### MovieGen

MovieGen is our single-shot setting. It is the cleanest place to compare per-prompt fidelity, realism, compression ratio, runtime, and peak VRAM under a shared prompt suite.

### StoryEval

StoryEval is our narrative / rollout stability setting. It is where drift and temporal degradation become easier to see, especially through the drift-last imaging-quality signal and prompt-level qualitative playback.

## Quality Is Measured On Two Axes

### Perceptual realism

Measured primarily with VBench-derived signals:

- `background_consistency`
- `imaging_quality`
- `subject_consistency`
- `aesthetic_quality`

### Structural fidelity

Measured relative to the BF16 baseline:

- `SSIM`
- `LPIPS`
- `PSNR`

We keep these separate deliberately. A method can still make a pleasing video while drifting structurally away from BF16.

## Method Coverage

We evaluate 33 method variants across several design families:

- `BF16`: uncompressed reference
- `RTN`: naive low-bit round-to-nearest baselines, plus refresh/recent-context variants
- `KIVI`: asymmetric key/value quantization
- `QuaRot`: Hadamard-rotation quantization for outlier suppression
- `PRQ`, `QAQ`, `TPTQ`: custom higher-fidelity or outlier-aware quantizers
- `Age-Tier`: recency-aware temporal quantization
- `FlowCache variants`: hybrid, adaptive, prune, soft-prune, and native-style reuse ideas
- `Spatial mixed precision`: foreground/background precision partitioning

The full grouped catalog, rationale, and method-by-method description are in [docs/method_catalog.md](docs/method_catalog.md).

## Repository Highlights

### 1. Benchmark harness

The `scripts/` directory contains the full experiment flow:

- environment bootstrap
- dependency clone and patch application
- generation
- fidelity evaluation
- VBench evaluation
- drift evaluation
- summary building
- method-specific experiment launchers
- combined registry and dataset construction
- analysis figure generation
- dashboard launch

Notable entry points:

- [scripts/09_run_full_research_pipeline.sh](scripts/09_run_full_research_pipeline.sh): end-to-end research pipeline
- [scripts/13_launch_dashboard.sh](scripts/13_launch_dashboard.sh): Streamlit presentation launcher
- [scripts/30_build_combined_comparison_dataset.py](scripts/30_build_combined_comparison_dataset.py): unified comparison dataset
- [scripts/26_generate_analysis_figures.py](scripts/26_generate_analysis_figures.py): paper/deck-friendly plots
- [scripts/34_generate_static_analysis_assets.py](scripts/34_generate_static_analysis_assets.py): README-ready static benchmark plots, traces, and method tables

### 2. Combined comparison dataset

The public-facing comparison layer is built around [results/combined/combined_comparison_dataset.csv](results/combined/combined_comparison_dataset.csv), which merges prompt-level records, method summaries, evaluation outputs, and provenance across runs.

This is what powers the dashboard and most of the comparative analysis in the repo.

### 3. Presentation dashboard

The dashboard at [dashboard/app.py](dashboard/app.py) provides:

- benchmark and run selection
- method filtering across the combined dataset
- a presentation page with synchronized videos, focused metrics, highlighted plots, and a decision tree
- executive summaries and recommendation cards
- Pareto frontier analysis
- constraint-based rankings
- detailed method exploration
- systems traces and KV-footprint plots
- quality and drift analysis
- prompt-level tables
- raw method tables
- caveats and provenance views

A full tab-by-tab guide is in [docs/dashboard_guide.md](docs/dashboard_guide.md).

## Figures

These are the static figures we used repeatedly while explaining the systems/quality trade space:

### Memory vs compression

![VRAM vs compression](docs/figures/vram_compression.png)

### Temporal drift

![Temporal drift](docs/figures/temporal_drift.png)

## Static Dashboard Analysis

The dashboard already exposes a richer decision layer than the headline figures above. To make the public repo self-contained, the same benchmark-level static analysis pack is generated into `docs/analysis_assets/`.

Regeneration command:

```bash
/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python scripts/34_generate_static_analysis_assets.py
```

This pack includes:

- all four Pareto/frontier views used in the dashboard
- systems scatter plots
- full drift curves per benchmark
- representative VRAM and KV-cache traces
- benchmark-wide method tables for every method in the combined dataset

### MovieGen: Pareto and Systems Views

Balanced practical frontier:

![MovieGen balanced practical frontier](docs/analysis_assets/moviegen/balanced_practical.png)

Quality-preserving compression frontier:

![MovieGen quality-preserving compression frontier](docs/analysis_assets/moviegen/quality_preserving_compression.png)

Systems efficiency frontier:

![MovieGen systems efficiency frontier](docs/analysis_assets/moviegen/systems_efficiency.png)

Quality-first frontier:

![MovieGen quality-first frontier](docs/analysis_assets/moviegen/quality_first.png)

Peak VRAM vs quality:

![MovieGen peak VRAM vs quality](docs/analysis_assets/moviegen/peak_vram_vs_quality.png)

Peak VRAM vs runtime:

![MovieGen peak VRAM vs runtime](docs/analysis_assets/moviegen/vram_vs_runtime.png)

Compression vs peak VRAM:

![MovieGen compression vs peak VRAM](docs/analysis_assets/moviegen/compression_vs_peak_vram.png)

Drift curves across all available MovieGen methods:

![MovieGen drift curves](docs/analysis_assets/moviegen/drift_curves.png)

Representative all-method VRAM trace:

- prompt: `0`
- seed: `0`
- methods plotted: `33`

![MovieGen VRAM trace](docs/analysis_assets/moviegen/trace_vram_allocated.png)

Representative all-method KV-cache trace:

![MovieGen KV-cache trace](docs/analysis_assets/moviegen/trace_kv_compressed.png)

Method tables and trace summaries:

- [MovieGen derived method table CSV](docs/analysis_assets/moviegen/derived_method_table.csv)
- [MovieGen derived method table MD](docs/analysis_assets/moviegen/derived_method_table.md)
- [MovieGen trace peak summary CSV](docs/analysis_assets/moviegen/trace_peak_summary.csv)
- [MovieGen trace peak summary MD](docs/analysis_assets/moviegen/trace_peak_summary.md)

<details>
<summary>MovieGen method-wise table</summary>

| method | method_family | compression_ratio | peak_vram_gb | avg_runtime_s_per_prompt | imaging_quality | drift_last_imaging_quality | psnr | ssim | lpips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FLOWCACHE_PRUNE_INT2 | FLOWCACHE | 7.7774 | 11.1145 | 69.8677 | 0.6371 | 0.6334 | 15.2597 | 0.4666 | 0.4825 |
| FLOWCACHE_PRUNE_INT4 | FLOWCACHE | 5.4981 | 11.7083 | 72.2211 | 0.7269 | 0.7261 | 15.3004 | 0.4569 | 0.4119 |
| FLOWCACHE_SOFT_PRUNE_INT4 | FLOWCACHE | 5.4899 | 11.7114 | 74.9954 | 0.7390 | 0.7383 | 17.6734 | 0.5442 | 0.2975 |
| FLOWCACHE_SOFT_PRUNE_INT2 | FLOWCACHE | 6.8245 | 11.7114 | 76.1179 | 0.6623 | 0.6583 | 15.8380 | 0.4822 | 0.4398 |
| FLOWCACHE_NATIVE_SOFT_PRUNE_INT4 | FLOWCACHE | 5.4899 | 11.7387 | 63.6137 | 0.7259 | 0.7243 | 13.2571 | 0.4108 | 0.4755 |
| FLOWCACHE_ADAPTIVE_INT2 | FLOWCACHE | 4.2682 | 14.3755 | 92.6335 | 0.6155 | 0.6105 | 15.1943 | 0.4479 | 0.4645 |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | SPATIAL_MIXED | 3.4606 | 14.3760 | 224.8226 | 0.3987 | 0.3942 | 14.0597 | 0.4327 | 0.5696 |
| FLOWCACHE_HYBRID_INT2 | FLOWCACHE | 4.6066 | 14.3769 | 82.5925 | 0.6163 | 0.6116 | 15.6244 | 0.4707 | 0.4538 |
| AGE_TIER_INT4 | AGE_TIER | 3.1843 | 14.3775 | 103.8574 | 0.7351 | 0.7339 | 21.3176 | 0.6880 | 0.1804 |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4 | SPATIAL_MIXED | 3.1843 | 14.3775 | 105.4578 | 0.6934 | 0.6898 | 18.8866 | 0.5772 | 0.3099 |
| AGE_TIER_INT2 | AGE_TIER | 4.4145 | 14.3775 | 105.2508 | 0.5781 | 0.5731 | 15.1818 | 0.4566 | 0.4704 |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 | SPATIAL_MIXED | 3.6834 | 14.3775 | 106.6273 | 0.4113 | 0.4069 | 13.9269 | 0.4212 | 0.5580 |
| SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2 | SPATIAL_MIXED | 3.4528 | 14.3833 | 110.3590 | 0.5289 | 0.5210 | 13.7229 | 0.4268 | 0.6418 |
| QAQ_INT2 | QAQ | 5.1842 | 14.4214 | 109.7803 | 0.6200 | 0.6182 | 13.3364 | 0.3646 | 0.5299 |
| QAQ_INT4 | QAQ | 3.1448 | 14.4218 | 110.0065 | 0.5889 | 0.5863 | 11.9680 | 0.2617 | 0.6470 |
| BF16 | BF16 | 1.0000 | 19.2801 | 58.5726 | 0.7390 | 0.7394 | inf | 1.0000 | 0.0000 |
| FLOWCACHE_NATIVE | FLOWCACHE | 1.0000 | 19.3075 | 48.2873 | 0.7377 | 0.7373 | 13.2549 | 0.4115 | 0.4506 |
| TPTQ_INT2 | TPTQ | 2.7166 | 19.8546 | 167.2228 | 0.7237 | 0.7222 | 19.9062 | 0.6268 | 0.2397 |
| QUAROT_KV_INT4 | QUAROT | 3.2000 | 19.9831 | 236.6028 | 0.7376 | 0.7378 | 22.6420 | 0.7240 | 0.1483 |
| RTN_INT4 | RTN | 3.2000 | 19.9831 | 86.2636 | 0.7353 | 0.7341 | 21.3205 | 0.6880 | 0.1803 |
| RTN_INT2 | RTN | 5.3333 | 19.9831 | 87.1161 | 0.5668 | 0.5617 | 15.0444 | 0.4515 | 0.4750 |
| QUAROT_KV_INT2 | QUAROT | 5.3333 | 19.9831 | 242.0181 | 0.6008 | 0.5968 | 14.7310 | 0.4403 | 0.4670 |
| KIVI_INT4 | KIVI | 3.1933 | 19.9904 | 92.6862 | 0.6812 | 0.6784 | 13.0698 | 0.4048 | 0.5709 |
| KIVI_INT2 | KIVI | 5.3149 | 19.9904 | 95.4773 | 0.6211 | 0.6181 | 11.4240 | 0.2414 | 0.6714 |
| PRQ_INT4 | PRQ | 1.6000 | 20.6861 | 159.9706 | 0.7389 | 0.7393 | 26.5446 | 0.8239 | 0.0819 |
| PRQ_INT2 | PRQ | 2.0000 | 20.6861 | 156.6343 | 0.7392 | 0.7396 | 25.1333 | 0.7997 | 0.0938 |
| RTN_INT4_RECENT2 | RTN | 2.4348 | 21.3741 | 68.8637 | 0.7356 | 0.7351 | 23.6918 | 0.7320 | 0.1482 |
| QUAROT_KV_INT4_RECENT2 | QUAROT | 2.4348 | 21.6854 | 111.3048 | 0.7302 | 0.7290 | inf | 0.7058 | 0.1834 |
| KIVI_INT4_REFRESH | KIVI | 3.1933 | 22.6322 | 68.0524 | 0.7137 | 0.7116 | 13.7329 | 0.4203 | 0.5095 |
| RTN_INT4_REFRESH | RTN | 3.2000 | 22.6361 | 65.0466 | 0.7361 | 0.7352 | 21.4496 | 0.6934 | 0.1777 |
| KIVI_K2_V4 | KIVI |  | 22.6740 | 76.2940 | 0.6233 | 0.6186 | 13.0301 | 0.3742 | 0.5783 |
| RTN_K2_V4 | RTN |  | 22.6779 | 75.3167 | 0.5305 | 0.5242 | 14.7430 | 0.4340 | 0.4953 |
| QUAROT_KV_INT4_REFRESH | QUAROT |  | 22.8235 | 97.5132 | 0.7218 | 0.7192 | 19.6354 | 0.6129 | 0.2144 |

</details>

### StoryEval: Pareto and Systems Views

Balanced practical frontier:

![StoryEval balanced practical frontier](docs/analysis_assets/storyeval/balanced_practical.png)

Quality-preserving compression frontier:

![StoryEval quality-preserving compression frontier](docs/analysis_assets/storyeval/quality_preserving_compression.png)

Systems efficiency frontier:

![StoryEval systems efficiency frontier](docs/analysis_assets/storyeval/systems_efficiency.png)

Quality-first frontier:

![StoryEval quality-first frontier](docs/analysis_assets/storyeval/quality_first.png)

Peak VRAM vs quality:

![StoryEval peak VRAM vs quality](docs/analysis_assets/storyeval/peak_vram_vs_quality.png)

Peak VRAM vs runtime:

![StoryEval peak VRAM vs runtime](docs/analysis_assets/storyeval/vram_vs_runtime.png)

Compression vs peak VRAM:

![StoryEval compression vs peak VRAM](docs/analysis_assets/storyeval/compression_vs_peak_vram.png)

Drift curves across all available StoryEval methods:

![StoryEval drift curves](docs/analysis_assets/storyeval/drift_curves.png)

Representative all-method VRAM trace:

- prompt: `A_CD_is_inserted_into_a_player_and_then_spins_up`
- seed: `0`
- methods plotted: `30`

![StoryEval VRAM trace](docs/analysis_assets/storyeval/trace_vram_allocated.png)

Representative all-method KV-cache trace:

![StoryEval KV-cache trace](docs/analysis_assets/storyeval/trace_kv_compressed.png)

Method tables and trace summaries:

- [StoryEval derived method table CSV](docs/analysis_assets/storyeval/derived_method_table.csv)
- [StoryEval derived method table MD](docs/analysis_assets/storyeval/derived_method_table.md)
- [StoryEval trace peak summary CSV](docs/analysis_assets/storyeval/trace_peak_summary.csv)
- [StoryEval trace peak summary MD](docs/analysis_assets/storyeval/trace_peak_summary.md)

<details>
<summary>StoryEval method-wise table</summary>

| method | method_family | compression_ratio | peak_vram_gb | avg_runtime_s_per_prompt | imaging_quality | drift_last_imaging_quality | background_consistency | subject_consistency | aesthetic_quality |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FLOWCACHE_PRUNE_INT2 | FLOWCACHE | 7.6797 | 11.1395 | 70.1576 | 0.5161 | 0.5161 | 0.8652 | 0.7685 | 0.4552 |
| FLOWCACHE_PRUNE_INT4 | FLOWCACHE | 5.4315 | 11.7534 | 72.4063 | 0.6815 | 0.6797 | 0.8999 | 0.8725 | 0.5508 |
| FLOWCACHE_SOFT_PRUNE_INT4 | FLOWCACHE | 5.4236 | 11.7564 | 75.1512 | 0.6803 | 0.6789 | 0.9092 | 0.8998 | 0.5485 |
| FLOWCACHE_SOFT_PRUNE_INT2 | FLOWCACHE | 6.7224 | 11.7564 | 74.3772 | 0.5320 | 0.5361 | 0.8759 | 0.7935 | 0.4709 |
| FLOWCACHE_NATIVE_SOFT_PRUNE_INT4 | FLOWCACHE | 5.4236 | 11.7837 | 64.2319 | 0.6575 | 0.6572 | 0.9199 | 0.8761 | 0.5469 |
| FLOWCACHE_ADAPTIVE_INT2 | FLOWCACHE | 4.2575 | 14.3752 | 91.2923 | 0.4977 | 0.4960 | 0.8524 | 0.7361 | 0.4430 |
| SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 | SPATIAL_MIXED | 3.4606 | 14.3760 | 224.0546 | 0.3997 | 0.3979 | 0.8081 | 0.6245 | 0.3462 |
| FLOWCACHE_HYBRID_INT2 | FLOWCACHE | 4.5922 | 14.3766 | 82.1753 | 0.4921 | 0.4937 | 0.8712 | 0.7758 | 0.4567 |
| AGE_TIER_INT4 | AGE_TIER | 3.1843 | 14.3775 | 102.4500 | 0.6735 | 0.6756 | 0.9229 | 0.9118 | 0.5394 |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4 | SPATIAL_MIXED | 3.1843 | 14.3775 | 106.6432 | 0.6056 | 0.6066 | 0.8993 | 0.8558 | 0.5127 |
| AGE_TIER_INT2 | AGE_TIER | 4.4145 | 14.3775 | 101.9456 | 0.4691 | 0.4734 | 0.8618 | 0.7578 | 0.4566 |
| SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 | SPATIAL_MIXED | 3.6932 | 14.3775 | 106.5983 | 0.4214 | 0.4195 | 0.8103 | 0.6356 | 0.3524 |
| SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2 | SPATIAL_MIXED | 3.4528 | 14.3833 | 110.1912 | 0.4542 | 0.4509 | 0.8323 | 0.6644 | 0.4091 |
| QAQ_INT2 | QAQ | 5.1855 | 14.4209 | 109.8903 | 0.5790 | 0.5852 | 0.8387 | 0.7115 | 0.4614 |
| QAQ_INT4 | QAQ | 3.1458 | 14.4210 | 109.8911 | 0.5671 | 0.5706 | 0.8083 | 0.6302 | 0.4303 |
| BF16 | BF16 | 1.0000 | 19.2801 | 56.8107 | 0.6932 | 0.6951 | 0.9322 | 0.9207 | 0.5559 |
| FLOWCACHE_NATIVE | FLOWCACHE | 1.0000 | 19.3075 | 49.0442 | 0.6815 | 0.6821 | 0.9260 | 0.8886 | 0.5497 |
| TPTQ_INT2 | TPTQ | 2.7166 | 19.7662 | 166.5541 | 0.6537 | 0.6580 | 0.9205 | 0.9058 | 0.5321 |
| QUAROT_KV_INT4 | QUAROT | 3.2000 | 19.9831 | 239.5797 | 0.6870 | 0.6889 | 0.9262 | 0.9203 | 0.5451 |
| RTN_INT4 | RTN | 3.2000 | 19.9831 | 88.7893 | 0.6738 | 0.6753 | 0.9235 | 0.9118 | 0.5393 |
| RTN_INT2 | RTN | 5.3333 | 19.9831 | 86.1275 | 0.4644 | 0.4709 | 0.8591 | 0.7528 | 0.4525 |
| QUAROT_KV_INT2 | QUAROT | 5.3333 | 19.9831 | 239.0473 | 0.4775 | 0.4802 | 0.8607 | 0.7535 | 0.4586 |
| KIVI_INT4 | KIVI | 3.1933 | 19.9904 | 92.9985 | 0.6348 | 0.6352 | 0.8913 | 0.8354 | 0.5121 |
| KIVI_INT2 | KIVI | 5.3149 | 19.9904 | 94.7113 | 0.5312 | 0.5271 | 0.7984 | 0.6049 | 0.3797 |
| PRQ_INT2 | PRQ | 2.0000 | 20.6861 | 155.6378 | 0.6975 | 0.6982 | 0.9334 | 0.9273 | 0.5544 |
| PRQ_INT4 | PRQ | 1.6000 | 20.6861 | 157.9600 | 0.6989 | 0.6994 | 0.9313 | 0.9209 | 0.5568 |
| RTN_INT4_RECENT2 | RTN | 2.4348 | 21.3741 | 68.6420 | 0.6803 | 0.6836 | 0.9235 | 0.9142 | 0.5452 |
| QUAROT_KV_INT4_RECENT2 | QUAROT | 2.4348 | 21.6854 | 112.9255 | 0.6665 | 0.6698 | 0.9188 | 0.9049 | 0.5383 |
| KIVI_INT4_REFRESH | KIVI | 3.1933 | 22.6322 | 66.7335 | 0.6448 | 0.6414 | 0.8808 | 0.8295 | 0.4995 |
| RTN_INT4_REFRESH | RTN | 3.2000 | 22.6361 | 64.6068 | 0.6779 | 0.6787 | 0.9235 | 0.9136 | 0.5408 |

</details>

## Public-Facing Repo Layout

```text
.
├── README.md
├── dashboard/
├── kv_quant/
├── prompts/
├── scripts/
├── docs/
│   ├── environment_setup.md
│   ├── dashboard_guide.md
│   ├── method_catalog.md
│   ├── results_gallery.md
│   ├── analysis_assets/
│   ├── figures/
│   ├── presentations/
│   ├── reports/
│   └── dashboard/
├── results/
│   ├── benchmarks/
│   ├── combined/
│   └── ...
└── scripts/
    └── generate_deck.py
```

## Quick Start

### Environment setup

Use the detailed environment notes in [docs/environment_setup.md](docs/environment_setup.md).

Minimal flow:

```bash
./scripts/10_clone_deps.sh
./scripts/11_apply_self_forcing_patch.sh
conda create -n qvg_sf_infer python=3.10 -y
conda activate qvg_sf_infer
pip install -r requirements-inference.txt
```

Optional evaluation and dashboard environments are documented in the same setup guide.

### Launch the dashboard

```bash
./scripts/13_launch_dashboard.sh
```

### Build the combined dataset and figures

```bash
python scripts/30_build_combined_comparison_dataset.py
python scripts/26_generate_analysis_figures.py
```

## Dashboard: What It Gives You

If you are visiting this repo mainly to understand the results, the dashboard is the fastest path.

Use it to:

- compare prompt-matched videos across methods
- inspect systems tradeoffs with highlighted presentation methods
- switch between MovieGen and StoryEval from the same UI
- apply recommendation presets and constraint thresholds
- see Pareto-surviving methods under different objectives
- study VRAM traces and compressed-KV traces over time
- see BF16-relative deltas for fidelity and drift
- drill down to prompt-level rows and provenance

The detailed guide is in [docs/dashboard_guide.md](docs/dashboard_guide.md).

## Results Interpretation Guide

The repo is intentionally opinionated about how to read the study:

- `BF16` is the reference, not the deployable answer
- `FLOWCACHE_SOFT_PRUNE_INT4` is the strongest practical single-GPU operating point in the current stack
- `FLOWCACHE_PRUNE_INT4` is the stronger raw compression / memory point if you accept more quality loss
- `QUAROT_KV_INT4` is the strongest quantized fidelity baseline among the selected presentation methods
- `RTN_INT4_RECENT2` is the best practical recency-aware RTN result
- `RTN_INT4_REFRESH` is the cleanest simple policy ablation for refresh cadence

## Important Public-Repo Notes

- Local checkpoint and model directories are expected to be created with the provided setup scripts rather than bundled directly.
- Some MovieGen source videos referenced in the combined dataset came from external run roots during the original study. The repo includes curated derived media assets for presentation, and the dashboard is the canonical place to browse the full prompt-level comparisons.
- The dashboard and docs are presentation-oriented, but the raw tables and scripts are preserved so others can adapt the harness later.

## Additional Reading

- [docs/dashboard_guide.md](docs/dashboard_guide.md): dashboard capabilities and analysis surfaces
- [docs/method_catalog.md](docs/method_catalog.md): grouped description of the 33 methods
- [docs/results_gallery.md](docs/results_gallery.md): curated demos used in presentation
- [docs/reports/reportv2.md](docs/reports/reportv2.md): fuller narrative write-up of the study
- [docs/reports/report.md](docs/reports/report.md): compact report-style summary
- [docs/reports/qa_defense.md](docs/reports/qa_defense.md): defense notes and anticipated questions
- [docs/presentations/presentation.md](docs/presentations/presentation.md): deck-oriented summary and talk structure
- [docs/presentations/final_presentation.pptx](docs/presentations/final_presentation.pptx): generated slide deck

## Future Work

- Reproduce newer long-video KV-cache methods such as QVG / QVG-Pro within the same harness
- Extend the study beyond 10-second settings to stronger long-horizon drift evaluation
- Test generalization beyond the current self-forcing stack
- Push into first-frame-grounded, embodied, and stronger consistency-sensitive settings
