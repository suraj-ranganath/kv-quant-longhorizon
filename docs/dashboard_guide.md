# Dashboard Guide

This dashboard is designed as both a research analysis tool and a presentation surface. It sits on top of the combined comparison dataset and lets you move from benchmark-level summaries to prompt-level evidence without leaving the same UI.

## What The Dashboard Covers

The dashboard integrates:

- prompt-level and method-level metrics from the combined dataset
- systems measurements such as runtime, peak VRAM, compression ratio, and compressed KV size
- fidelity metrics relative to BF16: `PSNR`, `SSIM`, and `LPIPS`
- perceptual metrics from VBench-style evaluation
- drift signals from longer rollout evaluation
- run provenance and source-catalog context

The main application entry point is [dashboard/app.py](../dashboard/app.py).

## Sidebar Controls

The sidebar is the control plane for the entire dashboard.

It lets you:

- choose the benchmark (`MovieGen` or `StoryEval`)
- filter to specific runs inside the combined dataset
- filter the active method set
- choose the recommendation focus preset
- adjust threshold controls for runtime, VRAM, SSIM loss, LPIPS increase, drift loss, and minimum compression

These controls update the downstream recommendation cards, Pareto frontiers, plots, and tables.

## Presentation Page

The presentation page is the first tab and the most presentation-oriented surface in the app.

It provides:

- a curated method selector for the six presentation methods
- a prompt dropdown so you can switch to a single matched prompt across methods
- synchronized video playback controls
- method cards with systems and quality metrics beside each clip
- a focused comparison table for the pinned methods
- the core tradeoff plots with the pinned methods highlighted directly on the graphs
- prompt-level records and run provenance
- a method-selection decision tree

This is the tab we used for deck-style presentation because it combines qualitative and quantitative evidence in one place.

## Executive Summary

The executive summary compresses the current benchmark slice into recommendation-oriented summaries.

It surfaces:

- best memory-reduction candidates
- strongest quality-retention candidates
- best raw compression points
- best runtime points
- benchmark-level context for the current recommendation focus

Use this tab when you want the dashboard’s current “best read” of the method landscape under a chosen preset.

## Pareto Analysis

The Pareto tab answers: which methods survive multi-objective tradeoffs?

It includes several frontiers:

- balanced practical frontier
- quality-preserving compression frontier
- systems efficiency frontier
- quality-first frontier

This is the main place to separate methods that remain competitive under multiple objectives from methods that only look good on one isolated metric.

## Constraint Rankings

The constraint tab answers deployment-style questions.

Examples:

- best method under a peak-VRAM cap
- best compression while keeping SSIM loss bounded
- fastest method under a quality tolerance
- best method under direct PSNR, LPIPS, and drift constraints

The defaults are calibrated from the current top non-BF16 methods so the tab starts in a reasonable region instead of arbitrary thresholds.

## Detailed Method Explorer

The method explorer is the drilldown view for a single method.

It shows:

- headline metric cards
- BF16-relative deltas
- the method’s position on the main tradeoff plots
- frontier membership
- explanation text for why the method lands where it does

This is the easiest place to narrate the strengths and weaknesses of one method at a time.

## Systems Analysis

The systems tab focuses on performance and memory behavior.

It includes:

- compression ratio comparisons
- peak VRAM comparisons
- runtime comparisons
- KV-cache size over time
- VRAM traces over time
- prompt-level trace summaries when they exist

This is where you see the distinction between nominal KV compression and realized peak-memory behavior.

## Quality / Drift Analysis

The quality tab focuses on output quality and temporal stability.

It includes:

- imaging quality
- drift-last imaging quality
- `PSNR`, `SSIM`, and `LPIPS`
- BF16-relative deltas
- StoryEval drift curves when available

This tab is intentionally separate from systems analysis because the project’s main argument is that quality and systems behavior must be evaluated jointly but read distinctly.

## Video Explorer

The video explorer gives a prompt-matched playback view outside the presentation page.

Use it to:

- compare a larger method set than the pinned presentation methods
- inspect prompt-level metrics beside the videos
- browse results more freely than the presentation tab allows

## Prompt Analytics

The prompt analytics tab preserves prompt-level evidence.

It provides:

- per-prompt rows
- prompt-level fidelity metrics
- prompt-level perceptual metrics
- run-level metadata tied to each video record

This is useful when a method’s benchmark average hides important prompt-specific failures or wins.

## Raw Method Table

The raw method table is the closest thing to the full benchmark matrix inside the app.

It includes:

- all method-level metrics used in the current dashboard view
- frontier flags
- recommendation annotations
- BF16-relative deltas
- systems metrics
- quality metrics

If you want the dashboard as a data browser rather than as a guided narrative, this is the main table to use.

## Notes / Caveats

The notes tab captures provenance and incomplete-data context.

It includes:

- source catalog information
- discovered CSV sources
- gap summaries
- caveats on what is or is not present in the combined dataset

This matters because the study merges results from multiple runs, backfills, and benchmark passes.

## Analysis Philosophy

The dashboard is opinionated in a few important ways:

- `BF16` is the reference baseline, not the practical answer
- `PSNR`, `SSIM`, and `LPIPS` are treated as direct metrics rather than hidden behind an invented composite
- perceptual realism and structural fidelity are shown separately
- peak VRAM is not treated as interchangeable with compressed KV size
- recommendation presets are explicit so the user can see what objective ordering is driving the ranking

## Typical Demo Flow

If you are presenting from the dashboard, the cleanest sequence is:

1. Start on the `Presentation Page`
2. Show the synced videos for a matched prompt
3. Use the focused table for exact numbers
4. Move to the highlighted tradeoff plots
5. Open the decision tree
6. Use `Systems Analysis` or `Quality / Drift Analysis` only when you need deeper evidence

## Related Files

- [dashboard/app.py](../dashboard/app.py)
- [dashboard/decision_analysis.py](../dashboard/decision_analysis.py)
- [dashboard/decision_plots.py](../dashboard/decision_plots.py)
- [results/combined/combined_comparison_dataset.csv](../results/combined/combined_comparison_dataset.csv)
