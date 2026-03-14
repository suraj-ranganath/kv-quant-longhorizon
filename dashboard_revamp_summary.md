# Dashboard Revamp Summary

## What changed

- Rebuilt dataset mode around decision-making instead of static metric display.
- Added modular support code in `dashboard/data_sources.py`, `dashboard/decision_analysis.py`, and `dashboard/decision_plots.py` for source discovery, derived metrics, Pareto analysis, scoring, recommendations, and presentation-ready plots.
- Added top-level guidance panels, rule-based recommendation cards, Pareto frontier views, constraint-based ranking tables, a method explorer, systems/quality sections, and a derived raw method table while preserving video exploration and prompt analytics.
- Recommendation cards now explicitly consider active peak-VRAM caps and use lower `peak_vram_gb` as an explicit tie-breaker when ranking candidates.

## How recommendations are computed

- The dashboard auto-discovers CSV sources in the repo, scores them for completeness, and uses the strongest merged comparison table as the primary source while retaining supporting exports and registries for provenance.
- Benchmark-level method summaries are rebuilt from prompt-level rows, with benchmark-appropriate fidelity, VBench, runtime, VRAM, KV, and drift fields.
- Every method receives BF16-relative deltas, Pareto flags for four frontiers, and a configurable composite helper score. The score is only a helper: the recommendation logic still uses explicit quality, drift, runtime, fidelity, and now VRAM-cap filters.
- Default practical recommendations require near-BF16 imaging and drift, a structural-fidelity guard, and a preference for methods that fit inside the active runtime/VRAM budget. Aggressive-compression recommendations now prefer near-max compression methods that also minimize peak VRAM within the active caps. Fastest and quality-first cards also use peak VRAM as an explicit secondary factor.

## Main takeaways from current data

### Moviegen

- **Default practical recommendation**: `RTN_INT4_RECENT2` — compression `2.43x`, runtime `68.9s`, peak VRAM `21.37 GB`, imaging delta `-0.003`, drift delta `-0.004`.
- **Best aggressive-compression option**: `FLOWCACHE_PRUNE_INT4` — compression `5.50x`, runtime `72.2s`, peak VRAM `11.71 GB`, imaging delta `-0.012`, drift delta `-0.013`.
- **Fastest option**: `FLOWCACHE_NATIVE_SOFT_PRUNE_INT4` — compression `5.49x`, runtime `63.6s`, peak VRAM `11.74 GB`, imaging delta `-0.013`, drift delta `-0.015`.
- **Quality-first option**: `FLOWCACHE_SOFT_PRUNE_INT4` — compression `5.49x`, runtime `75.0s`, peak VRAM `11.71 GB`, imaging delta `-0.000`, drift delta `-0.001`.
- **BF16 reference baseline**: `BF16` — compression `1.00x`, runtime `58.6s`, peak VRAM `19.28 GB`, imaging delta `+0.000`, drift delta `+0.000`.

### Storyeval

- **Default practical recommendation**: `RTN_INT4_RECENT2` — compression `2.43x`, runtime `68.6s`, peak VRAM `21.37 GB`, imaging delta `-0.013`, drift delta `-0.011`.
- **Best aggressive-compression option**: `FLOWCACHE_PRUNE_INT4` — compression `5.43x`, runtime `72.4s`, peak VRAM `11.75 GB`, imaging delta `-0.012`, drift delta `-0.015`.
- **Fastest option**: `FLOWCACHE_NATIVE_SOFT_PRUNE_INT4` — compression `5.42x`, runtime `64.2s`, peak VRAM `11.78 GB`, imaging delta `-0.036`, drift delta `-0.038`.
- **Quality-first option**: `FLOWCACHE_SOFT_PRUNE_INT4` — compression `5.42x`, runtime `75.2s`, peak VRAM `11.76 GB`, imaging delta `-0.013`, drift delta `-0.016`.
- **BF16 reference baseline**: `BF16` — compression `1.00x`, runtime `56.8s`, peak VRAM `19.28 GB`, imaging delta `+0.000`, drift delta `+0.000`.

## Pareto-optimal methods for the main frontiers

### Moviegen

- **Balanced practical frontier**: `BF16`, `FLOWCACHE_NATIVE`, `FLOWCACHE_NATIVE_SOFT_PRUNE_INT4`, `FLOWCACHE_PRUNE_INT2`, `FLOWCACHE_PRUNE_INT4`, `FLOWCACHE_SOFT_PRUNE_INT2`, `FLOWCACHE_SOFT_PRUNE_INT4`, `PRQ_INT2`, `RTN_INT4_RECENT2`, `RTN_INT4_REFRESH`
- **Quality-preserving compression frontier**: `FLOWCACHE_PRUNE_INT2`, `FLOWCACHE_PRUNE_INT4`, `FLOWCACHE_SOFT_PRUNE_INT2`, `FLOWCACHE_SOFT_PRUNE_INT4`, `PRQ_INT2`
- **Systems efficiency frontier**: `BF16`, `FLOWCACHE_NATIVE`, `FLOWCACHE_NATIVE_SOFT_PRUNE_INT4`, `FLOWCACHE_PRUNE_INT2`
- **Quality-first frontier**: `BF16`, `FLOWCACHE_NATIVE`, `PRQ_INT2`

### Storyeval

- **Balanced practical frontier**: `BF16`, `FLOWCACHE_NATIVE`, `FLOWCACHE_NATIVE_SOFT_PRUNE_INT4`, `FLOWCACHE_PRUNE_INT2`, `FLOWCACHE_PRUNE_INT4`, `FLOWCACHE_SOFT_PRUNE_INT2`, `PRQ_INT2`, `PRQ_INT4`, `QUAROT_KV_INT4`, `RTN_INT4_RECENT2`, `RTN_INT4_REFRESH`
- **Quality-preserving compression frontier**: `FLOWCACHE_PRUNE_INT2`, `FLOWCACHE_PRUNE_INT4`, `FLOWCACHE_SOFT_PRUNE_INT2`, `PRQ_INT2`, `PRQ_INT4`, `QUAROT_KV_INT4`
- **Systems efficiency frontier**: `BF16`, `FLOWCACHE_NATIVE`, `FLOWCACHE_NATIVE_SOFT_PRUNE_INT4`, `FLOWCACHE_PRUNE_INT2`
- **Quality-first frontier**: `BF16`, `FLOWCACHE_NATIVE`, `PRQ_INT2`, `PRQ_INT4`

## Assumptions and caveats

- Current runs are short-horizon proxies rather than definitive long-horizon validation; drift is used as the nearest available stability signal.
- Lower compressed KV bytes do not always translate into lower peak VRAM because the current stack still pays for temporary allocations and dequantization buffers.
- Some recommendation outcomes are benchmark-specific. The dashboard therefore recommends by operating regime instead of declaring a single global winner.
- Recommendations should be interpreted as the best choices under the current codepath and current runs, not as architecture-agnostic conclusions.
