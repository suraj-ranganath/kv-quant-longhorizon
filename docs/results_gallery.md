# Results Gallery

This document collects the qualitative demos we used most often while presenting the study. The goal is to make the repo feel like a research artifact rather than only a code dump.

## Presentation Method Set

These are the six methods we pinned most often in the dashboard and presentation deck:

- `BF16`
- `FLOWCACHE_SOFT_PRUNE_INT4`
- `FLOWCACHE_PRUNE_INT4`
- `RTN_INT4_RECENT2`
- `RTN_INT4_REFRESH`
- `QUAROT_KV_INT4`

Why this set:

- `BF16` is the reference
- `FLOWCACHE_SOFT_PRUNE_INT4` is the strongest practical deployment candidate
- `FLOWCACHE_PRUNE_INT4` is the stronger raw-memory reduction branch
- `RTN_INT4_RECENT2` is the strongest practical RTN policy result
- `RTN_INT4_REFRESH` is the cleanest simple cadence ablation
- `QUAROT_KV_INT4` is the strongest high-fidelity quantized baseline

## Curated Comparison Videos

### MovieGen: fluffy character by a flame

Prompt ID:

- `4`

Poster:

[![MovieGen flame comparison](assets/media/moviegen_flame_selected_methods.png)](assets/media/moviegen_flame_selected_methods.mp4)

Why we used it:

- strong lighting
- obvious scene structure
- easy to notice drift, instability, and detail loss

### MovieGen: coral reef / fish

Prompt ID:

- `5`

Poster:

[![MovieGen fish comparison](assets/media/moviegen_fish_selected_methods.png)](assets/media/moviegen_fish_selected_methods.mp4)

Why we used it:

- rich texture and fine detail
- good stress test for structure-preserving compression
- visually appealing enough for a front-page demo

### StoryEval: bear in the pond

Prompt:

- `A bear bathes in a pond, shakes off water, and then rolls in grass.`

Poster:

[![StoryEval bear comparison](assets/media/storyeval_bear_selected_methods.png)](assets/media/storyeval_bear_selected_methods.mp4)

Why we used it:

- longer action progression
- easy to notice temporal drift
- strong qualitative companion to the StoryEval drift metrics

## How These Media Assets Were Built

The three linked MP4s in `docs/assets/media/` are derived comparison videos created from six prompt-matched runs and arranged into a labeled 2x3 grid:

- top row: `BF16`, `FLOWCACHE_SOFT_PRUNE_INT4`, `FLOWCACHE_PRUNE_INT4`
- bottom row: `RTN_INT4_RECENT2`, `RTN_INT4_REFRESH`, `QUAROT_KV_INT4`

They are not the dashboard itself; they are small, repository-friendly derivatives meant for GitHub presentation.

## Recommended Dashboard Demo Path

If you want the full interactive version rather than the compact repo media:

1. Launch the dashboard with `./scripts/13_launch_dashboard.sh`
2. Open the `Presentation Page`
3. Keep the default six methods pinned
4. Select:
   - MovieGen prompt `4` for the flame clip
   - MovieGen prompt `5` for the fish clip
   - StoryEval bear prompt for the narrative example
5. Use the play-all controls and the highlighted plots

## Why We Like These Three Prompts

Together they show different failure modes:

- `flame`: expressive lighting and local detail
- `fish`: dense texture and scene composition
- `bear`: temporal progression and longer-horizon stability

That combination makes them useful as a compact public-facing gallery for the repo.
