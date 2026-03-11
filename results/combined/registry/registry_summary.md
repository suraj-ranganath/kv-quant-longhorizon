# Combined KV-Quant Registry Audit

## Dashboard-required artifact interpretation

### MovieGen
- overview table: `efficiency`, `vbench`, and `fidelity` (except BF16, which is the reference)
- video explorer: method video directory with `prompt_*_seed_*.mp4`
- prompt analytics: `generation_<method>.jsonl` plus `vram_trace_<method>.jsonl`
- optional long-horizon extension: `drift_<method>.json`

### StoryEval
- overview: `summary/summary.json` (or `runner_summary.json`), `summary/config.json`, `metrics/vbench.json`, `metrics/drift_imaging_quality.json`
- video explorer: `videos/*.mp4` plus `per_prompt/*.json`
- prompt analytics: `per_prompt/*.json` plus `logs/vram_trace_storyeval.jsonl`

## Audit counts
- MovieGen method-runs: **269**
- StoryEval method-runs: **39**
- Unique quantization configurations: **67**
- Runs with dashboard gaps: **78**
- Configurations missing any 10-second MovieGen run: **0**

## Configurations missing 10-second MovieGen coverage

- None

## Runs with missing dashboard requirements

| source | benchmark | run | method | long10 | missing |
|---|---|---|---|---|---|
| vaishak | moviegen | runs/tptq_ablation_smoke_1772779198 | AGE_TIER_INT2 | no | fidelity;vbench |
| suraj | moviegen | legacy_root | BF16 | no | vram_trace |
| suraj | moviegen | runs/1772742389_baseline10_memfix_clean | BF16 | no | vram_trace |
| suraj | moviegen | runs/1773037963_newideas10s_10prompts | BF16 | yes | efficiency;vbench |
| vaishak | moviegen | legacy_root | BF16 | no | vram_trace |
| vaishak | moviegen | runs/1772742389_baseline10_memfix_clean | BF16 | no | vram_trace |
| vaishak | moviegen | runs/spatial_mixed_smoke_1772758158 | BF16 | no | vbench |
| vaishak | moviegen | runs/tptq_ablation_smoke_1772779198 | BF16 | no | vbench |
| suraj | moviegen | runs/1773035807_newideas10s_10prompts | BF16 | yes | vbench |
| vaishak | moviegen | runs/flowcache_native_calib_1772867470 | FLOWCACHE_NATIVE | no | fidelity;vbench |
| vaishak | moviegen | runs/flowcache_native_smoke_1772867470 | FLOWCACHE_NATIVE | no | fidelity |
| vaishak | moviegen | runs/flowcache_native_smoke_1772867470 | FLOWCACHE_NATIVE_SOFT_PRUNE_INT4 | no | fidelity |
| vaishak | moviegen | runs/flowcache_native_smoke_1772867470 | FLOWCACHE_SOFT_PRUNE_INT4 | no | fidelity |
| suraj | moviegen | legacy_root | KIVI_INT2 | no | vram_trace |
| suraj | moviegen | runs/1772742389_baseline10_memfix_clean | KIVI_INT2 | no | vram_trace |
| vaishak | moviegen | legacy_root | KIVI_INT2 | no | vram_trace |
| vaishak | moviegen | runs/1772742389_baseline10_memfix_clean | KIVI_INT2 | no | vram_trace |
| suraj | moviegen | legacy_root | KIVI_INT4 | no | vram_trace |
| suraj | moviegen | runs/1772742389_baseline10_memfix_clean | KIVI_INT4 | no | vram_trace |
| vaishak | moviegen | legacy_root | KIVI_INT4 | no | vram_trace |
| vaishak | moviegen | runs/1772742389_baseline10_memfix_clean | KIVI_INT4 | no | vram_trace |
| vaishak | moviegen | runs/spatial_mixed_smoke_1772758158 | KIVI_INT4 | no | fidelity;vbench |
| suraj | moviegen | runs/1773035807_newideas10s_10prompts | KIVI_INT4_REFRESH | yes | efficiency;fidelity;vbench;generation_log;videos |
| suraj | moviegen | runs/1773037963_newideas10s_10prompts | KIVI_INT4_REFRESH | yes | efficiency;fidelity;vbench |
| suraj | moviegen | runs/1773038789_newideas10s_10prompts | KIVI_K2_V4 | yes | efficiency;fidelity;vbench |
| suraj | moviegen | runs/1773110004_presentation_moviegen_fullmatrix | KIVI_K2_V4 | yes | efficiency;fidelity;vbench |
| combined | moviegen | runs/_debug_prq_generate | PRQ_INT2 | no | efficiency;fidelity;vbench;generation_log;videos |
| combined | moviegen | runs/_debug_prq_generate2 | PRQ_INT2 | no | efficiency;fidelity;vbench;generation_log;videos |
| combined | moviegen | runs/_debug_prq_generate3 | PRQ_INT2 | no | efficiency;fidelity;vbench;generation_log;videos |
| combined | moviegen | runs/_debug_prq_generate4 | PRQ_INT2 | no | efficiency;fidelity;vbench;generation_log;videos |
| combined | moviegen | runs/_debug_prq_generate5 | PRQ_INT2 | yes | fidelity;vbench |
| combined | moviegen | runs/_debug_prq_generate_lowmem | PRQ_INT2 | no | efficiency;fidelity;vbench;generation_log;videos |
| vaishak | moviegen | runs/tptq_ablation_smoke_1772779198 | QAQ_INT2 | no | fidelity;vbench |
| suraj | moviegen | legacy_root | QUAROT_KV_INT2 | no | vram_trace |
| suraj | moviegen | runs/1772742389_baseline10_memfix_clean | QUAROT_KV_INT2 | no | vram_trace |
| vaishak | moviegen | legacy_root | QUAROT_KV_INT2 | no | vram_trace |
| vaishak | moviegen | runs/1772742389_baseline10_memfix_clean | QUAROT_KV_INT2 | no | vram_trace |
| vaishak | moviegen | runs/1772751420_baseline10s_10prompts_v3 | QUAROT_KV_INT2 | yes | fidelity |
| suraj | moviegen | legacy_root | QUAROT_KV_INT4 | no | vram_trace |
| suraj | moviegen | runs/1772742389_baseline10_memfix_clean | QUAROT_KV_INT4 | no | vram_trace |
| vaishak | moviegen | legacy_root | QUAROT_KV_INT4 | no | vram_trace |
| vaishak | moviegen | runs/1772742389_baseline10_memfix_clean | QUAROT_KV_INT4 | no | vram_trace |
| suraj | moviegen | runs/1773035807_newideas10s_10prompts | QUAROT_KV_INT4_REFRESH | yes | efficiency;fidelity;vbench;generation_log;videos |
| suraj | moviegen | runs/1773037963_newideas10s_10prompts | QUAROT_KV_INT4_REFRESH | yes | efficiency;fidelity;vbench |
| suraj | moviegen | runs/1773110004_presentation_moviegen_fullmatrix | QUAROT_KV_INT4_REFRESH | yes | efficiency;fidelity;vbench |
| suraj | moviegen | legacy_root | RTN_INT2 | no | vram_trace |
| suraj | moviegen | runs/1772742389_baseline10_memfix_clean | RTN_INT2 | no | vram_trace |
| vaishak | moviegen | legacy_root | RTN_INT2 | no | vram_trace |
| vaishak | moviegen | runs/1772742389_baseline10_memfix_clean | RTN_INT2 | no | vram_trace |
| suraj | moviegen | legacy_root | RTN_INT4 | no | vram_trace |
| suraj | moviegen | runs/1772742389_baseline10_memfix_clean | RTN_INT4 | no | vram_trace |
| vaishak | moviegen | legacy_root | RTN_INT4 | no | vram_trace |
| vaishak | moviegen | runs/1772742389_baseline10_memfix_clean | RTN_INT4 | no | vram_trace |
| vaishak | moviegen | runs/flowcache_native_smoke_1772867470 | RTN_INT4 | no | fidelity |
| vaishak | moviegen | runs/spatial_mixed_smoke_1772758158 | RTN_INT4 | no | fidelity;vbench |
| vaishak | moviegen | runs/tptq_ablation_smoke_1772779198 | RTN_INT4 | no | fidelity;vbench |
| suraj | moviegen | runs/1773035807_newideas10s_10prompts | RTN_INT4_REFRESH | yes | efficiency;fidelity;vbench;generation_log;videos |
| suraj | moviegen | runs/1773037963_newideas10s_10prompts | RTN_INT4_REFRESH | yes | efficiency;fidelity;vbench |
| suraj | moviegen | runs/1773038789_newideas10s_10prompts | RTN_K2_V4 | yes | efficiency;fidelity;vbench |
| suraj | moviegen | runs/1773110004_presentation_moviegen_fullmatrix | RTN_K2_V4 | yes | efficiency;fidelity;vbench |
| vaishak | moviegen | runs/spatial_mixed_smoke_1772758158 | SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 | no | efficiency;fidelity;vbench;generation_log;videos |
| vaishak | moviegen | runs/tptq_ablation_smoke_1772779198 | TPTQ_INT2 | no | efficiency;fidelity;vbench;generation_log;videos |
| suraj | storyeval | storyeval/debug_storyeval_bf16_gpu | BF16 | yes | summary;vbench;drift;per_prompt;videos |
| suraj | storyeval | storyeval/storyeval_BF16_10prompts_10s_1772778462 | BF16 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_BF16_10prompts_10s_1772778517 | BF16 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_KIVI_INT2_10prompts_10s_1772778462 | KIVI_INT2 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_KIVI_INT4_10prompts_10s_1772778462 | KIVI_INT4 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_KIVI_INT4_10prompts_10s_1772778517 | KIVI_INT4 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_KIVI_K2_V4_presentation_fullmatrix_1773110004 | KIVI_K2_V4 | no | vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/debug_storyeval_method | METHOD | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_QUAROT_KV_INT2_10prompts_10s_1772778462 | QUAROT_KV_INT2 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_QUAROT_KV_INT4_10prompts_10s_1772778462 | QUAROT_KV_INT4 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_QUAROT_KV_INT4_REFRESH_presentation_fullmatrix_1773110004 | QUAROT_KV_INT4_REFRESH | no | vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_RTN_INT2_10prompts_10s_1772778462 | RTN_INT2 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_RTN_INT2_10prompts_10s_1772778517 | RTN_INT2 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_RTN_INT4_10prompts_10s_1772778462 | RTN_INT4 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_RTN_INT4_10prompts_10s_1772778517 | RTN_INT4 | no | summary;config;vbench;drift;per_prompt;videos;vram_trace |
| suraj | storyeval | storyeval/storyeval_RTN_K2_V4_presentation_fullmatrix_1773110004 | RTN_K2_V4 | no | vbench;drift;per_prompt;videos;vram_trace |
