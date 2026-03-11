#!/usr/bin/env bash
set -euo pipefail

# AGE_TIER_INT2 :: 9bbee19b8cfd
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_age_tier_int2_9bbee19b8cfd_10s --run-name backfill_age_tier_int2_9bbee19b8cfd_10s --method AGE_TIER_INT2 --config-id 9bbee19b8cfd -- --device cuda:0 --use-ema --age-tier-recent-ratio 0.3 --age-tier-recent-bits 4 --age-tier-recent-method RTN --age-tier-old-method RTN

# AGE_TIER_INT2 :: e8623fd4255c
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_age_tier_int2_e8623fd4255c_10s --run-name backfill_age_tier_int2_e8623fd4255c_10s --method AGE_TIER_INT2 --config-id e8623fd4255c -- --device cuda:0 --use-ema --age-tier-recent-ratio 0.25 --age-tier-recent-bits 4 --age-tier-recent-method RTN --age-tier-old-method RTN

# AGE_TIER_INT4 :: 74dcf380574b
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_age_tier_int4_74dcf380574b_10s --run-name backfill_age_tier_int4_74dcf380574b_10s --method AGE_TIER_INT4 --config-id 74dcf380574b -- --device cuda:0 --use-ema --age-tier-recent-ratio 0.3 --age-tier-recent-bits 4 --age-tier-recent-method RTN --age-tier-old-method RTN

# FLOWCACHE_ADAPTIVE_INT2 :: 1a41612d204d
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_adaptive_int2_1a41612d204d_10s --run-name backfill_flowcache_adaptive_int2_1a41612d204d_10s --method FLOWCACHE_ADAPTIVE_INT2 --config-id 1a41612d204d --profile-flowcache --flowcache-profile-recent-ratio 0.25 --flowcache-profile-min-scale 0.7 --flowcache-profile-max-scale 1.3 -- --device cuda:0 --use-ema --flowcache-recent-ratio 0.25 --flowcache-recent-bits 4 --flowcache-recent-method RTN --flowcache-old-method RTN --flowcache-min-layer-budget-scale 0.7 --flowcache-max-layer-budget-scale 1.3 --flowcache-important-old-ratio 0.2 --flowcache-importance-alpha 0.7 --flowcache-importance-beta 0.3

# FLOWCACHE_ADAPTIVE_INT2 :: a154b903b16c
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_adaptive_int2_a154b903b16c_10s --run-name backfill_flowcache_adaptive_int2_a154b903b16c_10s --method FLOWCACHE_ADAPTIVE_INT2 --config-id a154b903b16c --profile-flowcache --flowcache-profile-recent-ratio 0.25 --flowcache-profile-min-scale 0.7 --flowcache-profile-max-scale 1.3 -- --device cuda:0 --use-ema --flowcache-recent-ratio 0.25 --flowcache-recent-bits 4 --flowcache-recent-method RTN --flowcache-old-method RTN --flowcache-min-layer-budget-scale 0.7 --flowcache-max-layer-budget-scale 1.3 --flowcache-important-old-ratio 0.1 --flowcache-importance-alpha 0.7 --flowcache-importance-beta 0.3

# FLOWCACHE_HYBRID_INT2 :: 973ebc836fd7,9cf4621557e3
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_hybrid_int2_973ebc836fd7_10s --run-name backfill_flowcache_hybrid_int2_973ebc836fd7_10s --method FLOWCACHE_HYBRID_INT2 --config-id 973ebc836fd7 --profile-flowcache --flowcache-profile-recent-ratio 0.25 --flowcache-profile-min-scale 0.7 --flowcache-profile-max-scale 1.3 -- --device cuda:0 --use-ema --flowcache-recent-ratio 0.25 --flowcache-recent-bits 4 --flowcache-recent-method RTN --flowcache-old-method RTN --flowcache-min-layer-budget-scale 0.7 --flowcache-max-layer-budget-scale 1.3

# FLOWCACHE_HYBRID_INT2 :: bfbf5df8b757
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_hybrid_int2_bfbf5df8b757_10s --run-name backfill_flowcache_hybrid_int2_bfbf5df8b757_10s --method FLOWCACHE_HYBRID_INT2 --config-id bfbf5df8b757 -- --device cuda:0 --use-ema --flowcache-recent-ratio 0.25 --flowcache-recent-bits 4 --flowcache-recent-method RTN --flowcache-old-method RTN --flowcache-min-layer-budget-scale 0.75 --flowcache-max-layer-budget-scale 1.25

# FLOWCACHE_NATIVE :: 7743d4e12ea8
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_native_7743d4e12ea8_10s --run-name backfill_flowcache_native_7743d4e12ea8_10s --method FLOWCACHE_NATIVE --config-id 7743d4e12ea8 --profile-flowcache --flowcache-profile-recent-ratio 0.25 --flowcache-profile-min-scale 0.7 --flowcache-profile-max-scale 1.3 -- --device cuda:0 --use-ema --flowcache-native-rel-l1-thresh 1.5 --flowcache-native-warmup-steps 0

# FLOWCACHE_NATIVE_SOFT_PRUNE_INT4 :: 56d7f051182b
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_native_soft_prune_int4_56d7f051182b_10s --run-name backfill_flowcache_native_soft_prune_int4_56d7f051182b_10s --method FLOWCACHE_NATIVE_SOFT_PRUNE_INT4 --config-id 56d7f051182b --profile-flowcache --flowcache-profile-recent-ratio 0.25 --flowcache-profile-min-scale 0.7 --flowcache-profile-max-scale 1.3 -- --device cuda:0 --use-ema --flowcache-recent-ratio 0.25 --flowcache-recent-bits 4 --flowcache-recent-method RTN --flowcache-old-method RTN --flowcache-min-layer-budget-scale 0.7 --flowcache-max-layer-budget-scale 1.3 --flowcache-important-old-ratio 0.1 --flowcache-importance-alpha 0.7 --flowcache-importance-beta 0.3 --flowcache-prune-retained-old-ratio 0.45 --flowcache-prune-refresh-gap-chunks 1 --flowcache-native-rel-l1-thresh 1.5 --flowcache-native-warmup-steps 0

# FLOWCACHE_PRUNE_INT2 :: 990b60e0bbd4
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_prune_int2_990b60e0bbd4_10s --run-name backfill_flowcache_prune_int2_990b60e0bbd4_10s --method FLOWCACHE_PRUNE_INT2 --config-id 990b60e0bbd4 --profile-flowcache --flowcache-profile-recent-ratio 0.25 --flowcache-profile-min-scale 0.7 --flowcache-profile-max-scale 1.3 -- --device cuda:0 --use-ema --flowcache-recent-ratio 0.25 --flowcache-recent-bits 4 --flowcache-recent-method RTN --flowcache-old-method RTN --flowcache-min-layer-budget-scale 0.7 --flowcache-max-layer-budget-scale 1.3 --flowcache-important-old-ratio 0.1 --flowcache-importance-alpha 0.7 --flowcache-importance-beta 0.3 --flowcache-prune-retained-old-ratio 0.3 --flowcache-prune-refresh-gap-chunks 1

# FLOWCACHE_PRUNE_INT2 :: f163dc4eb7a9
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_prune_int2_f163dc4eb7a9_10s --run-name backfill_flowcache_prune_int2_f163dc4eb7a9_10s --method FLOWCACHE_PRUNE_INT2 --config-id f163dc4eb7a9 --profile-flowcache --flowcache-profile-recent-ratio 0.25 --flowcache-profile-min-scale 0.7 --flowcache-profile-max-scale 1.3 -- --device cuda:0 --use-ema --flowcache-recent-ratio 0.25 --flowcache-recent-bits 4 --flowcache-recent-method RTN --flowcache-old-method RTN --flowcache-min-layer-budget-scale 0.7 --flowcache-max-layer-budget-scale 1.3 --flowcache-important-old-ratio 0.1 --flowcache-importance-alpha 0.7 --flowcache-importance-beta 0.3 --flowcache-prune-retained-old-ratio 0.45 --flowcache-prune-refresh-gap-chunks 1

# FLOWCACHE_PRUNE_INT4 :: 27bbc04277fe
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_prune_int4_27bbc04277fe_10s --run-name backfill_flowcache_prune_int4_27bbc04277fe_10s --method FLOWCACHE_PRUNE_INT4 --config-id 27bbc04277fe --profile-flowcache --flowcache-profile-recent-ratio 0.25 --flowcache-profile-min-scale 0.7 --flowcache-profile-max-scale 1.3 -- --device cuda:0 --use-ema --flowcache-recent-ratio 0.25 --flowcache-recent-bits 4 --flowcache-recent-method RTN --flowcache-old-method RTN --flowcache-min-layer-budget-scale 0.7 --flowcache-max-layer-budget-scale 1.3 --flowcache-important-old-ratio 0.1 --flowcache-importance-alpha 0.7 --flowcache-importance-beta 0.3 --flowcache-prune-retained-old-ratio 0.45 --flowcache-prune-refresh-gap-chunks 1

# FLOWCACHE_SOFT_PRUNE_INT2 :: 24b7f6c1cfbb
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_soft_prune_int2_24b7f6c1cfbb_10s --run-name backfill_flowcache_soft_prune_int2_24b7f6c1cfbb_10s --method FLOWCACHE_SOFT_PRUNE_INT2 --config-id 24b7f6c1cfbb --profile-flowcache --flowcache-profile-recent-ratio 0.25 --flowcache-profile-min-scale 0.7 --flowcache-profile-max-scale 1.3 -- --device cuda:0 --use-ema --flowcache-recent-ratio 0.25 --flowcache-recent-bits 4 --flowcache-recent-method RTN --flowcache-old-method RTN --flowcache-min-layer-budget-scale 0.7 --flowcache-max-layer-budget-scale 1.3 --flowcache-important-old-ratio 0.1 --flowcache-importance-alpha 0.7 --flowcache-importance-beta 0.3 --flowcache-prune-retained-old-ratio 0.45 --flowcache-prune-refresh-gap-chunks 1

# FLOWCACHE_SOFT_PRUNE_INT4 :: 5f5a998802bd
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_flowcache_soft_prune_int4_5f5a998802bd_10s --run-name backfill_flowcache_soft_prune_int4_5f5a998802bd_10s --method FLOWCACHE_SOFT_PRUNE_INT4 --config-id 5f5a998802bd --profile-flowcache --flowcache-profile-recent-ratio 0.25 --flowcache-profile-min-scale 0.7 --flowcache-profile-max-scale 1.3 -- --device cuda:0 --use-ema --flowcache-recent-ratio 0.25 --flowcache-recent-bits 4 --flowcache-recent-method RTN --flowcache-old-method RTN --flowcache-min-layer-budget-scale 0.7 --flowcache-max-layer-budget-scale 1.3 --flowcache-important-old-ratio 0.1 --flowcache-importance-alpha 0.7 --flowcache-importance-beta 0.3 --flowcache-prune-retained-old-ratio 0.45 --flowcache-prune-refresh-gap-chunks 1

# PRQ_INT2 :: c800cfa2a981
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_prq_int2_c800cfa2a981_10s --run-name backfill_prq_int2_c800cfa2a981_10s --method PRQ_INT2 --config-id c800cfa2a981 -- --device cuda:0 --use-ema --prq-residual-bits 4

# PRQ_INT4 :: 23a612bd2862
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_prq_int4_23a612bd2862_10s --run-name backfill_prq_int4_23a612bd2862_10s --method PRQ_INT4 --config-id 23a612bd2862 -- --device cuda:0 --use-ema --prq-residual-bits 4

# QAQ_INT2 :: dcc080e7d18f
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_qaq_int2_dcc080e7d18f_10s --run-name backfill_qaq_int2_dcc080e7d18f_10s --method QAQ_INT2 --config-id dcc080e7d18f -- --device cuda:0 --use-ema --qaq-outlier-threshold 6.0

# QAQ_INT4 :: aa6e42af3d16
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_qaq_int4_aa6e42af3d16_10s --run-name backfill_qaq_int4_aa6e42af3d16_10s --method QAQ_INT4 --config-id aa6e42af3d16 -- --device cuda:0 --use-ema --qaq-outlier-threshold 6.0

# SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2 :: 3d327a450ca5
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_spatial_mixed_fg_kivi_int4_bg_kivi_int2_3d327a450ca5_10s --run-name backfill_spatial_mixed_fg_kivi_int4_bg_kivi_int2_3d327a450ca5_10s --method SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2 --config-id 3d327a450ca5 -- --device cuda:0 --use-ema --spatial-fg-method KIVI --spatial-fg-bits 4 --spatial-bg-method KIVI --spatial-bg-bits 2 --spatial-mask-policy topk --spatial-variance-threshold 0.02 --spatial-min-foreground-ratio 0.6 --spatial-max-foreground-ratio 0.9 --spatial-target-foreground-ratio 0.8

# SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 :: af0e1883d58d
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_spatial_mixed_fg_quarot_kv_int4_bg_rtn_int2_af0e1883d58d_10s --run-name backfill_spatial_mixed_fg_quarot_kv_int4_bg_rtn_int2_af0e1883d58d_10s --method SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2 --config-id af0e1883d58d -- --device cuda:0 --use-ema --spatial-fg-method QUAROT_KV --spatial-fg-bits 4 --spatial-bg-method RTN --spatial-bg-bits 2 --spatial-mask-policy topk --spatial-variance-threshold 0.02 --spatial-min-foreground-ratio 0.6 --spatial-max-foreground-ratio 0.9 --spatial-target-foreground-ratio 0.8

# SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 :: 1f080d8dcc0e,c64825ec3ea8
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_spatial_mixed_fg_rtn_int4_bg_rtn_int2_1f080d8dcc0e_10s --run-name backfill_spatial_mixed_fg_rtn_int4_bg_rtn_int2_1f080d8dcc0e_10s --method SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 --config-id 1f080d8dcc0e -- --device cuda:0 --use-ema --spatial-fg-method RTN --spatial-fg-bits 4 --spatial-bg-method RTN --spatial-bg-bits 2 --spatial-mask-policy hybrid --spatial-variance-threshold 0.02 --spatial-min-foreground-ratio 0.45 --spatial-max-foreground-ratio 0.85 --spatial-target-foreground-ratio 0.65

# SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 :: bae80e29fbb6
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_spatial_mixed_fg_rtn_int4_bg_rtn_int2_bae80e29fbb6_10s --run-name backfill_spatial_mixed_fg_rtn_int4_bg_rtn_int2_bae80e29fbb6_10s --method SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2 --config-id bae80e29fbb6 -- --device cuda:0 --use-ema --spatial-fg-method RTN --spatial-fg-bits 4 --spatial-bg-method RTN --spatial-bg-bits 2 --spatial-mask-policy topk --spatial-variance-threshold 0.02 --spatial-min-foreground-ratio 0.6 --spatial-max-foreground-ratio 0.9 --spatial-target-foreground-ratio 0.8

# SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4 :: 04d60f52e048
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_spatial_mixed_fg_rtn_int4_bg_rtn_int4_04d60f52e048_10s --run-name backfill_spatial_mixed_fg_rtn_int4_bg_rtn_int4_04d60f52e048_10s --method SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4 --config-id 04d60f52e048 -- --device cuda:0 --use-ema --spatial-fg-method RTN --spatial-fg-bits 4 --spatial-bg-method RTN --spatial-bg-bits 4 --spatial-mask-policy hybrid --spatial-variance-threshold 0.02 --spatial-min-foreground-ratio 0.45 --spatial-max-foreground-ratio 0.85 --spatial-target-foreground-ratio 0.65

# SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4 :: 580e67e08d8e
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_spatial_mixed_fg_rtn_int4_bg_rtn_int4_580e67e08d8e_10s --run-name backfill_spatial_mixed_fg_rtn_int4_bg_rtn_int4_580e67e08d8e_10s --method SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4 --config-id 580e67e08d8e -- --device cuda:0 --use-ema --spatial-fg-method RTN --spatial-fg-bits 4 --spatial-bg-method RTN --spatial-bg-bits 4 --spatial-mask-policy topk --spatial-variance-threshold 0.02 --spatial-min-foreground-ratio 0.6 --spatial-max-foreground-ratio 0.9 --spatial-target-foreground-ratio 0.8

# TPTQ_INT2 :: 089f989f38f6
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_tptq_int2_089f989f38f6_10s --run-name backfill_tptq_int2_089f989f38f6_10s --method TPTQ_INT2 --config-id 089f989f38f6 -- --device cuda:0 --use-ema --tptq-recent-ratio 0.15 --tptq-recent-bits 4 --tptq-recent-method RTN --tptq-residual-bits 2 --tptq-outlier-threshold 6.0 --tptq-outlier-max-ratio 0.001

# TPTQ_INT2 :: 2babecc6bab8
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_tptq_int2_2babecc6bab8_10s --run-name backfill_tptq_int2_2babecc6bab8_10s --method TPTQ_INT2 --config-id 2babecc6bab8 -- --device cuda:0 --use-ema --tptq-recent-ratio 0.2 --tptq-recent-bits 4 --tptq-recent-method RTN --tptq-residual-bits 2 --tptq-outlier-threshold 6.0 --tptq-outlier-max-ratio 0.002

# TPTQ_INT2 :: 351b43b37ce8
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_tptq_int2_351b43b37ce8_10s --run-name backfill_tptq_int2_351b43b37ce8_10s --method TPTQ_INT2 --config-id 351b43b37ce8 -- --device cuda:0 --use-ema --tptq-recent-ratio 0.25 --tptq-recent-bits 4 --tptq-recent-method RTN --tptq-residual-bits 2 --tptq-outlier-threshold 6.0 --tptq-outlier-max-ratio 0.003

# TPTQ_INT2 :: 44c4da374609
python3 scripts/23_run_moviegen_backfill.py --run-root results/runs/backfill_tptq_int2_44c4da374609_10s --run-name backfill_tptq_int2_44c4da374609_10s --method TPTQ_INT2 --config-id 44c4da374609 -- --device cuda:0 --use-ema --tptq-recent-ratio 0.3 --tptq-recent-bits 4 --tptq-recent-method RTN --tptq-residual-bits 2 --tptq-outlier-threshold 6.0 --tptq-outlier-max-ratio 0.005
