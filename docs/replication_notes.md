# Replication Notes

## Current status
- README planning gate is complete and pushed.
- Project scaffold, quantization modules, generation/evaluation scripts, and baseline matrix runner are implemented.
- Wan2.1-T2V-1.3B and Self-Forcing DMD checkpoint are downloaded.

## A5000 / runtime constraints observed
- Physical GPUs 6 and 7 currently have only ~3.8 GiB free VRAM each.
- BF16 initialization on GPU 6 failed with CUDA OOM while loading text encoder + model stack.
- No compute was run on GPUs 0-5.

## Required to unblock baseline runs
- Free at least one of GPU 6 or 7 sufficiently for Self-Forcing-Wan-1.3B inference.
- Then run `scripts/06_run_baseline_matrix.sh` with `GPU_ID=6` or `GPU_ID=7`.

## Known implementation deviations
- KV quantization is integrated at causal KV-cache boundary via an optional quantizer attached to each cache block.
- QVG implementation is intentionally not started yet; baseline-first policy is preserved.
