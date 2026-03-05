# Replication Notes

## Current status
- README planning gate is complete and pushed.
- Project scaffold, quantization modules, generation/evaluation scripts, and baseline matrix runner are implemented.
- Wan2.1-T2V-1.3B and Self-Forcing DMD checkpoint are downloaded.
- KV-cache quantization path has been updated to avoid persistent BF16+quantized dual cache residency.
- Named run isolation is enabled via `results/runs/<unix_ts>_<run_name>/`.

## A5000 / runtime constraints observed
- Full 480p generation is feasible on A5000 24GB for 5s baseline runs.
- Long-horizon runs still require careful memory/runtime planning.

## KV memory behavior note
- Previous integration kept BF16 KV tensors resident while also storing quantized state, inflating peak VRAM.
- Current integration stores quantized state as the primary cache representation and drops persistent BF16 cache residency in quantized modes.
- In smoke verification, quantized methods now peak near BF16-level VRAM rather than substantially above it.

## Known implementation deviations
- KV quantization is integrated at causal KV-cache boundary via an optional quantizer attached to each cache block.
- QVG implementation is intentionally not started yet; baseline-first policy is preserved.
