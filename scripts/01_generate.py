#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF_FORCING_ROOT = REPO_ROOT / "third_party" / "Self-Forcing"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SELF_FORCING_ROOT) not in sys.path:
    sys.path.insert(0, str(SELF_FORCING_ROOT))

from kv_quant.factory import create_quantizer
from pipeline import CausalInferencePipeline, CausalDiffusionInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb


def parse_method(method: str, bits: Optional[int], block_size: int):
    method = method.upper()
    if method == "BF16":
        return "BF16", None

    if bits is not None:
        if method in ("RTN", "KIVI", "QUAROT_KV"):
            return f"{method}_INT{bits}", create_quantizer(method, bits=bits, block_size=block_size)
        raise ValueError(f"Unsupported method={method} with explicit bits")

    m = re.fullmatch(r"(RTN|KIVI|QUAROT_KV)_INT(2|4)", method)
    if not m:
        raise ValueError(
            "Method must be one of BF16, RTN, KIVI, QUAROT_KV, or explicit names like RTN_INT4/KIVI_INT2/QUAROT_KV_INT4"
        )
    base = m.group(1)
    parsed_bits = int(m.group(2))
    return method, create_quantizer(base, bits=parsed_bits, block_size=block_size)


def load_prompts(prompt_path: Path, max_prompts: Optional[int]) -> List[Tuple[int, str]]:
    prompts: List[Tuple[int, str]] = []
    with prompt_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            prompt = line.strip()
            if not prompt:
                continue
            prompts.append((idx, prompt))
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    return prompts


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])  # noqa: S603
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def reset_kv_state(pipeline, quantizer):
    if pipeline.kv_cache1 is None:
        return
    for block in pipeline.kv_cache1:
        block["global_end_index"].fill_(0)
        block["local_end_index"].fill_(0)
        if isinstance(block.get("k"), torch.Tensor) and block["k"].numel() > 0:
            block["k"].zero_()
        if isinstance(block.get("v"), torch.Tensor) and block["v"].numel() > 0:
            block["v"].zero_()
        if quantizer is not None:
            block["quantizer"] = quantizer
            block["quant_state"] = None


def initialize_pipeline(
    config_path: Path,
    default_config_path: Path,
    checkpoint_path: Path,
    use_ema: bool,
    device: torch.device,
    low_memory: bool,
):
    config = OmegaConf.load(str(default_config_path))
    config = OmegaConf.merge(config, OmegaConf.load(str(config_path)))

    if hasattr(config, "denoising_step_list"):
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    key = "generator_ema" if use_ema else "generator"
    if key not in state_dict:
        raise KeyError(f"Checkpoint missing key '{key}'. Keys: {list(state_dict.keys())[:10]}")
    pipeline.generator.load_state_dict(state_dict[key])

    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device)
    pipeline.generator.to(device)
    pipeline.vae.to(device)
    return pipeline


def tensor_shape_to_resolution(video: torch.Tensor) -> Tuple[int, int]:
    # [B, T, C, H, W]
    _, _, _, h, w = video.shape
    return int(h), int(w)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Self-Forcing generation.")

    device = torch.device(args.device)
    method_name, quantizer = parse_method(args.method, args.bits, args.block_size)

    results_root = args.results_root if args.results_root.is_absolute() else (REPO_ROOT / args.results_root)
    output_dir = results_root / "videos" / method_name
    logs_dir = results_root / "logs"
    metrics_dir = results_root / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompt_path, args.max_prompts)
    if not prompts:
        raise RuntimeError(f"No prompts found in {args.prompt_path}")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "method": method_name,
                    "num_prompts": len(prompts),
                    "checkpoint": str(args.checkpoint_path),
                    "config": str(args.config_path),
                    "prompt_path": str(args.prompt_path),
                },
                indent=2,
            )
        )
        return

    set_seed(args.seed)
    low_memory = args.low_memory or get_cuda_free_memory_gb(device) < 40
    pipeline = initialize_pipeline(
        config_path=args.config_path,
        default_config_path=args.default_config_path,
        checkpoint_path=args.checkpoint_path,
        use_ema=args.use_ema,
        device=device,
        low_memory=low_memory,
    )

    # Pre-initialize cache once so we can attach quantizer handles.
    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    if quantizer is not None:
        quantizer.reset_stats()
        for block in pipeline.kv_cache1:
            block["kv_cache_size"] = int(block["k"].shape[1])
            block["batch_size"] = int(block["k"].shape[0])
            block["num_heads"] = int(block["k"].shape[2])
            block["head_dim"] = int(block["k"].shape[3])
            block["quantizer"] = quantizer
            block["quant_state"] = None
            # Keep quantized state as the primary cache representation.
            # This avoids persistent BF16 KV residency for quantized methods.
            block["k"] = torch.empty(0, dtype=torch.bfloat16, device=device)
            block["v"] = torch.empty(0, dtype=torch.bfloat16, device=device)

    run_log_path = logs_dir / f"generation_{method_name}.jsonl"
    run_log_f = run_log_path.open("a", encoding="utf-8")

    total_runtime_s = 0.0
    peak_vram_bytes = 0
    first_video_shape = None

    try:
        for prompt_id, prompt in prompts:
            prompt_seed = args.seed + prompt_id
            set_seed(prompt_seed)
            reset_kv_state(pipeline, quantizer)

            sampled_noise = torch.randn(
                [args.num_samples, args.num_output_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            )

            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with torch.no_grad():
                video, latents = pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=[prompt] * args.num_samples,
                    return_latents=True,
                    low_memory=low_memory,
                )
            runtime_s = time.perf_counter() - start
            peak = int(torch.cuda.max_memory_allocated(device))

            total_runtime_s += runtime_s
            peak_vram_bytes = max(peak_vram_bytes, peak)
            first_video_shape = tuple(video.shape)

            video_uint8 = (255.0 * rearrange(video, "b t c h w -> b t h w c")).clamp(0, 255).to(torch.uint8).cpu()
            h, w = tensor_shape_to_resolution(video)

            for sample_idx in range(args.num_samples):
                out_path = output_dir / f"prompt_{prompt_id:04d}_seed_{prompt_seed + sample_idx}.mp4"
                write_video(str(out_path), video_uint8[sample_idx], fps=args.fps)

                record = {
                    "method": method_name,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "seed": prompt_seed + sample_idx,
                    "model_config": str(args.config_path),
                    "checkpoint_path": str(args.checkpoint_path),
                    "git_commit_hash": git_commit_hash(),
                    "total_frames": int(video.shape[1]),
                    "resolution": [h, w],
                    "wall_clock_runtime_s": runtime_s,
                    "peak_vram_bytes": peak,
                    "output_video": str(out_path.relative_to(REPO_ROOT)),
                    "latents_shape": list(latents.shape),
                }
                run_log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                run_log_f.flush()

    finally:
        run_log_f.close()

    if pipeline.kv_cache1 is not None:
        bf16_kv_bytes = 0
        for b in pipeline.kv_cache1:
            if all(k in b for k in ("kv_cache_size", "batch_size", "num_heads", "head_dim")):
                elems = int(b["batch_size"]) * int(b["kv_cache_size"]) * int(b["num_heads"]) * int(b["head_dim"])
                bf16_kv_bytes += elems * 2 * 2  # K+V, bf16
            elif isinstance(b.get("k"), torch.Tensor) and isinstance(b.get("v"), torch.Tensor):
                bf16_kv_bytes += int((b["k"].numel() + b["v"].numel()) * 2)
    else:
        bf16_kv_bytes = 0

    if quantizer is None:
        quant_time = 0.0
        dequant_time = 0.0
        compressed_kv_bytes = bf16_kv_bytes
    else:
        quant_time = float(quantizer.stats.quantize_time_s)
        dequant_time = float(quantizer.stats.dequantize_time_s)
        compressed_kv_bytes = 0
        for b in pipeline.kv_cache1:
            if b.get("quant_state") is not None:
                compressed_kv_bytes += int(quantizer.memory_bytes(b["quant_state"]))
            elif isinstance(b.get("k"), torch.Tensor) and isinstance(b.get("v"), torch.Tensor):
                compressed_kv_bytes += int((b["k"].numel() + b["v"].numel()) * 2)

    efficiency = {
        "method": method_name,
        "num_prompts": len(prompts),
        "num_samples": args.num_samples,
        "total_runtime_s": total_runtime_s,
        "avg_runtime_s_per_prompt": total_runtime_s / max(len(prompts), 1),
        "peak_vram_bytes": peak_vram_bytes,
        "quantize_time_s": quant_time,
        "dequantize_time_s": dequant_time,
        "bf16_kv_bytes": bf16_kv_bytes,
        "compressed_kv_bytes": compressed_kv_bytes,
        "compression_ratio": (bf16_kv_bytes / compressed_kv_bytes) if compressed_kv_bytes > 0 else 0.0,
        "first_video_shape": list(first_video_shape) if first_video_shape is not None else None,
    }

    metrics_path = metrics_dir / f"efficiency_{method_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(efficiency, f, indent=2)

    print(json.dumps(efficiency, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate videos with BF16 or KV quantization baselines.")
    parser.add_argument("--method", type=str, default="BF16", help="BF16, RTN_INT4, RTN_INT2, KIVI_INT4, KIVI_INT2, QUAROT_KV_INT4")
    parser.add_argument("--bits", type=int, default=None, help="Optional bit-width when using method names RTN/KIVI/QUAROT_KV")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--config-path", type=Path, default=SELF_FORCING_ROOT / "configs" / "self_forcing_dmd.yaml")
    parser.add_argument("--default-config-path", type=Path, default=SELF_FORCING_ROOT / "configs" / "default_config.yaml")
    parser.add_argument("--checkpoint-path", type=Path, default=REPO_ROOT / "checkpoints" / "self_forcing_dmd.pt")
    parser.add_argument("--prompt-path", type=Path, default=REPO_ROOT / "prompts" / "MovieGenVideoBench_extended.txt")
    parser.add_argument("--num-output-frames", type=int, default=21)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--low-memory", action="store_true", help="Enable official dynamic-swap low-memory mode.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
