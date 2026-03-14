#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import threading
import math
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


def parse_method(
    method: str,
    bits: Optional[int],
    block_size: int,
    prq_residual_bits: int,
    qaq_outlier_threshold: float,
    age_tier_recent_ratio: float,
    age_tier_recent_bits: int,
    age_tier_recent_method: str,
    age_tier_old_method: str,
    tptq_recent_ratio: float,
    tptq_recent_bits: int,
    tptq_recent_method: str,
    tptq_residual_bits: int,
    tptq_outlier_threshold: float,
    tptq_outlier_max_ratio: float,
    flowcache_recent_ratio: float,
    flowcache_recent_bits: int,
    flowcache_recent_method: str,
    flowcache_old_method: str,
    flowcache_min_layer_budget_scale: float,
    flowcache_max_layer_budget_scale: float,
    flowcache_important_old_ratio: float,
    flowcache_importance_alpha: float,
    flowcache_importance_beta: float,
    flowcache_layer_budget_path: Optional[str],
    flowcache_prune_retained_old_ratio: float,
    flowcache_prune_refresh_gap_chunks: int,
    spatial_fg_method: str,
    spatial_fg_bits: int,
    spatial_bg_method: str,
    spatial_bg_bits: int,
    spatial_mask_policy: str,
    spatial_variance_threshold: float,
    spatial_min_foreground_ratio: float,
    spatial_max_foreground_ratio: float,
    spatial_target_foreground_ratio: float,
):
    method = method.upper()
    flowcache_layer_budget_table = load_layer_budget_table(flowcache_layer_budget_path)
    if re.fullmatch(
        r"SPATIAL_MIXED_FG_(RTN|KIVI|QUAROT_KV)_INT(2|4)_BG_(RTN|KIVI|QUAROT_KV)_INT(2|4)",
        method,
    ):
        method = "SPATIAL_MIXED"
    if method == "BF16":
        return "BF16", None
    if method == "FLOWCACHE_NATIVE":
        return "FLOWCACHE_NATIVE", None
    if method == "FLOWCACHE_NATIVE_SOFT_PRUNE_INT4":
        quantizer = create_quantizer(
            "FLOWCACHE_SOFT_PRUNE",
            bits=4,
            block_size=block_size,
            chunk_recent_ratio=flowcache_recent_ratio,
            recent_bits=flowcache_recent_bits,
            recent_method=flowcache_recent_method,
            old_method=flowcache_old_method,
            layer_budget_table=flowcache_layer_budget_table,
            min_layer_budget_scale=flowcache_min_layer_budget_scale,
            max_layer_budget_scale=flowcache_max_layer_budget_scale,
            important_old_ratio=flowcache_important_old_ratio,
            retained_old_ratio=flowcache_prune_retained_old_ratio,
            importance_alpha=flowcache_importance_alpha,
            importance_beta=flowcache_importance_beta,
            refresh_gap_chunks=flowcache_prune_refresh_gap_chunks,
        )
        return "FLOWCACHE_NATIVE_SOFT_PRUNE_INT4", quantizer
    if method == "FLOWCACHE_PROFILE":
        return "FLOWCACHE_PROFILE", create_quantizer(
            "FLOWCACHE_PROFILE",
            bits=4,
            block_size=block_size,
            profile_recent_ratio=flowcache_recent_ratio,
        )
    if method == "SPATIAL_MIXED":
        quantizer = create_quantizer(
            "SPATIAL_MIXED",
            bits=spatial_fg_bits,
            block_size=block_size,
            fg_method=spatial_fg_method,
            fg_bits=spatial_fg_bits,
            bg_method=spatial_bg_method,
            bg_bits=spatial_bg_bits,
            mask_policy=spatial_mask_policy,
            variance_threshold=spatial_variance_threshold,
            min_foreground_ratio=spatial_min_foreground_ratio,
            max_foreground_ratio=spatial_max_foreground_ratio,
            target_foreground_ratio=spatial_target_foreground_ratio,
        )
        return quantizer.name(), quantizer

    if bits is not None:
        if method in ("RTN", "KIVI", "QUAROT_KV"):
            return f"{method}_INT{bits}", create_quantizer(method, bits=bits, block_size=block_size)
        if method == "PRQ":
            return f"{method}_INT{bits}", create_quantizer(
                method, bits=bits, block_size=block_size, residual_bits=prq_residual_bits
            )
        if method == "QAQ":
            return f"{method}_INT{bits}", create_quantizer(
                method, bits=bits, block_size=block_size, outlier_threshold=qaq_outlier_threshold
            )
        if method == "AGE_TIER":
            return f"{method}_INT{bits}", create_quantizer(
                method,
                bits=bits,
                block_size=block_size,
                recent_ratio=age_tier_recent_ratio,
                recent_bits=age_tier_recent_bits,
                recent_method=age_tier_recent_method,
                old_method=age_tier_old_method,
            )
        if method == "TPTQ":
            return f"{method}_INT{bits}", create_quantizer(
                method,
                bits=bits,
                block_size=block_size,
                recent_ratio=tptq_recent_ratio,
                recent_bits=tptq_recent_bits,
                recent_method=tptq_recent_method,
                residual_bits=tptq_residual_bits,
                outlier_threshold=tptq_outlier_threshold,
                outlier_max_ratio=tptq_outlier_max_ratio,
            )
        if method == "FLOWCACHE_HYBRID":
            quantizer = create_quantizer(
                method,
                bits=bits,
                block_size=block_size,
                chunk_recent_ratio=flowcache_recent_ratio,
                recent_bits=flowcache_recent_bits,
                recent_method=flowcache_recent_method,
                old_method=flowcache_old_method,
                layer_budget_table=flowcache_layer_budget_table,
                min_layer_budget_scale=flowcache_min_layer_budget_scale,
                max_layer_budget_scale=flowcache_max_layer_budget_scale,
            )
            return quantizer.name(), quantizer
        if method == "FLOWCACHE_ADAPTIVE":
            quantizer = create_quantizer(
                method,
                bits=bits,
                block_size=block_size,
                chunk_recent_ratio=flowcache_recent_ratio,
                recent_bits=flowcache_recent_bits,
                recent_method=flowcache_recent_method,
                old_method=flowcache_old_method,
                layer_budget_table=flowcache_layer_budget_table,
                min_layer_budget_scale=flowcache_min_layer_budget_scale,
                max_layer_budget_scale=flowcache_max_layer_budget_scale,
                important_old_ratio=flowcache_important_old_ratio,
                importance_alpha=flowcache_importance_alpha,
                importance_beta=flowcache_importance_beta,
            )
            return quantizer.name(), quantizer
        if method == "FLOWCACHE_PRUNE":
            quantizer = create_quantizer(
                method,
                bits=bits,
                block_size=block_size,
                chunk_recent_ratio=flowcache_recent_ratio,
                recent_bits=flowcache_recent_bits,
                recent_method=flowcache_recent_method,
                old_method=flowcache_old_method,
                layer_budget_table=flowcache_layer_budget_table,
                min_layer_budget_scale=flowcache_min_layer_budget_scale,
                max_layer_budget_scale=flowcache_max_layer_budget_scale,
                important_old_ratio=flowcache_important_old_ratio,
                retained_old_ratio=flowcache_prune_retained_old_ratio,
                importance_alpha=flowcache_importance_alpha,
                importance_beta=flowcache_importance_beta,
                refresh_gap_chunks=flowcache_prune_refresh_gap_chunks,
            )
            return quantizer.name(), quantizer
        if method == "FLOWCACHE_SOFT_PRUNE":
            quantizer = create_quantizer(
                method,
                bits=bits,
                block_size=block_size,
                chunk_recent_ratio=flowcache_recent_ratio,
                recent_bits=flowcache_recent_bits,
                recent_method=flowcache_recent_method,
                old_method=flowcache_old_method,
                layer_budget_table=flowcache_layer_budget_table,
                min_layer_budget_scale=flowcache_min_layer_budget_scale,
                max_layer_budget_scale=flowcache_max_layer_budget_scale,
                important_old_ratio=flowcache_important_old_ratio,
                retained_old_ratio=flowcache_prune_retained_old_ratio,
                importance_alpha=flowcache_importance_alpha,
                importance_beta=flowcache_importance_beta,
                refresh_gap_chunks=flowcache_prune_refresh_gap_chunks,
            )
            return quantizer.name(), quantizer
        raise ValueError(f"Unsupported method={method} with explicit bits")

    m = re.fullmatch(r"(RTN|KIVI|QUAROT_KV|PRQ|QAQ|AGE_TIER|TPTQ|FLOWCACHE_HYBRID|FLOWCACHE_ADAPTIVE|FLOWCACHE_PRUNE|FLOWCACHE_SOFT_PRUNE)_INT(2|4)", method)
    if not m:
        raise ValueError(
            "Method must be one of BF16, FLOWCACHE_PROFILE, SPATIAL_MIXED, RTN, KIVI, QUAROT_KV, PRQ, QAQ, AGE_TIER, TPTQ, FLOWCACHE_HYBRID, FLOWCACHE_ADAPTIVE, FLOWCACHE_PRUNE, FLOWCACHE_SOFT_PRUNE, or explicit names like RTN_INT4/KIVI_INT2/TPTQ_INT2/FLOWCACHE_SOFT_PRUNE_INT2"
        )
    base = m.group(1)
    parsed_bits = int(m.group(2))
    if base == "PRQ":
        return method, create_quantizer(base, bits=parsed_bits, block_size=block_size, residual_bits=prq_residual_bits)
    if base == "QAQ":
        return method, create_quantizer(
            base, bits=parsed_bits, block_size=block_size, outlier_threshold=qaq_outlier_threshold
        )
    if base == "AGE_TIER":
        return method, create_quantizer(
            base,
            bits=parsed_bits,
            block_size=block_size,
            recent_ratio=age_tier_recent_ratio,
            recent_bits=age_tier_recent_bits,
            recent_method=age_tier_recent_method,
            old_method=age_tier_old_method,
        )
    if base == "TPTQ":
        return method, create_quantizer(
            base,
            bits=parsed_bits,
            block_size=block_size,
            recent_ratio=tptq_recent_ratio,
            recent_bits=tptq_recent_bits,
            recent_method=tptq_recent_method,
            residual_bits=tptq_residual_bits,
            outlier_threshold=tptq_outlier_threshold,
            outlier_max_ratio=tptq_outlier_max_ratio,
        )
    if base == "FLOWCACHE_HYBRID":
        quantizer = create_quantizer(
            base,
            bits=parsed_bits,
            block_size=block_size,
            chunk_recent_ratio=flowcache_recent_ratio,
            recent_bits=flowcache_recent_bits,
            recent_method=flowcache_recent_method,
            old_method=flowcache_old_method,
            layer_budget_table=flowcache_layer_budget_table,
            min_layer_budget_scale=flowcache_min_layer_budget_scale,
            max_layer_budget_scale=flowcache_max_layer_budget_scale,
        )
        return quantizer.name(), quantizer
    if base == "FLOWCACHE_ADAPTIVE":
        quantizer = create_quantizer(
            base,
            bits=parsed_bits,
            block_size=block_size,
            chunk_recent_ratio=flowcache_recent_ratio,
            recent_bits=flowcache_recent_bits,
            recent_method=flowcache_recent_method,
            old_method=flowcache_old_method,
            layer_budget_table=flowcache_layer_budget_table,
            min_layer_budget_scale=flowcache_min_layer_budget_scale,
            max_layer_budget_scale=flowcache_max_layer_budget_scale,
            important_old_ratio=flowcache_important_old_ratio,
            importance_alpha=flowcache_importance_alpha,
            importance_beta=flowcache_importance_beta,
        )
        return quantizer.name(), quantizer
    if base == "FLOWCACHE_PRUNE":
        quantizer = create_quantizer(
            base,
            bits=parsed_bits,
            block_size=block_size,
            chunk_recent_ratio=flowcache_recent_ratio,
            recent_bits=flowcache_recent_bits,
            recent_method=flowcache_recent_method,
            old_method=flowcache_old_method,
            layer_budget_table=flowcache_layer_budget_table,
            min_layer_budget_scale=flowcache_min_layer_budget_scale,
            max_layer_budget_scale=flowcache_max_layer_budget_scale,
            important_old_ratio=flowcache_important_old_ratio,
            retained_old_ratio=flowcache_prune_retained_old_ratio,
            importance_alpha=flowcache_importance_alpha,
            importance_beta=flowcache_importance_beta,
            refresh_gap_chunks=flowcache_prune_refresh_gap_chunks,
        )
        return quantizer.name(), quantizer
    if base == "FLOWCACHE_SOFT_PRUNE":
        quantizer = create_quantizer(
            base,
            bits=parsed_bits,
            block_size=block_size,
            chunk_recent_ratio=flowcache_recent_ratio,
            recent_bits=flowcache_recent_bits,
            recent_method=flowcache_recent_method,
            old_method=flowcache_old_method,
            layer_budget_table=flowcache_layer_budget_table,
            min_layer_budget_scale=flowcache_min_layer_budget_scale,
            max_layer_budget_scale=flowcache_max_layer_budget_scale,
            important_old_ratio=flowcache_important_old_ratio,
            retained_old_ratio=flowcache_prune_retained_old_ratio,
            importance_alpha=flowcache_importance_alpha,
            importance_beta=flowcache_importance_beta,
            refresh_gap_chunks=flowcache_prune_refresh_gap_chunks,
        )
        return quantizer.name(), quantizer
    return method, create_quantizer(base, bits=parsed_bits, block_size=block_size)


def load_layer_budget_table(path_str: Optional[str]) -> Dict[int, float] | None:
    if path_str is None or str(path_str).strip() == "":
        return None
    payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Layer budget table must be a JSON object: {path_str}")
    return {int(k): float(v) for k, v in payload.items()}


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
    if quantizer is not None and hasattr(quantizer, "reset_prompt_state"):
        quantizer.reset_prompt_state()
    if getattr(pipeline, "flowcache_reuse_manager", None) is not None:
        pipeline.flowcache_reuse_manager.reset_prompt_state()
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
            recent_k = block.get("recent_k")
            recent_v = block.get("recent_v")
            if isinstance(recent_k, torch.Tensor):
                block["recent_k"] = recent_k.new_empty((recent_k.shape[0], 0, recent_k.shape[2], recent_k.shape[3]))
            if isinstance(recent_v, torch.Tensor):
                block["recent_v"] = recent_v.new_empty((recent_v.shape[0], 0, recent_v.shape[2], recent_v.shape[3]))
            block["recent_start_index"] = 0
            block["recent_end_index"] = 0
            block["quantize_on_write"] = block.get("quantize_cadence", "per_step") == "per_step"


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


def attach_flowcache_native(pipeline, method_name: str, args: argparse.Namespace) -> None:
    if not method_name.startswith("FLOWCACHE_NATIVE"):
        if hasattr(pipeline, "flowcache_reuse_manager"):
            pipeline.flowcache_reuse_manager = None
        return

    if not hasattr(pipeline, "denoising_step_list"):
        raise RuntimeError("FLOWCACHE_NATIVE methods require a denoising-step Self-Forcing config.")

    from utils.flowcache_reuse import FlowCacheReuseManager

    pipeline.flowcache_reuse_manager = FlowCacheReuseManager(
        rel_l1_thresh=args.flowcache_native_rel_l1_thresh,
        warmup_steps=args.flowcache_native_warmup_steps,
    )


def tensor_shape_to_resolution(video: torch.Tensor) -> Tuple[int, int]:
    # [B, T, C, H, W]
    _, _, _, h, w = video.shape
    return int(h), int(w)


def ensure_kv_cache_capacity(pipeline, num_output_frames: int, dtype: torch.dtype, device: torch.device) -> None:
    if not hasattr(pipeline, "kv_cache1") or pipeline.kv_cache1 is None:
        return
    frame_seq_length = int(getattr(pipeline, "frame_seq_length", 0))
    if frame_seq_length <= 0:
        return
    required_tokens = int(num_output_frames) * frame_seq_length
    for block in pipeline.kv_cache1:
        k = block.get("k")
        v = block.get("v")
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor) or k.ndim != 4 or v.ndim != 4:
            continue
        current_tokens = int(k.shape[1])
        if required_tokens <= current_tokens:
            continue
        batch_size, _, num_heads, head_dim = k.shape
        new_k = torch.zeros([batch_size, required_tokens, num_heads, head_dim], dtype=dtype, device=device)
        new_v = torch.zeros_like(new_k)
        if current_tokens > 0:
            new_k[:, :current_tokens] = k
            new_v[:, :current_tokens] = v
        block["k"] = new_k
        block["v"] = new_v


def _current_active_kv_bytes(pipeline, quantizer) -> tuple[int, int]:
    kv_cache = getattr(pipeline, "kv_cache1", None)
    if kv_cache is None:
        return 0, 0

    bf16_bytes = 0
    compressed_bytes = 0
    for block in kv_cache:
        active_tokens = int(block.get("local_end_index", torch.tensor([0])).item())
        if isinstance(block.get("k"), torch.Tensor) and block["k"].ndim == 4 and block["k"].numel() > 0:
            batch_size, _, num_heads, head_dim = block["k"].shape
        else:
            batch_size = int(block.get("batch_size", 0))
            num_heads = int(block.get("num_heads", 0))
            head_dim = int(block.get("head_dim", 0))
        if active_tokens <= 0 or batch_size <= 0 or num_heads <= 0 or head_dim <= 0:
            continue
        bf16_bytes += batch_size * active_tokens * num_heads * head_dim * 2 * 2
        if quantizer is None:
            compressed_bytes += batch_size * active_tokens * num_heads * head_dim * 2 * 2
        else:
            frame_seq_length = int(block.get("frame_seq_length", 0))
            num_frame_per_block = int(block.get("num_frame_per_block", 1))
            recent_blocks = int(block.get("recent_blocks", 0))
            recent_tokens = 0
            if frame_seq_length > 0 and recent_blocks > 0:
                recent_tokens = min(active_tokens, recent_blocks * num_frame_per_block * frame_seq_length)
            old_tokens = max(active_tokens - recent_tokens, 0)
            compressed_bytes += int(
                quantizer.estimate_active_kv_bytes(
                    active_tokens=old_tokens,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                )
            )
            compressed_bytes += batch_size * recent_tokens * num_heads * head_dim * 2 * 2
    return int(bf16_bytes), int(compressed_bytes)


def _sample_trace(device: torch.device, pipeline, quantizer, start_time: float, out_samples: List[Dict[str, float]]) -> None:
    bf16_kv_bytes, compressed_kv_bytes = _current_active_kv_bytes(pipeline, quantizer)
    out_samples.append(
        {
            "t_s": time.perf_counter() - start_time,
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "bf16_kv_bytes": bf16_kv_bytes,
            "compressed_kv_bytes": compressed_kv_bytes,
        }
    )


def collect_vram_trace(
    device: torch.device,
    pipeline,
    quantizer,
    interval_s: float,
    stop_event: threading.Event,
    out_samples: List[Dict[str, float]],
) -> None:
    start_time = time.perf_counter()
    _sample_trace(device, pipeline, quantizer, start_time, out_samples)
    while not stop_event.is_set():
        time.sleep(interval_s)
        _sample_trace(device, pipeline, quantizer, start_time, out_samples)
    _sample_trace(device, pipeline, quantizer, start_time, out_samples)


def downsample_trace(samples: List[Dict[str, float]], max_points: int) -> List[Dict[str, float]]:
    if max_points <= 0 or len(samples) <= max_points:
        return samples
    stride = int(math.ceil(len(samples) / max_points))
    reduced = samples[::stride]
    if reduced[-1]["t_s"] != samples[-1]["t_s"]:
        reduced.append(samples[-1])
    return reduced


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Self-Forcing generation.")

    device = torch.device(args.device)
    method_name, quantizer = parse_method(
        args.method,
        args.bits,
        args.block_size,
        args.prq_residual_bits,
        args.qaq_outlier_threshold,
        args.age_tier_recent_ratio,
        args.age_tier_recent_bits,
        args.age_tier_recent_method,
        args.age_tier_old_method,
        args.tptq_recent_ratio,
        args.tptq_recent_bits,
        args.tptq_recent_method,
        args.tptq_residual_bits,
        args.tptq_outlier_threshold,
        args.tptq_outlier_max_ratio,
        args.flowcache_recent_ratio,
        args.flowcache_recent_bits,
        args.flowcache_recent_method,
        args.flowcache_old_method,
        args.flowcache_min_layer_budget_scale,
        args.flowcache_max_layer_budget_scale,
        args.flowcache_important_old_ratio,
        args.flowcache_importance_alpha,
        args.flowcache_importance_beta,
        args.flowcache_layer_budget_path,
        args.flowcache_prune_retained_old_ratio,
        args.flowcache_prune_refresh_gap_chunks,
        args.spatial_fg_method,
        args.spatial_fg_bits,
        args.spatial_bg_method,
        args.spatial_bg_bits,
        args.spatial_mask_policy,
        args.spatial_variance_threshold,
        args.spatial_min_foreground_ratio,
        args.spatial_max_foreground_ratio,
        args.spatial_target_foreground_ratio,
    )

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
    attach_flowcache_native(pipeline, method_name, args)
    if quantizer is not None and hasattr(quantizer, "set_runtime_context"):
        quantizer.set_runtime_context(frame_seq_length=int(getattr(pipeline, "frame_seq_length", 0)))
    num_frame_per_block = int(getattr(pipeline, "num_frame_per_block", 1))
    if args.num_output_frames % num_frame_per_block != 0:
        lower = args.num_output_frames - (args.num_output_frames % num_frame_per_block)
        upper = lower + num_frame_per_block
        raise ValueError(
            f"--num-output-frames={args.num_output_frames} must be divisible by num_frame_per_block={num_frame_per_block}. "
            f"Try {lower} or {upper}."
        )

    # Pre-initialize cache once so we can attach quantizer handles.
    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    ensure_kv_cache_capacity(pipeline, args.num_output_frames, dtype=torch.bfloat16, device=device)
    if quantizer is not None:
        quantizer.reset_stats()
        num_layers = len(pipeline.kv_cache1)
        for block_idx, block in enumerate(pipeline.kv_cache1):
            block["kv_cache_size"] = int(block["k"].shape[1])
            block["batch_size"] = int(block["k"].shape[0])
            block["num_heads"] = int(block["k"].shape[2])
            block["head_dim"] = int(block["k"].shape[3])
            block["layer_id"] = int(block_idx)
            block["num_layers"] = int(num_layers)
            block["quantizer"] = quantizer
            block["quant_state"] = None
            block["quantize_cadence"] = cache_policy["cadence"]
            block["recent_blocks"] = int(cache_policy["recent_blocks"])
            block["frame_seq_length"] = int(getattr(pipeline, "frame_seq_length", 0))
            block["num_frame_per_block"] = int(num_frame_per_block)
            block["quantize_on_write"] = cache_policy["cadence"] == "per_step"
            block["recent_k"] = block["k"][:, :0].clone()
            block["recent_v"] = block["v"][:, :0].clone()
            block["recent_start_index"] = 0
            block["recent_end_index"] = 0
            # Keep quantized state as the primary cache representation.
            # This avoids persistent BF16 KV residency for quantized methods.
            block["k"] = torch.empty(0, dtype=torch.bfloat16, device=device)
            block["v"] = torch.empty(0, dtype=torch.bfloat16, device=device)

    run_log_path = logs_dir / f"generation_{method_name}.jsonl"
    run_log_f = run_log_path.open("a", encoding="utf-8")
    vram_trace_f = None
    if args.log_vram_trace:
        vram_trace_path = logs_dir / f"vram_trace_{method_name}.jsonl"
        vram_trace_f = vram_trace_path.open("a", encoding="utf-8")

    total_runtime_s = 0.0
    peak_vram_bytes = 0
    peak_compressed_kv_bytes_seen = 0
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
            vram_samples: List[Dict[str, float]] = []
            trace_stop_event = threading.Event()
            trace_thread = None
            if args.log_vram_trace and args.vram_sample_interval_s > 0:
                trace_thread = threading.Thread(
                    target=collect_vram_trace,
                    args=(device, pipeline, quantizer, args.vram_sample_interval_s, trace_stop_event, vram_samples),
                    daemon=True,
                )
                trace_thread.start()
            start = time.perf_counter()
            with torch.no_grad():
                video, latents = pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=[prompt] * args.num_samples,
                    return_latents=True,
                    low_memory=low_memory,
                )
            runtime_s = time.perf_counter() - start
            if trace_thread is not None:
                trace_stop_event.set()
                trace_thread.join(timeout=5.0)
            peak = int(torch.cuda.max_memory_allocated(device))
            if not vram_samples:
                vram_samples = [
                    {
                        "t_s": 0.0,
                        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                        "bf16_kv_bytes": 0,
                        "compressed_kv_bytes": 0,
                    }
                ]
            vram_samples = downsample_trace(vram_samples, args.vram_max_points)
            peak_compressed_kv_bytes_seen = max(
                peak_compressed_kv_bytes_seen,
                max((int(sample.get("compressed_kv_bytes", 0)) for sample in vram_samples), default=0),
            )

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
                    "vram_trace_points": len(vram_samples),
                }
                run_log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                run_log_f.flush()
                if vram_trace_f is not None:
                    vram_trace_f.write(
                        json.dumps(
                            {
                                "method": method_name,
                                "prompt_id": prompt_id,
                                "seed": prompt_seed + sample_idx,
                                "runtime_s": runtime_s,
                                "peak_vram_bytes": peak,
                                "samples": vram_samples,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    vram_trace_f.flush()

    finally:
        run_log_f.close()
        if vram_trace_f is not None:
            vram_trace_f.close()

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
            recent_k = b.get("recent_k")
            recent_v = b.get("recent_v")
            if isinstance(recent_k, torch.Tensor) and isinstance(recent_v, torch.Tensor):
                compressed_kv_bytes += int((recent_k.numel() + recent_v.numel()) * 2)
            elif isinstance(b.get("k"), torch.Tensor) and isinstance(b.get("v"), torch.Tensor):
                compressed_kv_bytes += int((b["k"].numel() + b["v"].numel()) * 2)
        compressed_kv_bytes = max(compressed_kv_bytes, peak_compressed_kv_bytes_seen)

    efficiency = {
        "method": method_name,
        "num_prompts": len(prompts),
        "num_samples": args.num_samples,
        "total_runtime_s": total_runtime_s,
        "avg_runtime_s_per_prompt": total_runtime_s / max(len(prompts), 1),
        "peak_vram_bytes": peak_vram_bytes,
        "quantize_time_s": quant_time,
        "dequantize_time_s": dequant_time,
        "quantize_calls": 0.0 if quantizer is None else int(quantizer.stats.quantize_calls),
        "dequantize_calls": 0.0 if quantizer is None else int(quantizer.stats.dequantize_calls),
        "bf16_kv_bytes": bf16_kv_bytes,
        "compressed_kv_bytes": compressed_kv_bytes,
        "compression_ratio": (bf16_kv_bytes / compressed_kv_bytes) if compressed_kv_bytes > 0 else 0.0,
        "first_video_shape": list(first_video_shape) if first_video_shape is not None else None,
        "cache_policy": cache_policy,
    }
    if quantizer is not None and hasattr(quantizer, "diagnostics"):
        extra_metrics = quantizer.diagnostics()
        if isinstance(extra_metrics, dict):
            efficiency.update(extra_metrics)
    if getattr(pipeline, "flowcache_reuse_manager", None) is not None:
        efficiency.update(pipeline.flowcache_reuse_manager.diagnostics())

    metrics_path = metrics_dir / f"efficiency_{method_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(efficiency, f, indent=2)

    print(json.dumps(efficiency, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate videos with BF16 or KV quantization baselines.")
    parser.add_argument(
        "--method",
        type=str,
        default="BF16",
        help="BF16, FLOWCACHE_NATIVE, FLOWCACHE_NATIVE_SOFT_PRUNE_INT4, FLOWCACHE_PROFILE, SPATIAL_MIXED, RTN_INT4, KIVI_INT4, QUAROT_KV_INT4, PRQ_INT2, QAQ_INT2, AGE_TIER_INT2, TPTQ_INT2, FLOWCACHE_HYBRID_INT2, FLOWCACHE_ADAPTIVE_INT2, FLOWCACHE_PRUNE_INT2, FLOWCACHE_SOFT_PRUNE_INT2",
    )
    parser.add_argument(
        "--bits", type=int, default=None, help="Optional bit-width when using method names RTN/KIVI/QUAROT_KV/PRQ/QAQ/AGE_TIER/TPTQ/FLOWCACHE_HYBRID/FLOWCACHE_ADAPTIVE/FLOWCACHE_PRUNE/FLOWCACHE_SOFT_PRUNE"
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prq-residual-bits", type=int, default=4, choices=[2, 4])
    parser.add_argument("--qaq-outlier-threshold", type=float, default=6.0)
    parser.add_argument("--age-tier-recent-ratio", type=float, default=0.3)
    parser.add_argument("--age-tier-recent-bits", type=int, default=4, choices=[2, 4])
    parser.add_argument("--age-tier-recent-method", type=str, default="RTN", choices=["RTN", "KIVI", "QUAROT_KV"])
    parser.add_argument("--age-tier-old-method", type=str, default="RTN", choices=["RTN", "KIVI", "QUAROT_KV"])
    parser.add_argument("--tptq-recent-ratio", type=float, default=0.3)
    parser.add_argument("--tptq-recent-bits", type=int, default=4, choices=[2, 4])
    parser.add_argument("--tptq-recent-method", type=str, default="RTN", choices=["RTN", "KIVI", "QUAROT_KV"])
    parser.add_argument("--tptq-residual-bits", type=int, default=2, choices=[2, 4])
    parser.add_argument("--tptq-outlier-threshold", type=float, default=6.0)
    parser.add_argument("--tptq-outlier-max-ratio", type=float, default=0.005)
    parser.add_argument("--flowcache-recent-ratio", type=float, default=0.25)
    parser.add_argument("--flowcache-recent-bits", type=int, default=4, choices=[2, 4])
    parser.add_argument("--flowcache-recent-method", type=str, default="RTN", choices=["RTN", "KIVI", "QUAROT_KV"])
    parser.add_argument("--flowcache-old-method", type=str, default="RTN", choices=["RTN", "KIVI", "QUAROT_KV"])
    parser.add_argument("--flowcache-min-layer-budget-scale", type=float, default=0.75)
    parser.add_argument("--flowcache-max-layer-budget-scale", type=float, default=1.25)
    parser.add_argument("--flowcache-important-old-ratio", type=float, default=0.20)
    parser.add_argument("--flowcache-importance-alpha", type=float, default=0.7)
    parser.add_argument("--flowcache-importance-beta", type=float, default=0.3)
    parser.add_argument("--flowcache-layer-budget-path", type=str, default=None)
    parser.add_argument("--flowcache-prune-retained-old-ratio", type=float, default=0.30)
    parser.add_argument("--flowcache-prune-refresh-gap-chunks", type=int, default=1)
    parser.add_argument("--flowcache-native-rel-l1-thresh", type=float, default=1.50)
    parser.add_argument("--flowcache-native-warmup-steps", type=int, default=0)
    parser.add_argument("--spatial-fg-method", type=str, default="RTN", choices=["RTN", "KIVI", "QUAROT_KV"])
    parser.add_argument("--spatial-fg-bits", type=int, default=4, choices=[2, 4])
    parser.add_argument("--spatial-bg-method", type=str, default="RTN", choices=["RTN", "KIVI", "QUAROT_KV"])
    parser.add_argument("--spatial-bg-bits", type=int, default=2, choices=[2, 4])
    parser.add_argument("--spatial-mask-policy", type=str, default="hybrid", choices=["threshold", "topk", "hybrid"])
    parser.add_argument("--spatial-variance-threshold", type=float, default=0.02)
    parser.add_argument("--spatial-min-foreground-ratio", type=float, default=0.45)
    parser.add_argument("--spatial-max-foreground-ratio", type=float, default=0.85)
    parser.add_argument("--spatial-target-foreground-ratio", type=float, default=0.65)
    parser.add_argument("--config-path", type=Path, default=SELF_FORCING_ROOT / "configs" / "self_forcing_dmd.yaml")
    parser.add_argument("--default-config-path", type=Path, default=SELF_FORCING_ROOT / "configs" / "default_config.yaml")
    parser.add_argument("--checkpoint-path", type=Path, default=REPO_ROOT / "checkpoints" / "self_forcing_dmd.pt")
    parser.add_argument("--prompt-path", type=Path, default=REPO_ROOT / "prompts" / "MovieGenVideoBench_extended.txt")
    parser.add_argument("--num-output-frames", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--low-memory", action="store_true", help="Enable official dynamic-swap low-memory mode.")
    parser.add_argument(
        "--log-vram-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-prompt VRAM usage trace logging to logs/vram_trace_<method>.jsonl",
    )
    parser.add_argument("--vram-sample-interval-s", type=float, default=0.2, help="VRAM trace sampling interval in seconds.")
    parser.add_argument("--vram-max-points", type=int, default=1000, help="Maximum stored points per prompt trace.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
