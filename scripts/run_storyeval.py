#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from benchmarks.t2v.storyeval import StoryEvalLoader, storyeval_video_name
from demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
from kv_quant.factory import create_quantizer
from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline
from utils.misc import set_seed


def parse_method(method: str, bits: int | None, block_size: int):
    method = method.upper()
    cache_policy = {"cadence": "per_step", "recent_blocks": 0}
    if method == "BF16":
        return "BF16", None, cache_policy

    if bits is not None:
        if method in ("RTN", "KIVI", "QUAROT_KV"):
            method_name = f"{method}_INT{bits}"
            return method_name, create_quantizer(method, bits=bits, block_size=block_size, name=method_name), cache_policy
        raise ValueError(f"Unsupported method={method} with explicit bits")

    m = __import__("re").fullmatch(r"(RTN|KIVI|QUAROT_KV)_INT(2|4)(?:_(REFRESH))?(?:_RECENT(\d+))?", method)
    if m:
        base = m.group(1)
        parsed_bits = int(m.group(2))
        if m.group(3):
            cache_policy["cadence"] = "refresh_only"
        if m.group(4):
            cache_policy["recent_blocks"] = int(m.group(4))
        return method, create_quantizer(base, bits=parsed_bits, block_size=block_size, name=method), cache_policy

    m = __import__("re").fullmatch(r"(RTN|KIVI)_K(2|4)_V(2|4)(?:_(REFRESH))?(?:_RECENT(\d+))?", method)
    if m:
        base = m.group(1)
        key_bits = int(m.group(2))
        value_bits = int(m.group(3))
        if m.group(4):
            cache_policy["cadence"] = "refresh_only"
        if m.group(5):
            cache_policy["recent_blocks"] = int(m.group(5))
        return (
            method,
            create_quantizer(
                base,
                bits=max(key_bits, value_bits),
                block_size=block_size,
                key_bits=key_bits,
                value_bits=value_bits,
                name=method,
            ),
            cache_policy,
        )

    raise ValueError(
        "Method must be BF16, *_INT{2|4}[ _REFRESH ][ _RECENTW ], or RTN/KIVI asymmetric forms like RTN_K2_V4"
    )


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])  # noqa: S603
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def reset_kv_state(pipeline, quantizer=None) -> None:
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


def _current_active_kv_bytes(pipeline, quantizer=None) -> tuple[int, int]:
    kv_cache = getattr(pipeline, "kv_cache1", None)
    if kv_cache is None:
        return 0, 0
    total_bytes = 0
    compressed_bytes = 0
    for block in kv_cache:
        active_tokens = int(block.get("local_end_index", torch.tensor([0])).item())
        k = block.get("k")
        if isinstance(k, torch.Tensor) and k.ndim == 4 and k.numel() > 0:
            batch_size, _, num_heads, head_dim = k.shape
        else:
            batch_size = int(block.get("batch_size", 0))
            num_heads = int(block.get("num_heads", 0))
            head_dim = int(block.get("head_dim", 0))
        if active_tokens <= 0 or batch_size <= 0 or num_heads <= 0 or head_dim <= 0:
            continue
        total_bytes += batch_size * active_tokens * num_heads * head_dim * 2 * 2
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
    return int(total_bytes), int(compressed_bytes)


def _sample_trace(device: torch.device, pipeline, quantizer, start_time: float, out_samples: list[dict[str, float]]) -> None:
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
    out_samples: list[dict[str, float]],
) -> None:
    start_time = time.perf_counter()
    _sample_trace(device, pipeline, quantizer, start_time, out_samples)
    while not stop_event.is_set():
        time.sleep(interval_s)
        _sample_trace(device, pipeline, quantizer, start_time, out_samples)
    _sample_trace(device, pipeline, quantizer, start_time, out_samples)


def downsample_trace(samples: list[dict[str, float]], max_points: int) -> list[dict[str, float]]:
    if max_points <= 0 or len(samples) <= max_points:
        return samples
    stride = int(math.ceil(len(samples) / max_points))
    reduced = samples[::stride]
    if reduced[-1]["t_s"] != samples[-1]["t_s"]:
        reduced.append(samples[-1])
    return reduced


def _resolve_default_fps(config: Any) -> int:
    for key in ("fps", "video_fps", "frame_rate"):
        val = config.get(key) if hasattr(config, "get") else None
        if isinstance(val, (int, float)) and val > 0:
            return int(round(float(val)))
    return 16


def _resolve_default_chunk_size(config: Any) -> int:
    for key in ("chunk_size", "num_frame_per_block"):
        val = config.get(key) if hasattr(config, "get") else None
        if isinstance(val, int) and val > 0:
            return val
        if isinstance(val, float) and val > 0:
            return int(round(val))
    return 16


def _resolve_latent_shape(config: Any) -> tuple[int, int, int]:
    shape = config.get("image_or_video_shape") if hasattr(config, "get") else None
    if isinstance(shape, list) and len(shape) >= 5:
        latent_c = int(shape[2])
        latent_h = int(shape[3])
        latent_w = int(shape[4])
        return latent_c, latent_h, latent_w
    return 16, 60, 104


def _select_prompts(loader: StoryEvalLoader, start_idx: int, end_idx: int | None, max_prompts: int | None):
    prompts = loader.load()
    n = len(prompts)
    start = max(start_idx, 0)
    end = n if end_idx is None else min(end_idx, n)
    selected = prompts[start:end]
    if max_prompts is not None:
        selected = selected[: max(max_prompts, 0)]
    if not selected:
        raise RuntimeError(f"No StoryEval prompts selected (start_idx={start_idx}, end_idx={end_idx}, max_prompts={max_prompts}).")
    return selected


def _compute_frame_targets(
    duration_sec: float, fps: int, chunk_size: int
) -> tuple[int, int, int, int, float]:
    # Self-Forcing/Wan latent-to-pixel mapping is approximately:
    # output_frames = 4 * latent_frames - 3
    raw_output_frames = max(1, int(round(duration_sec * fps)))
    raw_latent_frames = max(1, int(math.ceil((raw_output_frames + 3) / 4.0)))
    target_latent_frames = int(math.ceil(raw_latent_frames / chunk_size) * chunk_size)
    target_output_frames = int(4 * target_latent_frames - 3)
    effective_duration_sec = target_output_frames / float(fps)
    return (
        raw_output_frames,
        raw_latent_frames,
        target_latent_frames,
        target_output_frames,
        effective_duration_sec,
    )


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StoryEval generation.")

    method_name, quantizer, cache_policy = parse_method(args.method, args.bits, args.block_size)
    out_root = args.out_root if args.out_root.is_absolute() else (REPO_ROOT / args.out_root)
    run_id = args.run_id or f"storyeval_{int(time.time())}"
    run_dir = out_root / run_id

    videos_dir = run_dir / "videos"
    per_prompt_dir = run_dir / "per_prompt"
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    logs_dir = run_dir / "logs"
    summary_dir = run_dir / "summary"
    for d in (videos_dir, per_prompt_dir, metrics_dir, plots_dir, logs_dir, summary_dir):
        d.mkdir(parents=True, exist_ok=True)

    merged_cfg = OmegaConf.merge(
        OmegaConf.load(str(args.sf_default_config_path)),
        OmegaConf.load(str(args.sf_config_path)),
    )
    fps = int(args.fps) if args.fps is not None else _resolve_default_fps(merged_cfg)
    chunk_size = int(args.chunk_size) if args.chunk_size is not None else _resolve_default_chunk_size(merged_cfg)
    (
        raw_output_frames,
        raw_latent_frames,
        target_latent_frames,
        target_output_frames,
        effective_duration_sec,
    ) = _compute_frame_targets(args.duration_sec, fps, chunk_size)

    loader = StoryEvalLoader(args.prompt_file)
    prompts = _select_prompts(loader, args.start_idx, args.end_idx, args.max_prompts)
    print(f"StoryEval prompts selected: {len(prompts)}")

    device = torch.device(args.device)
    set_seed(args.seed)
    low_memory = args.low_memory or get_cuda_free_memory_gb(device) < 40
    pipeline = initialize_pipeline(
        config_path=args.sf_config_path,
        default_config_path=args.sf_default_config_path,
        checkpoint_path=args.sf_checkpoint_path,
        use_ema=args.use_ema,
        device=device,
        low_memory=low_memory,
    )

    num_frame_per_block = int(getattr(pipeline, "num_frame_per_block", 1))
    if target_latent_frames % num_frame_per_block != 0:
        target_latent_frames = int(math.ceil(target_latent_frames / num_frame_per_block) * num_frame_per_block)
        target_output_frames = int(4 * target_latent_frames - 3)
        effective_duration_sec = target_output_frames / float(fps)

    latent_c, latent_h, latent_w = _resolve_latent_shape(merged_cfg)

    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    # For >block-length generation, allocate enough KV to avoid cache-index rollover errors.
    # At the current 10s default (~42 latent frames), this fits on A5000.
    ensure_kv_cache_capacity(pipeline, target_latent_frames, dtype=torch.bfloat16, device=device)
    if quantizer is not None:
        quantizer.reset_stats()
        for block in pipeline.kv_cache1:
            block["kv_cache_size"] = int(block["k"].shape[1])
            block["batch_size"] = int(block["k"].shape[0])
            block["num_heads"] = int(block["k"].shape[2])
            block["head_dim"] = int(block["k"].shape[3])
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
            block["k"] = torch.empty(0, dtype=torch.bfloat16, device=device)
            block["v"] = torch.empty(0, dtype=torch.bfloat16, device=device)

    run_config = {
        "benchmark": "storyeval",
        "method": method_name,
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_file": str(args.prompt_file),
        "num_prompts_selected": len(prompts),
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "max_prompts": args.max_prompts,
        "seed": args.seed,
        "seeds_per_prompt": args.seeds_per_prompt,
        "fps": fps,
        "chunk_size": chunk_size,
        "num_frame_per_block": num_frame_per_block,
        "duration_sec_requested": args.duration_sec,
        "raw_output_frames": raw_output_frames,
        "raw_latent_frames": raw_latent_frames,
        "target_latent_frames": target_latent_frames,
        "target_frames": target_output_frames,
        "effective_duration_sec": effective_duration_sec,
        "latent_shape_cthw": [latent_c, target_latent_frames, latent_h, latent_w],
        "sf_config_path": str(args.sf_config_path),
        "sf_default_config_path": str(args.sf_default_config_path),
        "sf_checkpoint_path": str(args.sf_checkpoint_path),
        "use_ema": bool(args.use_ema),
        "low_memory": bool(low_memory),
        "device": str(args.device),
        "git_commit_hash": git_commit_hash(),
        "resume": bool(args.resume),
        "cache_policy": cache_policy,
    }
    (summary_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    log_jsonl = logs_dir / "generation_storyeval.jsonl"
    vram_trace_jsonl = logs_dir / "vram_trace_storyeval.jsonl"

    total_jobs = len(prompts) * max(args.seeds_per_prompt, 1)
    completed = 0
    failed = 0
    skipped = 0
    runtimes: list[float] = []
    peaks_mb: list[float] = []

    with log_jsonl.open("a", encoding="utf-8") as log_f, vram_trace_jsonl.open("a", encoding="utf-8") as trace_f:
        for prompt_obj in prompts:
            line_index = int(prompt_obj.meta.get("line_index", 0))
            for sample_idx in range(max(args.seeds_per_prompt, 1)):
                seed = args.seed + line_index * max(args.seeds_per_prompt, 1) + sample_idx
                video_name = storyeval_video_name(prompt_obj.prompt_id, seed)
                out_video = videos_dir / video_name
                out_json = per_prompt_dir / f"{prompt_obj.prompt_id}_seed{seed}.json"
                if args.resume and out_json.exists() and out_video.exists():
                    skipped += 1
                    continue

                reset_kv_state(pipeline, quantizer)
                set_seed(seed)
                sampled_noise = torch.randn(
                    [1, target_latent_frames, latent_c, latent_h, latent_w],
                    device=device,
                    dtype=torch.bfloat16,
                )

                error_text = None
                runtime_s = None
                peak_bytes = None
                total_frames = None
                resolution = None
                vram_samples: list[dict[str, float]] = []
                trace_stop_event = threading.Event()
                trace_thread = None
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                    if args.vram_sample_interval_s > 0:
                        trace_thread = threading.Thread(
                            target=collect_vram_trace,
                            args=(device, pipeline, quantizer, args.vram_sample_interval_s, trace_stop_event, vram_samples),
                            daemon=True,
                        )
                        trace_thread.start()

                    start_t = time.perf_counter()
                    with torch.no_grad():
                        video, _latents = pipeline.inference(
                            noise=sampled_noise,
                            text_prompts=[prompt_obj.prompt],
                            return_latents=True,
                            low_memory=low_memory,
                        )
                    runtime_s = float(time.perf_counter() - start_t)
                    peak_bytes = int(torch.cuda.max_memory_allocated(device))
                    total_frames = int(video.shape[1])
                    resolution = [int(video.shape[3]), int(video.shape[4])]

                    video_uint8 = (
                        (255.0 * rearrange(video, "b t c h w -> b t h w c")).clamp(0, 255).to(torch.uint8).cpu()
                    )
                    write_video(str(out_video), video_uint8[0], fps=fps)

                    completed += 1
                    runtimes.append(runtime_s)
                    peaks_mb.append(peak_bytes / (1024.0 * 1024.0))
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    error_text = f"{type(exc).__name__}: {exc}"
                finally:
                    if trace_thread is not None:
                        trace_stop_event.set()
                        trace_thread.join(timeout=5.0)
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

                record = {
                    "benchmark": "storyeval",
                    "method": method_name,
                    "run_id": run_id,
                    "prompt_id": prompt_obj.prompt_id,
                    "prompt": prompt_obj.prompt,
                    "seed": seed,
                    "fps": fps,
                    "chunk_size": chunk_size,
                    "num_frame_per_block": num_frame_per_block,
                    "duration_sec_requested": args.duration_sec,
                    "effective_duration_sec": effective_duration_sec,
                    "raw_output_frames": raw_output_frames,
                    "raw_latent_frames": raw_latent_frames,
                    "target_latent_frames": target_latent_frames,
                    "target_frames": target_output_frames,
                    "total_frames": total_frames,
                    "resolution": resolution,
                    "wall_time_sec": runtime_s,
                    "peak_vram_mb": (peak_bytes / (1024.0 * 1024.0)) if peak_bytes is not None else None,
                    "peak_vram_bytes": peak_bytes,
                    "generated_video_path": str(out_video.relative_to(REPO_ROOT)),
                    "sf_config_path": str(args.sf_config_path),
                    "sf_checkpoint_path": str(args.sf_checkpoint_path),
                    "git_commit_hash": run_config["git_commit_hash"],
                    "line_index": line_index,
                    "error": error_text,
                }
                out_json.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
                log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace_f.write(
                    json.dumps(
                        {
                            "prompt_id": prompt_obj.prompt_id,
                            "method": method_name,
                            "seed": seed,
                            "runtime_s": runtime_s,
                            "peak_vram_bytes": peak_bytes,
                            "samples": vram_samples,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                log_f.flush()
                trace_f.flush()

    summary = {
        "benchmark": "storyeval",
        "method": method_name,
        "run_id": run_id,
        "created_utc": run_config["created_utc"],
        "counts": {
            "total_jobs": total_jobs,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
        },
        "avg_runtime_sec": (sum(runtimes) / len(runtimes)) if runtimes else None,
        "avg_peak_vram_mb": (sum(peaks_mb) / len(peaks_mb)) if peaks_mb else None,
        "max_peak_vram_mb": max(peaks_mb) if peaks_mb else None,
        "fps": fps,
        "duration_sec_requested": args.duration_sec,
        "raw_output_frames": raw_output_frames,
        "raw_latent_frames": raw_latent_frames,
        "target_latent_frames": target_latent_frames,
        "target_frames": target_output_frames,
        "effective_duration_sec": effective_duration_sec,
    }
    (summary_dir / "runner_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StoryEval (T2V only) on Self-Forcing-Wan-1.3B.")
    parser.add_argument("--method", type=str, default="BF16", help="BF16, RTN_INT4, RTN_INT2, KIVI_INT4, KIVI_INT2, QUAROT_KV_INT4")
    parser.add_argument("--bits", type=int, default=None, help="Optional bit-width when using method names RTN/KIVI/QUAROT_KV")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--out_root", type=Path, default=Path("results/benchmarks/storyeval"))
    parser.add_argument("--run_id", type=str, default=f"storyeval_{int(time.time())}")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds_per_prompt", type=int, default=1)
    parser.add_argument("--duration_sec", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--sf_config_path", type=Path, default=SELF_FORCING_ROOT / "configs" / "self_forcing_dmd.yaml")
    parser.add_argument("--sf_default_config_path", type=Path, default=SELF_FORCING_ROOT / "configs" / "default_config.yaml")
    parser.add_argument("--sf_checkpoint_path", type=Path, default=REPO_ROOT / "checkpoints" / "self_forcing_dmd.pt")
    parser.add_argument("--prompt_file", type=Path, default=REPO_ROOT / "data" / "prompts" / "storyeval" / "all_prompts.txt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--low_memory", action="store_true", help="Enable Self-Forcing dynamic swap memory mode.")
    parser.add_argument("--resume", action="store_true", help="Skip completed per_prompt json/video pairs.")
    parser.add_argument("--vram_sample_interval_s", type=float, default=0.2, help="VRAM trace sampling interval in seconds.")
    parser.add_argument("--vram_max_points", type=int, default=1000, help="Maximum trace samples to store per video.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
