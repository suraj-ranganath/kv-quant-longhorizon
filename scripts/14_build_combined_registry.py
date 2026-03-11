#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKFILL_MANIFEST_PATH = REPO_ROOT / "results" / "combined" / "registry" / "backfill_manifest.json"
DEFAULT_SOURCES: dict[str, Path] = {
    "vaishak": Path("/data/vaishak/kv-quant-longhorizon/results"),
    "suraj": Path("/data/suraj/kv-quant-longhorizon/results"),
}

METRIC_PREFIXES = ("efficiency", "fidelity", "vbench", "drift")
VIDEO_RE = re.compile(r"prompt_(\d+)_seed_(\d+)\.mp4$")
BITS_RE = re.compile(r"_INT(\d+)")
GENERIC_META_KEYS = {
    "run_name",
    "safe_run_name",
    "run_timestamp_unix",
    "run_id",
    "run_root",
    "run_timestamp",
    "run_start_utc",
    "run_end_utc",
    "created_utc",
    "benchmark",
    "prompt_file",
    "model_config",
    "checkpoint_path",
    "git_commit_hash",
    "sf_config_path",
    "sf_default_config_path",
    "sf_checkpoint_path",
    "device",
    "resume",
    "seed",
    "seeds_per_prompt",
    "num_prompts_selected",
    "num_output_frames",
    "raw_output_frames",
    "raw_latent_frames",
    "target_latent_frames",
    "target_frames",
    "effective_duration_sec",
    "duration_sec_requested",
    "fps",
    "chunk_size",
    "num_frame_per_block",
    "start_idx",
    "end_idx",
    "max_prompts",
    "use_ema",
    "low_memory",
}
QUANT_META_PREFIXES = (
    "flowcache_",
    "spatial_",
    "tptq_",
    "qaq_",
    "prq_",
    "age_tier_",
    "cache_policy",
)
STATIC_QUANT_SUFFIXES = (
    "_config_recent_ratio",
    "_recent_bits",
    "_old_bits",
    "_residual_bits",
    "_outlier_threshold",
    "_outlier_max_ratio",
    "_important_old_ratio",
    "_importance_alpha",
    "_importance_beta",
    "_min_layer_budget_scale",
    "_max_layer_budget_scale",
    "_profile_min_scale",
    "_profile_max_scale",
    "_recent_method",
    "_old_method",
    "_prune_retained_old_ratio",
    "_prune_refresh_gap_chunks",
    "_native_rel_l1_thresh",
    "_native_warmup_steps",
    "_config_chunk_recent_ratio",
    "_config_important_old_ratio",
    "_config_retained_old_ratio",
    "_fg_method",
    "_fg_bits",
    "_bg_method",
    "_bg_bits",
    "_mask_policy",
    "_variance_threshold",
    "_min_foreground_ratio",
    "_max_foreground_ratio",
    "_target_foreground_ratio",
)
DYNAMIC_QUANT_MARKERS = (
    "_avg_",
    "_events",
    "_rate",
    "_delta",
    "_reuse_ratio",
    "_reuse_steps",
    "_total_steps",
    "_compute_steps",
    "_total_blocks",
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def read_first_jsonl_record(path: Path) -> tuple[int, dict[str, Any] | None]:
    if not path.exists():
        return 0, None
    count = 0
    first: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                count += 1
                if first is None:
                    first = payload
    return count, first


def parse_method_family(method: str) -> str:
    if method == "BF16":
        return "BF16"
    if method.startswith("SPATIAL_MIXED"):
        return "SPATIAL_MIXED"
    for prefix in (
        "QUAROT_KV",
        "FLOWCACHE_NATIVE_SOFT_PRUNE",
        "FLOWCACHE_SOFT_PRUNE",
        "FLOWCACHE_ADAPTIVE",
        "FLOWCACHE_HYBRID",
        "FLOWCACHE_PRUNE",
        "FLOWCACHE_NATIVE",
        "FLOWCACHE_PROFILE",
        "AGE_TIER",
        "PRQ",
        "QAQ",
        "TPTQ",
        "RTN",
        "KIVI",
    ):
        if method == prefix or method.startswith(f"{prefix}_"):
            return prefix
    if "_INT" in method:
        return method.split("_INT", 1)[0]
    return method


def parse_bits(method: str) -> int | None:
    match = BITS_RE.search(method)
    return int(match.group(1)) if match else None


def is_static_quant_key(key: str) -> bool:
    if any(marker in key for marker in DYNAMIC_QUANT_MARKERS):
        return False
    return key.endswith(STATIC_QUANT_SUFFIXES) or "_config_" in key


def extract_quant_meta(meta: dict[str, Any], config: dict[str, Any] | None, efficiency: dict[str, Any] | None) -> dict[str, Any]:
    quant_meta: dict[str, Any] = {}
    for key, value in meta.items():
        if key in GENERIC_META_KEYS:
            continue
        if key.startswith(QUANT_META_PREFIXES) and is_static_quant_key(key):
            quant_meta[key] = value
    if config:
        cache_policy = config.get("cache_policy")
        if isinstance(cache_policy, dict):
            quant_meta["cache_policy"] = cache_policy
    if efficiency:
        cache_policy = efficiency.get("cache_policy")
        if isinstance(cache_policy, dict):
            quant_meta["cache_policy"] = cache_policy
        for key, value in efficiency.items():
            if (
                key.startswith(QUANT_META_PREFIXES)
                and key not in quant_meta
                and is_static_quant_key(key)
            ):
                quant_meta[key] = value
    return quant_meta


def filter_quant_meta_for_method(method_family: str, quant_meta: dict[str, Any]) -> dict[str, Any]:
    allowed_fragments = {
        "AGE_TIER": (
            "age_tier_config_recent_ratio",
            "age_tier_recent_bits",
            "age_tier_old_bits",
            "age_tier_recent_method",
            "age_tier_old_method",
        ),
        "PRQ": ("prq_residual_bits",),
        "QAQ": ("qaq_outlier_threshold",),
        "TPTQ": (
            "tptq_config_recent_ratio",
            "tptq_recent_bits",
            "tptq_old_bits",
            "tptq_recent_method",
            "tptq_residual_bits",
            "tptq_outlier_threshold",
            "tptq_outlier_max_ratio",
        ),
        "SPATIAL_MIXED": (
            "spatial_fg_method",
            "spatial_fg_bits",
            "spatial_bg_method",
            "spatial_bg_bits",
            "spatial_mask_policy",
            "spatial_variance_threshold",
            "spatial_min_foreground_ratio",
            "spatial_max_foreground_ratio",
            "spatial_target_foreground_ratio",
        ),
        "FLOWCACHE_HYBRID": (
            "flowcache_config_chunk_recent_ratio",
            "flowcache_recent_bits",
            "flowcache_recent_method",
            "flowcache_old_method",
            "flowcache_min_layer_budget_scale",
            "flowcache_max_layer_budget_scale",
            "flowcache_profile_min_scale",
            "flowcache_profile_max_scale",
        ),
        "FLOWCACHE_ADAPTIVE": (
            "flowcache_adaptive_config_chunk_recent_ratio",
            "flowcache_adaptive_config_important_old_ratio",
            "flowcache_adaptive_recent_bits",
            "flowcache_adaptive_old_bits",
            "flowcache_adaptive_importance_alpha",
            "flowcache_adaptive_importance_beta",
            "flowcache_recent_bits",
            "flowcache_recent_method",
            "flowcache_old_method",
            "flowcache_profile_min_scale",
            "flowcache_profile_max_scale",
        ),
        "FLOWCACHE_PRUNE": (
            "flowcache_prune_config_chunk_recent_ratio",
            "flowcache_prune_config_important_old_ratio",
            "flowcache_prune_config_retained_old_ratio",
            "flowcache_prune_recent_bits",
            "flowcache_prune_old_bits",
            "flowcache_prune_refresh_gap_chunks",
            "flowcache_recent_bits",
            "flowcache_recent_method",
            "flowcache_old_method",
            "flowcache_importance_alpha",
            "flowcache_importance_beta",
        ),
        "FLOWCACHE_SOFT_PRUNE": (
            "flowcache_soft_prune_config_chunk_recent_ratio",
            "flowcache_soft_prune_config_important_old_ratio",
            "flowcache_soft_prune_config_retained_old_ratio",
            "flowcache_soft_prune_recent_bits",
            "flowcache_soft_prune_old_bits",
            "flowcache_soft_prune_refresh_gap_chunks",
            "flowcache_recent_bits",
            "flowcache_recent_method",
            "flowcache_old_method",
            "flowcache_importance_alpha",
            "flowcache_importance_beta",
        ),
        "FLOWCACHE_NATIVE": (
            "flowcache_native_rel_l1_thresh",
            "flowcache_native_warmup_steps",
        ),
        "FLOWCACHE_NATIVE_SOFT_PRUNE": (
            "flowcache_native_rel_l1_thresh",
            "flowcache_native_warmup_steps",
            "flowcache_soft_prune_config_chunk_recent_ratio",
            "flowcache_soft_prune_config_important_old_ratio",
            "flowcache_soft_prune_config_retained_old_ratio",
            "flowcache_soft_prune_recent_bits",
            "flowcache_soft_prune_old_bits",
            "flowcache_soft_prune_refresh_gap_chunks",
            "flowcache_recent_bits",
            "flowcache_recent_method",
            "flowcache_old_method",
            "flowcache_importance_alpha",
            "flowcache_importance_beta",
        ),
    }.get(method_family, ())
    if not allowed_fragments:
        return {}
    return {
        key: value
        for key, value in quant_meta.items()
        if any(fragment in key for fragment in allowed_fragments)
    }


def build_config_identity(
    benchmark: str,
    method: str,
    method_family: str,
    bits: int | None,
    cadence: str | None,
    recent_blocks: int | None,
    quant_meta: dict[str, Any],
) -> tuple[str, str]:
    payload = {
        "benchmark": benchmark,
        "method": method,
        "method_family": method_family,
        "bits": bits,
        "cache_policy": {
            "cadence": cadence,
            "recent_blocks": recent_blocks,
        },
        "quant_meta": quant_meta,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    return digest, canonical


def load_backfill_aliases() -> dict[str, list[str]]:
    if not BACKFILL_MANIFEST_PATH.exists():
        return {}
    try:
        manifest = json.loads(BACKFILL_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(manifest, list):
        return {}
    aliases: dict[str, list[str]] = {}
    for row in manifest:
        if not isinstance(row, dict):
            continue
        config_ids = [str(value) for value in row.get("config_ids", []) if value]
        if not config_ids:
            primary = row.get("primary_config_id")
            if primary:
                config_ids = [str(primary)]
        if not config_ids:
            continue
        run_name = row.get("run_name")
        run_root = row.get("run_root")
        if run_name:
            aliases[f"run_name:{run_name}"] = config_ids
        if run_root:
            aliases[f"run_root:{Path(str(run_root)).as_posix()}"] = config_ids
    return aliases


def resolve_combined_backfill_config_ids(
    run_name: str,
    run_root: Path,
    run_meta: dict[str, Any],
    derived_config_id: str,
    backfill_aliases: dict[str, list[str]],
) -> list[str]:
    if run_meta.get("experiment_type") != "combined_backfill":
        return [derived_config_id]

    alias_ids = backfill_aliases.get(f"run_name:{run_name}") or backfill_aliases.get(
        f"run_root:{run_root.relative_to(REPO_ROOT).as_posix()}"
    )
    if alias_ids:
        return alias_ids

    explicit_ids = run_meta.get("backfill_config_ids")
    if isinstance(explicit_ids, list):
        normalized_ids = [str(value) for value in explicit_ids if value]
        if normalized_ids:
            return normalized_ids

    return [derived_config_id]


def count_storyeval_videos(videos_dir: Path) -> int:
    if not videos_dir.exists():
        return 0
    return len(list(videos_dir.glob("*.mp4")))


def max_moviegen_videos(video_dirs: list[Path], method: str) -> int:
    count = 0
    for base in video_dirs:
        method_dir = base / method
        if method_dir.exists():
            count = max(count, len(list(method_dir.glob("prompt_*_seed_*.mp4"))))
    return count


def find_file(search_dirs: list[Path], name: str) -> Path | None:
    for base in search_dirs:
        candidate = base / name
        if candidate.exists():
            return candidate
    return None


def infer_moviegen_run_layouts(results_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []

    current_metrics = results_root / "metrics"
    current_logs = results_root / "logs"
    current_videos = results_root / "videos"
    current_tables = results_root / "tables"

    current_has_data = False
    if current_videos.exists():
        current_has_data = any(current_videos.glob("*/*.mp4"))
    if not current_has_data and current_logs.exists():
        current_has_data = any(current_logs.glob("generation_*.jsonl"))
    if not current_has_data and current_metrics.exists():
        current_has_data = any(current_metrics.glob("*.json"))
    if current_has_data:
        runs.append(
            {
                "run_label": "legacy_root",
                "run_root": results_root,
                "metric_dirs": [current_metrics],
                "log_dirs": [current_logs],
                "video_dirs": [current_videos],
                "table_dirs": [current_tables],
            }
        )

    for run_parent in (results_root / "runs", results_root / "archive"):
        if not run_parent.exists():
            continue
        for run_root in sorted([path for path in run_parent.iterdir() if path.is_dir()], reverse=True):
            metric_dir = run_root / "metrics"
            log_dir = run_root / "logs"
            video_dir = run_root / "videos"
            table_dir = run_root / "tables"
            has_data = (
                (video_dir.exists() and any(video_dir.glob("*/*.mp4")))
                or (log_dir.exists() and any(log_dir.glob("generation_*.jsonl")))
                or (metric_dir.exists() and any(metric_dir.glob("*.json")))
            )
            if not has_data:
                continue
            runs.append(
                {
                    "run_label": f"{run_parent.name}/{run_root.name}",
                    "run_root": run_root,
                    "metric_dirs": [path for path in (metric_dir, run_root) if path.exists()],
                    "log_dirs": [path for path in (log_dir, run_root) if path.exists()],
                    "video_dirs": [path for path in (video_dir, run_root) if path.exists()],
                    "table_dirs": [path for path in (table_dir, run_root) if path.exists()],
                }
            )
    return runs


def list_moviegen_methods(layout: dict[str, Any]) -> set[str]:
    methods: set[str] = set()
    for metric_dir in layout["metric_dirs"]:
        for prefix in METRIC_PREFIXES:
            for path in metric_dir.glob(f"{prefix}_*.json"):
                methods.add(path.stem[len(prefix) + 1 :])
    for log_dir in layout["log_dirs"]:
        for path in log_dir.glob("generation_*.jsonl"):
            methods.add(path.stem[len("generation_") :])
        for path in log_dir.glob("vram_trace_*.jsonl"):
            methods.add(path.stem[len("vram_trace_") :])
    for video_dir in layout["video_dirs"]:
        if not video_dir.exists():
            continue
        for sub in video_dir.iterdir():
            if sub.is_dir() and any(sub.glob("prompt_*_seed_*.mp4")):
                methods.add(sub.name)
    return methods


def parse_storyeval_method(run_root: Path) -> str:
    config = load_json(run_root / "summary" / "config.json") or {}
    method = config.get("method")
    if isinstance(method, str) and method:
        return method
    match = re.match(r"^storyeval_(.+?)_\d+prompts_\d+s_\d+$", run_root.name)
    if match:
        return match.group(1)
    if run_root.name.startswith("debug_storyeval_"):
        return run_root.name[len("debug_storyeval_") :].upper()
    return run_root.name


def infer_storyeval_runs(results_root: Path) -> list[Path]:
    storyeval_root = results_root / "benchmarks" / "storyeval"
    if not storyeval_root.exists():
        return []
    return sorted([path for path in storyeval_root.iterdir() if path.is_dir()], reverse=True)


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def infer_moviegen_total_frames(
    efficiency: dict[str, Any] | None,
    generation_first: dict[str, Any] | None,
    run_meta: dict[str, Any],
) -> int | None:
    if efficiency:
        first_shape = efficiency.get("first_video_shape")
        if isinstance(first_shape, list) and len(first_shape) >= 2:
            frames = safe_int(first_shape[1])
            if frames:
                return frames
    if generation_first:
        frames = safe_int(generation_first.get("total_frames"))
        if frames:
            return frames
    latent_frames = safe_int(run_meta.get("num_output_frames"))
    if latent_frames is None:
        latent_frames = safe_int(run_meta.get("target_num_output_frames"))
    if latent_frames in {41, 42}:
        return 165
    return None


def infer_moviegen_prompt_count(
    efficiency: dict[str, Any] | None,
    generation_count: int,
    video_count: int,
) -> int | None:
    if efficiency:
        prompts = safe_int(efficiency.get("num_prompts"))
        if prompts:
            return prompts
    if generation_count > 0:
        return generation_count
    if video_count > 0:
        return video_count
    return None


def infer_is_ten_second_moviegen(run_name: str, total_frames: int | None) -> bool:
    if total_frames is not None and total_frames >= 160:
        return True
    return "10s" in run_name.lower()


def infer_is_ten_second_storyeval(config: dict[str, Any], summary: dict[str, Any], first_record: dict[str, Any] | None) -> bool:
    requested = safe_float(config.get("duration_sec_requested"))
    if requested is not None and requested >= 9.5:
        return True
    target_frames = safe_int(config.get("target_frames"))
    if target_frames is not None and target_frames >= 160:
        return True
    summary_duration = safe_float(summary.get("effective_duration_sec"))
    if summary_duration is not None and summary_duration >= 9.5:
        return True
    if first_record:
        record_duration = safe_float(first_record.get("effective_duration_sec"))
        if record_duration is not None and record_duration >= 9.5:
            return True
        record_frames = safe_int(first_record.get("target_frames"))
        if record_frames is not None and record_frames >= 160:
            return True
    return False


def build_moviegen_record(
    source: str,
    results_root: Path,
    layout: dict[str, Any],
    method: str,
    *,
    backfill_aliases: dict[str, list[str]],
) -> list[dict[str, Any]]:
    run_root = layout["run_root"]
    run_meta = load_json(run_root / "run_meta.json") or {}
    run_name = str(run_meta.get("run_name") or run_root.name)

    metric_dirs: list[Path] = layout["metric_dirs"]
    log_dirs: list[Path] = layout["log_dirs"]
    video_dirs: list[Path] = layout["video_dirs"]
    table_dirs: list[Path] = layout["table_dirs"]

    efficiency_path = find_file(metric_dirs, f"efficiency_{method}.json")
    fidelity_path = find_file(metric_dirs, f"fidelity_{method}.json")
    vbench_path = find_file(metric_dirs, f"vbench_{method}.json")
    drift_path = find_file(metric_dirs, f"drift_{method}.json")
    generation_path = find_file(log_dirs, f"generation_{method}.jsonl")
    vram_path = find_file(log_dirs, f"vram_trace_{method}.jsonl")

    efficiency = load_json(efficiency_path) if efficiency_path else None
    fidelity = load_json(fidelity_path) if fidelity_path else None
    vbench = load_json(vbench_path) if vbench_path else None
    drift = load_json(drift_path) if drift_path else None
    generation_count, generation_first = read_first_jsonl_record(generation_path) if generation_path else (0, None)

    video_count = max_moviegen_videos(video_dirs, method)
    total_frames = infer_moviegen_total_frames(efficiency, generation_first, run_meta)
    prompt_count = infer_moviegen_prompt_count(efficiency, generation_count, video_count)

    cache_policy = {}
    if efficiency and isinstance(efficiency.get("cache_policy"), dict):
        cache_policy = dict(efficiency["cache_policy"])
    cadence = cache_policy.get("cadence") if isinstance(cache_policy.get("cadence"), str) else None
    recent_blocks = safe_int(cache_policy.get("recent_blocks"))

    method_family = parse_method_family(method)
    bits = parse_bits(method)
    quant_meta = filter_quant_meta_for_method(method_family, extract_quant_meta(run_meta, None, efficiency))
    config_id, config_payload = build_config_identity(
        benchmark="moviegen",
        method=method,
        method_family=method_family,
        bits=bits,
        cadence=cadence,
        recent_blocks=recent_blocks,
        quant_meta=quant_meta,
    )

    missing: list[str] = []
    has_efficiency = efficiency is not None
    has_fidelity = fidelity is not None or method == "BF16"
    has_vbench = vbench is not None
    has_generation = generation_path is not None and generation_count > 0
    has_vram_trace = vram_path is not None
    has_videos = video_count > 0
    has_summary = any((table_dir / "baseline_summary.csv").exists() for table_dir in table_dirs)

    if not has_efficiency:
        missing.append("efficiency")
    if method != "BF16" and fidelity is None:
        missing.append("fidelity")
    if not has_vbench:
        missing.append("vbench")
    if not has_generation:
        missing.append("generation_log")
    if not has_vram_trace:
        missing.append("vram_trace")
    if not has_videos:
        missing.append("videos")
    dashboard_overview_ready = has_efficiency and has_vbench and has_fidelity
    dashboard_video_ready = has_videos
    dashboard_prompt_ready = has_generation and has_vram_trace
    dashboard_ready = dashboard_overview_ready and dashboard_video_ready
    is_ten_second = infer_is_ten_second_moviegen(run_name, total_frames)
    long10_dashboard_ready = is_ten_second and dashboard_ready and dashboard_prompt_ready

    fidelity_agg = fidelity.get("aggregate", {}) if isinstance(fidelity, dict) else {}
    base_record = {
        "source_repo": source,
        "results_root": str(results_root),
        "benchmark": "moviegen",
        "run_label": layout["run_label"],
        "run_root": str(run_root),
        "run_name": run_name,
        "method": method,
        "method_family": method_family,
        "bits": bits,
        "cadence": cadence,
        "recent_blocks": recent_blocks,
        "config_id": config_id,
        "config_payload": config_payload,
        "experiment_type": run_meta.get("experiment_type"),
        "quant_meta": json.dumps(quant_meta, sort_keys=True),
        "prompt_count": prompt_count,
        "video_count": video_count,
        "generation_records": generation_count,
        "has_vram_trace": has_vram_trace,
        "has_efficiency": efficiency is not None,
        "has_fidelity": fidelity is not None,
        "has_vbench": vbench is not None,
        "has_drift": drift is not None,
        "has_summary_csv": has_summary,
        "decoded_frames": total_frames,
        "approx_duration_sec": (float(total_frames) / 16.0) if total_frames is not None else None,
        "is_ten_second": is_ten_second,
        "dashboard_overview_ready": dashboard_overview_ready,
        "dashboard_video_ready": dashboard_video_ready,
        "dashboard_prompt_ready": dashboard_prompt_ready,
        "dashboard_ready": dashboard_ready,
        "long10_dashboard_ready": long10_dashboard_ready,
        "missing_requirements": ";".join(missing),
        "psnr": fidelity_agg.get("psnr"),
        "ssim": fidelity_agg.get("ssim"),
        "lpips": fidelity_agg.get("lpips"),
        "background_consistency": (vbench or {}).get("background_consistency", [None])[0] if isinstance((vbench or {}).get("background_consistency"), list) else (vbench or {}).get("background_consistency"),
        "imaging_quality": (vbench or {}).get("imaging_quality", [None])[0] if isinstance((vbench or {}).get("imaging_quality"), list) else (vbench or {}).get("imaging_quality"),
        "subject_consistency": (vbench or {}).get("subject_consistency", [None])[0] if isinstance((vbench or {}).get("subject_consistency"), list) else (vbench or {}).get("subject_consistency"),
        "aesthetic_quality": (vbench or {}).get("aesthetic_quality", [None])[0] if isinstance((vbench or {}).get("aesthetic_quality"), list) else (vbench or {}).get("aesthetic_quality"),
        "compression_ratio": (efficiency or {}).get("compression_ratio"),
        "total_runtime_s": (efficiency or {}).get("total_runtime_s"),
        "peak_vram_bytes": (efficiency or {}).get("peak_vram_bytes"),
    }
    config_ids = resolve_combined_backfill_config_ids(
        run_name,
        run_root,
        run_meta,
        config_id,
        backfill_aliases,
    )
    records: list[dict[str, Any]] = []
    for resolved_config_id in config_ids:
        record = dict(base_record)
        record["config_id"] = resolved_config_id
        records.append(record)
    return records


def build_storyeval_record(source: str, results_root: Path, run_root: Path) -> dict[str, Any]:
    method = parse_storyeval_method(run_root)
    summary = load_json(run_root / "summary" / "summary.json") or load_json(run_root / "summary" / "runner_summary.json") or {}
    config = load_json(run_root / "summary" / "config.json") or {}
    vbench = load_json(run_root / "metrics" / "vbench.json") or {}
    drift = load_json(run_root / "metrics" / "drift_imaging_quality.json") or {}
    per_prompt_dir = run_root / "per_prompt"
    per_prompt_files = sorted(per_prompt_dir.glob("*.json")) if per_prompt_dir.exists() else []
    first_record = load_json(per_prompt_files[0]) if per_prompt_files else None
    video_count = count_storyeval_videos(run_root / "videos")
    method_family = parse_method_family(method)
    bits = parse_bits(method)
    cache_policy = config.get("cache_policy", {}) if isinstance(config.get("cache_policy"), dict) else {}
    cadence = cache_policy.get("cadence") if isinstance(cache_policy.get("cadence"), str) else None
    recent_blocks = safe_int(cache_policy.get("recent_blocks"))
    quant_meta = extract_quant_meta({}, config, None)
    quant_meta = filter_quant_meta_for_method(method_family, quant_meta)
    config_id, config_payload = build_config_identity(
        benchmark="storyeval",
        method=method,
        method_family=method_family,
        bits=bits,
        cadence=cadence,
        recent_blocks=recent_blocks,
        quant_meta=quant_meta,
    )

    has_summary = bool(summary)
    has_config = bool(config)
    has_vbench = bool(vbench)
    has_drift = bool(drift)
    has_records = len(per_prompt_files) > 0
    has_videos = video_count > 0
    has_vram_trace = (run_root / "logs" / "vram_trace_storyeval.jsonl").exists()
    is_ten_second = infer_is_ten_second_storyeval(config, summary, first_record)

    missing: list[str] = []
    if not has_summary:
        missing.append("summary")
    if not has_config:
        missing.append("config")
    if not has_vbench:
        missing.append("vbench")
    if not has_drift:
        missing.append("drift")
    if not has_records:
        missing.append("per_prompt")
    if not has_videos:
        missing.append("videos")
    if not has_vram_trace:
        missing.append("vram_trace")

    prompt_count = safe_int(summary.get("num_prompts")) or safe_int(config.get("num_prompts_selected")) or len(per_prompt_files)
    return {
        "source_repo": source,
        "results_root": str(results_root),
        "benchmark": "storyeval",
        "run_label": f"storyeval/{run_root.name}",
        "run_root": str(run_root),
        "run_name": run_root.name,
        "method": method,
        "method_family": method_family,
        "bits": bits,
        "cadence": cadence,
        "recent_blocks": recent_blocks,
        "config_id": config_id,
        "config_payload": config_payload,
        "experiment_type": "storyeval",
        "quant_meta": json.dumps(quant_meta, sort_keys=True),
        "prompt_count": prompt_count,
        "video_count": video_count,
        "generation_records": len(per_prompt_files),
        "has_vram_trace": has_vram_trace,
        "has_efficiency": False,
        "has_fidelity": False,
        "has_vbench": has_vbench,
        "has_drift": has_drift,
        "has_summary_csv": False,
        "decoded_frames": safe_int(config.get("target_frames")) or safe_int(first_record.get("target_frames") if first_record else None),
        "approx_duration_sec": safe_float(config.get("effective_duration_sec")) or safe_float(config.get("duration_sec_requested")) or safe_float(first_record.get("effective_duration_sec") if first_record else None),
        "is_ten_second": is_ten_second,
        "dashboard_overview_ready": has_summary and has_vbench,
        "dashboard_video_ready": has_videos,
        "dashboard_prompt_ready": has_records and has_vram_trace,
        "dashboard_ready": has_summary and has_config and has_vbench and has_drift and has_records and has_videos,
        "long10_dashboard_ready": is_ten_second and has_summary and has_config and has_vbench and has_drift and has_records and has_videos,
        "missing_requirements": ";".join(missing),
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "background_consistency": summary.get("vbench_background_consistency"),
        "imaging_quality": summary.get("vbench_imaging_quality"),
        "subject_consistency": summary.get("vbench_subject_consistency"),
        "aesthetic_quality": summary.get("vbench_aesthetic_quality"),
        "compression_ratio": None,
        "total_runtime_s": summary.get("avg_runtime_sec"),
        "peak_vram_bytes": None,
    }


def discover_records(source: str, results_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    backfill_aliases = load_backfill_aliases()
    for layout in infer_moviegen_run_layouts(results_root):
        for method in sorted(list_moviegen_methods(layout)):
            records.extend(
                build_moviegen_record(
                    source,
                    results_root,
                    layout,
                    method,
                    backfill_aliases=backfill_aliases,
                )
            )
    for run_root in infer_storyeval_runs(results_root):
        records.append(build_storyeval_record(source, results_root, run_root))
    return records


def config_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("benchmark") or "",
        record.get("method_family") or "",
        record.get("method") or "",
        record.get("cadence") or "",
        record.get("recent_blocks") if record.get("recent_blocks") is not None else -1,
        record.get("source_repo") or "",
        record.get("run_label") or "",
    )


def aggregate_configs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        group = grouped.setdefault(
            record["config_id"],
            {
                "config_id": record["config_id"],
                "benchmark": record["benchmark"],
                "method": record["method"],
                "method_family": record["method_family"],
                "bits": record["bits"],
                "cadence": record["cadence"],
                "recent_blocks": record["recent_blocks"],
                "config_payload": record["config_payload"],
                "sources": set(),
                "run_labels": set(),
                "moviegen_short_runs": 0,
                "moviegen_ten_second_runs": 0,
                "storyeval_ten_second_runs": 0,
                "dashboard_ready_runs": 0,
                "long10_dashboard_ready_runs": 0,
                "representative_quant_meta": record["quant_meta"],
                "missing_reasons": set(),
            },
        )
        group["sources"].add(record["source_repo"])
        group["run_labels"].add(f"{record['source_repo']}::{record['run_label']}")
        if record["benchmark"] == "moviegen":
            if record["is_ten_second"]:
                group["moviegen_ten_second_runs"] += 1
            else:
                group["moviegen_short_runs"] += 1
        if record["benchmark"] == "storyeval" and record["is_ten_second"]:
            group["storyeval_ten_second_runs"] += 1
        if record["dashboard_ready"]:
            group["dashboard_ready_runs"] += 1
        if record["long10_dashboard_ready"]:
            group["long10_dashboard_ready_runs"] += 1
        for token in (record.get("missing_requirements") or "").split(";"):
            if token:
                group["missing_reasons"].add(token)

    rows: list[dict[str, Any]] = []
    for group in grouped.values():
        group["sources"] = ",".join(sorted(group["sources"]))
        group["run_labels"] = ",".join(sorted(group["run_labels"]))
        group["missing_reasons"] = ";".join(sorted(group["missing_reasons"]))
        group["needs_moviegen_10s_backfill"] = (
            group["benchmark"] == "moviegen"
            and group["moviegen_short_runs"] > 0
            and group["moviegen_ten_second_runs"] == 0
        )
        rows.append(group)
    return sorted(rows, key=lambda row: (row["benchmark"], row["method_family"], row["method"], row["config_id"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cleaned: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, Path):
                cleaned[key] = str(value)
            else:
                cleaned[key] = value
        out.append(cleaned)
    return out


def build_summary_markdown(records: list[dict[str, Any]], configs: list[dict[str, Any]]) -> str:
    moviegen_records = [row for row in records if row["benchmark"] == "moviegen"]
    storyeval_records = [row for row in records if row["benchmark"] == "storyeval"]
    gaps = [row for row in records if row.get("missing_requirements")]
    missing_long10 = [row for row in configs if row["needs_moviegen_10s_backfill"]]

    lines = [
        "# Combined KV-Quant Registry Audit",
        "",
        "## Dashboard-required artifact interpretation",
        "",
        "### MovieGen",
        "- overview table: `efficiency`, `vbench`, and `fidelity` (except BF16, which is the reference)",
        "- video explorer: method video directory with `prompt_*_seed_*.mp4`",
        "- prompt analytics: `generation_<method>.jsonl` plus `vram_trace_<method>.jsonl`",
        "- optional long-horizon extension: `drift_<method>.json`",
        "",
        "### StoryEval",
        "- overview: `summary/summary.json` (or `runner_summary.json`), `summary/config.json`, `metrics/vbench.json`, `metrics/drift_imaging_quality.json`",
        "- video explorer: `videos/*.mp4` plus `per_prompt/*.json`",
        "- prompt analytics: `per_prompt/*.json` plus `logs/vram_trace_storyeval.jsonl`",
        "",
        "## Audit counts",
        f"- MovieGen method-runs: **{len(moviegen_records)}**",
        f"- StoryEval method-runs: **{len(storyeval_records)}**",
        f"- Unique quantization configurations: **{len(configs)}**",
        f"- Runs with dashboard gaps: **{len(gaps)}**",
        f"- Configurations missing any 10-second MovieGen run: **{len(missing_long10)}**",
        "",
        "## Configurations missing 10-second MovieGen coverage",
        "",
    ]

    if not missing_long10:
        lines.append("- None")
    else:
        lines.append("| method | family | cadence | recent_blocks | sources | short_runs | missing |")
        lines.append("|---|---|---|---:|---|---:|---|")
        for row in missing_long10:
            lines.append(
                f"| {row['method']} | {row['method_family']} | {row.get('cadence') or ''} | "
                f"{'' if row.get('recent_blocks') is None else row.get('recent_blocks')} | {row['sources']} | "
                f"{row['moviegen_short_runs']} | {row['missing_reasons']} |"
            )

    lines.extend(
        [
            "",
            "## Runs with missing dashboard requirements",
            "",
        ]
    )
    if not gaps:
        lines.append("- None")
    else:
        lines.append("| source | benchmark | run | method | long10 | missing |")
        lines.append("|---|---|---|---|---|---|")
        for row in sorted(gaps, key=config_sort_key):
            lines.append(
                f"| {row['source_repo']} | {row['benchmark']} | {row['run_label']} | {row['method']} | "
                f"{'yes' if row['is_ten_second'] else 'no'} | {row['missing_requirements']} |"
            )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cross-repo KV quantization runs and dashboard completeness.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Additional source in the form name=/absolute/path/to/results",
    )
    parser.add_argument(
        "--include-combined",
        action="store_true",
        help="Also scan this workspace's results directory as a third source named 'combined'.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "results" / "combined" / "registry",
        help="Directory where registry CSV/JSON/Markdown outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = dict(DEFAULT_SOURCES)
    if args.include_combined:
        sources["combined"] = REPO_ROOT / "results"
    for item in args.source:
        if "=" not in item:
            raise ValueError(f"Invalid --source value: {item!r}")
        name, raw_path = item.split("=", 1)
        sources[name] = Path(raw_path)

    records: list[dict[str, Any]] = []
    for source_name, results_root in sources.items():
        if not results_root.exists():
            continue
        records.extend(discover_records(source_name, results_root))

    records = sorted(records, key=config_sort_key)
    configs = aggregate_configs(records)
    gaps = [row for row in records if row.get("missing_requirements")]
    missing_long10 = [row for row in configs if row["needs_moviegen_10s_backfill"]]

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    registry_json = out_root / "combined_registry.json"
    registry_csv = out_root / "combined_registry.csv"
    configs_json = out_root / "unique_configurations.json"
    configs_csv = out_root / "unique_configurations.csv"
    gaps_csv = out_root / "dashboard_gaps.csv"
    missing_csv = out_root / "missing_moviegen_10s.csv"
    summary_md = out_root / "registry_summary.md"

    registry_json.write_text(json.dumps(json_ready_rows(records), indent=2), encoding="utf-8")
    write_csv(registry_csv, records)
    configs_json.write_text(json.dumps(json_ready_rows(configs), indent=2), encoding="utf-8")
    write_csv(configs_csv, configs)
    write_csv(gaps_csv, gaps)
    write_csv(missing_csv, missing_long10)
    summary_md.write_text(build_summary_markdown(records, configs), encoding="utf-8")

    print(f"Wrote {registry_json}")
    print(f"Wrote {registry_csv}")
    print(f"Wrote {configs_json}")
    print(f"Wrote {configs_csv}")
    print(f"Wrote {gaps_csv}")
    print(f"Wrote {missing_csv}")
    print(f"Wrote {summary_md}")
    print(f"Records: {len(records)} | Unique configs: {len(configs)} | Missing 10s MovieGen configs: {len(missing_long10)}")


if __name__ == "__main__":
    main()
