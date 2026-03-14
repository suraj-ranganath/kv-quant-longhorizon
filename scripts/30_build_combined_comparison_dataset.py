#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backfill_paths import metric_backfill_path

COMBINED_RESULTS_ROOT = REPO_ROOT / "results"
SURAJ_RESULTS_ROOT = Path("/data/suraj/kv-quant-longhorizon/results")
REGISTRY_DIR = COMBINED_RESULTS_ROOT / "combined" / "registry"
DEFAULT_BACKFILL_MANIFEST = REGISTRY_DIR / "backfill_manifest.json"
DEFAULT_STORYEVAL_MANIFEST = REGISTRY_DIR / "storyeval_manifest.json"
DEFAULT_UNIQUE_CONFIGS = REGISTRY_DIR / "unique_configurations.json"
DEFAULT_OUTPUT_DIR = COMBINED_RESULTS_ROOT / "combined"

VIDEO_RE = re.compile(r"^prompt_(\d+)_seed_(\d+)\.mp4$")
BASELINE_METHODS = {"BF16", "RTN_INT4", "QUAROT_KV_INT4"}
CANONICAL_MOVIEGEN_RUN = "1773110004_presentation_moviegen_fullmatrix"
ID_COLUMNS = {
    "comparison_key",
    "source_user",
    "source_repo",
    "source_results_root",
    "benchmark",
    "run_label",
    "run_root",
    "run_name",
    "method",
    "method_display",
    "method_family",
    "config_id",
    "config_ids",
    "config_payload_json",
    "quant_meta_json",
    "prompt_id",
    "prompt_index",
    "prompt",
    "seed",
    "video_name",
    "video_path",
    "video_rel_path",
}
MOVIEGEN_REQUIRED_FIELDS = [
    "avg_runtime_s_per_prompt",
    "compression_ratio",
    "moviegen_fidelity_psnr_agg",
    "moviegen_fidelity_ssim_agg",
    "moviegen_fidelity_lpips_agg",
    "moviegen_background_consistency_agg",
    "moviegen_imaging_quality_agg",
    "moviegen_subject_consistency_agg",
    "moviegen_aesthetic_quality_agg",
]
STORYEVAL_REQUIRED_FIELDS = [
    "storyeval_avg_runtime_sec",
    "storyeval_avg_peak_vram_mb",
    "storyeval_background_consistency_agg",
    "storyeval_imaging_quality_agg",
    "storyeval_subject_consistency_agg",
    "storyeval_aesthetic_quality_agg",
    "storyeval_drift_points",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def resolve_metric_json_path(
    *,
    benchmark: str,
    metric_kind: str,
    run_root: Path,
    method: str,
    run_local_path: Path | None = None,
) -> Path | None:
    if run_local_path is not None and run_local_path.exists():
        return run_local_path
    backfill_path = metric_backfill_path(
        benchmark=benchmark,
        metric_kind=metric_kind,
        run_root=run_root,
        method=method,
    )
    if backfill_path.exists():
        return backfill_path
    return run_local_path if run_local_path is not None and run_local_path.exists() else None


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def has_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def method_family_from_name(method: str) -> str:
    if method == "BF16":
        return "BF16"
    if method.startswith("SPATIAL_MIXED"):
        return "SPATIAL_MIXED"
    for prefix in (
        "FLOWCACHE_NATIVE_SOFT_PRUNE",
        "FLOWCACHE_NATIVE",
        "FLOWCACHE_SOFT_PRUNE",
        "FLOWCACHE_ADAPTIVE",
        "FLOWCACHE_HYBRID",
        "FLOWCACHE_PRUNE",
        "QUAROT_KV",
        "AGE_TIER",
        "KIVI",
        "RTN",
        "PRQ",
        "QAQ",
        "TPTQ",
    ):
        if method.startswith(prefix):
            return prefix
    return method


def is_ten_second_duration(duration_sec: float | None, run_name: str | None) -> bool:
    if duration_sec is not None and duration_sec >= 10.0:
        return True
    if isinstance(run_name, str) and "10s" in run_name.lower():
        return True
    return False


def build_config_maps(unique_configs_path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    rows = load_json(unique_configs_path)
    by_config_id = {str(row["config_id"]): row for row in rows if row.get("config_id") is not None}
    counts = Counter(str(row["method"]) for row in rows if row.get("sources") and "combined" in str(row["sources"]))
    return by_config_id, counts


def build_manifest_maps(manifest_path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[str]]:
    rows = load_json(manifest_path)
    by_run_method: dict[tuple[str, str], dict[str, Any]] = {}
    counts = Counter()
    for row in rows:
        run_root = str((REPO_ROOT / row["run_root"]).resolve())
        method = str(row["method"])
        by_run_method[(run_root, method)] = row
        counts[method] += 1
    return by_run_method, counts


def parse_moviegen_vbench(vbench: dict[str, Any] | None) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    if not isinstance(vbench, dict):
        return {}, {}
    aggregate = {
        "background_consistency": safe_float((vbench.get("background_consistency") or [None])[0] if isinstance(vbench.get("background_consistency"), list) else vbench.get("background_consistency")),
        "imaging_quality": safe_float((vbench.get("imaging_quality") or [None])[0] if isinstance(vbench.get("imaging_quality"), list) else vbench.get("imaging_quality")),
        "subject_consistency": safe_float((vbench.get("subject_consistency") or [None])[0] if isinstance(vbench.get("subject_consistency"), list) else vbench.get("subject_consistency")),
        "aesthetic_quality": safe_float((vbench.get("aesthetic_quality") or [None])[0] if isinstance(vbench.get("aesthetic_quality"), list) else vbench.get("aesthetic_quality")),
    }
    per_video: dict[str, dict[str, Any]] = {}
    for metric in ("background_consistency", "imaging_quality", "subject_consistency", "aesthetic_quality"):
        payload = vbench.get(metric)
        details = payload[1] if isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], list) else []
        for item in details:
            video_name = Path(item["video_path"]).name
            row = per_video.setdefault(video_name, {})
            row[metric] = item.get("video_results")
    return aggregate, per_video


def parse_moviegen_fidelity(fidelity: dict[str, Any] | None) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    if not isinstance(fidelity, dict):
        return {}, {}
    aggregate_payload = fidelity.get("aggregate", {})
    aggregate = {
        "psnr": safe_float(aggregate_payload.get("psnr")) if isinstance(aggregate_payload, dict) else None,
        "ssim": safe_float(aggregate_payload.get("ssim")) if isinstance(aggregate_payload, dict) else None,
        "lpips": safe_float(aggregate_payload.get("lpips")) if isinstance(aggregate_payload, dict) else None,
    }
    per_video: dict[str, dict[str, Any]] = {}
    for item in fidelity.get("per_video", []) if isinstance(fidelity.get("per_video"), list) else []:
        video_name = str(item.get("video"))
        per_video[video_name] = {
            "psnr": safe_float(item.get("psnr")),
            "ssim": safe_float(item.get("ssim")),
            "lpips": safe_float(item.get("lpips")),
            "num_frames": safe_int(item.get("num_frames")),
            "resolution": item.get("resolution"),
        }
    return aggregate, per_video


def parse_storyeval_system_metrics(payload: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not isinstance(payload, dict):
        return {}, {}
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    per_video = payload.get("per_video") if isinstance(payload.get("per_video"), dict) else {}
    return aggregate, per_video


def parse_moviegen_drift(drift: dict[str, Any] | None) -> tuple[int | None, float | None]:
    if not isinstance(drift, dict):
        return None, None
    curve = drift.get("curve", [])
    if not isinstance(curve, list) or not curve:
        return 0, None
    last = curve[-1].get("imaging_quality")
    if isinstance(last, list) and last:
        last = last[0]
    return len(curve), safe_float(last)


def method_display_name(method: str, config_id: str | None, *, duplicate_counts: Counter[str]) -> str:
    if config_id and duplicate_counts.get(method, 0) > 1:
        return f"{method} [{config_id}]"
    return method


def list_methods_in_run(run_root: Path) -> list[str]:
    methods: set[str] = set()
    metrics_dir = run_root / "metrics"
    if metrics_dir.exists():
        for prefix in ("efficiency", "fidelity", "vbench", "drift"):
            for path in metrics_dir.glob(f"{prefix}_*.json"):
                methods.add(path.stem[len(prefix) + 1 :])
    videos_dir = run_root / "videos"
    if videos_dir.exists():
        for sub in videos_dir.iterdir():
            if sub.is_dir():
                methods.add(sub.name)
    return sorted(methods)


def generation_video_prompt_seed(video_name: str) -> tuple[int | None, int | None]:
    match = VIDEO_RE.match(video_name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def build_moviegen_rows(
    *,
    run_root: Path,
    run_label: str,
    source_user: str,
    source_repo: str,
    duplicate_counts: Counter[str],
    manifest_map: dict[tuple[str, str], dict[str, Any]],
    config_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_root_resolved = str(run_root.resolve())
    run_name = run_root.name if run_root.name else run_label
    run_meta = load_json(run_root / "run_meta.json") if (run_root / "run_meta.json").exists() else {}
    methods = list_methods_in_run(run_root)
    for method in methods:
        manifest_row = manifest_map.get((run_root_resolved, method), {})
        config_id = manifest_row.get("primary_config_id")
        config_meta = config_map.get(str(config_id), {}) if config_id is not None else {}
        config_payload = config_meta.get("config_payload")
        representative_quant_meta = config_meta.get("representative_quant_meta")
        display_method = method_display_name(method, str(config_id) if config_id else None, duplicate_counts=duplicate_counts)

        generation_records = load_jsonl(run_root / "logs" / f"generation_{method}.jsonl")
        efficiency = load_json_if_exists(run_root / "metrics" / f"efficiency_{method}.json")
        fidelity = load_json_if_exists(
            resolve_metric_json_path(
                benchmark="moviegen",
                metric_kind="fidelity",
                run_root=run_root,
                method=method,
                run_local_path=run_root / "metrics" / f"fidelity_{method}.json",
            )
        )
        vbench = load_json_if_exists(
            resolve_metric_json_path(
                benchmark="moviegen",
                metric_kind="vbench",
                run_root=run_root,
                method=method,
                run_local_path=run_root / "metrics" / f"vbench_{method}.json",
            )
        )
        drift = load_json_if_exists(
            resolve_metric_json_path(
                benchmark="moviegen",
                metric_kind="drift",
                run_root=run_root,
                method=method,
                run_local_path=run_root / "metrics" / f"drift_{method}.json",
            )
        )
        vbench_agg, vbench_per_video = parse_moviegen_vbench(vbench)
        fidelity_agg, fidelity_per_video = parse_moviegen_fidelity(fidelity)
        drift_points, drift_last = parse_moviegen_drift(drift)

        for record in generation_records:
            video_rel = str(record.get("output_video"))
            video_path = (REPO_ROOT / video_rel).resolve()
            video_name = video_path.name
            prompt_idx = safe_int(record.get("prompt_id"))
            _parsed_prompt_idx, parsed_seed = generation_video_prompt_seed(video_name)
            resolution = record.get("resolution")
            fidelity_video = fidelity_per_video.get(video_name, {})
            vbench_video = vbench_per_video.get(video_name, {})
            total_frames = safe_int(record.get("total_frames")) or safe_int(fidelity_video.get("num_frames"))
            fps = 16
            duration_sec = (float(total_frames) / float(fps)) if total_frames else None
            rows.append(
                {
                    "comparison_key": f"{source_user}:moviegen:{run_root_resolved}:{display_method}:{prompt_idx}:{parsed_seed}",
                    "source_user": source_user,
                    "source_repo": source_repo,
                    "source_results_root": str(run_root.parents[1] if run_root.parent.name == 'runs' else run_root.parent),
                    "benchmark": "moviegen",
                    "run_label": run_label,
                    "run_root": run_root_resolved,
                    "run_name": run_name,
                    "method": method,
                    "method_display": display_method,
                    "method_family": str(config_meta.get("method_family") or manifest_row.get("method_family") or method_family_from_name(method)),
                    "config_id": str(config_id) if config_id else None,
                    "config_ids": ",".join(str(x) for x in manifest_row.get("config_ids", []) or []),
                    "config_payload_json": config_payload,
                    "quant_meta_json": representative_quant_meta,
                    "prompt_id": str(prompt_idx) if prompt_idx is not None else None,
                    "prompt_index": prompt_idx,
                    "prompt": record.get("prompt"),
                    "seed": parsed_seed,
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "video_rel_path": video_rel,
                    "resolution_h": resolution[0] if isinstance(resolution, list) and len(resolution) > 0 else None,
                    "resolution_w": resolution[1] if isinstance(resolution, list) and len(resolution) > 1 else None,
                    "num_frames": total_frames,
                    "fps": fps,
                    "duration_sec": duration_sec,
                    "is_ten_second": is_ten_second_duration(duration_sec, run_name),
                    "wall_time_sec": safe_float(record.get("wall_clock_runtime_s")),
                    "peak_vram_bytes": safe_int(record.get("peak_vram_bytes")),
                    "peak_vram_mb": (safe_float(record.get("peak_vram_bytes")) / (1024.0 * 1024.0)) if record.get("peak_vram_bytes") is not None else None,
                    "compression_ratio": safe_float(efficiency.get("compression_ratio")),
                    "bf16_kv_bytes": safe_int(efficiency.get("bf16_kv_bytes")),
                    "compressed_kv_bytes": safe_int(efficiency.get("compressed_kv_bytes")),
                    "quantize_time_s": safe_float(efficiency.get("quantize_time_s")),
                    "dequantize_time_s": safe_float(efficiency.get("dequantize_time_s")),
                    "total_runtime_s": safe_float(efficiency.get("total_runtime_s")),
                    "avg_runtime_s_per_prompt": safe_float(efficiency.get("avg_runtime_s_per_prompt")),
                    "moviegen_fidelity_psnr": fidelity_video.get("psnr"),
                    "moviegen_fidelity_ssim": fidelity_video.get("ssim"),
                    "moviegen_fidelity_lpips": fidelity_video.get("lpips"),
                    "moviegen_fidelity_psnr_agg": fidelity_agg.get("psnr"),
                    "moviegen_fidelity_ssim_agg": fidelity_agg.get("ssim"),
                    "moviegen_fidelity_lpips_agg": fidelity_agg.get("lpips"),
                    "moviegen_background_consistency": safe_float(vbench_video.get("background_consistency")),
                    "moviegen_imaging_quality": safe_float(vbench_video.get("imaging_quality")),
                    "moviegen_subject_consistency": safe_float(vbench_video.get("subject_consistency")),
                    "moviegen_aesthetic_quality": safe_float(vbench_video.get("aesthetic_quality")),
                    "moviegen_background_consistency_agg": vbench_agg.get("background_consistency"),
                    "moviegen_imaging_quality_agg": vbench_agg.get("imaging_quality"),
                    "moviegen_subject_consistency_agg": vbench_agg.get("subject_consistency"),
                    "moviegen_aesthetic_quality_agg": vbench_agg.get("aesthetic_quality"),
                    "moviegen_drift_points": drift_points,
                    "moviegen_drift_last_imaging_quality": drift_last,
                    "source_moviegen_run_root": str(run_meta.get("linked_baseline_run_root") or manifest_row.get("source_moviegen_run_root") or ""),
                }
            )
    return rows


def collect_suraj_moviegen_run_roots(results_root: Path) -> list[tuple[Path, str]]:
    runs: list[tuple[Path, str]] = []
    if any((results_root / sub).exists() for sub in ("videos", "metrics", "logs")):
        runs.append((results_root, "legacy_root"))
    runs_root = results_root / "runs"
    if runs_root.exists():
        for path in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
            runs.append((path, f"runs/{path.name}"))
    archive_root = results_root / "archive"
    if archive_root.exists():
        for path in sorted([p for p in archive_root.iterdir() if p.is_dir()]):
            runs.append((path, f"archive/{path.name}"))
    return runs


def storyeval_method_from_root(root: Path, config: dict[str, Any], run_meta: dict[str, Any]) -> str:
    method = config.get("method") if isinstance(config, dict) else None
    if isinstance(method, str) and method:
        return method
    method = run_meta.get("method") if isinstance(run_meta, dict) else None
    if isinstance(method, str) and method:
        return method
    match = re.match(r"^storyeval_(.+?)(?:_[0-9a-f]{12})?_10prompts_10s.*$", root.name)
    if match:
        return match.group(1).upper()
    return root.name


def build_storyeval_rows(
    *,
    run_root: Path,
    source_user: str,
    source_repo: str,
    duplicate_counts: Counter[str],
    manifest_map: dict[tuple[str, str], dict[str, Any]],
    config_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_root_resolved = str(run_root.resolve())
    config = load_json(run_root / "summary" / "config.json") if (run_root / "summary" / "config.json").exists() else {}
    summary = None
    for candidate in (run_root / "summary" / "summary.json", run_root / "summary" / "runner_summary.json"):
        if candidate.exists():
            summary = load_json(candidate)
            break
    if summary is None:
        summary = {}
    vbench = load_json_if_exists(run_root / "metrics" / "vbench.json")
    run_meta = load_json(run_root / "run_meta.json") if (run_root / "run_meta.json").exists() else {}
    method = storyeval_method_from_root(run_root, config, run_meta)
    drift = load_json_if_exists(
        resolve_metric_json_path(
            benchmark="storyeval",
            metric_kind="drift",
            run_root=run_root,
            method=method,
            run_local_path=run_root / "metrics" / "drift_imaging_quality.json",
        )
    )
    config_id = config.get("config_id") or run_meta.get("config_id")
    config_meta = config_map.get(str(config_id), {}) if config_id else {}
    display_method = method_display_name(method, str(config_id) if config_id else None, duplicate_counts=duplicate_counts)
    manifest_row = manifest_map.get((run_root_resolved, method), {})
    storyeval_system_metrics = load_json_if_exists(
        resolve_metric_json_path(
            benchmark="storyeval",
            metric_kind="storyeval_system",
            run_root=run_root,
            method=method,
            run_local_path=None,
        )
    )
    storyeval_fidelity = load_json_if_exists(
        resolve_metric_json_path(
            benchmark="storyeval",
            metric_kind="fidelity",
            run_root=run_root,
            method=method,
            run_local_path=run_root / "metrics" / f"fidelity_{method}.json",
        )
    )
    vbench_agg = vbench.get("aggregate", {}) if isinstance(vbench.get("aggregate"), dict) else {}
    per_video = vbench.get("per_video", {}) if isinstance(vbench.get("per_video"), dict) else {}
    fidelity_agg, fidelity_per_video = parse_moviegen_fidelity(storyeval_fidelity)
    system_agg, system_per_video = parse_storyeval_system_metrics(storyeval_system_metrics)
    drift_curve = drift.get("curve", []) if isinstance(drift.get("curve"), list) else []
    drift_last = safe_float(drift_curve[-1].get("imaging_quality")) if drift_curve else None
    per_prompt_dir = run_root / "per_prompt"
    for path in sorted(per_prompt_dir.glob("*.json")):
        record = load_json(path)
        video_rel = record.get("generated_video_path")
        video_path = (REPO_ROOT / video_rel).resolve() if isinstance(video_rel, str) else None
        video_name = video_path.name if video_path is not None else (Path(video_rel).name if isinstance(video_rel, str) and video_rel else None)
        resolution = record.get("resolution")
        video_metrics = per_video.get(video_name, {}) if video_name else {}
        fidelity_video = fidelity_per_video.get(video_name, {}) if video_name else {}
        system_video = system_per_video.get(video_name, {}) if video_name else {}
        rows.append(
            {
                "comparison_key": f"{source_user}:storyeval:{run_root_resolved}:{display_method}:{record.get('prompt_id')}:{record.get('seed')}",
                "source_user": source_user,
                "source_repo": source_repo,
                "source_results_root": str(run_root.parents[2] if run_root.parent.name == "storyeval" else run_root.parent),
                "benchmark": "storyeval",
                "run_label": f"storyeval/{run_root.name}",
                "run_root": run_root_resolved,
                "run_name": run_root.name,
                "method": method,
                "method_display": display_method,
                "method_family": str(config_meta.get("method_family") or manifest_row.get("method_family") or method_family_from_name(method)),
                "config_id": str(config_id) if config_id else None,
                "config_ids": ",".join(str(x) for x in manifest_row.get("config_ids", []) or []),
                "config_payload_json": config_meta.get("config_payload"),
                "quant_meta_json": config_meta.get("representative_quant_meta"),
                "prompt_id": str(record.get("prompt_id")) if record.get("prompt_id") is not None else None,
                "prompt_index": safe_int(record.get("line_index")),
                "prompt": record.get("prompt"),
                "seed": safe_int(record.get("seed")),
                "video_name": video_name,
                "video_path": str(video_path) if video_path is not None else None,
                "video_rel_path": video_rel,
                "resolution_h": resolution[0] if isinstance(resolution, list) and len(resolution) > 0 else None,
                "resolution_w": resolution[1] if isinstance(resolution, list) and len(resolution) > 1 else None,
                "num_frames": safe_int(record.get("total_frames")),
                "fps": safe_int(record.get("fps")),
                "duration_sec": safe_float(record.get("effective_duration_sec")),
                "is_ten_second": is_ten_second_duration(safe_float(record.get("effective_duration_sec")), run_root.name),
                "wall_time_sec": first_non_null(
                    safe_float(record.get("wall_time_sec")),
                    safe_float(system_video.get("wall_time_sec")),
                ),
                "peak_vram_bytes": first_non_null(
                    safe_int(record.get("peak_vram_bytes")),
                    safe_int(system_video.get("peak_vram_bytes")),
                ),
                "peak_vram_mb": first_non_null(
                    safe_float(record.get("peak_vram_mb")),
                    safe_float(system_video.get("peak_vram_mb")),
                ),
                "compression_ratio": first_non_null(
                    safe_float(system_video.get("compression_ratio")),
                    safe_float(system_agg.get("compression_ratio")),
                ),
                "bf16_kv_bytes": first_non_null(
                    safe_int(system_video.get("bf16_kv_bytes")),
                    safe_int(system_agg.get("bf16_kv_bytes")),
                ),
                "compressed_kv_bytes": first_non_null(
                    safe_int(system_video.get("compressed_kv_bytes")),
                    safe_int(system_agg.get("compressed_kv_bytes")),
                ),
                "quantize_time_s": safe_float(system_agg.get("quantize_time_s")),
                "dequantize_time_s": safe_float(system_agg.get("dequantize_time_s")),
                "total_runtime_s": safe_float(system_agg.get("total_runtime_s")),
                "avg_runtime_s_per_prompt": safe_float(system_agg.get("avg_runtime_s_per_prompt")),
                "storyeval_fidelity_psnr": fidelity_video.get("psnr"),
                "storyeval_fidelity_ssim": fidelity_video.get("ssim"),
                "storyeval_fidelity_lpips": fidelity_video.get("lpips"),
                "storyeval_fidelity_psnr_agg": fidelity_agg.get("psnr"),
                "storyeval_fidelity_ssim_agg": fidelity_agg.get("ssim"),
                "storyeval_fidelity_lpips_agg": fidelity_agg.get("lpips"),
                "storyeval_background_consistency": safe_float(video_metrics.get("background_consistency")),
                "storyeval_imaging_quality": safe_float(video_metrics.get("imaging_quality")),
                "storyeval_subject_consistency": safe_float(video_metrics.get("subject_consistency")),
                "storyeval_aesthetic_quality": safe_float(video_metrics.get("aesthetic_quality")),
                "storyeval_background_consistency_agg": safe_float(vbench_agg.get("background_consistency")),
                "storyeval_imaging_quality_agg": safe_float(vbench_agg.get("imaging_quality")),
                "storyeval_subject_consistency_agg": safe_float(vbench_agg.get("subject_consistency")),
                "storyeval_aesthetic_quality_agg": safe_float(vbench_agg.get("aesthetic_quality")),
                "storyeval_avg_runtime_sec": first_non_null(
                    safe_float(summary.get("avg_runtime_sec")),
                    safe_float(system_agg.get("avg_runtime_s_per_prompt")),
                ),
                "storyeval_avg_peak_vram_mb": first_non_null(
                    safe_float(summary.get("avg_peak_vram_mb")),
                    safe_float(system_agg.get("avg_peak_vram_mb")),
                ),
                "storyeval_max_peak_vram_mb": first_non_null(
                    safe_float(summary.get("max_peak_vram_mb")),
                    safe_float(system_agg.get("max_peak_vram_mb")),
                ),
                "storyeval_num_records": safe_int(summary.get("num_records")),
                "storyeval_num_prompts": safe_int(summary.get("num_prompts")),
                "storyeval_num_success": safe_int(summary.get("num_success")),
                "storyeval_num_failed": safe_int(summary.get("num_failed")),
                "storyeval_drift_points": len(drift_curve),
                "storyeval_drift_last_imaging_quality": drift_last,
                "source_moviegen_run_root": config.get("source_moviegen_run_root") or run_meta.get("source_moviegen_run_root"),
            }
        )
    return rows


def build_gap_report(
    *,
    dataset_rows: list[dict[str, Any]],
    storyeval_manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for row in storyeval_manifest:
        run_root = (REPO_ROOT / row["run_root"]).resolve()
        missing: list[str] = []
        if not run_root.exists():
            missing.append("run_root")
        else:
            if not (run_root / "per_prompt").exists() or not any((run_root / "per_prompt").glob("*.json")):
                missing.append("per_prompt")
            if not (run_root / "metrics" / "vbench.json").exists():
                missing.append("vbench")
            if not (run_root / "metrics" / "drift_imaging_quality.json").exists():
                missing.append("drift")
            if not (run_root / "summary" / "summary.json").exists():
                missing.append("summary")
        if missing:
            gaps.append(
                {
                    "scope": "storyeval_parity",
                    "source_user": "vaishak",
                    "benchmark": "storyeval",
                    "method": row["method"],
                    "config_id": row.get("primary_config_id"),
                    "run_root": str(run_root),
                    "issue": "missing_storyeval_artifacts",
                    "details": ";".join(missing),
                }
            )

    for row in dataset_rows:
        required = []
        if not row.get("benchmark"):
            required.append("benchmark")
        if not row.get("video_path"):
            required.append("video_path")
        if row.get("wall_time_sec") is None:
            required.append("wall_time_sec")
        if row.get("peak_vram_bytes") is None:
            required.append("peak_vram_bytes")
        if required:
            gaps.append(
                {
                    "scope": "dataset_row",
                    "source_user": row.get("source_user"),
                    "benchmark": row.get("benchmark"),
                    "method": row.get("method"),
                    "config_id": row.get("config_id"),
                    "run_root": row.get("run_root"),
                    "issue": "missing_required_fields",
                    "details": ";".join(required),
                }
            )
    return gaps


def group_has_fields(rows: list[dict[str, Any]], fields: list[str]) -> bool:
    return all(any(has_value(row.get(field)) for row in rows) for field in fields)


def metric_population_score(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    score = 0
    for key in rows[0].keys():
        if key in ID_COLUMNS or key.endswith("_json"):
            continue
        if any(has_value(row.get(key)) for row in rows):
            score += 1
    return score


def summarize_run_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("source_user")),
            str(row.get("benchmark")),
            str(row.get("method")),
            str(row.get("config_id") or ""),
            str(row.get("run_root")),
        )
        summary = grouped.setdefault(
            key,
            {
                "key": key,
                "source_user": row.get("source_user"),
                "benchmark": row.get("benchmark"),
                "method": row.get("method"),
                "config_id": row.get("config_id"),
                "run_name": row.get("run_name"),
                "run_root": row.get("run_root"),
                "rows": [],
            },
        )
        summary["rows"].append(row)
    summaries: list[dict[str, Any]] = []
    for summary in grouped.values():
        rows_in_group = summary["rows"]
        summary["videos"] = len({str(row.get("video_name") or row.get("comparison_key")) for row in rows_in_group})
        summary["metric_score"] = metric_population_score(rows_in_group)
        summary["is_ten_second"] = bool(rows_in_group) and all(bool(row.get("is_ten_second")) for row in rows_in_group)
        summary["moviegen_required_ok"] = group_has_fields(rows_in_group, MOVIEGEN_REQUIRED_FIELDS)
        summary["storyeval_required_ok"] = group_has_fields(rows_in_group, STORYEVAL_REQUIRED_FIELDS)
        summaries.append(summary)
    return summaries


def purge_redundant_backfill_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if (
            row.get("source_user") == "vaishak"
            and str(row.get("run_name") or "").startswith("backfill_")
            and row.get("method") in {"BF16", "RTN_INT4"}
        ):
            continue
        filtered.append(row)
    return filtered


def baseline_run_rank(summary: dict[str, Any]) -> int:
    run_name = str(summary["run_name"] or "").lower()
    if summary["benchmark"] == "moviegen":
        if summary["run_name"] == CANONICAL_MOVIEGEN_RUN:
            return 0
        if "baseline10s" in run_name:
            return 1
        if "baseline10" in run_name:
            return 2
        if run_name == "results":
            return 4
        return 3
    if "presentation_fullmatrix" in run_name:
        return 0
    if "10prompts_10s" in run_name:
        return 1
    return 2


def select_canonical_baseline_group(candidates: list[dict[str, Any]], *, benchmark: str) -> dict[str, Any] | None:
    if not candidates:
        return None
    required_field_ok = "moviegen_required_ok" if benchmark == "moviegen" else "storyeval_required_ok"
    presentation_rank = 0
    preferred = [c for c in candidates if baseline_run_rank(c) == presentation_rank and c.get(required_field_ok)]
    if preferred:
        pool = preferred
    else:
        fallback = [c for c in candidates if c.get(required_field_ok)]
        pool = fallback or candidates
    return sorted(
        pool,
        key=lambda c: (
            baseline_run_rank(c),
            0 if c["videos"] == 10 else abs(int(c["videos"]) - 10),
            -int(c["metric_score"]),
            str(c["run_name"]),
        ),
    )[0]


def experiment_source_rank(summary: dict[str, Any]) -> int:
    run_name = str(summary["run_name"] or "").lower()
    if summary["source_user"] == "vaishak" and run_name.startswith("backfill_"):
        return 0
    if summary["source_user"] == "suraj" and summary["run_name"] == CANONICAL_MOVIEGEN_RUN:
        return 0
    if summary["source_user"] == "suraj" and "presentation" in run_name:
        return 1
    if summary["source_user"] == "suraj" and "baseline10s" in run_name:
        return 2
    if run_name == "results":
        return 5
    return 3


def select_primary_experimental_group(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    moviegen_candidates = [c for c in candidates if c["benchmark"] == "moviegen"]
    pool = moviegen_candidates or candidates
    return sorted(
        pool,
        key=lambda c: (
            0 if c["benchmark"] == "moviegen" else 1,
            0 if c["is_ten_second"] else 1,
            0 if c["videos"] == 10 else abs(int(c["videos"]) - 10) + 1,
            -int(c["metric_score"]),
            experiment_source_rank(c),
            str(c["run_name"]),
        ),
    )[0]


def select_counterpart_group(
    primary: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    benchmark: str,
) -> dict[str, Any] | None:
    pool = [c for c in candidates if c["benchmark"] == benchmark and c["source_user"] == primary["source_user"] and c["method"] == primary["method"]]
    if primary.get("config_id"):
        same_config = [c for c in pool if c.get("config_id") == primary.get("config_id")]
        if same_config:
            pool = same_config
    if not pool:
        return None
    if primary["source_user"] == "suraj":
        run_name = str(primary["run_name"] or "").lower()
        if primary["benchmark"] == "moviegen" and primary["run_name"] == CANONICAL_MOVIEGEN_RUN:
            preferred = [c for c in pool if "presentation_fullmatrix" in str(c["run_name"] or "").lower()]
            if preferred:
                pool = preferred
        elif primary["benchmark"] == "moviegen" and "baseline10s" in run_name:
            preferred = [c for c in pool if "10prompts_10s" in str(c["run_name"] or "").lower()]
            if preferred:
                pool = preferred
    return sorted(
        pool,
        key=lambda c: (
            0 if c["is_ten_second"] else 1,
            0 if c["videos"] == 10 else abs(int(c["videos"]) - 10) + 1,
            -int(c["metric_score"]),
            str(c["run_name"]),
        ),
    )[0]


def apply_strict_deduplication(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pruned_rows = purge_redundant_backfill_baselines(rows)
    summaries = summarize_run_groups(pruned_rows)
    summaries_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        summaries_by_method[str(summary["method"])].append(summary)

    selected_keys: set[tuple[str, str, str, str, str]] = set()
    selected_meta: list[dict[str, Any]] = []

    for method in sorted(BASELINE_METHODS):
        method_summaries = summaries_by_method.get(method, [])
        for benchmark in ("moviegen", "storyeval"):
            candidates = [
                summary
                for summary in method_summaries
                if summary["source_user"] == "suraj" and summary["benchmark"] == benchmark
            ]
            chosen = select_canonical_baseline_group(candidates, benchmark=benchmark)
            if not chosen:
                continue
            selected_keys.add(chosen["key"])
            selected_meta.append(
                {
                    "selection_kind": "canonical_baseline",
                    "method": method,
                    "benchmark": benchmark,
                    "source_user": chosen["source_user"],
                    "run_name": chosen["run_name"],
                    "config_id": chosen.get("config_id"),
                    "videos": chosen["videos"],
                    "metric_score": chosen["metric_score"],
                }
            )

    for method in sorted(set(summaries_by_method) - BASELINE_METHODS):
        method_summaries = summaries_by_method[method]
        primary = select_primary_experimental_group(method_summaries)
        if not primary:
            continue
        selected_keys.add(primary["key"])
        selected_meta.append(
            {
                "selection_kind": "experimental_primary",
                "method": method,
                "benchmark": primary["benchmark"],
                "source_user": primary["source_user"],
                "run_name": primary["run_name"],
                "config_id": primary.get("config_id"),
                "videos": primary["videos"],
                "metric_score": primary["metric_score"],
            }
        )
        counterpart_benchmark = "storyeval" if primary["benchmark"] == "moviegen" else "moviegen"
        counterpart = select_counterpart_group(primary, method_summaries, benchmark=counterpart_benchmark)
        if counterpart:
            selected_keys.add(counterpart["key"])
            selected_meta.append(
                {
                    "selection_kind": "experimental_counterpart",
                    "method": method,
                    "benchmark": counterpart["benchmark"],
                    "source_user": counterpart["source_user"],
                    "run_name": counterpart["run_name"],
                    "config_id": counterpart.get("config_id"),
                    "videos": counterpart["videos"],
                    "metric_score": counterpart["metric_score"],
                }
            )

    selected_groups = [summary for summary in summaries if summary["key"] in selected_keys]
    deduped_rows: list[dict[str, Any]] = []
    for summary in sorted(selected_groups, key=lambda s: (str(s["benchmark"]), str(s["method"]), str(s["run_name"]))):
        sorted_rows = sorted(
            summary["rows"],
            key=lambda row: (
                safe_int(row.get("prompt_index")) if row.get("prompt_index") is not None else 10**9,
                safe_int(row.get("seed")) if row.get("seed") is not None else 10**9,
                str(row.get("video_name") or ""),
            ),
        )
        for row in sorted_rows:
            deduped = dict(row)
            deduped["method_display"] = str(deduped["method"])
            deduped_rows.append(deduped)
    return deduped_rows, selected_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build combined MovieGen + StoryEval comparison dataset.")
    parser.add_argument("--backfill-manifest", type=Path, default=DEFAULT_BACKFILL_MANIFEST)
    parser.add_argument("--storyeval-manifest", type=Path, default=DEFAULT_STORYEVAL_MANIFEST)
    parser.add_argument("--unique-configs", type=Path, default=DEFAULT_UNIQUE_CONFIGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-incomplete", action="store_true", help="Write outputs even if parity gaps remain.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_map, duplicate_counts = build_config_maps(args.unique_configs)
    combined_manifest_map, combined_duplicate_counts = build_manifest_maps(args.backfill_manifest)
    storyeval_manifest = load_json(args.storyeval_manifest)
    storyeval_manifest_map, storyeval_duplicate_counts = build_manifest_maps(args.storyeval_manifest)
    duplicate_counts.update(combined_duplicate_counts)
    duplicate_counts.update(storyeval_duplicate_counts)

    rows: list[dict[str, Any]] = []

    for row in load_json(args.backfill_manifest):
        run_root = (REPO_ROOT / row["run_root"]).resolve()
        if run_root.exists():
            rows.extend(
                build_moviegen_rows(
                    run_root=run_root,
                    run_label=str(row["run_name"]),
                    source_user="vaishak",
                    source_repo="combined-kv-quant",
                    duplicate_counts=duplicate_counts,
                    manifest_map=combined_manifest_map,
                    config_map=config_map,
                )
            )

    for run_root, run_label in collect_suraj_moviegen_run_roots(SURAJ_RESULTS_ROOT):
        rows.extend(
            build_moviegen_rows(
                run_root=run_root,
                run_label=run_label,
                source_user="suraj",
                source_repo="kv-quant-longhorizon",
                duplicate_counts=Counter(),
                manifest_map={},
                config_map={},
            )
        )

    for row in storyeval_manifest:
        run_root = (REPO_ROOT / row["run_root"]).resolve()
        if run_root.exists():
            rows.extend(
                build_storyeval_rows(
                    run_root=run_root,
                    source_user="vaishak",
                    source_repo="combined-kv-quant",
                    duplicate_counts=duplicate_counts,
                    manifest_map=storyeval_manifest_map,
                    config_map=config_map,
                )
            )

    suraj_storyeval_root = SURAJ_RESULTS_ROOT / "benchmarks" / "storyeval"
    if suraj_storyeval_root.exists():
        for run_root in sorted([p for p in suraj_storyeval_root.iterdir() if p.is_dir()]):
            rows.extend(
                build_storyeval_rows(
                    run_root=run_root,
                    source_user="suraj",
                    source_repo="kv-quant-longhorizon",
                    duplicate_counts=Counter(),
                    manifest_map={},
                    config_map={},
                )
            )

    rows, selected_runs = apply_strict_deduplication(rows)
    gaps = build_gap_report(dataset_rows=rows, storyeval_manifest=storyeval_manifest)

    dataset_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_rows": len(rows),
        "num_selected_runs": len(selected_runs),
        "num_gaps": len(gaps),
        "parity_complete": not gaps,
        "selection_rules": {
            "purged_vaishak_backfill_baselines": ["BF16", "RTN_INT4"],
            "canonical_baseline_methods": sorted(BASELINE_METHODS),
            "canonical_moviegen_run_preference": CANONICAL_MOVIEGEN_RUN,
            "experimental_dedup_rule": "Prefer 10-second runs with 10 videos and the most comprehensive metric population; keep one selected run family per method.",
        },
        "selected_runs": selected_runs,
        "rows": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_json = args.output_dir / "combined_comparison_dataset.json"
    dataset_csv = args.output_dir / "combined_comparison_dataset.csv"
    gaps_json = args.output_dir / "combined_comparison_gaps.json"
    gaps_csv = args.output_dir / "combined_comparison_gaps.csv"
    write_json(dataset_json, dataset_payload)
    write_csv(dataset_csv, rows)
    write_json(gaps_json, gaps)
    write_csv(gaps_csv, gaps)
    print(f"Wrote {dataset_json}")
    print(f"Wrote {dataset_csv}")
    print(f"Wrote {gaps_json}")
    print(f"Wrote {gaps_csv}")
    print(f"Selected run groups: {len(selected_runs)}")

    if gaps and not args.allow_incomplete:
        raise SystemExit(
            f"Dataset built with {len(gaps)} gap(s). Finish StoryEval parity or rerun with --allow-incomplete."
        )


if __name__ == "__main__":
    main()
