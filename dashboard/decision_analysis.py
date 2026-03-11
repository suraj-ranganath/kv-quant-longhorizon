from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np
import pandas as pd

SCORE_PRESETS: dict[str, dict[str, float]] = {
    "Balanced": {
        "imaging_quality": 0.35,
        "drift_last_imaging_quality": 0.25,
        "peak_vram_gb": 0.20,
        "avg_runtime_s_per_prompt": 0.10,
        "compression_ratio": 0.10,
    },
    "Quality-first": {
        "imaging_quality": 0.40,
        "drift_last_imaging_quality": 0.25,
        "peak_vram_gb": 0.10,
        "avg_runtime_s_per_prompt": 0.10,
        "compression_ratio": 0.15,
    },
    "Runtime-first": {
        "imaging_quality": 0.20,
        "drift_last_imaging_quality": 0.15,
        "peak_vram_gb": 0.10,
        "avg_runtime_s_per_prompt": 0.40,
        "compression_ratio": 0.15,
    },
    "Memory-first": {
        "imaging_quality": 0.20,
        "drift_last_imaging_quality": 0.20,
        "peak_vram_gb": 0.35,
        "avg_runtime_s_per_prompt": 0.10,
        "compression_ratio": 0.15,
    },
    "Aggressive compression": {
        "imaging_quality": 0.20,
        "drift_last_imaging_quality": 0.20,
        "peak_vram_gb": 0.15,
        "avg_runtime_s_per_prompt": 0.05,
        "compression_ratio": 0.40,
    },
}

FRONTIER_DEFINITIONS: dict[str, dict[str, Any]] = {
    "balanced_practical": {
        "label": "Balanced practical frontier",
        "objectives": {
            "imaging_quality": "max",
            "drift_last_imaging_quality": "max",
            "compression_ratio": "max",
            "avg_runtime_s_per_prompt": "min",
            "peak_vram_gb": "min",
        },
    },
    "quality_preserving_compression": {
        "label": "Quality-preserving compression frontier",
        "objectives": {
            "compression_ratio": "max",
            "imaging_quality": "max",
            "drift_last_imaging_quality": "max",
        },
    },
    "systems_efficiency": {
        "label": "Systems efficiency frontier",
        "objectives": {
            "avg_runtime_s_per_prompt": "min",
            "peak_vram_gb": "min",
            "compression_ratio": "max",
        },
    },
    "quality_first": {
        "label": "Quality-first frontier",
        "objectives": {
            "imaging_quality": "max",
            "drift_last_imaging_quality": "max",
            "avg_runtime_s_per_prompt": "min",
        },
    },
}

FAMILY_PREFIXES = [
    "SPATIAL_MIXED",
    "FLOWCACHE",
    "QUAROT_KV",
    "AGE_TIER",
    "KIVI",
    "RTN",
    "PRQ",
    "QAQ",
    "TPTQ",
    "BF16",
]

DEFAULT_THRESHOLDS = {
    "acceptable_imaging_drop": 0.015,
    "acceptable_drift_drop": 0.020,
    "min_compression": 1.10,
}

OBJECTIVE_DIRECTIONS = {
    "imaging_quality": "max",
    "drift_last_imaging_quality": "max",
    "compression_ratio": "max",
    "avg_runtime_s_per_prompt": "min",
    "peak_vram_gb": "min",
    "ssim": "max",
    "lpips": "min",
}


@dataclass
class DecisionAnalysis:
    benchmark: str
    run_summary: pd.DataFrame
    method_summary: pd.DataFrame
    explainability_table: pd.DataFrame
    recommendations: dict[str, dict[str, Any]]
    takeaways: list[str]
    frontier_members: dict[str, list[str]]
    constraint_tables: dict[str, pd.DataFrame]
    score_weights: dict[str, float]
    thresholds: dict[str, float]
    source_catalog: pd.DataFrame
    primary_source_path: str | None


def _safe_string(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _first_nonnull(series: pd.Series) -> Any:
    cleaned = series.dropna()
    return None if cleaned.empty else cleaned.iloc[0]


def _unique_join(series: pd.Series) -> str:
    values = sorted({_safe_string(value) for value in series if _safe_string(value)})
    return ", ".join(values)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _benchmark_prefix(benchmark: str) -> str:
    return "storyeval" if str(benchmark).strip().lower() == "storyeval" else "moviegen"


def _value_from_group(group: pd.DataFrame, candidates: list[str], reducer: str = "first") -> Any:
    for candidate in candidates:
        if candidate not in group.columns:
            continue
        series = group[candidate]
        if reducer == "max":
            value = series.max(skipna=True)
            return None if pd.isna(value) else value
        if reducer == "mean":
            value = series.mean(skipna=True)
            return None if pd.isna(value) else value
        value = _first_nonnull(series)
        if value is not None:
            return value
    return None


def standardize_method_name(method_name: Any) -> str:
    method = _safe_string(method_name).upper().replace(" ", "_")
    method = re.sub(r"__+", "_", method)
    return method


def infer_method_family(method_name: str, raw_family: str | None = None) -> str:
    if raw_family:
        raw = standardize_method_name(raw_family)
        if raw.startswith("FLOWCACHE"):
            return "FLOWCACHE"
        if raw.startswith("QUAROT"):
            return "QUAROT"
        for prefix in FAMILY_PREFIXES:
            if raw.startswith(prefix):
                return prefix
    method = standardize_method_name(method_name)
    for prefix in FAMILY_PREFIXES:
        if method.startswith(prefix):
            if prefix == "QUAROT_KV":
                return "QUAROT"
            if prefix.startswith("FLOWCACHE"):
                return "FLOWCACHE"
            return prefix
    return method.split("_", 1)[0] if method else "UNKNOWN"


def parse_method_metadata(method_name: str, raw_family: str | None = None, display_label: str | None = None) -> dict[str, Any]:
    method = standardize_method_name(method_name)
    broad_family = infer_method_family(method, raw_family)
    bit_matches = re.findall(r"INT\d+", method)
    bit_label = ", ".join(bit_matches) if bit_matches else ("BF16" if method == "BF16" else "mixed / implicit")

    variant_order = [
        ("SOFT_PRUNE", "soft-prune"),
        ("PRUNE", "prune"),
        ("NATIVE", "native reuse"),
        ("ADAPTIVE", "adaptive"),
        ("HYBRID", "hybrid"),
        ("REFRESH", "refresh"),
        ("RECENT2", "recent2"),
        ("K2_V4", "K2/V4"),
    ]
    variants = [label for token, label in variant_order if token in method]

    detail = ""
    label_text = _safe_string(display_label)
    if "|" in label_text:
        detail = label_text.split("|", 1)[1].strip()
    elif method.startswith("SPATIAL_MIXED_FG_") and "_BG_" in method:
        fg_part, bg_part = method.split("_BG_", 1)
        fg_desc = fg_part.replace("SPATIAL_MIXED_FG_", "").replace("_", " ")
        bg_desc = bg_part.replace("_", " ")
        detail = f"FG: {fg_desc}; BG: {bg_desc}"
    elif broad_family == "FLOWCACHE":
        variant_desc = ", ".join(variants) if variants else "retention/reuse policy"
        detail = f"{variant_desc}; bits={bit_label}"
    elif broad_family == "BF16":
        detail = "Uncompressed BF16 reference KV cache"
    else:
        variant_desc = ", ".join(variants) if variants else broad_family.lower()
        detail = f"{variant_desc}; bits={bit_label}"

    return {
        "method_family_broad": broad_family,
        "bit_width_label": bit_label,
        "variant_tags": ", ".join(variants),
        "quantization_details": detail,
    }


def build_run_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    required_cols = [
        "source_user",
        "source_repo",
        "benchmark",
        "run_name",
        "run_root",
        "method",
        "method_display",
        "config_id",
        "method_family",
    ]
    present_group_cols = [column for column in required_cols if column in df.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(present_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = dict(zip(present_group_cols, keys))
        benchmark = _safe_string(values.get("benchmark") or _first_nonnull(group.get("benchmark", pd.Series(dtype=str))))
        prefix = _benchmark_prefix(benchmark)
        raw_method = standardize_method_name(values.get("method") or _first_nonnull(group.get("method", pd.Series(dtype=str))))
        display_method = _safe_string(values.get("method_display") or _first_nonnull(group.get("method_display", pd.Series(dtype=str)))) or raw_method
        raw_family = _safe_string(values.get("method_family") or _first_nonnull(group.get("method_family", pd.Series(dtype=str))))
        meta = parse_method_metadata(raw_method, raw_family=raw_family, display_label=display_method)

        prompt_count = 0
        for column in ["prompt_id", "prompt_index"]:
            if column in group.columns:
                prompt_count = max(prompt_count, int(group[column].dropna().astype(str).nunique()))
        if prompt_count == 0:
            prompt_count = int(len(group))

        seed_count = int(group["seed"].dropna().astype(str).nunique()) if "seed" in group.columns else 0
        video_count = int(group["video_name"].dropna().astype(str).nunique()) if "video_name" in group.columns else int(len(group))

        if prefix == "storyeval":
            peak_vram_gb = _to_float(_value_from_group(group, ["storyeval_max_peak_vram_mb"], reducer="first"))
            if peak_vram_gb is not None:
                peak_vram_gb /= 1024.0
        else:
            peak_vram_gb = None
        if peak_vram_gb is None:
            peak_vram_bytes = _value_from_group(group, ["peak_vram_bytes"], reducer="max")
            peak_vram_mb = _value_from_group(group, ["peak_vram_mb"], reducer="max")
            if peak_vram_bytes is not None:
                peak_vram_gb = float(peak_vram_bytes) / (1024**3)
            elif peak_vram_mb is not None:
                peak_vram_gb = float(peak_vram_mb) / 1024.0

        avg_runtime = _value_from_group(
            group,
            [
                f"{prefix}_avg_runtime_sec",
                "avg_runtime_s_per_prompt",
                "total_runtime_s",
            ],
            reducer="first",
        )
        if avg_runtime is None and "wall_time_sec" in group.columns:
            avg_runtime = _value_from_group(group, ["wall_time_sec"], reducer="mean")

        row = {
            "benchmark": benchmark,
            "source_user": _safe_string(values.get("source_user") or _first_nonnull(group.get("source_user", pd.Series(dtype=str)))),
            "source_repo": _safe_string(values.get("source_repo") or _first_nonnull(group.get("source_repo", pd.Series(dtype=str)))),
            "run_name": _safe_string(values.get("run_name") or _first_nonnull(group.get("run_name", pd.Series(dtype=str)))),
            "run_root": _safe_string(values.get("run_root") or _first_nonnull(group.get("run_root", pd.Series(dtype=str)))),
            "method": raw_method,
            "method_display": display_method,
            "raw_method": raw_method,
            "config_id": _safe_string(values.get("config_id") or _first_nonnull(group.get("config_id", pd.Series(dtype=str)))),
            "method_family_raw": raw_family,
            "method_family": meta["method_family_broad"],
            "bit_width_label": meta["bit_width_label"],
            "variant_tags": meta["variant_tags"],
            "quantization_details": meta["quantization_details"],
            "display_label": _safe_string(_value_from_group(group, ["method_display", "display_label"], reducer="first")) or display_method,
            "config_payload_json": _safe_string(_value_from_group(group, ["config_payload_json"], reducer="first")),
            "quant_meta_json": _safe_string(_value_from_group(group, ["quant_meta_json"], reducer="first")),
            "prompt_count": prompt_count,
            "seed_count": seed_count,
            "run_count": 1,
            "source_count": 1,
            "video_count": video_count,
            "logged_prompts": int(len(group)),
            "psnr": _value_from_group(group, [f"{prefix}_fidelity_psnr_agg", "psnr"], reducer="first"),
            "ssim": _value_from_group(group, [f"{prefix}_fidelity_ssim_agg", "ssim"], reducer="first"),
            "lpips": _value_from_group(group, [f"{prefix}_fidelity_lpips_agg", "lpips"], reducer="first"),
            "background_consistency": _value_from_group(group, [f"{prefix}_background_consistency_agg", "background_consistency"], reducer="first"),
            "imaging_quality": _value_from_group(group, [f"{prefix}_imaging_quality_agg", "imaging_quality"], reducer="first"),
            "subject_consistency": _value_from_group(group, [f"{prefix}_subject_consistency_agg", "subject_consistency"], reducer="first"),
            "aesthetic_quality": _value_from_group(group, [f"{prefix}_aesthetic_quality_agg", "aesthetic_quality"], reducer="first"),
            "drift_last_imaging_quality": _value_from_group(group, [f"{prefix}_drift_last_imaging_quality", "drift_last_imaging_quality"], reducer="first"),
            "drift_points": _value_from_group(group, [f"{prefix}_drift_points", "drift_points"], reducer="first"),
            "compression_ratio": _value_from_group(group, ["compression_ratio"], reducer="first"),
            "bf16_kv_bytes": _value_from_group(group, ["bf16_kv_bytes"], reducer="first"),
            "compressed_kv_bytes": _value_from_group(group, ["compressed_kv_bytes"], reducer="first"),
            "peak_vram_gb": peak_vram_gb,
            "peak_allocated_gb": peak_vram_gb,
            "peak_compressed_kv_gb": (
                float(_value_from_group(group, ["compressed_kv_bytes"], reducer="first")) / (1024**3)
                if _value_from_group(group, ["compressed_kv_bytes"], reducer="first") is not None
                else None
            ),
            "bf16_kv_gb": (
                float(_value_from_group(group, ["bf16_kv_bytes"], reducer="first")) / (1024**3)
                if _value_from_group(group, ["bf16_kv_bytes"], reducer="first") is not None
                else None
            ),
            "avg_runtime_s_per_prompt": avg_runtime,
            "runtime_per_prompt": avg_runtime,
            "dataset_provenance_path": _safe_string(_value_from_group(group, ["dataset_provenance_path"], reducer="first")),
        }
        rows.append(row)

    run_summary = pd.DataFrame(rows)
    if run_summary.empty:
        return run_summary
    numeric_columns = [
        "psnr",
        "ssim",
        "lpips",
        "background_consistency",
        "imaging_quality",
        "subject_consistency",
        "aesthetic_quality",
        "drift_last_imaging_quality",
        "drift_points",
        "compression_ratio",
        "bf16_kv_bytes",
        "compressed_kv_bytes",
        "peak_vram_gb",
        "peak_allocated_gb",
        "peak_compressed_kv_gb",
        "bf16_kv_gb",
        "avg_runtime_s_per_prompt",
        "runtime_per_prompt",
        "prompt_count",
        "seed_count",
        "run_count",
        "source_count",
        "video_count",
        "logged_prompts",
    ]
    for column in numeric_columns:
        if column in run_summary.columns:
            run_summary[column] = pd.to_numeric(run_summary[column], errors="coerce")
    return run_summary


def build_method_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    metric_columns = [
        "psnr",
        "ssim",
        "lpips",
        "background_consistency",
        "imaging_quality",
        "subject_consistency",
        "aesthetic_quality",
        "drift_last_imaging_quality",
        "compression_ratio",
        "peak_vram_gb",
        "peak_allocated_gb",
        "peak_compressed_kv_gb",
        "bf16_kv_gb",
        "avg_runtime_s_per_prompt",
        "runtime_per_prompt",
        "prompt_count",
        "seed_count",
        "video_count",
        "logged_prompts",
    ]
    for (benchmark, raw_method), group in run_summary.groupby(["benchmark", "raw_method"], dropna=False):
        first = group.iloc[0]
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "method": raw_method,
            "method_display": raw_method,
            "raw_method": raw_method,
            "method_family": first.get("method_family"),
            "method_family_raw": first.get("method_family_raw"),
            "bit_width_label": first.get("bit_width_label"),
            "variant_tags": first.get("variant_tags"),
            "quantization_details": first.get("quantization_details"),
            "display_label": first.get("display_label") or raw_method,
            "run_count": int(group["run_name"].nunique()),
            "source_count": int(group["source_user"].nunique()),
            "source_users": _unique_join(group["source_user"]),
            "source_repos": _unique_join(group["source_repo"]),
            "source_runs": _unique_join(group["run_name"]),
            "config_ids": _unique_join(group["config_id"]),
            "dataset_provenance_paths": _unique_join(group["dataset_provenance_path"]),
        }
        for column in metric_columns:
            if column in group.columns:
                row[column] = float(group[column].mean(skipna=True)) if group[column].notna().any() else None
                row[f"{column}_std"] = float(group[column].std(skipna=True)) if group[column].notna().sum() > 1 else 0.0
        row["prompt_count"] = int(group["prompt_count"].sum()) if "prompt_count" in group.columns else 0
        row["seed_count"] = int(group["seed_count"].sum()) if "seed_count" in group.columns else 0
        row["video_count"] = int(group["video_count"].sum()) if "video_count" in group.columns else 0
        row["logged_prompts"] = int(group["logged_prompts"].sum()) if "logged_prompts" in group.columns else 0
        row["available_runs"] = int(len(group))
        rows.append(row)

    method_summary = pd.DataFrame(rows)
    if method_summary.empty:
        return method_summary
    numeric_columns = [column for column in method_summary.columns if column.endswith("_std") or column in metric_columns or column in {"run_count", "source_count", "prompt_count", "seed_count", "video_count", "logged_prompts", "available_runs"}]
    for column in numeric_columns:
        method_summary[column] = pd.to_numeric(method_summary[column], errors="coerce")
    return method_summary


def add_bf16_relative_metrics(summary_df: pd.DataFrame, method_column: str = "method") -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    result = summary_df.copy()
    for benchmark, group in result.groupby("benchmark"):
        mask = (result["benchmark"] == benchmark)
        bf16_rows = group[group[method_column] == "BF16"]
        if bf16_rows.empty:
            continue
        baseline = bf16_rows.iloc[0]

        baseline_imaging = _to_float(baseline.get("imaging_quality"))
        baseline_drift = _to_float(baseline.get("drift_last_imaging_quality"))
        baseline_psnr = baseline.get("psnr")
        baseline_ssim = _to_float(baseline.get("ssim"))
        baseline_lpips = _to_float(baseline.get("lpips"))
        baseline_runtime = _to_float(baseline.get("avg_runtime_s_per_prompt"))
        baseline_vram = _to_float(baseline.get("peak_vram_gb"))
        baseline_compressed = _to_float(baseline.get("peak_compressed_kv_gb"))
        baseline_compression = _to_float(baseline.get("compression_ratio"))

        result.loc[mask, "is_bf16_reference"] = result.loc[mask, method_column] == "BF16"
        if baseline_imaging is not None:
            result.loc[mask, "imaging_quality_delta_vs_bf16"] = result.loc[mask, "imaging_quality"] - baseline_imaging
            result.loc[mask, "imaging_quality_drop_vs_bf16"] = baseline_imaging - result.loc[mask, "imaging_quality"]
        if baseline_drift is not None:
            result.loc[mask, "drift_last_imaging_quality_delta_vs_bf16"] = result.loc[mask, "drift_last_imaging_quality"] - baseline_drift
            result.loc[mask, "drift_last_imaging_quality_drop_vs_bf16"] = baseline_drift - result.loc[mask, "drift_last_imaging_quality"]
        if baseline_ssim is not None:
            result.loc[mask, "ssim_delta_vs_bf16"] = result.loc[mask, "ssim"] - baseline_ssim
            result.loc[mask, "ssim_drop_vs_bf16"] = baseline_ssim - result.loc[mask, "ssim"]
        if baseline_lpips is not None:
            result.loc[mask, "lpips_delta_vs_bf16"] = result.loc[mask, "lpips"] - baseline_lpips
        if baseline_runtime is not None and baseline_runtime > 0:
            result.loc[mask, "runtime_speedup_vs_bf16"] = baseline_runtime / result.loc[mask, "avg_runtime_s_per_prompt"]
            result.loc[mask, "runtime_overhead_vs_bf16_pct"] = 100.0 * (result.loc[mask, "avg_runtime_s_per_prompt"] - baseline_runtime) / baseline_runtime
        if baseline_vram is not None and baseline_vram > 0:
            result.loc[mask, "peak_vram_delta_vs_bf16_gb"] = result.loc[mask, "peak_vram_gb"] - baseline_vram
            result.loc[mask, "peak_vram_reduction_vs_bf16_pct"] = 100.0 * (1.0 - (result.loc[mask, "peak_vram_gb"] / baseline_vram))
        if baseline_compressed is not None:
            result.loc[mask, "peak_compressed_kv_delta_vs_bf16_gb"] = result.loc[mask, "peak_compressed_kv_gb"] - baseline_compressed
            if baseline_compressed > 0:
                result.loc[mask, "peak_compressed_kv_reduction_vs_bf16_pct"] = 100.0 * (1.0 - (result.loc[mask, "peak_compressed_kv_gb"] / baseline_compressed))
        if baseline_compression is not None and baseline_compression > 0:
            result.loc[mask, "compression_gain_vs_bf16"] = result.loc[mask, "compression_ratio"] / baseline_compression

        psnr_baseline_finite = _to_float(baseline_psnr)
        if psnr_baseline_finite is not None:
            result.loc[mask, "psnr_delta_vs_bf16"] = result.loc[mask, "psnr"] - psnr_baseline_finite
        else:
            result.loc[mask, "psnr_delta_vs_bf16"] = np.nan
            result.loc[mask, "psnr_delta_note"] = "Undefined because the BF16 reference PSNR is non-finite."

    derived_cols = [
        "imaging_quality_delta_vs_bf16",
        "imaging_quality_drop_vs_bf16",
        "drift_last_imaging_quality_delta_vs_bf16",
        "drift_last_imaging_quality_drop_vs_bf16",
        "psnr_delta_vs_bf16",
        "ssim_delta_vs_bf16",
        "ssim_drop_vs_bf16",
        "lpips_delta_vs_bf16",
        "runtime_speedup_vs_bf16",
        "runtime_overhead_vs_bf16_pct",
        "peak_vram_delta_vs_bf16_gb",
        "peak_vram_reduction_vs_bf16_pct",
        "peak_compressed_kv_delta_vs_bf16_gb",
        "peak_compressed_kv_reduction_vs_bf16_pct",
        "compression_gain_vs_bf16",
    ]
    for column in derived_cols:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return result


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {metric: max(float(value), 0.0) for metric, value in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        default = SCORE_PRESETS["Balanced"]
        total = sum(default.values())
        return {metric: value / total for metric, value in default.items()}
    return {metric: value / total for metric, value in cleaned.items()}


def _normalize_metric(series: pd.Series, direction: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    if finite.notna().sum() <= 1:
        return finite.notna().astype(float)
    min_value = finite.min(skipna=True)
    max_value = finite.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value) or math.isclose(float(min_value), float(max_value)):
        return finite.notna().astype(float)
    scaled = (finite - min_value) / (max_value - min_value)
    if direction == "min":
        scaled = 1.0 - scaled
    return scaled


def apply_scores(method_summary: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if method_summary.empty:
        return method_summary.copy()

    result = method_summary.copy()
    normalized_weights = normalize_weights(weights)
    score_components = []
    total_weight = 0.0
    for metric, weight in normalized_weights.items():
        if metric not in result.columns:
            continue
        direction = OBJECTIVE_DIRECTIONS[metric]
        normalized = _normalize_metric(result[metric], direction)
        result[f"score_component_{metric}"] = normalized
        score_components.append(normalized * weight)
        total_weight += weight
    if score_components and total_weight > 0:
        result["utility_score"] = sum(score_components) / total_weight
    else:
        result["utility_score"] = np.nan

    quality_weights = normalize_weights(
        {
            "imaging_quality": 0.30,
            "drift_last_imaging_quality": 0.20,
            "ssim": 0.35,
            "lpips": 0.15,
        }
    )
    quality_components = []
    quality_total = 0.0
    for metric, weight in quality_weights.items():
        if metric not in result.columns:
            continue
        direction = OBJECTIVE_DIRECTIONS[metric]
        normalized = _normalize_metric(result[metric], direction)
        result[f"quality_component_{metric}"] = normalized
        quality_components.append(normalized * weight)
        quality_total += weight
    if quality_components and quality_total > 0:
        result["quality_reference_score"] = sum(quality_components) / quality_total
    else:
        result["quality_reference_score"] = np.nan
    return result


def _dominates(left: pd.Series, right: pd.Series, objectives: dict[str, str], epsilon: float = 1e-9) -> bool:
    strictly_better = False
    for column, direction in objectives.items():
        left_value = _to_float(left.get(column))
        right_value = _to_float(right.get(column))
        if left_value is None or right_value is None:
            return False
        if direction == "max":
            if left_value + epsilon < right_value:
                return False
            if left_value > right_value + epsilon:
                strictly_better = True
        else:
            if left_value > right_value + epsilon:
                return False
            if left_value + epsilon < right_value:
                strictly_better = True
    return strictly_better


def attach_frontier_results(method_summary: pd.DataFrame) -> pd.DataFrame:
    if method_summary.empty:
        return method_summary.copy()

    result = method_summary.copy()
    for frontier_key, frontier_config in FRONTIER_DEFINITIONS.items():
        objectives = frontier_config["objectives"]
        required_columns = list(objectives.keys())
        eligible = result[required_columns].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        frontier_flags: list[bool] = []
        dominated_counts: list[int | None] = []
        dominated_examples: list[str] = []
        reasons: list[str] = []
        for idx, row in result.iterrows():
            if not bool(eligible.loc[idx]):
                frontier_flags.append(False)
                dominated_counts.append(None)
                dominated_examples.append("")
                missing = [column for column in required_columns if pd.isna(row.get(column)) or not np.isfinite(row.get(column))]
                reasons.append(f"Missing objective values: {', '.join(missing)}")
                continue
            dominators: list[str] = []
            for other_idx, other_row in result.iterrows():
                if idx == other_idx or not bool(eligible.loc[other_idx]):
                    continue
                if _dominates(other_row, row, objectives):
                    dominators.append(str(other_row.get("method", other_row.get("raw_method", other_idx))))
            is_frontier = len(dominators) == 0
            frontier_flags.append(is_frontier)
            dominated_counts.append(len(dominators))
            dominated_examples.append(", ".join(dominators[:4]))
            if is_frontier:
                reasons.append("Survives this frontier because no other method improves every selected objective simultaneously.")
            else:
                reasons.append(f"Dominated by {', '.join(dominators[:3])}.")
        result[f"pareto_{frontier_key}"] = frontier_flags
        result[f"dominated_by_{frontier_key}_count"] = dominated_counts
        result[f"dominated_by_{frontier_key}"] = dominated_examples
        result[f"pareto_{frontier_key}_explanation"] = reasons
    return result


def _sorted_candidates(df: pd.DataFrame, sort_columns: list[str], ascending: list[bool], exclude_methods: set[str]) -> pd.DataFrame:
    candidates = df.copy()
    if exclude_methods:
        candidates = candidates[~candidates["method"].isin(sorted(exclude_methods))]
    if candidates.empty:
        return candidates
    return candidates.sort_values(sort_columns, ascending=ascending, na_position="last")


def _pick_candidate(df: pd.DataFrame, sort_columns: list[str], ascending: list[bool], exclude_methods: set[str]) -> pd.Series | None:
    sorted_df = _sorted_candidates(df, sort_columns, ascending, exclude_methods)
    if sorted_df.empty:
        return None
    return sorted_df.iloc[0]


def _apply_recommendation_caps(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    capped = df.copy()
    runtime_max = thresholds.get("runtime_max")
    if runtime_max is not None and "avg_runtime_s_per_prompt" in capped.columns:
        capped = capped[capped["avg_runtime_s_per_prompt"].fillna(np.inf) <= runtime_max]
    vram_max = thresholds.get("vram_max")
    if vram_max is not None and "peak_vram_gb" in capped.columns:
        capped = capped[capped["peak_vram_gb"].fillna(np.inf) <= vram_max]
    return capped


def _prefer_capped_candidates(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    capped = _apply_recommendation_caps(df, thresholds)
    return capped if not capped.empty else df


def _recommendation_reason(kind: str, row: pd.Series) -> str:
    method = row.get("method")
    compression = _to_float(row.get("compression_ratio"))
    runtime = _to_float(row.get("avg_runtime_s_per_prompt"))
    peak_vram = _to_float(row.get("peak_vram_gb"))
    imaging_delta = _to_float(row.get("imaging_quality_delta_vs_bf16"))
    drift_delta = _to_float(row.get("drift_last_imaging_quality_delta_vs_bf16"))
    if kind == "default_practical":
        return (
            f"Best balanced option under the current filters: {method} keeps imaging and drift close to BF16 "
            f"while delivering {compression:.2f}x compression, {runtime:.1f}s per prompt, and {peak_vram:.2f} GB peak VRAM."
            if compression is not None and runtime is not None and peak_vram is not None
            else f"Best balanced option under the current filters: {method}."
        )
    if kind == "aggressive_compression":
        return (
            f"Highest-compression option that still stays within the active quality and drift tolerance: {compression:.2f}x compression, "
            f"{peak_vram:.2f} GB peak VRAM."
            if compression is not None and peak_vram is not None
            else f"Highest-compression option that still meets the active quality and drift tolerance: {method}."
        )
    if kind == "fastest":
        return (
            f"Fastest non-BF16 option at {runtime:.1f}s per prompt with {peak_vram:.2f} GB peak VRAM."
            if runtime is not None and peak_vram is not None
            else f"Fastest non-BF16 option in the current comparison: {method}."
        )
    if kind == "quality_first":
        return (
            f"Strongest non-BF16 quality-retention candidate: imaging delta {imaging_delta:+.3f}, drift delta {drift_delta:+.3f}, peak VRAM {peak_vram:.2f} GB."
            if imaging_delta is not None and drift_delta is not None and peak_vram is not None
            else f"Strongest non-BF16 quality-retention candidate in the current comparison: {method}."
        )
    return "Reference baseline for all BF16-relative comparisons."


def _build_caution(row: pd.Series, thresholds: dict[str, float]) -> str:
    cautions: list[str] = []
    peak_vram_reduction = _to_float(row.get("peak_vram_reduction_vs_bf16_pct"))
    compression_ratio = _to_float(row.get("compression_ratio"))
    runtime_overhead = _to_float(row.get("runtime_overhead_vs_bf16_pct"))
    imaging_drop = _to_float(row.get("imaging_quality_drop_vs_bf16"))
    drift_drop = _to_float(row.get("drift_last_imaging_quality_drop_vs_bf16"))
    ssim_drop = _to_float(row.get("ssim_drop_vs_bf16"))

    if compression_ratio is not None and compression_ratio > 1.0 and peak_vram_reduction is not None and peak_vram_reduction <= 0:
        cautions.append("Compressed KV bytes do not translate into lower peak VRAM in the current stack.")
    if runtime_overhead is not None and runtime_overhead > 100:
        cautions.append("Runtime is more than 2x BF16 under the current implementation.")
    if imaging_drop is not None and imaging_drop > thresholds["acceptable_imaging_drop"]:
        cautions.append("Imaging quality drops beyond the current tolerance.")
    if drift_drop is not None and drift_drop > thresholds["acceptable_drift_drop"]:
        cautions.append("Drift stability drops beyond the current tolerance.")
    if ssim_drop is not None and ssim_drop > 0.20:
        cautions.append("Structural fidelity drops more than perceptual quality alone suggests.")
    return " ".join(cautions[:2])


def recommend_methods(method_summary: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, dict[str, Any]]:
    recommendations: dict[str, dict[str, Any]] = {}
    if method_summary.empty:
        return recommendations

    base_df = method_summary.copy()
    non_bf16 = base_df[base_df["method"] != "BF16"].copy()
    exclude: set[str] = set()

    quality_tolerant = non_bf16[
        (non_bf16["compression_ratio"].fillna(0) >= thresholds["min_compression"])
        & (non_bf16["imaging_quality_drop_vs_bf16"].fillna(np.inf) <= thresholds["acceptable_imaging_drop"])
        & (non_bf16["drift_last_imaging_quality_drop_vs_bf16"].fillna(np.inf) <= thresholds["acceptable_drift_drop"])
    ]
    fidelity_tolerant = quality_tolerant[quality_tolerant["ssim_drop_vs_bf16"].fillna(0) <= 0.35] if "ssim_drop_vs_bf16" in quality_tolerant.columns else quality_tolerant
    quality_tolerant_practical = _prefer_capped_candidates(fidelity_tolerant, thresholds)

    default_candidates = quality_tolerant_practical[quality_tolerant_practical["pareto_balanced_practical"].fillna(False)]
    if default_candidates.empty:
        default_candidates = quality_tolerant_practical
    if default_candidates.empty:
        default_candidates = _prefer_capped_candidates(non_bf16[non_bf16["pareto_balanced_practical"].fillna(False)], thresholds)
    if default_candidates.empty:
        default_candidates = _prefer_capped_candidates(non_bf16, thresholds)
    if default_candidates.empty:
        default_candidates = non_bf16
    default_row = _pick_candidate(
        default_candidates,
        ["utility_score", "peak_vram_gb", "quality_reference_score", "compression_ratio", "avg_runtime_s_per_prompt"],
        [False, True, False, False, True],
        exclude,
    )
    if default_row is not None:
        exclude.add(str(default_row["method"]))
        recommendations["default_practical"] = {
            "label": "Default practical recommendation",
            "method": str(default_row["method"]),
            "row": default_row,
            "reason": _recommendation_reason("default_practical", default_row),
            "caution": _build_caution(default_row, thresholds),
        }

    aggressive_candidates = _prefer_capped_candidates(quality_tolerant, thresholds)
    if aggressive_candidates.empty:
        aggressive_candidates = _prefer_capped_candidates(non_bf16, thresholds)
    if aggressive_candidates.empty:
        aggressive_candidates = non_bf16
    if not aggressive_candidates.empty and aggressive_candidates["compression_ratio"].notna().any():
        max_compression = aggressive_candidates["compression_ratio"].max(skipna=True)
        near_max_mask = aggressive_candidates["compression_ratio"] >= (0.98 * float(max_compression))
        near_max_candidates = aggressive_candidates[near_max_mask]
        if not near_max_candidates.empty:
            aggressive_candidates = near_max_candidates
    aggressive_row = _pick_candidate(
        aggressive_candidates,
        ["compression_ratio", "peak_vram_gb", "quality_reference_score", "utility_score", "imaging_quality", "drift_last_imaging_quality"],
        [False, True, False, False, False, False],
        exclude,
    )
    if aggressive_row is not None:
        exclude.add(str(aggressive_row["method"]))
        recommendations["aggressive_compression"] = {
            "label": "Best aggressive-compression option",
            "method": str(aggressive_row["method"]),
            "row": aggressive_row,
            "reason": _recommendation_reason("aggressive_compression", aggressive_row),
            "caution": _build_caution(aggressive_row, thresholds),
        }

    fastest_candidates = _prefer_capped_candidates(non_bf16, thresholds)
    if fastest_candidates.empty:
        fastest_candidates = non_bf16.copy()
    fastest_row = _pick_candidate(
        fastest_candidates,
        ["avg_runtime_s_per_prompt", "peak_vram_gb", "imaging_quality", "drift_last_imaging_quality"],
        [True, True, False, False],
        exclude,
    )
    if fastest_row is not None:
        exclude.add(str(fastest_row["method"]))
        recommendations["fastest"] = {
            "label": "Fastest option",
            "method": str(fastest_row["method"]),
            "row": fastest_row,
            "reason": _recommendation_reason("fastest", fastest_row),
            "caution": _build_caution(fastest_row, thresholds),
        }

    quality_candidates = _prefer_capped_candidates(non_bf16, thresholds)
    if quality_candidates.empty:
        quality_candidates = non_bf16.copy()
    quality_row = _pick_candidate(
        quality_candidates,
        ["quality_reference_score", "ssim", "lpips", "imaging_quality", "drift_last_imaging_quality", "peak_vram_gb", "avg_runtime_s_per_prompt"],
        [False, False, True, False, False, True, True],
        exclude,
    )
    if quality_row is not None:
        exclude.add(str(quality_row["method"]))
        recommendations["quality_first"] = {
            "label": "Quality-first option",
            "method": str(quality_row["method"]),
            "row": quality_row,
            "reason": _recommendation_reason("quality_first", quality_row),
            "caution": _build_caution(quality_row, thresholds),
        }

    bf16_rows = base_df[base_df["method"] == "BF16"]
    if not bf16_rows.empty:
        bf16_row = bf16_rows.iloc[0]
        recommendations["bf16_reference"] = {
            "label": "BF16 reference baseline",
            "method": "BF16",
            "row": bf16_row,
            "reason": _recommendation_reason("bf16_reference", bf16_row),
            "caution": "Reference only: no KV compression is applied.",
        }
    return recommendations


def annotate_recommendations(method_summary: pd.DataFrame, recommendations: dict[str, dict[str, Any]], thresholds: dict[str, float]) -> pd.DataFrame:
    if method_summary.empty:
        return method_summary.copy()
    result = method_summary.copy()
    role_map: dict[str, list[str]] = {}
    caution_map: dict[str, list[str]] = {}
    for payload in recommendations.values():
        role_map.setdefault(payload["method"], []).append(payload["label"])
        if payload.get("caution"):
            caution_map.setdefault(payload["method"], []).append(str(payload["caution"]))

    result["recommended_for"] = result["method"].map(lambda method: "; ".join(role_map.get(method, [])))
    result["caution_label"] = result["method"].map(lambda method: " ".join(caution_map.get(method, [])) or _build_caution(result[result["method"] == method].iloc[0], thresholds))
    explanations = []
    for _, row in result.iterrows():
        if row.get("recommended_for"):
            explanations.append(f"{row['recommended_for']}: {_build_caution(row, thresholds) or 'Selected by the current rule-based recommendation engine.'}")
        elif bool(row.get("pareto_balanced_practical")):
            explanations.append("Balanced-frontier survivor: no other method improves quality, drift, runtime, memory, and compression simultaneously.")
        elif bool(row.get("pareto_systems_efficiency")):
            explanations.append("Systems-frontier survivor: efficient on runtime/memory/compression, but not the strongest quality-retention choice.")
        else:
            dominators = _safe_string(row.get("dominated_by_balanced_practical"))
            if dominators:
                explanations.append(f"Dominated in the balanced frontier by {dominators} under the current objective mix.")
            else:
                explanations.append("Useful reference point, but not selected under the current objectives.")
    result["auto_explanation"] = explanations
    return result


def build_constraint_rankings(method_summary: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, pd.DataFrame]:
    if method_summary.empty:
        return {}
    df = method_summary.copy()
    rankings: dict[str, pd.DataFrame] = {}

    def _prepare(table: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "method",
            "method_family",
            "compression_ratio",
            "peak_vram_gb",
            "avg_runtime_s_per_prompt",
            "imaging_quality",
            "drift_last_imaging_quality",
            "imaging_quality_delta_vs_bf16",
            "drift_last_imaging_quality_delta_vs_bf16",
            "runtime_overhead_vs_bf16_pct",
            "peak_vram_reduction_vs_bf16_pct",
            "recommended_for",
            "caution_label",
        ]
        present = [column for column in columns if column in table.columns]
        return table[present].reset_index(drop=True)

    runtime_max = thresholds.get("runtime_max", np.inf)
    runtime_max = np.inf if runtime_max is None else runtime_max
    vram_max = thresholds.get("vram_max", np.inf)
    vram_max = np.inf if vram_max is None else vram_max
    image_drop = thresholds["acceptable_imaging_drop"]
    drift_drop = thresholds["acceptable_drift_drop"]
    min_compression = thresholds["min_compression"]

    rankings["Best quality under runtime <= X"] = _prepare(
        df[df["avg_runtime_s_per_prompt"].fillna(np.inf) <= runtime_max].sort_values(
            ["imaging_quality", "drift_last_imaging_quality", "quality_reference_score"],
            ascending=[False, False, False],
            na_position="last",
        )
    )
    rankings["Best quality under peak VRAM <= Y"] = _prepare(
        df[df["peak_vram_gb"].fillna(np.inf) <= vram_max].sort_values(
            ["imaging_quality", "drift_last_imaging_quality", "quality_reference_score"],
            ascending=[False, False, False],
            na_position="last",
        )
    )
    rankings["Best compression under imaging-quality drop <= Z"] = _prepare(
        df[df["imaging_quality_drop_vs_bf16"].fillna(np.inf) <= image_drop].sort_values(
            ["compression_ratio", "imaging_quality", "drift_last_imaging_quality"],
            ascending=[False, False, False],
            na_position="last",
        )
    )
    rankings["Best compression under drift drop <= Z"] = _prepare(
        df[df["drift_last_imaging_quality_drop_vs_bf16"].fillna(np.inf) <= drift_drop].sort_values(
            ["compression_ratio", "drift_last_imaging_quality", "imaging_quality"],
            ascending=[False, False, False],
            na_position="last",
        )
    )
    rankings["Best runtime under imaging-quality drop <= Z"] = _prepare(
        df[df["imaging_quality_drop_vs_bf16"].fillna(np.inf) <= image_drop].sort_values(
            ["avg_runtime_s_per_prompt", "compression_ratio", "imaging_quality"],
            ascending=[True, False, False],
            na_position="last",
        )
    )
    rankings["Best runtime under drift drop <= Z"] = _prepare(
        df[df["drift_last_imaging_quality_drop_vs_bf16"].fillna(np.inf) <= drift_drop].sort_values(
            ["avg_runtime_s_per_prompt", "compression_ratio", "drift_last_imaging_quality"],
            ascending=[True, False, False],
            na_position="last",
        )
    )
    rankings["Best balanced score under configurable weights"] = _prepare(
        df[
            (df["avg_runtime_s_per_prompt"].fillna(np.inf) <= runtime_max)
            & (df["peak_vram_gb"].fillna(np.inf) <= vram_max)
            & (df["compression_ratio"].fillna(0) >= min_compression)
            & (df["imaging_quality_drop_vs_bf16"].fillna(np.inf) <= image_drop)
            & (df["drift_last_imaging_quality_drop_vs_bf16"].fillna(np.inf) <= drift_drop)
        ].sort_values(["utility_score", "imaging_quality", "drift_last_imaging_quality"], ascending=[False, False, False], na_position="last")
    )
    return rankings


def build_explainability_table(method_summary: pd.DataFrame) -> pd.DataFrame:
    if method_summary.empty:
        return pd.DataFrame()
    columns = [
        "method",
        "method_family",
        "bit_width_label",
        "variant_tags",
        "quantization_details",
        "pareto_balanced_practical",
        "pareto_quality_preserving_compression",
        "pareto_systems_efficiency",
        "pareto_quality_first",
        "recommended_for",
        "caution_label",
        "auto_explanation",
    ]
    present = [column for column in columns if column in method_summary.columns]
    return method_summary[present].sort_values("method").reset_index(drop=True)


def generate_experiment_takeaways(method_summary: pd.DataFrame, recommendations: dict[str, dict[str, Any]], thresholds: dict[str, float]) -> list[str]:
    takeaways: list[str] = []
    default_payload = recommendations.get("default_practical")
    aggressive_payload = recommendations.get("aggressive_compression")
    fastest_payload = recommendations.get("fastest")
    quality_payload = recommendations.get("quality_first")
    if default_payload:
        row = default_payload["row"]
        takeaways.append(
            f"Default operating point: `{row['method']}` best balances visual quality, drift stability, runtime, and KV compression under the current filters."
        )
    if aggressive_payload:
        row = aggressive_payload["row"]
        takeaways.append(
            f"Aggressive compression option: `{row['method']}` reaches {row['compression_ratio']:.2f}x compression while staying inside the active quality/drift tolerance."
        )
    if quality_payload:
        row = quality_payload["row"]
        takeaways.append(
            f"Quality-first note: `{row['method']}` is the strongest non-BF16 fidelity/quality retainer, but its runtime cost may still limit default deployment."
        )
    if fastest_payload:
        row = fastest_payload["row"]
        caution = _build_caution(row, thresholds)
        if caution:
            takeaways.append(f"Runtime note: `{row['method']}` is the fastest non-BF16 option, but {caution.lower()}")
        else:
            takeaways.append(f"Runtime note: `{row['method']}` is the fastest non-BF16 option in the current stack.")

    kv_not_vram = method_summary[
        (method_summary["compression_ratio"].fillna(0) > 1.5)
        & (method_summary["peak_vram_reduction_vs_bf16_pct"].fillna(-np.inf) <= 0)
    ]
    if not kv_not_vram.empty:
        row = kv_not_vram.sort_values("compression_ratio", ascending=False).iloc[0]
        takeaways.append(
            f"Implementation caveat: `{row['method']}` materially compresses KV state without lowering measured peak VRAM, which indicates temporary buffers and integration overhead still matter at the current horizon length."
        )
    takeaways.append(
        "Current runs are short-horizon proxies rather than definitive long-horizon validation, so drift metrics are used as the best available stability signal."
    )
    return takeaways[:5]


def build_dashboard_analysis(
    filtered_df: pd.DataFrame,
    source_catalog: pd.DataFrame,
    benchmark: str,
    score_weights: dict[str, float],
    thresholds: dict[str, float],
    primary_source_path: str | None = None,
) -> DecisionAnalysis:
    run_summary = build_run_level_summary(filtered_df)
    run_summary = add_bf16_relative_metrics(run_summary)
    method_summary = build_method_summary(run_summary)
    method_summary = add_bf16_relative_metrics(method_summary)
    method_summary = attach_frontier_results(method_summary)
    method_summary = apply_scores(method_summary, score_weights)
    recommendations = recommend_methods(method_summary, thresholds)
    method_summary = annotate_recommendations(method_summary, recommendations, thresholds)
    explainability = build_explainability_table(method_summary)
    constraint_tables = build_constraint_rankings(method_summary, thresholds)
    frontier_members = {
        frontier_key: method_summary.loc[method_summary[f"pareto_{frontier_key}"].fillna(False), "method"].astype(str).tolist()
        for frontier_key in FRONTIER_DEFINITIONS
    }
    takeaways = generate_experiment_takeaways(method_summary, recommendations, thresholds)
    return DecisionAnalysis(
        benchmark=benchmark,
        run_summary=run_summary,
        method_summary=method_summary,
        explainability_table=explainability,
        recommendations=recommendations,
        takeaways=takeaways,
        frontier_members=frontier_members,
        constraint_tables=constraint_tables,
        score_weights=normalize_weights(score_weights),
        thresholds=thresholds,
        source_catalog=source_catalog,
        primary_source_path=primary_source_path,
    )
