#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is unavailable in the default interpreter; run this script with "
        "/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python."
    ) from exc

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "BF16",
    "RTN_INT4",
    "RTN_INT2",
    "KIVI_INT4",
    "KIVI_INT2",
    "QUAROT_KV_INT4",
    "QUAROT_KV_INT2",
]

SOURCE_RANK = {"combined": 0, "suraj": 1, "vaishak": 2}
HELPER_BASELINE_METHODS = {"BF16", "RTN_INT4"}

META_LABEL_KEYS: dict[str, list[tuple[str, str]]] = {
    "AGE_TIER": [
        ("age_tier_config_recent_ratio", "recent"),
        ("age_tier_old_bits", "old"),
        ("age_tier_recent_bits", "new"),
    ],
    "FLOWCACHE_ADAPTIVE": [
        ("flowcache_adaptive_config_important_old_ratio", "old"),
        ("flowcache_adaptive_config_chunk_recent_ratio", "recent"),
    ],
    "FLOWCACHE_HYBRID": [
        ("flowcache_config_chunk_recent_ratio", "recent"),
        ("flowcache_min_layer_budget_scale", "min"),
        ("flowcache_max_layer_budget_scale", "max"),
    ],
    "FLOWCACHE_NATIVE": [
        ("flowcache_native_rel_l1_thresh", "thr"),
        ("flowcache_native_warmup_steps", "warm"),
    ],
    "FLOWCACHE_NATIVE_SOFT_PRUNE": [
        ("flowcache_native_rel_l1_thresh", "thr"),
        ("flowcache_prune_retained_old_ratio", "keep"),
    ],
    "FLOWCACHE_PRUNE": [
        ("flowcache_prune_retained_old_ratio", "keep"),
        ("flowcache_prune_refresh_gap_chunks", "gap"),
    ],
    "FLOWCACHE_SOFT_PRUNE": [
        ("flowcache_prune_retained_old_ratio", "keep"),
        ("flowcache_prune_refresh_gap_chunks", "gap"),
    ],
    "SPATIAL_MIXED": [
        ("spatial_variance_threshold", "var"),
        ("spatial_target_foreground_ratio", "fg"),
    ],
    "TPTQ": [
        ("tptq_config_recent_ratio", "recent"),
        ("tptq_outlier_max_ratio", "out"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate long-10 MovieGen analysis figures from the combined registry."
    )
    parser.add_argument(
        "--registry-csv",
        type=Path,
        default=Path("results/combined/registry/combined_registry.csv"),
        help="Path to the combined registry CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/combined/analysis_figures"),
        help="Directory where figures and summary tables will be written.",
    )
    return parser.parse_args()


def _safe_json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_json_parse(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def _extract_vbench_scalar(value: Any) -> float:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, (int, float)):
            return float(first)
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def _canonical_payload_key(method: str, raw_payload: Any) -> str:
    if method == "BF16":
        return "BF16"
    payload = _safe_json_parse(raw_payload)
    if payload:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"{method}::{raw_payload}"


def _format_meta_value(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        if abs(value) < 0.01:
            return f"{value:.3g}"
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _method_family_for_row(row: pd.Series, payload: dict[str, Any]) -> str:
    family = row.get("method_family")
    if isinstance(family, str) and family:
        return family
    family = payload.get("method_family")
    if isinstance(family, str) and family:
        return family
    method = str(row.get("method") or "")
    if method.startswith("FLOWCACHE_NATIVE_SOFT_PRUNE"):
        return "FLOWCACHE_NATIVE_SOFT_PRUNE"
    if method.startswith("FLOWCACHE_SOFT_PRUNE"):
        return "FLOWCACHE_SOFT_PRUNE"
    if method.startswith("FLOWCACHE_PRUNE"):
        return "FLOWCACHE_PRUNE"
    if method.startswith("FLOWCACHE_"):
        return method.removeprefix("FLOWCACHE_").split("_INT")[0]
    return method


def _build_display_label(
    row: pd.Series,
    method_counts: dict[str, int],
) -> str:
    method = str(row["method"])
    if method == "BF16":
        return "BF16"

    family = str(row["method_family"])
    quant_meta = row["_quant_meta"]
    cache_policy = row["_cache_policy"]

    suffix_parts: list[str] = []
    if method_counts.get(method, 0) > 1:
        for key, label in META_LABEL_KEYS.get(family, []):
            if key in quant_meta:
                suffix_parts.append(f"{label}={_format_meta_value(quant_meta[key])}")
        cadence = cache_policy.get("cadence")
        recent_blocks = cache_policy.get("recent_blocks")
        if not suffix_parts and cadence:
            suffix_parts.append(f"cad={cadence}")
        if not suffix_parts and recent_blocks not in (None, "", 0, "0"):
            suffix_parts.append(f"recent={recent_blocks}")
        if not suffix_parts:
            suffix_parts.append(row["config_id"][:6])

    if suffix_parts:
        return f"{method} | {', '.join(suffix_parts[:2])}"
    return method


def _uniquify_labels(df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    labels: list[str] = []
    for _, row in df.iterrows():
        base = str(row["display_label"])
        index = counts.get(base, 0)
        counts[base] = index + 1
        if index == 0:
            labels.append(base)
        else:
            labels.append(f"{base} [{row['config_id'][:6]}]")
    out = df.copy()
    out["display_label"] = labels
    return out


def select_representative_rows(registry_csv: Path) -> pd.DataFrame:
    registry = pd.read_csv(registry_csv)
    mask = (
        registry["benchmark"].eq("moviegen")
        & registry["is_ten_second"].map(_as_bool)
        & registry["long10_dashboard_ready"].map(_as_bool)
    )
    rows = registry.loc[mask].copy()

    helper_mask = (
        rows["source_repo"].eq("combined")
        & rows["experiment_type"].eq("combined_backfill")
        & rows["method"].isin(HELPER_BASELINE_METHODS)
    )
    rows = rows.loc[~helper_mask].copy()

    rows["prompt_count_num"] = pd.to_numeric(rows["prompt_count"], errors="coerce").fillna(0)
    rows["video_count_num"] = pd.to_numeric(rows["video_count"], errors="coerce").fillna(0)
    rows["source_rank"] = rows["source_repo"].map(SOURCE_RANK).fillna(9)
    rows["group_key"] = rows.apply(
        lambda rec: _canonical_payload_key(str(rec["method"]), rec.get("config_payload")),
        axis=1,
    )

    rows = rows.sort_values(
        ["group_key", "source_rank", "prompt_count_num", "video_count_num", "run_name"],
        ascending=[True, True, False, False, True],
    )
    representatives = rows.groupby("group_key", as_index=False, sort=False).first()

    enriched_rows: list[dict[str, Any]] = []
    for _, row in representatives.iterrows():
        payload = _safe_json_parse(row.get("config_payload"))
        quant_meta = _safe_json_parse(row.get("quant_meta")) or payload.get("quant_meta", {})
        cache_policy = payload.get("cache_policy", {}) if isinstance(payload.get("cache_policy"), dict) else {}

        run_root = Path(row["run_root"])
        method = str(row["method"])
        efficiency = _safe_json_load(run_root / "metrics" / f"efficiency_{method}.json")
        fidelity = _safe_json_load(run_root / "metrics" / f"fidelity_{method}.json")
        vbench = _safe_json_load(run_root / "metrics" / f"vbench_{method}.json")

        aggregate = fidelity.get("aggregate", {}) if isinstance(fidelity.get("aggregate"), dict) else {}

        prompt_count = _as_float(row.get("prompt_count"))
        total_runtime_s = _as_float(efficiency.get("total_runtime_s", row.get("total_runtime_s")))
        avg_runtime_s_per_prompt = _as_float(efficiency.get("avg_runtime_s_per_prompt"))
        if np.isnan(avg_runtime_s_per_prompt) and prompt_count > 0:
            avg_runtime_s_per_prompt = total_runtime_s / prompt_count

        peak_vram_bytes = _as_float(efficiency.get("peak_vram_bytes", row.get("peak_vram_bytes")))
        record = {
            "source_repo": row["source_repo"],
            "run_name": row["run_name"],
            "run_root": row["run_root"],
            "config_id": str(row["config_id"]),
            "method": method,
            "method_family": _method_family_for_row(row, payload),
            "config_payload": json.dumps(payload, sort_keys=True) if payload else "",
            "quant_meta": json.dumps(quant_meta, sort_keys=True) if quant_meta else "{}",
            "compression_ratio": _as_float(efficiency.get("compression_ratio", row.get("compression_ratio"))),
            "peak_vram_bytes": peak_vram_bytes,
            "peak_vram_gb": peak_vram_bytes / (1024**3) if np.isfinite(peak_vram_bytes) else float("nan"),
            "total_runtime_s": total_runtime_s,
            "avg_runtime_s_per_prompt": avg_runtime_s_per_prompt,
            "quantize_time_s": _as_float(efficiency.get("quantize_time_s")),
            "dequantize_time_s": _as_float(efficiency.get("dequantize_time_s")),
            "psnr": _as_float(aggregate.get("psnr")),
            "ssim": _as_float(aggregate.get("ssim")),
            "lpips": _as_float(aggregate.get("lpips")),
            "background_consistency": _extract_vbench_scalar(vbench.get("background_consistency")),
            "imaging_quality": _extract_vbench_scalar(vbench.get("imaging_quality")),
            "subject_consistency": _extract_vbench_scalar(vbench.get("subject_consistency")),
            "aesthetic_quality": _extract_vbench_scalar(vbench.get("aesthetic_quality")),
            "_quant_meta": quant_meta if isinstance(quant_meta, dict) else {},
            "_cache_policy": cache_policy,
        }
        enriched_rows.append(record)

    frame = pd.DataFrame(enriched_rows)
    metric_cols = [
        "compression_ratio",
        "peak_vram_bytes",
        "peak_vram_gb",
        "total_runtime_s",
        "avg_runtime_s_per_prompt",
        "quantize_time_s",
        "dequantize_time_s",
        "psnr",
        "ssim",
        "lpips",
        "background_consistency",
        "imaging_quality",
        "subject_consistency",
        "aesthetic_quality",
    ]
    frame[metric_cols] = frame[metric_cols].replace([np.inf, -np.inf], np.nan)

    method_counts = frame["method"].value_counts().to_dict()
    frame["display_label"] = frame.apply(_build_display_label, axis=1, method_counts=method_counts)
    frame = _uniquify_labels(frame)
    return frame


def _family_palette(df: pd.DataFrame) -> dict[str, Any]:
    families = sorted(df["method_family"].dropna().unique().tolist(), key=lambda family: (family not in METHOD_ORDER, family))
    cmap = plt.colormaps.get_cmap("tab20")
    colors: dict[str, Any] = {}
    for index, family in enumerate(families):
        colors[family] = cmap(index % cmap.N)
    colors["BF16"] = "#6b7280"
    return colors


def _legend_handles(df: pd.DataFrame, palette: dict[str, Any]) -> list[Patch]:
    families = sorted(df["method_family"].dropna().unique().tolist())
    return [Patch(facecolor=palette.get(family, "#333333"), label=family) for family in families]


def _plot_bar_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    palette: dict[str, Any],
    ascending: bool,
    reference: float | None = None,
) -> None:
    sub = frame.loc[frame[metric].notna()].sort_values(metric, ascending=ascending)
    if sub.empty:
        ax.set_visible(False)
        return
    y = np.arange(len(sub))
    ax.barh(
        y,
        sub[metric],
        color=[palette.get(family, "#333333") for family in sub["method_family"]],
        edgecolor="black",
        linewidth=0.2,
    )
    ax.set_yticks(y, sub["display_label"])
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    if reference is not None and np.isfinite(reference):
        ax.axvline(reference, color="#111827", linestyle="--", linewidth=1.0, alpha=0.85)


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_memory_figure(df: pd.DataFrame, out_dir: Path, palette: dict[str, Any]) -> None:
    height = max(12.0, 0.38 * len(df) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(18, height), constrained_layout=True)
    bf16_vram = df.loc[df["method"].eq("BF16"), "peak_vram_gb"].dropna()
    _plot_bar_panel(
        axes[0],
        df,
        metric="compression_ratio",
        title="KV compression ratio",
        xlabel="compression ratio vs BF16 (higher is better)",
        palette=palette,
        ascending=False,
        reference=1.0,
    )
    _plot_bar_panel(
        axes[1],
        df,
        metric="peak_vram_gb",
        title="Peak VRAM",
        xlabel="peak VRAM (GB, lower is better)",
        palette=palette,
        ascending=True,
        reference=float(bf16_vram.iloc[0]) if not bf16_vram.empty else None,
    )
    fig.suptitle("MovieGen 10s long-horizon memory footprint", fontsize=16)
    fig.legend(
        handles=_legend_handles(df, palette),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
    )
    _save_figure(fig, out_dir, "moviegen_long10_memory_footprint")


def generate_quality_figure(df: pd.DataFrame, out_dir: Path, palette: dict[str, Any]) -> None:
    quality_df = df.loc[~df["method"].eq("BF16")].copy()
    height = max(12.0, 0.36 * len(quality_df) + 2.0)
    fig, axes = plt.subplots(2, 2, figsize=(18, height), constrained_layout=True)
    panels = [
        ("lpips", "LPIPS", "LPIPS vs BF16 (lower is better)", True),
        ("ssim", "SSIM", "SSIM vs BF16 (higher is better)", False),
        ("imaging_quality", "VBench imaging quality", "VBench imaging quality (higher is better)", False),
        ("subject_consistency", "VBench subject consistency", "VBench subject consistency (higher is better)", False),
    ]
    for ax, (metric, title, xlabel, ascending) in zip(axes.flatten(), panels, strict=True):
        _plot_bar_panel(
            ax,
            quality_df,
            metric=metric,
            title=title,
            xlabel=xlabel,
            palette=palette,
            ascending=ascending,
        )
    fig.suptitle("MovieGen 10s long-horizon generation quality", fontsize=16)
    fig.legend(
        handles=_legend_handles(quality_df, palette),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
    )
    _save_figure(fig, out_dir, "moviegen_long10_generation_quality")


def generate_performance_figure(df: pd.DataFrame, out_dir: Path, palette: dict[str, Any]) -> None:
    height = max(12.0, 0.38 * len(df) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(18, height), constrained_layout=True)
    bf16_runtime = df.loc[df["method"].eq("BF16"), "avg_runtime_s_per_prompt"].dropna()
    overhead = df.copy()
    overhead["quant_overhead_s"] = (
        overhead["quantize_time_s"].fillna(0.0) + overhead["dequantize_time_s"].fillna(0.0)
    )
    _plot_bar_panel(
        axes[0],
        df,
        metric="avg_runtime_s_per_prompt",
        title="Average runtime per prompt",
        xlabel="seconds per prompt (lower is better)",
        palette=palette,
        ascending=True,
        reference=float(bf16_runtime.iloc[0]) if not bf16_runtime.empty else None,
    )
    _plot_bar_panel(
        axes[1],
        overhead,
        metric="quant_overhead_s",
        title="Quantization + dequantization overhead",
        xlabel="seconds (lower is better)",
        palette=palette,
        ascending=True,
        reference=0.0,
    )
    fig.suptitle("MovieGen 10s long-horizon performance", fontsize=16)
    fig.legend(
        handles=_legend_handles(df, palette),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
    )
    _save_figure(fig, out_dir, "moviegen_long10_performance")


def write_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    export = df.drop(columns=["_quant_meta", "_cache_policy"]).copy()
    export = export.sort_values(["method_family", "method", "display_label"]).reset_index(drop=True)
    export.to_csv(out_dir / "moviegen_long10_representative_metrics.csv", index=False)
    export.to_json(out_dir / "moviegen_long10_representative_metrics.json", orient="records", indent=2)

    metadata = {
        "selection_rules": [
            "benchmark == moviegen",
            "is_ten_second == True",
            "long10_dashboard_ready == True",
            "exclude combined_backfill helper BF16/RTN_INT4 rows",
            "collapse duplicate source runs by canonical config payload",
            "collapse BF16 to a single reference row",
        ],
        "num_representative_rows": int(len(export)),
        "methods": sorted(export["method"].unique().tolist()),
        "sources": sorted(export["source_repo"].unique().tolist()),
    }
    (out_dir / "moviegen_long10_representative_metrics.metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = select_representative_rows(args.registry_csv.resolve())
    if frame.empty:
        raise RuntimeError("No completed 10-second MovieGen rows were found in the registry.")

    palette = _family_palette(frame)
    write_outputs(frame, out_dir)
    generate_memory_figure(frame, out_dir, palette)
    generate_quality_figure(frame, out_dir, palette)
    generate_performance_figure(frame, out_dir, palette)

    print(f"Wrote outputs to {out_dir}")
    print(f"Representative rows: {len(frame)}")
    print(f"Methods: {', '.join(sorted(frame['method'].unique().tolist()))}")


if __name__ == "__main__":
    main()
