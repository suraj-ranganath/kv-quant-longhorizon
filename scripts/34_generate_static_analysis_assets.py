#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT = REPO_ROOT / "docs" / "analysis_assets"
DATASET_PATH = REPO_ROOT / "results" / "combined" / "combined_comparison_dataset.csv"
RECOMMENDATION_FOCUS = "Single-GPU practical"
SOURCE_RANK = {"combined": 0, "suraj": 1, "vaishak": 2}

from dashboard.data_sources import load_dashboard_workspace
from dashboard.decision_analysis import (
    DEFAULT_THRESHOLDS,
    FRONTIER_DEFINITIONS,
    build_dashboard_analysis,
    get_recommendation_sort,
)

BENCHMARK_SPECS = {
    "moviegen": {
        "prefix": "moviegen",
        "pretty": "MovieGen",
        "drift_metric": "moviegen_drift_last_imaging_quality",
    },
    "storyeval": {
        "prefix": "storyeval",
        "pretty": "StoryEval",
        "drift_metric": "storyeval_drift_last_imaging_quality",
    },
}

FRONTIER_PLOT_SPECS = {
    "balanced_practical": {
        "x": "compression_ratio",
        "y": "drift_last_imaging_quality",
        "x_label": "Compression ratio",
        "y_label": "Drift-last imaging quality",
    },
    "quality_preserving_compression": {
        "x": "compression_ratio",
        "y": "ssim",
        "x_label": "Compression ratio",
        "y_label": "SSIM",
    },
    "systems_efficiency": {
        "x": "compression_ratio",
        "y": "peak_vram_gb",
        "x_label": "Compression ratio",
        "y_label": "Peak VRAM (GB)",
    },
    "quality_first": {
        "x": "avg_runtime_s_per_prompt",
        "y": "ssim",
        "x_label": "Runtime / prompt (s)",
        "y_label": "SSIM",
    },
}

GENERAL_PLOT_SPECS = {
    "peak_vram_vs_quality": {
        "x": "peak_vram_gb",
        "y": "ssim",
        "x_label": "Peak VRAM (GB)",
        "y_label": "SSIM",
        "size": "compression_ratio",
        "title": "Peak VRAM vs SSIM",
    },
    "vram_vs_runtime": {
        "x": "peak_vram_gb",
        "y": "avg_runtime_s_per_prompt",
        "x_label": "Peak VRAM (GB)",
        "y_label": "Runtime / prompt (s)",
        "size": None,
        "title": "Peak VRAM vs runtime",
    },
    "compression_vs_peak_vram": {
        "x": "compression_ratio",
        "y": "peak_vram_gb",
        "x_label": "Compression ratio",
        "y_label": "Peak VRAM (GB)",
        "size": None,
        "title": "Compression ratio vs peak VRAM",
    },
}

RAW_TABLE_COLUMNS = [
    "method",
    "method_family",
    "bit_width_label",
    "source_users",
    "run_count",
    "prompt_count",
    "seed_count",
    "compression_ratio",
    "peak_vram_gb",
    "peak_compressed_kv_gb",
    "avg_runtime_s_per_prompt",
    "imaging_quality",
    "drift_last_imaging_quality",
    "psnr",
    "ssim",
    "lpips",
    "psnr_delta_vs_bf16",
    "ssim_delta_vs_bf16",
    "lpips_delta_vs_bf16",
    "drift_last_imaging_quality_delta_vs_bf16",
    "runtime_overhead_vs_bf16_pct",
    "peak_vram_reduction_vs_bf16_pct",
    "compression_gain_vs_bf16",
    "pareto_balanced_practical",
    "pareto_quality_preserving_compression",
    "pareto_systems_efficiency",
    "pareto_quality_first",
    "recommended_for",
    "caution_label",
]

README_TABLE_COLUMNS = {
    "moviegen": [
        "method",
        "method_family",
        "compression_ratio",
        "peak_vram_gb",
        "avg_runtime_s_per_prompt",
        "imaging_quality",
        "drift_last_imaging_quality",
        "psnr",
        "ssim",
        "lpips",
    ],
    "storyeval": [
        "method",
        "method_family",
        "compression_ratio",
        "peak_vram_gb",
        "avg_runtime_s_per_prompt",
        "imaging_quality",
        "drift_last_imaging_quality",
        "background_consistency",
        "subject_consistency",
        "aesthetic_quality",
    ],
}


def _benchmark_df(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    out = df[df["benchmark"].astype(str).str.lower() == benchmark].copy()
    if "is_ten_second" in out.columns and out["is_ten_second"].notna().any():
        ten_second = out[out["is_ten_second"].astype(bool)]
        if not ten_second.empty:
            out = ten_second.copy()
    return out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _load_trace_records(run_root: Path, benchmark: str, method: str) -> list[dict[str, Any]]:
    if benchmark == "storyeval":
        path = run_root / "logs" / "vram_trace_storyeval.jsonl"
        rows = []
        for payload in _load_jsonl(path):
            item = dict(payload)
            item["method"] = item.get("method", method)
            rows.append(item)
        return rows
    return _load_jsonl(run_root / "logs" / f"vram_trace_{method}.jsonl")


def _pick_representative_runs(filtered_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (method, method_display), group in filtered_df.groupby(["method", "method_display"], dropna=False):
        rows = group.copy()
        rows["source_rank"] = rows["source_repo"].map(SOURCE_RANK).fillna(9)
        rows["video_count_num"] = 1
        if "video_name" in rows.columns:
            counts = rows.groupby("run_root")["video_name"].transform("nunique")
            rows["video_count_num"] = counts
        rows = rows.sort_values(
            ["source_rank", "video_count_num", "run_name"],
            ascending=[True, False, True],
            na_position="last",
        )
        records.append(rows.iloc[0].to_dict())
    return pd.DataFrame(records)


def _build_trace_df(filtered_df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    representatives = _pick_representative_runs(filtered_df)
    rows: list[dict[str, Any]] = []
    for rec in representatives.to_dict(orient="records"):
        run_root = Path(str(rec["run_root"]))
        method = str(rec["method"])
        method_display = str(rec["method_display"])
        trace_records = _load_trace_records(run_root, benchmark, method)
        fallback_bf16_kv = rec.get("bf16_kv_bytes")
        fallback_compressed_kv = rec.get("compressed_kv_bytes")
        for trace in trace_records:
            if str(trace.get("method", method)) != method:
                continue
            prompt_id = trace.get("prompt_id")
            if prompt_id is None:
                continue
            seed = trace.get("seed")
            for sample in trace.get("samples", []) or []:
                allocated = sample.get("allocated_bytes")
                reserved = sample.get("reserved_bytes")
                t_s = sample.get("t_s")
                if allocated is None or reserved is None or t_s is None:
                    continue
                bf16_kv = sample.get("bf16_kv_bytes")
                if bf16_kv in (None, 0, 0.0):
                    bf16_kv = fallback_bf16_kv
                compressed_kv = sample.get("compressed_kv_bytes")
                if compressed_kv in (None, 0, 0.0):
                    compressed_kv = fallback_compressed_kv
                rows.append(
                    {
                        "method": method,
                        "method_display": method_display,
                        "prompt_id": str(prompt_id),
                        "seed": int(seed) if seed is not None and not pd.isna(seed) else None,
                        "t_s": float(t_s),
                        "allocated_gb": float(allocated) / (1024**3),
                        "reserved_gb": float(reserved) / (1024**3),
                        "bf16_kv_gb": float(bf16_kv or 0) / (1024**3),
                        "compressed_kv_gb": float(compressed_kv or 0) / (1024**3),
                    }
                )
    return pd.DataFrame(rows)


def _select_trace_slice(trace_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if trace_df.empty:
        return trace_df, {}
    coverage = (
        trace_df.groupby(["prompt_id", "seed"], dropna=False)["method_display"]
        .nunique()
        .reset_index(name="method_count")
        .sort_values(["method_count", "prompt_id", "seed"], ascending=[False, True, True])
    )
    selected = coverage.iloc[0]
    prompt_id = str(selected["prompt_id"])
    seed = selected["seed"]
    slice_df = trace_df[(trace_df["prompt_id"].astype(str) == prompt_id) & (trace_df["seed"] == seed)].copy()
    meta = {
        "prompt_id": prompt_id,
        "seed": int(seed) if seed is not None and not pd.isna(seed) else None,
        "method_count": int(selected["method_count"]),
    }
    return slice_df, meta


def _load_drift_curves(filtered_df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    representatives = _pick_representative_runs(filtered_df)
    rows: list[dict[str, Any]] = []
    for rec in representatives.to_dict(orient="records"):
        run_root = Path(str(rec["run_root"]))
        method_display = str(rec["method_display"])
        if benchmark == "storyeval":
            path = run_root / "metrics" / "drift_imaging_quality.json"
        else:
            path = run_root / "metrics" / f"drift_{rec['method']}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for point in payload.get("curve", []) or []:
            imaging = point.get("imaging_quality")
            if isinstance(imaging, list) and imaging:
                imaging = imaging[0]
            if imaging is None:
                continue
            rows.append(
                {
                    "method": method_display,
                    "frame_cap": point.get("frame_cap"),
                    "seconds": point.get("seconds"),
                    "imaging_quality": float(imaging),
                }
            )
    return pd.DataFrame(rows)


def _format_markdown_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return f"{float(value):.4f}"
        return str(value)
    return str(value)


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    cols = [str(c) for c in df.columns.tolist()]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_markdown_value(row[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
    flag_col: str | None = None,
    size_col: str | None = None,
) -> None:
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    frontier_mask = plot_df[flag_col].fillna(False) if flag_col and flag_col in plot_df.columns else pd.Series(False, index=plot_df.index)
    bubble_sizes = None
    if size_col and size_col in plot_df.columns:
        size = pd.to_numeric(plot_df[size_col], errors="coerce").fillna(1.0).clip(lower=0.5)
        bubble_sizes = 40 + 35 * size
    else:
        bubble_sizes = np.full(len(plot_df), 70.0)

    dominated = plot_df[~frontier_mask]
    frontier = plot_df[frontier_mask]
    if not dominated.empty:
        ax.scatter(
            dominated[x_col],
            dominated[y_col],
            s=bubble_sizes[~frontier_mask.to_numpy()],
            c="#9ca3af",
            alpha=0.8,
            edgecolors="#1f2937",
            linewidths=0.6,
            label="Dominated",
        )
    if not frontier.empty:
        ax.scatter(
            frontier[x_col],
            frontier[y_col],
            s=bubble_sizes[frontier_mask.to_numpy()],
            c="#16a34a",
            alpha=0.95,
            edgecolors="#14532d",
            linewidths=0.9,
            label="Pareto-optimal" if flag_col else "Methods",
        )
    bf16 = plot_df[plot_df["method"] == "BF16"]
    if not bf16.empty:
        row = bf16.iloc[0]
        ax.axvline(float(row[x_col]), color="#111827", linestyle=":", linewidth=1.2)
        ax.axhline(float(row[y_col]), color="#111827", linestyle=":", linewidth=1.2)
        ax.scatter([row[x_col]], [row[y_col]], s=120, c="#111827", marker="D", label="BF16")
    for _, row in frontier.iterrows():
        ax.annotate(str(row["method"]), (row[x_col], row[y_col]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_trace_plot(
    trace_df: pd.DataFrame,
    y_col: str,
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    if trace_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    order = sorted(trace_df["method_display"].dropna().unique().tolist())
    cmap = plt.get_cmap("tab20")
    for idx, method in enumerate(order):
        rows = trace_df[trace_df["method_display"] == method].sort_values("t_s")
        ax.plot(rows["t_s"], rows[y_col], label=method, linewidth=1.8, color=cmap(idx % 20))
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_drift_plot(drift_df: pd.DataFrame, benchmark: str, out_path: Path) -> None:
    if drift_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    x_col = "seconds" if benchmark == "storyeval" and drift_df["seconds"].notna().any() else "frame_cap"
    order = sorted(drift_df["method"].dropna().unique().tolist())
    cmap = plt.get_cmap("tab20")
    for idx, method in enumerate(order):
        rows = drift_df[drift_df["method"] == method].sort_values(x_col)
        ax.plot(rows[x_col], rows["imaging_quality"], marker="o", label=method, linewidth=1.8, color=cmap(idx % 20))
    ax.set_title(f"{BENCHMARK_SPECS[benchmark]['pretty']} drift curves")
    ax.set_xlabel("seconds" if x_col == "seconds" else "frame_cap")
    ax.set_ylabel("imaging_quality")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _rounded_thresholds() -> dict[str, float | None]:
    return {
        "acceptable_ssim_drop": DEFAULT_THRESHOLDS["acceptable_ssim_drop"],
        "acceptable_lpips_increase": DEFAULT_THRESHOLDS["acceptable_lpips_increase"],
        "acceptable_drift_drop": DEFAULT_THRESHOLDS["acceptable_drift_drop"],
        "min_compression": DEFAULT_THRESHOLDS["min_compression"],
        "runtime_max": None,
        "vram_max": None,
    }


def _generate_for_benchmark(primary_df: pd.DataFrame, source_catalog: pd.DataFrame, benchmark: str) -> dict[str, Any]:
    filtered_df = _benchmark_df(primary_df, benchmark)
    analysis = build_dashboard_analysis(
        filtered_df=filtered_df,
        source_catalog=source_catalog,
        benchmark=benchmark,
        recommendation_focus=RECOMMENDATION_FOCUS,
        thresholds=_rounded_thresholds(),
        primary_source_path=str(DATASET_PATH.relative_to(REPO_ROOT)),
    )
    out_dir = OUT_ROOT / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    method_df = analysis.method_summary.copy()
    sort_cols, sort_asc = get_recommendation_sort(analysis.recommendation_focus)
    sort_cols = [c for c in sort_cols if c in method_df.columns]
    if sort_cols:
        method_df = method_df.sort_values(sort_cols, ascending=sort_asc[: len(sort_cols)], na_position="last")

    full_cols = [c for c in RAW_TABLE_COLUMNS if c in method_df.columns]
    full_table = method_df[full_cols].copy()
    full_table.to_csv(out_dir / "derived_method_table.csv", index=False)
    _write_markdown_table(full_table, out_dir / "derived_method_table.md")

    readme_cols = [c for c in README_TABLE_COLUMNS[benchmark] if c in method_df.columns]
    readme_table = method_df[readme_cols].copy()
    _write_markdown_table(readme_table, out_dir / "readme_method_table.md")

    for frontier_key, spec in FRONTIER_PLOT_SPECS.items():
        flag_col = f"pareto_{frontier_key}"
        _save_scatter(
            method_df,
            x_col=spec["x"],
            y_col=spec["y"],
            title=f"{BENCHMARK_SPECS[benchmark]['pretty']} {FRONTIER_DEFINITIONS[frontier_key]['label']}",
            x_label=spec["x_label"],
            y_label=spec["y_label"],
            out_path=out_dir / f"{frontier_key}.png",
            flag_col=flag_col,
        )

    for plot_name, spec in GENERAL_PLOT_SPECS.items():
        _save_scatter(
            method_df,
            x_col=spec["x"],
            y_col=spec["y"],
            title=f"{BENCHMARK_SPECS[benchmark]['pretty']} {spec['title']}",
            x_label=spec["x_label"],
            y_label=spec["y_label"],
            out_path=out_dir / f"{plot_name}.png",
            size_col=spec["size"],
        )

    drift_df = _load_drift_curves(filtered_df, benchmark)
    if not drift_df.empty:
        _save_drift_plot(drift_df, benchmark, out_dir / "drift_curves.png")
        drift_df.to_csv(out_dir / "drift_curve_points.csv", index=False)

    trace_df = _build_trace_df(filtered_df, benchmark)
    trace_slice, trace_meta = _select_trace_slice(trace_df)
    if not trace_slice.empty:
        _save_trace_plot(
            trace_slice,
            y_col="allocated_gb",
            title=f"{BENCHMARK_SPECS[benchmark]['pretty']} VRAM trace ({trace_meta['prompt_id']}, seed {trace_meta['seed']})",
            y_label="allocated_gb",
            out_path=out_dir / "trace_vram_allocated.png",
        )
        _save_trace_plot(
            trace_slice,
            y_col="compressed_kv_gb",
            title=f"{BENCHMARK_SPECS[benchmark]['pretty']} KV-cache trace ({trace_meta['prompt_id']}, seed {trace_meta['seed']})",
            y_label="compressed_kv_gb",
            out_path=out_dir / "trace_kv_compressed.png",
        )
        peak_summary = (
            trace_slice.groupby("method_display", as_index=False)[["allocated_gb", "reserved_gb", "compressed_kv_gb", "bf16_kv_gb"]]
            .max()
            .sort_values("allocated_gb", ascending=False)
        )
        peak_summary.to_csv(out_dir / "trace_peak_summary.csv", index=False)
        _write_markdown_table(peak_summary, out_dir / "trace_peak_summary.md")
        (out_dir / "trace_selection.json").write_text(json.dumps(trace_meta, indent=2), encoding="utf-8")

    summary = {
        "benchmark": benchmark,
        "method_count": int(method_df["method"].nunique()),
        "plots": sorted([p.name for p in out_dir.glob("*.png")]),
        "tables": sorted([p.name for p in out_dir.glob("*.csv")]) + sorted([p.name for p in out_dir.glob("*.md")]),
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    workspace = load_dashboard_workspace(REPO_ROOT)
    primary_df = workspace.get("primary_df", pd.DataFrame())
    source_catalog = workspace.get("source_catalog", pd.DataFrame())
    if primary_df.empty:
        raise SystemExit(f"No combined dataset available at {DATASET_PATH}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = {
        benchmark: _generate_for_benchmark(primary_df, source_catalog, benchmark)
        for benchmark in BENCHMARK_SPECS
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
