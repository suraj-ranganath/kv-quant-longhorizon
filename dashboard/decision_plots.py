from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

STATUS_COLORS = {
    "BF16 reference": "#111827",
    "Pareto-optimal": "#16a34a",
    "Dominated": "#9ca3af",
}


def _recommendation_methods(recommendations: dict[str, dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for key in ["bf16_reference", "default_practical", "aggressive_compression", "fastest", "quality_first"]:
        payload = recommendations.get(key)
        if not payload:
            continue
        method = str(payload["method"])
        if method in seen:
            continue
        seen.add(method)
        ordered.append(method)
    return ordered


def _hover_columns() -> dict[str, Any]:
    return {
        "method": True,
        "method_family": True,
        "compression_ratio": ":.2f",
        "peak_vram_gb": ":.2f",
        "avg_runtime_s_per_prompt": ":.1f",
        "imaging_quality": ":.3f",
        "drift_last_imaging_quality": ":.3f",
        "imaging_quality_delta_vs_bf16": ":+.3f",
        "drift_last_imaging_quality_delta_vs_bf16": ":+.3f",
        "peak_vram_reduction_vs_bf16_pct": ":+.1f",
        "runtime_overhead_vs_bf16_pct": ":+.1f",
    }


def _safe_bubble_size(series: pd.Series, fallback: float = 1.0) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.notna().any():
        min_positive = numeric[numeric > 0].min()
        fill_value = float(min_positive) if pd.notna(min_positive) else fallback
    else:
        fill_value = fallback
    return numeric.fillna(fill_value).clip(lower=fill_value)


def _annotate_methods(fig: go.Figure, df: pd.DataFrame, methods: list[str], x_col: str, y_col: str) -> None:
    for method in methods:
        match = df[df["method"] == method]
        if match.empty:
            continue
        row = match.iloc[0]
        x_value = row.get(x_col)
        y_value = row.get(y_col)
        if pd.isna(x_value) or pd.isna(y_value):
            continue
        fig.add_annotation(
            x=x_value,
            y=y_value,
            text=method,
            showarrow=True,
            arrowhead=2,
            ax=12,
            ay=-18,
            font={"size": 11},
            bgcolor="rgba(255,255,255,0.7)",
        )


def plot_compression_vs_quality(method_summary: pd.DataFrame, recommendations: dict[str, dict[str, Any]]) -> go.Figure:
    plot_df = method_summary.copy()
    plot_df["plot_status"] = np.where(
        plot_df["method"] == "BF16",
        "BF16 reference",
        np.where(plot_df["pareto_quality_preserving_compression"].fillna(False), "Pareto-optimal", "Dominated"),
    )
    fig = px.scatter(
        plot_df,
        x="compression_ratio",
        y="imaging_quality",
        color="plot_status",
        color_discrete_map=STATUS_COLORS,
        symbol="plot_status",
        hover_data=_hover_columns(),
        title="Compression ratio vs VBench imaging quality",
    )
    fig.update_traces(marker={"size": 13, "line": {"width": 1, "color": "#1f2937"}}, selector={"mode": "markers"})
    _annotate_methods(fig, plot_df, _recommendation_methods(recommendations), "compression_ratio", "imaging_quality")
    bf16 = plot_df[plot_df["method"] == "BF16"]
    if not bf16.empty:
        fig.add_hline(y=float(bf16.iloc[0]["imaging_quality"]), line_dash="dot", line_color="#111827")
        fig.add_vline(x=float(bf16.iloc[0]["compression_ratio"]), line_dash="dot", line_color="#111827")
    fig.update_layout(height=480, legend_title=None, xaxis_title="Compression ratio (higher is better)", yaxis_title="VBench imaging quality (higher is better)")
    return fig


def plot_compression_vs_drift(method_summary: pd.DataFrame, recommendations: dict[str, dict[str, Any]]) -> go.Figure:
    plot_df = method_summary.copy()
    plot_df["plot_status"] = np.where(
        plot_df["method"] == "BF16",
        "BF16 reference",
        np.where(plot_df["pareto_balanced_practical"].fillna(False), "Pareto-optimal", "Dominated"),
    )
    fig = px.scatter(
        plot_df,
        x="compression_ratio",
        y="drift_last_imaging_quality",
        color="plot_status",
        color_discrete_map=STATUS_COLORS,
        symbol="plot_status",
        hover_data=_hover_columns(),
        title="Compression ratio vs drift-last imaging quality",
    )
    fig.update_traces(marker={"size": 13, "line": {"width": 1, "color": "#1f2937"}}, selector={"mode": "markers"})
    _annotate_methods(fig, plot_df, _recommendation_methods(recommendations), "compression_ratio", "drift_last_imaging_quality")
    bf16 = plot_df[plot_df["method"] == "BF16"]
    if not bf16.empty:
        fig.add_hline(y=float(bf16.iloc[0]["drift_last_imaging_quality"]), line_dash="dot", line_color="#111827")
        fig.add_vline(x=float(bf16.iloc[0]["compression_ratio"]), line_dash="dot", line_color="#111827")
    fig.update_layout(height=480, legend_title=None, xaxis_title="Compression ratio (higher is better)", yaxis_title="Drift-last imaging quality (higher is better)")
    return fig


def plot_peak_vram_vs_quality(method_summary: pd.DataFrame, recommendations: dict[str, dict[str, Any]]) -> go.Figure:
    plot_df = method_summary.copy()
    plot_df["compression_ratio_for_size"] = _safe_bubble_size(plot_df["compression_ratio"])
    fig = px.scatter(
        plot_df,
        x="peak_vram_gb",
        y="imaging_quality",
        color="avg_runtime_s_per_prompt",
        size="compression_ratio_for_size",
        hover_data=_hover_columns(),
        title="Peak VRAM vs imaging quality",
        color_continuous_scale="Viridis_r",
    )
    fig.update_traces(marker={"sizemode": "area", "line": {"width": 1, "color": "#1f2937"}})
    _annotate_methods(fig, plot_df, _recommendation_methods(recommendations), "peak_vram_gb", "imaging_quality")
    fig.update_layout(height=480, xaxis_title="Peak VRAM (GB, lower is better)", yaxis_title="VBench imaging quality (higher is better)")
    return fig


def plot_runtime_vs_quality(method_summary: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        method_summary,
        x="avg_runtime_s_per_prompt",
        y="imaging_quality",
        color="method_family",
        symbol="pareto_quality_first",
        hover_data=_hover_columns(),
        title="Runtime vs imaging quality",
    )
    bf16 = method_summary[method_summary["method"] == "BF16"]
    if not bf16.empty:
        fig.add_hline(y=float(bf16.iloc[0]["imaging_quality"]), line_dash="dot", line_color="#111827")
        fig.add_vline(x=float(bf16.iloc[0]["avg_runtime_s_per_prompt"]), line_dash="dot", line_color="#111827")
    fig.update_layout(height=480, legend_title=None, xaxis_title="Runtime / prompt (s, lower is better)", yaxis_title="VBench imaging quality (higher is better)")
    return fig


def plot_vram_vs_runtime(method_summary: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        method_summary,
        x="peak_vram_gb",
        y="avg_runtime_s_per_prompt",
        color="imaging_quality_delta_vs_bf16",
        hover_data=_hover_columns(),
        title="Peak VRAM vs runtime",
        color_continuous_scale="RdYlGn",
    )
    fig.update_traces(marker={"size": 13, "line": {"width": 1, "color": "#1f2937"}})
    fig.update_layout(height=480, xaxis_title="Peak VRAM (GB, lower is better)", yaxis_title="Runtime / prompt (s, lower is better)")
    return fig


def plot_compression_vs_peak_vram(method_summary: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        method_summary,
        x="compression_ratio",
        y="peak_vram_gb",
        color="method_family",
        symbol="pareto_systems_efficiency",
        hover_data=_hover_columns(),
        title="Compression ratio vs peak VRAM",
    )
    fig.update_traces(marker={"size": 13, "line": {"width": 1, "color": "#1f2937"}})
    fig.update_layout(height=480, xaxis_title="Compression ratio (higher is better)", yaxis_title="Peak VRAM (GB, lower is better)")
    return fig


def plot_top_candidate_profile(method_summary: pd.DataFrame, recommendations: dict[str, dict[str, Any]]) -> go.Figure:
    selected_methods = _recommendation_methods(recommendations)
    selected_df = method_summary[method_summary["method"].isin(selected_methods)].copy()
    if selected_df.empty:
        return go.Figure()

    metrics = {
        "Imaging quality ↑": ("imaging_quality", "max"),
        "Drift proxy ↑": ("drift_last_imaging_quality", "max"),
        "Compression ↑": ("compression_ratio", "max"),
        "Runtime efficiency ↑": ("avg_runtime_s_per_prompt", "min"),
        "VRAM efficiency ↑": ("peak_vram_gb", "min"),
    }
    normalized_rows: list[dict[str, Any]] = []
    for label, (column, direction) in metrics.items():
        values = pd.to_numeric(selected_df[column], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan)
        if finite.notna().sum() <= 1:
            normalized = pd.Series([1.0 if pd.notna(value) else np.nan for value in finite], index=selected_df.index)
        else:
            min_value = finite.min(skipna=True)
            max_value = finite.max(skipna=True)
            normalized = (finite - min_value) / (max_value - min_value)
            if direction == "min":
                normalized = 1.0 - normalized
        for idx, row in selected_df.iterrows():
            normalized_rows.append({
                "method": row["method"],
                "metric": label,
                "normalized_value": float(normalized.loc[idx]) if pd.notna(normalized.loc[idx]) else None,
            })
    plot_df = pd.DataFrame(normalized_rows).dropna(subset=["normalized_value"])
    fig = px.bar(
        plot_df,
        x="metric",
        y="normalized_value",
        color="method",
        barmode="group",
        title="Normalized comparison of the top candidate methods",
    )
    fig.update_layout(height=420, yaxis_title="Normalized score (higher is better)", xaxis_title=None)
    return fig


def plot_family_summary(method_summary: pd.DataFrame) -> go.Figure:
    melt_df = method_summary.melt(
        id_vars=["method", "method_family"],
        value_vars=["compression_ratio", "avg_runtime_s_per_prompt", "imaging_quality", "drift_last_imaging_quality"],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    title_map = {
        "compression_ratio": "Compression ratio",
        "avg_runtime_s_per_prompt": "Runtime / prompt",
        "imaging_quality": "Imaging quality",
        "drift_last_imaging_quality": "Drift-last imaging quality",
    }
    melt_df["metric"] = melt_df["metric"].map(title_map).fillna(melt_df["metric"])
    fig = px.box(
        melt_df,
        x="method_family",
        y="value",
        color="method_family",
        points="all",
        facet_col="metric",
        title="Family-level summary across runtime, compression, quality, and drift",
    )
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.split("=", 1)[-1]))
    fig.update_layout(height=520, legend_title=None, xaxis_title=None)
    return fig
