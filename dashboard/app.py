#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from dashboard.data_sources import load_dashboard_workspace
from dashboard.decision_analysis import (
    DEFAULT_THRESHOLDS,
    FRONTIER_DEFINITIONS,
    DecisionAnalysis,
    RECOMMENDATION_FOCUS_PRESETS,
    build_dashboard_analysis,
    get_recommendation_sort,
    infer_method_family,
)
from dashboard.decision_plots import (
    plot_compression_vs_drift,
    plot_compression_vs_peak_vram,
    plot_frontier_position,
    plot_compression_vs_quality,
    plot_family_summary,
    plot_peak_vram_vs_quality,
    plot_runtime_vs_quality,
    plot_top_candidate_profile,
    plot_vram_vs_runtime,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
STORYEVAL_RESULTS_ROOT = RESULTS_ROOT / "benchmarks" / "storyeval"
DEFAULT_PROMPTS_FILE = REPO_ROOT / "prompts" / "MovieGenVideoBench_extended.txt"
COMBINED_DATASET_PATH = RESULTS_ROOT / "combined" / "combined_comparison_dataset.csv"
COMBINED_GAPS_PATH = RESULTS_ROOT / "combined" / "combined_comparison_gaps.json"

METHOD_ORDER = [
    "BF16",
    "RTN_INT4",
    "RTN_INT2",
    "KIVI_INT4",
    "KIVI_INT2",
    "QUAROT_KV_INT4",
    "QUAROT_KV_INT2",
    "PRQ_INT2",
    "PRQ_INT4",
    "QAQ_INT2",
    "QAQ_INT4",
    "AGE_TIER_INT2",
    "AGE_TIER_INT4",
    "TPTQ_INT2",
    "FLOWCACHE_HYBRID_INT2",
    "FLOWCACHE_ADAPTIVE_INT2",
    "FLOWCACHE_PRUNE_INT2",
    "FLOWCACHE_PRUNE_INT4",
    "FLOWCACHE_SOFT_PRUNE_INT2",
    "FLOWCACHE_SOFT_PRUNE_INT4",
    "FLOWCACHE_NATIVE",
    "FLOWCACHE_NATIVE_SOFT_PRUNE_INT4",
    "SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT4",
    "SPATIAL_MIXED_FG_RTN_INT4_BG_RTN_INT2",
    "SPATIAL_MIXED_FG_QUAROT_KV_INT4_BG_RTN_INT2",
    "SPATIAL_MIXED_FG_KIVI_INT4_BG_KIVI_INT2",
    "RTN_INT4_REFRESH",
    "KIVI_INT4_REFRESH",
    "QUAROT_KV_INT4_REFRESH",
    "RTN_K2_V4",
    "KIVI_K2_V4",
    "RTN_INT4_RECENT2",
    "QUAROT_KV_INT4_RECENT2",
]

PRESENTATION_METHOD_PREFERENCES = [
    ["BF16"],
    ["FLOWCACHE_SOFT_PRUNE_INT4"],
    ["FLOWCACHE_PRUNE_INT4"],
    ["RTN_INT4_RECENT2", "RTN_INT4"],
    ["RTN_INT4_REFRESH", "RTN_INT4"],
    ["QUAROT_KV_INT4", "QUAROT_KV_INT4_REFRESH", "QUAROT_KV_INT4_RECENT2"],
]

VIDEO_RE = re.compile(r"prompt_(\d+)_seed_(\d+)\.mp4$")

QUANT_VRAM_NOTE = (
    "Peak VRAM is an instantaneous maximum. Quantized methods still dequantize KV to BF16 "
    "for attention compute and allocate temporary quant/dequant scratch buffers, so peak VRAM "
    "can be similar to or slightly above BF16 even when compressed KV bytes are much lower. "
    "This integration has removed the earlier persistent BF16+quantized dual-cache residency."
)
KV_BYTES_NOTE = (
    "KV byte fields come from each method's efficiency log at run end. "
    "`bf16_kv_bytes` is the BF16 cache-size baseline for the same cache shape, and "
    "`compressed_kv_bytes` is the quantized cache footprint reported by the quantizer state."
)

PRESENTATION_TOOLTIP_TEXT: dict[str, str] = {
    "presentation_page": "Single-page presentation view that keeps the selected methods, prompt video, metrics, systems traces, and summary plots together in one place.",
    "presentation_methods": "Methods pinned to the presentation page. These are the rows, videos, and plot highlights shown throughout the page.",
    "presentation_input": "Prompt or input clip currently being shown in the presentation video comparison.",
    "presentation_videos": "Side-by-side video comparison for the selected prompt and the chosen presentation methods.",
    "presentation_cards": "Compact per-method summary cards for the presentation methods, showing the main systems and quality numbers together.",
    "presentation_graphs": "Core trade-off plots from the rest of the dashboard, with the presentation methods highlighted directly on the charts.",
    "presentation_focus_table": "Method-level comparison table restricted to the current presentation methods.",
    "presentation_prompt_records": "Prompt-level rows backing the currently displayed videos.",
    "presentation_traces": "VRAM and KV-cache traces for the selected prompt, restricted to the chosen presentation methods.",
    "presentation_provenance": "Run-level provenance rows behind the methods shown on the presentation page.",
    "research_decision_layer": "Presentation-focused layer that turns raw benchmark outputs into method recommendations, multi-objective frontiers, and deployment-style rankings.",
    "recommendation_focus": "Ranking preset that changes which trade-offs the dashboard prioritizes first when selecting the main recommended method.",
    "benchmark": "Evaluation surface currently in view. MovieGen is the single-shot 10-second benchmark, while StoryEval stresses longer narrative consistency.",
    "source_users": "Users whose runs contribute rows to the current comparison dataset after filtering.",
    "runs": "Named experiment groups contributing rows to the current benchmark slice.",
    "methods": "Quantization methods currently included in the visible comparison set.",
    "quality_goal": "The study treats perceptual quality and rollout stability as the first-order objective, then asks whether memory relief and runtime are good enough to be practical.",
    "methods_in_scope": "Number of distinct quantization methods still visible after the current benchmark, run, and method filters.",
    "balanced_frontier": "Methods that are not jointly beaten once quality, drift, runtime, VRAM, and compression are considered together.",
    "best_vram_reduction": "Largest percentage drop in peak VRAM relative to the BF16 baseline for the current benchmark.",
    "primary_benchmark": "Benchmark for which the current recommendation tables and plots are being computed.",
    "compression_ratio": "BF16 KV bytes divided by compressed KV bytes. Higher means the quantized cache is smaller relative to the BF16 baseline.",
    "runtime": "Average end-to-end generation wall-clock time per prompt. Lower is faster.",
    "peak_vram": "Highest GPU memory observed during generation. Lower is better for fitting longer runs on a given device.",
    "imaging_delta_vs_bf16": "Difference in VBench imaging quality relative to BF16. Values near zero preserve BF16-level visual quality; negative values lose quality.",
    "drift_delta_vs_bf16": "Difference in the last available drift imaging-quality point relative to BF16. Values near zero preserve temporal stability better.",
    "candidate_comparison": "Normalized comparison of the headline candidates. Every bar is scaled so higher is better, including runtime and VRAM after inversion into efficiency scores.",
    "family_summary": "Aggregates methods by family to show structural patterns rather than only individual winners.",
    "pareto_frontier": "A Pareto frontier contains methods that are not strictly outperformed on every objective in that frontier.",
    "frontier_membership_table": "Table showing which methods lie on each frontier and why they do or do not survive the balanced practical frontier.",
    "constraint_rankings": "Deployment-style ranking tables filtered by the active runtime, VRAM, and acceptable quality-loss caps from the sidebar.",
    "method_family": "High-level algorithm family the method belongs to, such as RTN, KIVI, PRQ, or FlowCache variants.",
    "bit_width_mode": "Bit-width or policy label summarizing how aggressively the cache is quantized or managed.",
    "quantization_details": "Short textual summary of the method's actual operating policy, not just its family name.",
    "recommended_for": "Use case or operating regime where the method is most defensible in the current dataset.",
    "caution_label": "Main caveat that should be kept in mind before recommending the method.",
    "pareto_status": "Whether the selected method survives each frontier's objective set and the explanation for that status.",
    "run_provenance": "Run-level rows backing the selected method summary, included so you can trace every recommendation back to its original experiments.",
    "explainability_table": "Derived feature table used to justify why the selected method scores well or poorly under the active recommendation focus.",
    "systems_tradeoffs": "Plots that compare nominal KV compression against realized VRAM and runtime behavior in the current integration.",
    "trace_preview": "Short systems sanity-check view showing how allocated VRAM and compressed KV size evolve during a sample prompt.",
    "quality_stability_table": "Method-level quality and temporal-stability summary, including BF16-relative deltas.",
    "storyeval_drift_curves": "StoryEval drift curves show how imaging quality changes over the longer narrative rollout checkpoints.",
    "raw_method_table": "Full derived benchmark-level method table used underneath the recommendation views.",
    "source_catalog": "Discovered CSV inputs and their role in the dashboard's merged analysis pipeline.",
    "metric_glossary": "Reference table defining the metrics and method families used throughout the dashboard.",
}


def _tooltip_text(key: str, fallback: str = "") -> str:
    return PRESENTATION_TOOLTIP_TEXT.get(key, fallback)


@dataclass
class RunLayout:
    label: str
    benchmark: str
    root: Path
    member_roots: list[Path]
    metric_dirs: list[Path]
    log_dirs: list[Path]
    video_dirs: list[Path]
    table_dirs: list[Path]


def _infer_run_benchmark(run: RunLayout) -> str:
    if isinstance(getattr(run, "benchmark", None), str) and run.benchmark.strip():
        return run.benchmark.strip().lower()
    meta = _read_json(run.root / "run_meta.json")
    if isinstance(meta, dict):
        benchmark = meta.get("benchmark")
        if isinstance(benchmark, str) and benchmark.strip():
            return benchmark.strip().lower()
        run_name = meta.get("run_name")
        if isinstance(run_name, str) and run_name.strip():
            lower_name = run_name.lower()
            if "storyeval" in lower_name:
                return "storyeval"
            if "moviegen" in lower_name:
                return "moviegen"

    label_lower = run.label.lower()
    path_lower = str(run.root).lower()
    if "storyeval" in label_lower or "storyeval" in path_lower:
        return "storyeval"
    return "moviegen"


def _parse_archive_timestamp(name: str) -> int:
    # Expected: YYYYMMDDTHHMMSSZ
    try:
        dt = datetime.strptime(name, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0


def _extract_run_unix_ts(run: RunLayout) -> int:
    if run.root.resolve() == RESULTS_ROOT.resolve():
        # Root-level results/ is legacy workspace data, not a timestamped run.
        return -1
    meta_path = run.root / "run_meta.json"
    if meta_path.exists():
        payload = _read_json(meta_path)
        if isinstance(payload, dict):
            ts = payload.get("run_timestamp_unix")
            if isinstance(ts, (int, float)):
                return int(ts)
    for root in getattr(run, "member_roots", []) or [run.root]:
        cfg_path = root / "summary" / "config.json"
        cfg = _read_json(cfg_path)
        if isinstance(cfg, dict):
            run_id = cfg.get("run_id")
            if isinstance(run_id, str):
                m_run_id = re.search(r"_(\d+)$", run_id)
                if m_run_id:
                    return int(m_run_id.group(1))
    m_prefix = re.match(r"^runs/(\d+)_", run.label)
    if m_prefix:
        return int(m_prefix.group(1))
    m_suffix = re.match(r"^runs/.+_(\d+)$", run.label)
    if m_suffix:
        return int(m_suffix.group(1))
    m_storyeval = re.match(r"^storyeval/.+_(\d+)$", run.label)
    if m_storyeval:
        return int(m_storyeval.group(1))
    if run.label.startswith("archive/"):
        return _parse_archive_timestamp(run.label.split("/", 1)[1])
    try:
        return int(run.root.stat().st_mtime)
    except Exception:
        return 0


def _is_deletable_run(run: RunLayout) -> bool:
    if run.root.resolve() == RESULTS_ROOT.resolve():
        return False
    return run.label.startswith("runs/") or run.label.startswith("archive/") or run.label.startswith("storyeval/")


def _delete_run_directory(run: RunLayout) -> tuple[bool, str]:
    if not _is_deletable_run(run):
        return False, "Only runs under runs/ or archive/ can be deleted from the dashboard."
    if not run.root.exists():
        return False, f"Run path does not exist: {run.root}"
    try:
        roots = [p.resolve() for p in (getattr(run, "member_roots", []) or [run.root])]
        if any(resolved == RESULTS_ROOT.resolve() for resolved in roots):
            return False, "Refusing to delete results root."
        for resolved in sorted(set(roots), key=lambda p: len(str(p)), reverse=True):
            if resolved.exists():
                shutil.rmtree(resolved)
        return True, f"Deleted {run.label}"
    except Exception as exc:
        return False, f"Delete failed for {run.label}: {exc}"


def _is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_file(search_dirs: list[Path], filename: str) -> Path | None:
    for base in search_dirs:
        candidate = base / filename
        if candidate.exists():
            return candidate
    return None


def _load_method_manifest(run: RunLayout) -> dict[str, dict[str, Any]]:
    meta_path = run.root / "run_meta.json"
    payload = _read_json(meta_path) if meta_path.exists() else None
    manifest = payload.get("method_manifest") if isinstance(payload, dict) else None
    if isinstance(manifest, dict):
        return {str(k): v for k, v in manifest.items() if isinstance(v, dict)}

    manifest_path = _find_file(run.table_dirs + [run.root], "method_manifest.json")
    file_payload = _read_json(manifest_path) if manifest_path else None
    if isinstance(file_payload, dict):
        return {str(k): v for k, v in file_payload.items() if isinstance(v, dict)}
    return {}


def _extract_vbench_scalar(value: Any) -> float | None:
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, (int, float)):
            return float(first)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_seconds(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    total = float(value)
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = int(total % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_metric_value(value: Any, precision: int = 3, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    value_f = float(value)
    if not pd.notna(value_f):
        return "-"
    return f"{value_f:.{precision}f}{suffix}"


def _format_psnr_value(value: Any, is_bf16_reference: bool = False, precision: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    value_f = float(value)
    if np.isposinf(value_f):
        return "Reference (self-comparison)" if is_bf16_reference else "Exact match to BF16"
    if np.isneginf(value_f) or not pd.notna(value_f):
        return "-"
    return f"{value_f:.{precision}f}"


def _format_signed_metric(value: Any, precision: int = 3, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "-"
    value_f = float(value)
    if not pd.notna(value_f):
        return "-"
    return f"{value_f:+.{precision}f}{suffix}"


def _tooltip_inline_html(label: str, tooltip: str) -> str:
    tooltip_attr = escape(tooltip, quote=True)
    return (
        f"<span class='tooltip-inline'>{escape(label)}"
        f"<span class='info-dot' tabindex='0' data-tooltip='{tooltip_attr}' aria-label='{tooltip_attr}'>i</span>"
        "</span>"
    )


def _render_heading(level: int, label: str, tooltip: str | None = None) -> None:
    heading_body = _tooltip_inline_html(label, tooltip) if tooltip else escape(label)
    st.markdown(
        f"<h{level} class='section-heading'>{heading_body}</h{level}>",
        unsafe_allow_html=True,
    )


def _metric_help(label_key: str, fallback: str) -> str:
    return _tooltip_text(label_key, fallback)


def _card_metric_line(label: str, value: str, tooltip_key: str, fallback: str) -> str:
    return f"<p><strong>{_tooltip_inline_html(label, _tooltip_text(tooltip_key, fallback))}:</strong> {value}</p>"


def _format_metric_for_tooltip(value: Any, precision: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    value_f = float(value)
    if not pd.notna(value_f):
        return "-"
    return f"{value_f:+.{precision}f}"


def _psnr_delta_tooltip_text(row: Any) -> str | None:
    note = row.get("psnr_delta_note") if hasattr(row, "get") else None
    if isinstance(note, str) and note.strip():
        return None
    return _format_metric_for_tooltip(row.get("psnr_delta_vs_bf16") if hasattr(row, "get") else None)


def _psnr_display_series(df: pd.DataFrame, value_column: str, method_column: str = "method") -> pd.Series:
    if value_column not in df.columns:
        return pd.Series(dtype="object")

    def _render(row: pd.Series) -> str:
        method_value = str(row.get(method_column, "")) if method_column in row else ""
        return _format_psnr_value(row.get(value_column), is_bf16_reference=(method_value == "BF16"))

    return df.apply(_render, axis=1)


def _prepare_psnr_display_df(df: pd.DataFrame, method_column: str = "method") -> pd.DataFrame:
    if df.empty:
        return df
    display_df = df.copy()
    for column in [
        "psnr",
        "moviegen_fidelity_psnr",
        "moviegen_fidelity_psnr_agg",
        "storyeval_fidelity_psnr",
        "storyeval_fidelity_psnr_agg",
    ]:
        if column in display_df.columns:
            display_df[column] = _psnr_display_series(display_df, column, method_column=method_column)
    return display_df

def _order_methods(methods: set[str]) -> list[str]:
    ordered = [m for m in METHOD_ORDER if m in methods]
    extras = sorted(m for m in methods if m not in METHOD_ORDER)
    return ordered + extras


def _resolve_presentation_methods(method_options: list[str]) -> list[str]:
    available = set(method_options)
    selected: list[str] = []
    for aliases in PRESENTATION_METHOD_PREFERENCES:
        match = next((alias for alias in aliases if alias in available and alias not in selected), None)
        if match:
            selected.append(match)
    if selected:
        return selected
    return method_options[: min(len(method_options), 6)]


def _ordered_focus_rows(df: pd.DataFrame, methods: list[str], method_column: str = "method") -> pd.DataFrame:
    if df.empty or method_column not in df.columns:
        return df
    order_map = {method: idx for idx, method in enumerate(methods)}
    ordered = df[df[method_column].astype(str).isin(methods)].copy()
    if ordered.empty:
        return ordered
    ordered["_presentation_order"] = ordered[method_column].astype(str).map(order_map).fillna(len(order_map))
    ordered = ordered.sort_values(["_presentation_order", method_column], na_position="last").drop(columns="_presentation_order")
    return ordered


def _highlight_focus_methods(
    fig: go.Figure,
    method_df: pd.DataFrame,
    focus_methods: list[str],
    x_col: str,
    y_col: str,
    label_col: str = "method",
) -> go.Figure:
    if fig is None or method_df.empty or x_col not in method_df.columns or y_col not in method_df.columns:
        return fig
    plot_df = method_df[method_df[label_col].astype(str).isin(focus_methods)].dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=plot_df[x_col],
            y=plot_df[y_col],
            mode="markers+text",
            text=plot_df[label_col].astype(str),
            textposition="top center",
            name="Presentation methods",
            marker={
                "size": 18,
                "color": "#ea580c",
                "line": {"width": 2, "color": "#7c2d12"},
                "symbol": "diamond",
            },
            hovertemplate=(
                "<b>%{text}</b><br>"
                + f"{x_col}: %{{x}}<br>"
                + f"{y_col}: %{{y}}<extra>Presentation focus</extra>"
            ),
        )
    )
    return fig


def _render_video_sync_controls(key: str) -> None:
    components.html(
        f"""
        <style>
        .sync-controls {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
            margin: 0.15rem 0 0.35rem 0;
            flex-wrap: wrap;
            font-family: 'Manrope', sans-serif;
        }}

        .sync-button {{
            border: 1px solid rgba(114, 128, 146, 0.34);
            background: rgba(114, 128, 146, 0.1);
            color: #475569;
            border-radius: 999px;
            padding: 0.38rem 0.78rem;
            font-size: 0.85rem;
            font-weight: 700;
            cursor: pointer;
        }}

        .sync-note {{
            color: #64748b;
            font-size: 0.82rem;
        }}
        </style>
        <div class="sync-controls">
            <button class="sync-button" id="play-all-{key}">Play all videos</button>
            <button class="sync-button" id="pause-all-{key}">Pause all videos</button>
            <button class="sync-button" id="restart-all-{key}">Restart all videos</button>
            <span class="sync-note">Loop stays enabled for every video on the page.</span>
        </div>
        <script>
        const parentDoc = window.parent.document;
        const getVideos = () => Array.from(parentDoc.querySelectorAll("video"));

        const applyLooping = () => {{
            getVideos().forEach((video) => {{
                video.loop = true;
            }});
        }};

        const syncPlay = (reset) => {{
            const videos = getVideos();
            if (!videos.length) return;
            let anchorTime = 0;
            if (!reset) {{
                const currentTimes = videos
                    .map((video) => Number.isFinite(video.currentTime) ? video.currentTime : 0)
                    .filter((value) => value >= 0);
                anchorTime = currentTimes.length ? Math.min(...currentTimes) : 0;
            }}
            videos.forEach((video) => {{
                video.loop = true;
                if (reset) {{
                    video.currentTime = 0;
                }} else {{
                    video.currentTime = anchorTime;
                }}
                const playPromise = video.play();
                if (playPromise && typeof playPromise.catch === "function") {{
                    playPromise.catch(() => {{}});
                }}
            }});
        }};

        const pauseAll = () => {{
            getVideos().forEach((video) => video.pause());
        }};

        document.getElementById("play-all-{key}").addEventListener("click", () => syncPlay(false));
        document.getElementById("pause-all-{key}").addEventListener("click", pauseAll);
        document.getElementById("restart-all-{key}").addEventListener("click", () => syncPlay(true));

        applyLooping();
        window.setInterval(applyLooping, 1500);
        </script>
        """,
        height=64,
    )


def _presentation_prompt_column_config(columns: list[str]) -> dict[str, Any]:
    configs: dict[str, Any] = {
        "source_user": st.column_config.TextColumn("Source user", help=_tooltip_text("source_users")),
        "run_name": st.column_config.TextColumn("Run", help=_tooltip_text("runs")),
        "method_display": st.column_config.TextColumn("Method", help="Presentation method name for this prompt-level record."),
        "prompt_id": st.column_config.TextColumn("Prompt ID", help=_tooltip_text("presentation_input")),
        "seed": st.column_config.NumberColumn("Seed", help="Random seed for this generated sample.", format="%d"),
        "wall_time_sec": st.column_config.NumberColumn("Runtime", help=_tooltip_text("runtime"), format="%.2f s"),
        "peak_vram_mb": st.column_config.NumberColumn("Peak VRAM", help="Maximum GPU memory observed for this prompt-level sample.", format="%.0f MB"),
        "moviegen_fidelity_psnr": st.column_config.TextColumn("PSNR", help="Prompt-level MovieGen PSNR relative to BF16. BF16 self-comparison is shown as a reference label instead of a raw infinite value."),
        "moviegen_fidelity_ssim": st.column_config.NumberColumn("SSIM", help="Prompt-level MovieGen SSIM relative to BF16. Higher is better.", format="%.4f"),
        "moviegen_fidelity_lpips": st.column_config.NumberColumn("LPIPS", help="Prompt-level MovieGen LPIPS relative to BF16. Lower is better.", format="%.4f"),
        "moviegen_imaging_quality": st.column_config.NumberColumn("Imaging quality", help="Prompt-level VBench imaging-quality score. Higher is better.", format="%.4f"),
        "storyeval_imaging_quality": st.column_config.NumberColumn("Imaging quality", help="Prompt-level StoryEval imaging-quality score. Higher is better.", format="%.4f"),
        "storyeval_subject_consistency": st.column_config.NumberColumn("Subject consistency", help="Prompt-level StoryEval subject-consistency score. Higher is better.", format="%.4f"),
    }
    return {column: config for column, config in configs.items() if column in columns}


def _presentation_tree_method_row(method_df: pd.DataFrame, method: str) -> pd.Series | None:
    matches = method_df[method_df["method"].astype(str) == method]
    if matches.empty:
        return None
    return matches.iloc[0]


def _tree_metric(row: pd.Series | None, column: str, precision: int = 2, suffix: str = "") -> str:
    if row is None:
        return "-"
    value = row.get(column)
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{precision}f}{suffix}"


def _graphviz_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_presentation_decision_tree(analysis: DecisionAnalysis) -> None:
    branch_options = {
        "All branches": set(),
        "Deployment path": {"start", "memory", "low_vram_quality", "memory_reduction", "flowcache_soft", "flowcache_prune"},
        "Quantized fidelity path": {"start", "memory", "quality", "highest_fidelity", "quality_runtime", "quarot", "rtn_recent2"},
        "Policy insight path": {"start", "memory", "quality", "policy_insight", "reference_result", "rtn_refresh"},
        "BF16 reference path": {"start", "memory", "quality", "policy_insight", "reference_result", "bf16"},
    }
    selected_branch = st.selectbox(
        "Tree focus",
        options=list(branch_options.keys()),
        index=0,
        key=f"presentation_tree_focus_{analysis.benchmark}",
        help="Highlight one decision path at a time while keeping the full tree visible.",
    )
    highlight_nodes = branch_options[selected_branch]

    node_specs = [
        {
            "id": "start",
            "label": "How to choose a KV-cache method\n\nStart: What do you need most?",
            "fill": "#e2e8f0",
            "line": "#94a3b8",
        },
        {
            "id": "memory",
            "label": "1. Do you need REAL memory relief\nor lower peak VRAM right now?",
            "fill": "#e2e8f0",
            "line": "#94a3b8",
        },
        {
            "id": "low_vram_quality",
            "label": "Want the best quality\namong the low-VRAM methods?",
            "fill": "#f8fafc",
            "line": "#cbd5e1",
        },
        {
            "id": "quality",
            "label": "Do you care most about preserving\nquality under quantization?",
            "fill": "#f8fafc",
            "line": "#cbd5e1",
        },
        {
            "id": "memory_reduction",
            "label": "Want the strongest raw memory reduction,\nand can accept more quality loss?",
            "fill": "#f8fafc",
            "line": "#cbd5e1",
        },
        {
            "id": "highest_fidelity",
            "label": "Want the highest-fidelity quantized method,\nruntime less important?",
            "fill": "#f8fafc",
            "line": "#cbd5e1",
        },
        {
            "id": "policy_insight",
            "label": "Do you want the simplest policy insight\nwith good speed?",
            "fill": "#f8fafc",
            "line": "#cbd5e1",
        },
        {
            "id": "quality_runtime",
            "label": "Want the best quality / runtime tradeoff?",
            "fill": "#f8fafc",
            "line": "#cbd5e1",
        },
        {
            "id": "reference_result",
            "label": "Need an upper-bound\nor reference result?",
            "fill": "#f8fafc",
            "line": "#cbd5e1",
        },
        {
            "id": "flowcache_soft",
            "label": "FLOWCACHE_SOFT_PRUNE_INT4\n\nWhy: ~5.49x compression, 11.71 GB peak VRAM,\nimaging quality 0.739\n\nUse when: deployment or single-GPU runs are\nmemory-limited",
            "fill": "#ecfdf5",
            "line": "#86efac",
        },
        {
            "id": "flowcache_prune",
            "label": "FLOWCACHE_PRUNE_INT4\n\nWhy: ~5.50x compression, 11.71 GB peak VRAM,\nfaster than soft-prune\n\nTradeoff: worse PSNR / SSIM / LPIPS than\nsoft-prune",
            "fill": "#f0fdf4",
            "line": "#86efac",
        },
        {
            "id": "quarot",
            "label": "QUAROT_KV_INT4\n\nWhy: strongest quality preservation among\nquantized baselines\n\nNumbers: 3.20x compression, LPIPS 0.1483,\nimaging quality 0.738\n\nTradeoff: very slow, ~236.6s/prompt",
            "fill": "#fff7ed",
            "line": "#fdba74",
        },
        {
            "id": "rtn_recent2",
            "label": "RTN_INT4_RECENT2\n\nWhy: best practical RTN policy variant\n\nNumbers: 2.43x compression, PSNR 23.692,\nSSIM 0.7320, LPIPS 0.1482, drift-last 0.735,\nruntime ~68.9s\n\nUse when: research says recent context matters and you want\na strong quality result without QuaRot's runtime cost",
            "fill": "#eff6ff",
            "line": "#93c5fd",
        },
        {
            "id": "rtn_refresh",
            "label": "RTN_INT4_REFRESH\n\nWhy: shows refresh-only cadence helps over\nplain RTN\n\nNumbers: 3.20x compression, runtime ~65.0s,\nimaging quality 0.736\n\nUse when: you want a cheap, interpretable\npolicy win",
            "fill": "#eff6ff",
            "line": "#93c5fd",
        },
        {
            "id": "bf16",
            "label": "BF16\n\nWhy: best reference quality, no compression\n\nUse when: you want the oracle baseline,\nnot a deployable method",
            "fill": "#f8fafc",
            "line": "#cbd5e1",
        },
    ]

    edges = [
        ("start", "memory", ""),
        ("memory", "low_vram_quality", "Yes"),
        ("memory", "quality", "No"),
        ("low_vram_quality", "flowcache_soft", "Yes"),
        ("low_vram_quality", "memory_reduction", "No"),
        ("memory_reduction", "flowcache_prune", "Yes"),
        ("quality", "highest_fidelity", "Yes"),
        ("quality", "policy_insight", "No"),
        ("highest_fidelity", "quarot", "Yes"),
        ("highest_fidelity", "quality_runtime", "No"),
        ("quality_runtime", "rtn_recent2", "Yes"),
        ("policy_insight", "rtn_refresh", "Yes"),
        ("policy_insight", "reference_result", "No"),
        ("reference_result", "bf16", "Yes"),
    ]

    node_styles: list[str] = []
    for node in node_specs:
        active = not highlight_nodes or node["id"] in highlight_nodes
        fontcolor = "#0f172a" if active else "#94a3b8"
        edgecolor = ("#0f766e" if active else "#cbd5e1") if node["id"] in {"flowcache_soft", "flowcache_prune", "quarot", "rtn_recent2", "rtn_refresh", "bf16"} else ("#475569" if active else "#cbd5e1")
        fillcolor = node["fill"] if active else "#f8fafc"
        penwidth = "2.2" if active else "1.2"
        node_styles.append(
            f'{node["id"]} [label="{_graphviz_escape(node["label"])}", fillcolor="{fillcolor}", color="{edgecolor}", fontcolor="{fontcolor}", penwidth={penwidth}];'
        )

    edge_styles: list[str] = []
    for source, target, label in edges:
        active = not highlight_nodes or (source in highlight_nodes and target in highlight_nodes)
        color = "#0f766e" if active else "#cbd5e1"
        fontcolor = "#475569" if active else "#94a3b8"
        penwidth = "2.0" if active else "1.0"
        edge_label = f', label="{label}", decorate=true, labelfloat=false, fontcolor="{fontcolor}", fontsize=11' if label else ""
        edge_styles.append(
            f'{source} -> {target} [color="{color}", penwidth={penwidth}{edge_label}];'
        )

    dot = f"""
digraph KVMethodTree {{
    graph [
        rankdir=TB,
        splines=ortho,
        nodesep=0.75,
        ranksep=0.95,
        pad=0.35,
        bgcolor="transparent"
    ];
    node [
        shape=box,
        style="rounded,filled",
        fontname="Helvetica",
        fontsize=11,
        margin="0.20,0.14"
    ];
    edge [
        arrowsize=0.7,
        fontname="Helvetica",
        fontsize=11
    ];

    {' '.join(node_styles)}
    {' '.join(edge_styles)}

    {{ rank=same; start; }}
    {{ rank=same; memory; }}
    {{ rank=same; low_vram_quality; quality; }}
    {{ rank=same; memory_reduction; highest_fidelity; policy_insight; }}
    {{ rank=same; quality_runtime; reference_result; }}
    {{ rank=same; flowcache_soft; flowcache_prune; quarot; rtn_recent2; rtn_refresh; bf16; }}

    low_vram_quality -> quality [style=invis, weight=10];
    memory_reduction -> highest_fidelity [style=invis, weight=10];
    highest_fidelity -> policy_insight [style=invis, weight=10];
    quality_runtime -> reference_result [style=invis, weight=10];
    flowcache_soft -> flowcache_prune [style=invis, weight=10];
    flowcache_prune -> quarot [style=invis, weight=10];
    quarot -> rtn_recent2 [style=invis, weight=10];
    rtn_recent2 -> rtn_refresh [style=invis, weight=10];
    rtn_refresh -> bf16 [style=invis, weight=10];
}}
"""

    st.graphviz_chart(dot, use_container_width=True)


def _project_sorted_table(
    df: pd.DataFrame,
    display_columns: list[str],
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    present_cols = [column for column in display_columns if column in df.columns]
    sort_pairs = [(column, direction) for column, direction in zip(sort_columns, ascending) if column in df.columns]
    ordered = df
    if sort_pairs:
        sort_by, sort_ascending = zip(*sort_pairs)
        ordered = df.sort_values(list(sort_by), ascending=list(sort_ascending), na_position="last")
    return ordered[present_cols]


def _safe_sort(df: pd.DataFrame, sort_columns: list[str], ascending: list[bool]) -> pd.DataFrame:
    sort_pairs = [(column, direction) for column, direction in zip(sort_columns, ascending) if column in df.columns]
    if not sort_pairs:
        return df
    sort_by, sort_ascending = zip(*sort_pairs)
    return df.sort_values(list(sort_by), ascending=list(sort_ascending), na_position="last")


def _top_constraint_calibration_rows(method_df: pd.DataFrame, recommendation_focus: str) -> pd.DataFrame:
    if method_df.empty:
        return method_df
    candidate_df = method_df[method_df["method"] != "BF16"].copy()
    if candidate_df.empty:
        candidate_df = method_df.copy()
    sort_columns, sort_ascending = get_recommendation_sort(recommendation_focus)
    candidate_df = _safe_sort(candidate_df, sort_columns, sort_ascending)
    return candidate_df.head(min(3, len(candidate_df)))


def _calibrated_constraint_defaults(method_df: pd.DataFrame, recommendation_focus: str) -> dict[str, float | str]:
    top_rows = _top_constraint_calibration_rows(method_df, recommendation_focus)
    source_methods = ", ".join(top_rows["method"].astype(str).tolist()) if not top_rows.empty else "-"

    def _series_max(column: str, fallback: float) -> float:
        if column not in top_rows.columns:
            return fallback
        series = pd.to_numeric(top_rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return float(series.max()) if not series.empty else fallback

    def _series_min(column: str, fallback: float) -> float:
        if column not in top_rows.columns:
            return fallback
        series = pd.to_numeric(top_rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        return float(series.min()) if not series.empty else fallback

    return {
        "runtime_max": _series_max("avg_runtime_s_per_prompt", 60.0),
        "vram_max": _series_max("peak_vram_gb", 16.0),
        "ssim_drop_max": max(_series_max("ssim_drop_vs_bf16", float(DEFAULT_THRESHOLDS["acceptable_ssim_drop"])), 0.0),
        "lpips_increase_max": max(_series_max("lpips_delta_vs_bf16", float(DEFAULT_THRESHOLDS["acceptable_lpips_increase"])), 0.0),
        "drift_drop_max": max(_series_max("drift_last_imaging_quality_drop_vs_bf16", float(DEFAULT_THRESHOLDS["acceptable_drift_drop"])), 0.0),
        "min_compression": max(_series_min("compression_ratio", float(DEFAULT_THRESHOLDS["min_compression"])), 1.0),
        "psnr_min": _series_min("psnr", 20.0),
        "source_methods": source_methods,
    }


def _prepare_constraint_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "method_family",
        "compression_ratio",
        "peak_vram_gb",
        "avg_runtime_s_per_prompt",
        "psnr",
        "ssim",
        "lpips",
        "psnr_delta_vs_bf16",
        "ssim_delta_vs_bf16",
        "lpips_delta_vs_bf16",
        "drift_last_imaging_quality",
        "drift_last_imaging_quality_delta_vs_bf16",
        "runtime_overhead_vs_bf16_pct",
        "peak_vram_reduction_vs_bf16_pct",
        "recommended_for",
        "caution_label",
    ]
    present = [column for column in columns if column in table.columns]
    return table[present].reset_index(drop=True)


def _build_live_constraint_rankings(method_df: pd.DataFrame, recommendation_focus: str, limits: dict[str, float]) -> dict[str, pd.DataFrame]:
    if method_df.empty:
        return {}

    sort_columns, sort_ascending = get_recommendation_sort(recommendation_focus)
    runtime_max = limits["runtime_max"]
    vram_max = limits["vram_max"]
    ssim_drop_max = limits["ssim_drop_max"]
    lpips_increase_max = limits["lpips_increase_max"]
    drift_drop_max = limits["drift_drop_max"]
    psnr_min = limits["psnr_min"]

    rankings: dict[str, pd.DataFrame] = {}

    rankings[f"Quality | runtime <= {runtime_max:.1f}s"] = _prepare_constraint_table(
        _safe_sort(
            method_df[method_df["avg_runtime_s_per_prompt"].fillna(np.inf) <= runtime_max],
            ["ssim_drop_vs_bf16", "lpips_delta_vs_bf16", "psnr", "drift_last_imaging_quality_drop_vs_bf16", "peak_vram_gb"],
            [True, True, False, True, True],
        )
    )
    rankings[f"Quality | peak VRAM <= {vram_max:.2f} GB"] = _prepare_constraint_table(
        _safe_sort(
            method_df[method_df["peak_vram_gb"].fillna(np.inf) <= vram_max],
            ["ssim_drop_vs_bf16", "lpips_delta_vs_bf16", "psnr", "avg_runtime_s_per_prompt"],
            [True, True, False, True],
        )
    )
    rankings[f"Compression | SSIM drop <= {ssim_drop_max:.3f}"] = _prepare_constraint_table(
        _safe_sort(
            method_df[method_df["ssim_drop_vs_bf16"].fillna(np.inf) <= ssim_drop_max],
            ["compression_ratio", "ssim", "lpips", "psnr"],
            [False, False, True, False],
        )
    )
    rankings[f"Compression | PSNR >= {psnr_min:.2f}"] = _prepare_constraint_table(
        _safe_sort(
            method_df[method_df["psnr"].fillna(-np.inf) >= psnr_min],
            ["compression_ratio", "ssim", "lpips", "psnr"],
            [False, False, True, False],
        )
    )
    rankings[f"Compression | LPIPS increase <= {lpips_increase_max:.3f}"] = _prepare_constraint_table(
        _safe_sort(
            method_df[method_df["lpips_delta_vs_bf16"].fillna(np.inf) <= lpips_increase_max],
            ["compression_ratio", "ssim", "lpips", "psnr"],
            [False, False, True, False],
        )
    )
    rankings[f"Runtime | SSIM drop <= {ssim_drop_max:.3f}"] = _prepare_constraint_table(
        _safe_sort(
            method_df[method_df["ssim_drop_vs_bf16"].fillna(np.inf) <= ssim_drop_max],
            ["avg_runtime_s_per_prompt", "compression_ratio", "ssim", "lpips"],
            [True, False, False, True],
        )
    )
    rankings[f"Runtime | PSNR >= {psnr_min:.2f}"] = _prepare_constraint_table(
        _safe_sort(
            method_df[method_df["psnr"].fillna(-np.inf) >= psnr_min],
            ["avg_runtime_s_per_prompt", "compression_ratio", "ssim", "lpips"],
            [True, False, False, True],
        )
    )
    rankings[f"Runtime | LPIPS increase <= {lpips_increase_max:.3f}"] = _prepare_constraint_table(
        _safe_sort(
            method_df[method_df["lpips_delta_vs_bf16"].fillna(np.inf) <= lpips_increase_max],
            ["avg_runtime_s_per_prompt", "compression_ratio", "ssim", "lpips"],
            [True, False, False, True],
        )
    )
    rankings[f"Stability | drift drop <= {drift_drop_max:.3f}"] = _prepare_constraint_table(
        _safe_sort(
            method_df[method_df["drift_last_imaging_quality_drop_vs_bf16"].fillna(np.inf) <= drift_drop_max],
            ["ssim_drop_vs_bf16", "lpips_delta_vs_bf16", "compression_ratio", "avg_runtime_s_per_prompt"],
            [True, True, False, True],
        )
    )
    rankings["Best under all active constraints"] = _prepare_constraint_table(
        _safe_sort(
            method_df[
                (method_df["avg_runtime_s_per_prompt"].fillna(np.inf) <= runtime_max)
                & (method_df["peak_vram_gb"].fillna(np.inf) <= vram_max)
                & (method_df["ssim_drop_vs_bf16"].fillna(np.inf) <= ssim_drop_max)
                & (method_df["lpips_delta_vs_bf16"].fillna(np.inf) <= lpips_increase_max)
                & (method_df["drift_last_imaging_quality_drop_vs_bf16"].fillna(np.inf) <= drift_drop_max)
                & (method_df["psnr"].fillna(-np.inf) >= psnr_min)
            ],
            sort_columns,
            sort_ascending,
        )
    )
    return rankings


def _run_member_roots(run: RunLayout) -> list[Path]:
    return getattr(run, "member_roots", []) or [run.root]


def _infer_storyeval_method_from_name(name: str) -> str:
    for method in sorted(METHOD_ORDER, key=len, reverse=True):
        prefix = f"storyeval_{method}_"
        if name.startswith(prefix):
            return method
    return "BF16"


def _storyeval_method_name(root: Path) -> str:
    cfg = _read_json(root / "summary" / "config.json")
    if isinstance(cfg, dict):
        method = cfg.get("method")
        if isinstance(method, str) and method.strip():
            return method.strip()
    return _infer_storyeval_method_from_name(root.name)


def _storyeval_group_name(root: Path) -> str:
    method = _storyeval_method_name(root)
    prefix = f"storyeval_{method}_"
    if root.name.startswith(prefix):
        return f"storyeval_{root.name[len(prefix):]}"
    return root.name


def _metric_column_config() -> dict[str, Any]:
    return {
        "method": st.column_config.TextColumn("method", help="Quantization method name."),
        "status": st.column_config.TextColumn("status", help="Method completion status in the selected run."),
        "source_run": st.column_config.TextColumn("source_run", help="Original run directory used to source this method."),
        "videos": st.column_config.NumberColumn(
            "videos",
            help="Number of generated videos discovered for the method in this run.",
            format="%d",
        ),
        "logged_prompts": st.column_config.NumberColumn(
            "logged_prompts",
            help="Number of prompt records found in generation logs.",
            format="%d",
        ),
        "psnr": st.column_config.TextColumn(
            "psnr",
            help="Peak Signal-to-Noise Ratio vs BF16 reference. Higher is better. BF16 self-comparison is shown as a reference label instead of a raw infinite value.",
        ),
        "ssim": st.column_config.NumberColumn(
            "ssim",
            help="Structural Similarity vs BF16 reference. Higher is better.",
            format="%.4f",
        ),
        "lpips": st.column_config.NumberColumn(
            "lpips",
            help="LPIPS vs BF16 reference. Lower is better.",
            format="%.4f",
        ),
        "background_consistency": st.column_config.NumberColumn(
            "background_consistency",
            help="VBench background_consistency. Higher is better.",
            format="%.4f",
        ),
        "imaging_quality": st.column_config.NumberColumn(
            "imaging_quality",
            help="VBench imaging_quality. Higher is better.",
            format="%.4f",
        ),
        "subject_consistency": st.column_config.NumberColumn(
            "subject_consistency",
            help="VBench subject_consistency. Higher is better.",
            format="%.4f",
        ),
        "aesthetic_quality": st.column_config.NumberColumn(
            "aesthetic_quality",
            help="VBench aesthetic_quality. Higher is better.",
            format="%.4f",
        ),
        "compression_ratio": st.column_config.NumberColumn(
            "compression_ratio",
            help="bf16_kv_bytes / compressed_kv_bytes from efficiency logs. Higher is better for compression.",
            format="%.4f",
        ),
        "bf16_kv_bytes_gb": st.column_config.NumberColumn(
            "bf16_kv_bytes_gb",
            help=f"BF16 KV bytes converted to GB. {KV_BYTES_NOTE}",
            format="%.4f GB",
        ),
        "compressed_kv_bytes_gb": st.column_config.NumberColumn(
            "compressed_kv_bytes_gb",
            help=f"Compressed KV bytes converted to GB. {KV_BYTES_NOTE}",
            format="%.4f GB",
        ),
        "compressed_kv_bytes": st.column_config.NumberColumn(
            "compressed_kv_bytes",
            help=f"Compressed KV bytes (raw integer). {KV_BYTES_NOTE}",
            format="%d",
        ),
        "avg_runtime_s_per_prompt": st.column_config.NumberColumn(
            "avg_runtime_s_per_prompt",
            help="Average end-to-end generation time per prompt in seconds. Lower is faster.",
            format="%.3f s",
        ),
        "runtime_overhead_pct_vs_bf16": st.column_config.NumberColumn(
            "runtime_overhead_pct_vs_bf16",
            help="Percent runtime change relative to BF16 in the same run.",
            format="%.2f%%",
        ),
        "peak_vram_gb": st.column_config.NumberColumn(
            "peak_vram_gb",
            help=f"Peak GPU memory in GB. {QUANT_VRAM_NOTE}",
            format="%.3f GB",
        ),
        "drift_last_imaging_quality": st.column_config.NumberColumn(
            "drift_last_imaging_quality",
            help="Last available imaging_quality value from the method's drift curve. Higher is better.",
            format="%.4f",
        ),
    }


def _decision_column_config(columns: list[str]) -> dict[str, Any]:
    configs: dict[str, Any] = {
        "method": st.column_config.TextColumn("Method", help="Quantization method name."),
        "method_family": st.column_config.TextColumn("Family", help=_tooltip_text("method_family")),
        "bit_width_label": st.column_config.TextColumn("Bit-width / mode", help=_tooltip_text("bit_width_mode")),
        "source_users": st.column_config.TextColumn("Source users", help=_tooltip_text("source_users")),
        "run_count": st.column_config.NumberColumn("Run count", help="Number of run-level summaries contributing to this benchmark-level method row.", format="%d"),
        "prompt_count": st.column_config.NumberColumn("Prompt count", help="Total prompt-level observations contributing to the method summary.", format="%d"),
        "seed_count": st.column_config.NumberColumn("Seed count", help="Number of prompt/seed records represented by the method summary.", format="%d"),
        "compression_ratio": st.column_config.NumberColumn("Compression", help=_tooltip_text("compression_ratio"), format="%.2fx"),
        "peak_vram_gb": st.column_config.NumberColumn("Peak VRAM", help=_tooltip_text("peak_vram"), format="%.2f GB"),
        "peak_compressed_kv_gb": st.column_config.NumberColumn("Compressed KV", help="Maximum compressed KV-cache footprint observed during the run, in gigabytes. Lower is smaller.", format="%.2f GB"),
        "avg_runtime_s_per_prompt": st.column_config.NumberColumn("Runtime / prompt", help=_tooltip_text("runtime"), format="%.1f s"),
        "imaging_quality": st.column_config.NumberColumn("Imaging quality", help="VBench imaging-quality score. Higher is better.", format="%.3f"),
        "drift_last_imaging_quality": st.column_config.NumberColumn("Drift last", help="Last available imaging-quality point from the drift curve. Higher is better for temporal stability.", format="%.3f"),
        "psnr": st.column_config.TextColumn("PSNR", help="Peak Signal-to-Noise Ratio relative to BF16. Higher is better. BF16 self-comparison is shown as a reference label instead of a raw infinite value."),
        "ssim": st.column_config.NumberColumn("SSIM", help="Structural Similarity Index relative to BF16. Higher is better.", format="%.3f"),
        "lpips": st.column_config.NumberColumn("LPIPS", help="Learned perceptual image distance relative to BF16. Lower is better.", format="%.3f"),
        "psnr_delta_vs_bf16": st.column_config.NumberColumn("PSNR Δ vs BF16", help="PSNR difference relative to BF16. This can be undefined when BF16 PSNR is infinite.", format="%+.3f"),
        "imaging_quality_delta_vs_bf16": st.column_config.NumberColumn("Imaging Δ vs BF16", help=_tooltip_text("imaging_delta_vs_bf16"), format="%+.3f"),
        "drift_last_imaging_quality_delta_vs_bf16": st.column_config.NumberColumn("Drift Δ vs BF16", help=_tooltip_text("drift_delta_vs_bf16"), format="%+.3f"),
        "ssim_delta_vs_bf16": st.column_config.NumberColumn("SSIM Δ vs BF16", help="SSIM difference relative to BF16. Values near zero preserve structure better.", format="%+.3f"),
        "lpips_delta_vs_bf16": st.column_config.NumberColumn("LPIPS Δ vs BF16", help="LPIPS difference relative to BF16. Smaller or negative is better.", format="%+.3f"),
        "runtime_overhead_vs_bf16_pct": st.column_config.NumberColumn("Runtime vs BF16", help="Relative runtime change compared with BF16. Lower or negative is better.", format="%+.1f%%"),
        "peak_vram_reduction_vs_bf16_pct": st.column_config.NumberColumn("VRAM reduction vs BF16", help="Peak-VRAM reduction relative to BF16. Higher means more memory saved.", format="%.1f%%"),
        "compression_gain_vs_bf16": st.column_config.NumberColumn("Compression gain vs BF16", help="Additional compression beyond the BF16 baseline. Higher means stronger compression improvement.", format="%.2fx"),
        "recommended_for": st.column_config.TextColumn("Recommended for", help=_tooltip_text("recommended_for")),
        "caution_label": st.column_config.TextColumn("Caution", help=_tooltip_text("caution_label")),
        "auto_explanation": st.column_config.TextColumn("Why it lands here", help="Auto-generated summary of the method's practical strengths and weaknesses under the current benchmark."),
        "pareto_balanced_practical": st.column_config.CheckboxColumn("Balanced frontier", help="Whether the method lies on the balanced practical frontier."),
        "pareto_quality_preserving_compression": st.column_config.CheckboxColumn("Quality-preserving frontier", help="Whether the method lies on the compression-versus-quality frontier."),
        "pareto_systems_efficiency": st.column_config.CheckboxColumn("Systems frontier", help="Whether the method lies on the runtime-versus-VRAM-versus-compression frontier."),
        "pareto_quality_first": st.column_config.CheckboxColumn("Quality-first frontier", help="Whether the method lies on the quality-first frontier."),
        "dominated_by_balanced_practical_count": st.column_config.NumberColumn("Balanced dominators", help="How many methods jointly beat this one on the balanced practical frontier objectives.", format="%d"),
        "dominated_by_balanced_practical": st.column_config.TextColumn("Dominated by", help="Methods that jointly dominate this one on the balanced practical frontier."),
        "pareto_balanced_practical_explanation": st.column_config.TextColumn("Balanced-frontier explanation", help="Why the method does or does not survive the balanced practical frontier."),
        "frontier": st.column_config.TextColumn("Frontier", help=_tooltip_text("pareto_frontier")),
        "methods": st.column_config.TextColumn("Methods", help="Methods currently lying on that frontier."),
        "path": st.column_config.TextColumn("Path", help="Relative path of the discovered source artifact."),
        "kind": st.column_config.TextColumn("Kind", help="Heuristic source category used during source discovery."),
        "analysis_role": st.column_config.TextColumn("Analysis role", help="Whether the source is the selected primary table or a supporting provenance artifact."),
        "selected_as_primary": st.column_config.CheckboxColumn("Primary", help="Whether this source was selected as the main comparison table."),
        "rows": st.column_config.NumberColumn("Rows", help="Number of rows read from the discovered CSV.", format="%d"),
        "column_count": st.column_config.NumberColumn("Columns", help="Number of columns present in the discovered CSV.", format="%d"),
        "note": st.column_config.TextColumn("Note", help="Discovery note, including skipped-inline-read or load-status details."),
    }
    return {column: config for column, config in configs.items() if column in columns}


def discover_runs_payload(results_root_str: str) -> list[dict[str, Any]]:
    results_root = Path(results_root_str)
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
        try:
            legacy_ts = int(results_root.stat().st_mtime)
        except Exception:
            legacy_ts = 0
        runs.append(
            {
                "label": f"runs/legacy_root_{legacy_ts}",
                "benchmark": "moviegen",
                "root": str(results_root),
                "metric_dirs": [str(current_metrics)],
                "log_dirs": [str(current_logs)],
                "video_dirs": [str(current_videos)],
                "table_dirs": [str(current_tables)],
            }
        )

    runs_root = results_root / "runs"
    if runs_root.exists():
        for d in sorted([p for p in runs_root.iterdir() if p.is_dir()], reverse=True):
            metric_dir = d / "metrics"
            log_dir = d / "logs"
            video_dir = d / "videos"
            table_dir = d / "tables"
            has_data = (
                (video_dir.exists() and any(video_dir.glob("*/*.mp4")))
                or (log_dir.exists() and any(log_dir.glob("generation_*.jsonl")))
                or (metric_dir.exists() and any(metric_dir.glob("*.json")))
            )
            if not has_data:
                continue
            runs.append(
                {
                    "label": f"runs/{d.name}",
                    "benchmark": "moviegen",
                    "root": str(d),
                    "metric_dirs": [str(p) for p in [metric_dir, d] if p.exists()],
                    "log_dirs": [str(p) for p in [log_dir, d] if p.exists()],
                    "video_dirs": [str(p) for p in [video_dir, d] if p.exists()],
                    "table_dirs": [str(p) for p in [table_dir, d] if p.exists()],
                }
            )

    archive_root = results_root / "archive"
    if archive_root.exists():
        for d in sorted([p for p in archive_root.iterdir() if p.is_dir()], reverse=True):
            metric_dirs = [d / "metrics", d]
            log_dirs = [d / "logs", d]
            video_dirs = [d / "videos", d]
            table_dirs = [d / "tables", d]
            runs.append(
                {
                    "label": f"archive/{d.name}",
                    "benchmark": "moviegen",
                    "root": str(d),
                    "metric_dirs": [str(p) for p in metric_dirs if p.exists()],
                    "log_dirs": [str(p) for p in log_dirs if p.exists()],
                    "video_dirs": [str(p) for p in video_dirs if p.exists()],
                    "table_dirs": [str(p) for p in table_dirs if p.exists()],
                }
            )

    storyeval_root = results_root / "benchmarks" / "storyeval"
    if storyeval_root.exists():
        grouped_storyeval: dict[str, dict[str, Any]] = {}
        for d in sorted([p for p in storyeval_root.iterdir() if p.is_dir()], reverse=True):
            has_data = (
                (d / "per_prompt").exists()
                and any((d / "per_prompt").glob("*.json"))
            ) or ((d / "videos").exists() and any((d / "videos").glob("*.mp4")))
            if not has_data:
                continue
            group_name = _storyeval_group_name(d)
            payload = grouped_storyeval.setdefault(
                group_name,
                {
                    "label": f"storyeval/{group_name}",
                    "benchmark": "storyeval",
                    "root": str(d),
                    "member_roots": [],
                    "metric_dirs": [],
                    "log_dirs": [],
                    "video_dirs": [],
                    "table_dirs": [],
                },
            )
            payload["member_roots"].append(str(d))
            payload["metric_dirs"].extend(str(p) for p in [d / "metrics", d] if p.exists())
            payload["log_dirs"].extend(str(p) for p in [d / "logs", d] if p.exists())
            payload["video_dirs"].extend(str(p) for p in [d / "videos", d] if p.exists())
            payload["table_dirs"].extend(str(p) for p in [d / "summary", d / "tables", d] if p.exists())
        runs.extend(grouped_storyeval.values())

    return runs


def discover_runs(results_root: Path) -> list[RunLayout]:
    payloads = discover_runs_payload(str(results_root))
    runs: list[RunLayout] = []
    for p in payloads:
        runs.append(
            RunLayout(
                label=p["label"],
                benchmark=p.get("benchmark", "moviegen"),
                root=Path(p["root"]),
                member_roots=[Path(x) for x in p.get("member_roots", [p["root"]])],
                metric_dirs=[Path(x) for x in p["metric_dirs"]],
                log_dirs=[Path(x) for x in p["log_dirs"]],
                video_dirs=[Path(x) for x in p["video_dirs"]],
                table_dirs=[Path(x) for x in p["table_dirs"]],
            )
        )
    return runs


@st.cache_data(show_spinner=False)
def load_prompts(prompts_path: Path) -> dict[int, str]:
    if not prompts_path.exists():
        return {}
    lines = prompts_path.read_text(encoding="utf-8").splitlines()
    return {idx: txt for idx, txt in enumerate(lines)}


@st.cache_data(show_spinner=False)
def list_methods(run: RunLayout) -> list[str]:
    methods: set[str] = set()

    if run.benchmark == "storyeval":
        for root in _run_member_roots(run):
            methods.add(_storyeval_method_name(root))
        return _order_methods(methods)

    for d in run.metric_dirs:
        for prefix in ["efficiency", "fidelity", "vbench"]:
            for path in d.glob(f"{prefix}_*.json"):
                methods.add(path.stem[len(prefix) + 1 :])

    for base in run.video_dirs:
        if not base.exists():
            continue
        for sub in base.iterdir():
            if sub.is_dir() and any(sub.glob("prompt_*_seed_*.mp4")):
                methods.add(sub.name)

    return _order_methods(methods)


@st.cache_data(show_spinner=False)
def _load_jsonl_records(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


@st.cache_data(show_spinner=False)
def load_generation_records(run: RunLayout, method: str) -> list[dict[str, Any]]:
    path = _find_file(run.log_dirs, f"generation_{method}.jsonl")
    if path is None:
        return []
    return _load_jsonl_records(str(path))


@st.cache_data(show_spinner=False)
def load_vram_trace_records(run: RunLayout, method: str) -> list[dict[str, Any]]:
    path = _find_file(run.log_dirs, f"vram_trace_{method}.jsonl")
    if path is None:
        return []
    return _load_jsonl_records(str(path))


@st.cache_data(show_spinner=False)
def load_storyeval_vram_trace_records(run: RunLayout) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in _run_member_roots(run):
        path = root / "logs" / "vram_trace_storyeval.jsonl"
        method = _storyeval_method_name(root)
        if not path.exists():
            continue
        for payload in _load_jsonl_records(str(path)):
            payload = dict(payload)
            payload["method"] = payload.get("method", method)
            rows.append(payload)
    return rows


@st.cache_data(show_spinner=False)
def load_dataset_vram_trace_records(run_root_str: str, benchmark: str, method: str) -> list[dict[str, Any]]:
    run_root = Path(run_root_str)
    if benchmark == "storyeval":
        rows: list[dict[str, Any]] = []
        for payload in _load_jsonl_records(str(run_root / "logs" / "vram_trace_storyeval.jsonl")):
            item = dict(payload)
            item["method"] = item.get("method", method)
            rows.append(item)
        return rows
    return _load_jsonl_records(str(run_root / "logs" / f"vram_trace_{method}.jsonl"))


@st.cache_data(show_spinner=False)
def load_metric_payload(run: RunLayout, prefix: str, method: str) -> dict[str, Any] | None:
    path = _find_file(run.metric_dirs, f"{prefix}_{method}.json")
    if path is None:
        return None
    return _read_json(path)


@st.cache_data(show_spinner=False)
def build_metric_table(run: RunLayout, methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    manifest = _load_method_manifest(run)

    for method in methods:
        efficiency = load_metric_payload(run, "efficiency", method) or {}
        fidelity = load_metric_payload(run, "fidelity", method) or {}
        vbench = load_metric_payload(run, "vbench", method) or {}
        drift = load_metric_payload(run, "drift", method) or {}
        method_meta = manifest.get(method, {})

        records = load_generation_records(run, method)
        num_videos = 0
        for base in run.video_dirs:
            method_dir = base / method
            if method_dir.exists():
                num_videos = max(num_videos, len(list(method_dir.glob("prompt_*_seed_*.mp4"))))

        fidelity_agg = fidelity.get("aggregate", {}) if isinstance(fidelity, dict) else {}
        drift_curve = drift.get("curve", []) if isinstance(drift, dict) else []
        drift_last = None
        if drift_curve:
            drift_value = drift_curve[-1].get("imaging_quality")
            if isinstance(drift_value, list) and drift_value:
                drift_last = drift_value[0]
            elif isinstance(drift_value, (int, float)):
                drift_last = drift_value

        row: dict[str, Any] = {
            "method": method,
            "status": method_meta.get("status"),
            "source_run": method_meta.get("source_run"),
            "videos": num_videos,
            "logged_prompts": len(records),
            "psnr": fidelity_agg.get("psnr"),
            "ssim": fidelity_agg.get("ssim"),
            "lpips": fidelity_agg.get("lpips"),
            "background_consistency": _extract_vbench_scalar(vbench.get("background_consistency") if isinstance(vbench, dict) else None),
            "imaging_quality": _extract_vbench_scalar(vbench.get("imaging_quality") if isinstance(vbench, dict) else None),
            "subject_consistency": _extract_vbench_scalar(vbench.get("subject_consistency") if isinstance(vbench, dict) else None),
            "aesthetic_quality": _extract_vbench_scalar(vbench.get("aesthetic_quality") if isinstance(vbench, dict) else None),
            "bf16_kv_bytes": efficiency.get("bf16_kv_bytes"),
            "compressed_kv_bytes": efficiency.get("compressed_kv_bytes"),
            "bf16_kv_bytes_gb": (float(efficiency["bf16_kv_bytes"]) / (1024**3)) if efficiency.get("bf16_kv_bytes") is not None else None,
            "compressed_kv_bytes_gb": (float(efficiency["compressed_kv_bytes"]) / (1024**3)) if efficiency.get("compressed_kv_bytes") is not None else None,
            "compression_ratio": efficiency.get("compression_ratio"),
            "total_runtime_s": efficiency.get("total_runtime_s"),
            "avg_runtime_s_per_prompt": efficiency.get("avg_runtime_s_per_prompt"),
            "peak_vram_gb": (float(efficiency["peak_vram_bytes"]) / (1024**3)) if efficiency.get("peak_vram_bytes") is not None else None,
            "quantize_time_s": efficiency.get("quantize_time_s"),
            "dequantize_time_s": efficiency.get("dequantize_time_s"),
            "drift_points": len(drift_curve),
            "drift_last_imaging_quality": drift_last,
            "note": method_meta.get("note"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    bf16_row = df[df["method"] == "BF16"]
    if not bf16_row.empty:
        bf16_runtime = bf16_row.iloc[0]["avg_runtime_s_per_prompt"]
        bf16_vram = bf16_row.iloc[0]["peak_vram_gb"]
        if pd.notna(bf16_runtime) and bf16_runtime > 0:
            df["runtime_overhead_pct_vs_bf16"] = 100.0 * (df["avg_runtime_s_per_prompt"] - bf16_runtime) / bf16_runtime
        else:
            df["runtime_overhead_pct_vs_bf16"] = None
        if pd.notna(bf16_vram) and bf16_vram > 0:
            df["peak_vram_delta_gb_vs_bf16"] = df["peak_vram_gb"] - bf16_vram
        else:
            df["peak_vram_delta_gb_vs_bf16"] = None
    else:
        df["runtime_overhead_pct_vs_bf16"] = None
        df["peak_vram_delta_gb_vs_bf16"] = None

    df["method"] = pd.Categorical(df["method"], categories=_order_methods(set(df["method"].tolist())), ordered=True)
    return df.sort_values("method").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_video_index(run: RunLayout, methods: list[str]) -> dict[str, dict[int, Path]]:
    out: dict[str, dict[int, Path]] = {}
    for method in methods:
        per_prompt: dict[int, Path] = {}
        for base in run.video_dirs:
            method_dir = base / method
            if not method_dir.exists():
                continue
            for video in sorted(method_dir.glob("prompt_*_seed_*.mp4")):
                m = VIDEO_RE.search(video.name)
                if not m:
                    continue
                prompt_id = int(m.group(1))
                if prompt_id not in per_prompt:
                    per_prompt[prompt_id] = video
        if per_prompt:
            out[method] = per_prompt
    return out


@st.cache_data(show_spinner=False)
def load_fidelity_per_video(run: RunLayout, method: str) -> dict[str, dict[str, Any]]:
    payload = load_metric_payload(run, "fidelity", method)
    if not payload:
        return {}
    entries = payload.get("per_video", [])
    out: dict[str, dict[str, Any]] = {}
    if isinstance(entries, list):
        for item in entries:
            video_name = item.get("video")
            if video_name:
                out[video_name] = item
    return out


@st.cache_data(show_spinner=False)
def load_run_meta(run: RunLayout) -> dict[str, Any]:
    if run.benchmark == "storyeval":
        payloads: list[dict[str, Any]] = []
        for root in _run_member_roots(run):
            summary_cfg = root / "summary" / "config.json"
            payload = _read_json(summary_cfg)
            if isinstance(payload, dict):
                payloads.append(payload)
        if payloads:
            merged = dict(payloads[0])
            merged["methods"] = _order_methods(
                {str(p.get("method")) for p in payloads if isinstance(p.get("method"), str)}
            )
            merged["run_id"] = run.label.split("/", 1)[1]
            return merged
    meta_path = run.root / "run_meta.json"
    if meta_path.exists():
        payload = _read_json(meta_path)
        if isinstance(payload, dict):
            return payload
    return {}


@st.cache_data(show_spinner=False)
def load_storyeval_records(run: RunLayout) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in _run_member_roots(run):
        per_prompt_dir = root / "per_prompt"
        method = _storyeval_method_name(root)
        if not per_prompt_dir.exists():
            continue
        for p in sorted(per_prompt_dir.glob("*.json")):
            rec = _read_json(p)
            if isinstance(rec, dict):
                row = dict(rec)
                row["method"] = row.get("method", method)
                row["run_root"] = str(root)
                rows.append(row)
    return rows


@st.cache_data(show_spinner=False)
def load_storyeval_summary(run: RunLayout) -> dict[str, Any]:
    payload: dict[str, Any] = {"methods": {}}
    for root in _run_member_roots(run):
        method = _storyeval_method_name(root)
        for p in [root / "summary" / "summary.json", root / "summary" / "runner_summary.json"]:
            method_payload = _read_json(p)
            if isinstance(method_payload, dict):
                payload["methods"][method] = method_payload
                break
    return payload


@st.cache_data(show_spinner=False)
def load_storyeval_vbench(run: RunLayout) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for root in _run_member_roots(run):
        method = _storyeval_method_name(root)
        method_payload = _read_json(root / "metrics" / "vbench.json")
        if isinstance(method_payload, dict):
            payload[method] = method_payload
    return payload


@st.cache_data(show_spinner=False)
def load_storyeval_drift(run: RunLayout) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for root in _run_member_roots(run):
        method = _storyeval_method_name(root)
        method_payload = _read_json(root / "metrics" / "drift_imaging_quality.json")
        if isinstance(method_payload, dict):
            payload[method] = method_payload
    return payload


def _storyeval_per_video_metrics(vbench_by_method: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for method, payload in vbench_by_method.items():
        per_video = payload.get("per_video", {}) if isinstance(payload, dict) else {}
        if isinstance(per_video, dict):
            out[method] = {k: v for k, v in per_video.items() if isinstance(v, dict)}
    return out


def build_storyeval_metric_table(run: RunLayout, methods: list[str]) -> pd.DataFrame:
    records = load_storyeval_records(run)
    summaries = load_storyeval_summary(run).get("methods", {})
    vbench_by_method = load_storyeval_vbench(run)
    drift_by_method = load_storyeval_drift(run)

    rows: list[dict[str, Any]] = []
    for method in methods:
        method_records = [r for r in records if r.get("method") == method]
        vbench = vbench_by_method.get(method, {})
        agg = vbench.get("aggregate", {}) if isinstance(vbench.get("aggregate"), dict) else {}
        summary = summaries.get(method, {}) if isinstance(summaries, dict) else {}
        drift_curve = drift_by_method.get(method, {}).get("curve", []) if isinstance(drift_by_method.get(method), dict) else []
        row = {
            "method": method,
            "status": summary.get("status"),
            "source_run": summary.get("source_run"),
            "videos": len([r for r in method_records if not r.get("error")]),
            "logged_prompts": len(method_records),
            "background_consistency": agg.get("background_consistency"),
            "imaging_quality": agg.get("imaging_quality"),
            "subject_consistency": agg.get("subject_consistency"),
            "aesthetic_quality": agg.get("aesthetic_quality"),
            "avg_runtime_s_per_prompt": summary.get("avg_runtime_sec"),
            "peak_vram_gb": (
                float(summary["avg_peak_vram_mb"]) / 1024.0 if summary.get("avg_peak_vram_mb") is not None else None
            ),
            "drift_points": len(drift_curve),
            "drift_last_imaging_quality": drift_curve[-1].get("imaging_quality") if drift_curve else None,
            "note": summary.get("note"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def render_storyeval_overview(run: RunLayout, methods: list[str]) -> None:
    summary = load_storyeval_summary(run)
    vbench = load_storyeval_vbench(run)
    drift = load_storyeval_drift(run)
    records = load_storyeval_records(run)
    summary_methods = summary.get("methods", {}) if isinstance(summary.get("methods"), dict) else {}
    metric_df = build_storyeval_metric_table(run, methods)
    total_prompts = len({(r.get("method"), r.get("prompt_id"), r.get("seed")) for r in records})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Methods", len(methods))
    c2.metric("Prompt records", total_prompts)
    c3.metric("Completed methods", sum(1 for method in methods if summary_methods.get(method)))
    avg_runtime = metric_df["avg_runtime_s_per_prompt"].mean(skipna=True) if not metric_df.empty else None
    c4.metric("Avg Runtime / Prompt", f"{float(avg_runtime):.2f}s" if pd.notna(avg_runtime) else "-")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Best Background", f"{metric_df['background_consistency'].max(skipna=True):.4f}" if not metric_df.empty and metric_df["background_consistency"].notna().any() else "-")
    c6.metric("Best Imaging", f"{metric_df['imaging_quality'].max(skipna=True):.4f}" if not metric_df.empty and metric_df["imaging_quality"].notna().any() else "-")
    c7.metric("Best Subject", f"{metric_df['subject_consistency'].max(skipna=True):.4f}" if not metric_df.empty and metric_df["subject_consistency"].notna().any() else "-")
    c8.metric("Best Aesthetic", f"{metric_df['aesthetic_quality'].max(skipna=True):.4f}" if not metric_df.empty and metric_df["aesthetic_quality"].notna().any() else "-")

    if not metric_df.empty:
        st.markdown("### Unified method table")
        st.dataframe(metric_df, use_container_width=True, hide_index=True, column_config=_metric_column_config())

    drift_rows: list[dict[str, Any]] = []
    for method, payload in drift.items():
        curve = payload.get("curve", []) if isinstance(payload, dict) else []
        for point in curve:
            row = dict(point)
            row["method"] = method
            drift_rows.append(row)
    if drift_rows:
        st.markdown("### Long-Horizon Drift (Imaging Quality)")
        drift_df = pd.DataFrame(drift_rows)
        x_col = "seconds" if "seconds" in drift_df.columns else "frame_cap"
        fig = px.line(drift_df, x=x_col, y="imaging_quality", color="method", markers=True, title="StoryEval drift curve")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    if not metric_df.empty:
        m = metric_df.melt(
            id_vars=["method"],
            value_vars=["background_consistency", "imaging_quality", "subject_consistency", "aesthetic_quality"],
            var_name="metric",
            value_name="value",
        ).dropna(subset=["value"])
        if not m.empty:
            fig = px.bar(m, x="method", y="value", color="metric", barmode="group", title="StoryEval VBench metrics")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### StoryEval run summary")
    summary_rows = []
    for method in methods:
        method_summary = summary_methods.get(method, {})
        summary_rows.append(
            {
                "run_group": run.label,
                "method": method,
                "run_root": next((str(root) for root in _run_member_roots(run) if _storyeval_method_name(root) == method), "-"),
                "num_records": method_summary.get("num_records"),
                "num_prompts": method_summary.get("num_prompts"),
                "num_success": method_summary.get("num_success", method_summary.get("counts", {}).get("completed")),
                "num_failed": method_summary.get("num_failed", method_summary.get("counts", {}).get("failed")),
                "avg_runtime_sec": method_summary.get("avg_runtime_sec", method_summary.get("avg_runtime_s")),
                "avg_peak_vram_mb": method_summary.get("avg_peak_vram_mb"),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    render_overview_explainers(metric_df)


def render_storyeval_video_explorer(run: RunLayout, methods: list[str]) -> None:
    records = load_storyeval_records(run)
    if not records:
        st.warning("No StoryEval per_prompt records found for this run.")
        return

    valid_records = [r for r in records if not r.get("error")]
    if not valid_records:
        st.warning("All StoryEval records in this run are marked as failed.")
        return

    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for rec in valid_records:
        pid = rec.get("prompt_id")
        if isinstance(pid, str):
            by_prompt.setdefault(pid, []).append(rec)
    prompt_ids = sorted(by_prompt.keys())
    selected_prompt_id = st.selectbox("Prompt ID", prompt_ids, index=0)
    prompt_records = sorted(by_prompt[selected_prompt_id], key=lambda r: (str(r.get("method")), int(r.get("seed", 0))))
    st.markdown(f"**Prompt:** {prompt_records[0].get('prompt', '-')}")

    available_methods = _order_methods({str(r.get("method")) for r in prompt_records if r.get("method")})
    selected_methods = st.multiselect(
        "Methods to display",
        available_methods,
        default=[m for m in ["BF16", "RTN_INT4", "KIVI_INT4", "QUAROT_KV_INT4"] if m in available_methods] or available_methods,
        key=f"storyeval_video_methods_{run.label}_{selected_prompt_id}",
    )
    if not selected_methods:
        st.info("Select at least one method.")
        return

    per_video = _storyeval_per_video_metrics(load_storyeval_vbench(run))
    cols = st.columns(min(4, max(1, len(selected_methods))))
    for idx, method in enumerate(selected_methods):
        col = cols[idx % len(cols)]
        with col:
            st.markdown(f"#### {method}")
            method_records = [r for r in prompt_records if r.get("method") == method]
            if not method_records:
                st.warning("Missing prompt for this method")
                continue
            seeds = [int(r.get("seed", 0)) for r in method_records]
            selected_seed = st.selectbox("Seed", seeds, index=0, key=f"storyeval_seed_{run.label}_{selected_prompt_id}_{method}")
            selected_rec = next(r for r in method_records if int(r.get("seed", 0)) == int(selected_seed))
            video_rel = selected_rec.get("generated_video_path")
            video_path = (REPO_ROOT / video_rel) if isinstance(video_rel, str) else None
            if video_path and video_path.exists():
                st.video(str(video_path))
            else:
                st.warning("Video file missing")
            video_name = Path(video_rel).name if isinstance(video_rel, str) else ""
            video_metrics = per_video.get(method, {}).get(video_name, {})
            if video_metrics:
                st.caption(
                    " | ".join(
                        [
                            f"bg: {video_metrics.get('background_consistency'):.4f}",
                            f"img: {video_metrics.get('imaging_quality'):.4f}",
                            f"subj: {video_metrics.get('subject_consistency'):.4f}",
                            f"aes: {video_metrics.get('aesthetic_quality'):.4f}",
                        ]
                    )
                )

    st.markdown("### Prompt/seed records")
    rows = []
    for rec in prompt_records:
        rows.append(
            {
                "method": rec.get("method"),
                "prompt_id": rec.get("prompt_id"),
                "seed": rec.get("seed"),
                "wall_time_sec": rec.get("wall_time_sec"),
                "peak_vram_mb": rec.get("peak_vram_mb"),
                "target_frames": rec.get("target_frames"),
                "effective_duration_sec": rec.get("effective_duration_sec"),
                "error": rec.get("error"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_storyeval_prompt_analytics(run: RunLayout, methods: list[str]) -> None:
    records = load_storyeval_records(run)
    if not records:
        st.warning("No StoryEval prompt-level records available.")
        return

    df = pd.DataFrame(records)
    if df.empty:
        st.warning("No StoryEval prompt-level records available.")
        return
    if "wall_time_sec" in df.columns:
        df["wall_time_sec"] = pd.to_numeric(df["wall_time_sec"], errors="coerce")
    if "peak_vram_mb" in df.columns:
        df["peak_vram_mb"] = pd.to_numeric(df["peak_vram_mb"], errors="coerce")
    if "line_index" in df.columns:
        df["line_index"] = pd.to_numeric(df["line_index"], errors="coerce")

    df = df[df["method"].isin(methods)]
    st.markdown("### Runtime and VRAM trends")
    c1, c2 = st.columns(2)
    if {"line_index", "wall_time_sec"}.issubset(set(df.columns)):
        with c1:
            fig = px.line(
                df.sort_values("line_index"),
                x="line_index",
                y="wall_time_sec",
                color="method" if "method" in df.columns else ("seed" if "seed" in df.columns else None),
                markers=True,
                title="Per-prompt runtime",
            )
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)
    if {"line_index", "peak_vram_mb"}.issubset(set(df.columns)):
        with c2:
            fig = px.line(
                df.sort_values("line_index"),
                x="line_index",
                y="peak_vram_mb",
                color="method" if "method" in df.columns else ("seed" if "seed" in df.columns else None),
                markers=True,
                title="Per-prompt peak VRAM (MB)",
            )
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)

    trace_rows: list[dict[str, Any]] = []
    for rec in load_storyeval_vram_trace_records(run):
        method = rec.get("method")
        if method not in methods:
            continue
        prompt_id = rec.get("prompt_id")
        seed = rec.get("seed")
        for sample in rec.get("samples", []):
            allocated_bytes = sample.get("allocated_bytes")
            reserved_bytes = sample.get("reserved_bytes")
            t_s = sample.get("t_s")
            if allocated_bytes is None or reserved_bytes is None or t_s is None:
                continue
            trace_rows.append(
                {
                    "method": method,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "t_s": float(t_s),
                    "allocated_gb": float(allocated_bytes) / (1024**3),
                    "reserved_gb": float(reserved_bytes) / (1024**3),
                    "bf16_kv_gb": float(sample.get("bf16_kv_bytes", 0)) / (1024**3),
                    "compressed_kv_gb": float(sample.get("compressed_kv_bytes", 0)) / (1024**3),
                }
            )
    if trace_rows:
        trace_df = pd.DataFrame(trace_rows)
        prompt_opts = sorted([str(x) for x in trace_df["prompt_id"].dropna().unique().tolist()])
        if prompt_opts:
            st.markdown("### VRAM and KV-cache traces")
            tc1, tc2, tc3, tc4 = st.columns([1, 1, 1, 1])
            with tc1:
                selected_prompt = st.selectbox("Trace prompt", prompt_opts, index=0, key=f"storyeval_trace_prompt_{run.label}")
            filtered_prompt = trace_df[trace_df["prompt_id"].astype(str) == selected_prompt]
            seed_opts = sorted([int(x) for x in filtered_prompt["seed"].dropna().unique().tolist()])
            with tc2:
                selected_seed = st.selectbox("Trace seed", seed_opts, index=0, key=f"storyeval_trace_seed_{run.label}")
            with tc3:
                vram_metric = st.selectbox("Trace VRAM metric", ["allocated_gb", "reserved_gb"], index=0, key=f"storyeval_trace_vram_{run.label}")
            with tc4:
                kv_metric = st.selectbox("Trace KV metric", ["compressed_kv_gb", "bf16_kv_gb"], index=0, key=f"storyeval_trace_kv_{run.label}")
            trace_methods = st.multiselect(
                "Trace methods",
                options=methods,
                default=[m for m in ["BF16", "RTN_INT4", "KIVI_INT4", "QUAROT_KV_INT4"] if m in methods] or methods,
                key=f"storyeval_trace_methods_{run.label}",
            )
            filtered = filtered_prompt[(filtered_prompt["seed"] == selected_seed) & (filtered_prompt["method"].isin(trace_methods))]
            if not filtered.empty:
                plot_cols = st.columns(2)
                with plot_cols[0]:
                    fig = px.line(
                        filtered.sort_values("t_s"),
                        x="t_s",
                        y=vram_metric,
                        color="method",
                        title=f"VRAM over time ({selected_prompt}, seed {selected_seed})",
                    )
                    fig.update_layout(height=360, xaxis_title="time (s)", yaxis_title=vram_metric.replace("_", " "))
                    st.plotly_chart(fig, use_container_width=True)
                with plot_cols[1]:
                    fig = px.line(
                        filtered.sort_values("t_s"),
                        x="t_s",
                        y=kv_metric,
                        color="method",
                        title=f"KV-cache size over time ({selected_prompt}, seed {selected_seed})",
                    )
                    fig.update_layout(height=360, xaxis_title="time (s)", yaxis_title=kv_metric.replace("_", " "))
                    st.plotly_chart(fig, use_container_width=True)
                peak_summary = (
                    filtered.groupby("method", as_index=False)[[vram_metric, kv_metric]]
                    .max()
                    .rename(columns={vram_metric: f"peak_{vram_metric}", kv_metric: f"peak_{kv_metric}"})
                )
                st.dataframe(peak_summary, use_container_width=True, hide_index=True)

    drift = load_storyeval_drift(run)
    drift_rows = []
    for method, payload in drift.items():
        if method not in methods:
            continue
        curve = payload.get("curve", []) if isinstance(payload, dict) else []
        for point in curve:
            row = dict(point)
            row["method"] = method
            drift_rows.append(row)
    if drift_rows:
        st.markdown("### Drift checkpoints")
        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)

    per_video = _storyeval_per_video_metrics(load_storyeval_vbench(run))
    metric_rows = []
    for method, videos in per_video.items():
        if method not in methods:
            continue
        for _video_name, rec in videos.items():
            metric_rows.append(
                {
                    "method": method,
                    "prompt_id": rec.get("prompt_id"),
                    "seed": rec.get("seed"),
                    "background_consistency": rec.get("background_consistency"),
                    "imaging_quality": rec.get("imaging_quality"),
                    "subject_consistency": rec.get("subject_consistency"),
                    "aesthetic_quality": rec.get("aesthetic_quality"),
                }
            )
    if metric_rows:
        st.markdown("### Per-video metric table")
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    st.markdown("### Prompt-level table")
    keep_cols = [
        "method",
        "prompt_id",
        "seed",
        "line_index",
        "wall_time_sec",
        "peak_vram_mb",
        "raw_output_frames",
        "target_frames",
        "effective_duration_sec",
        "generated_video_path",
        "error",
    ]
    cols = [c for c in keep_cols if c in df.columns]
    st.dataframe(df[cols].sort_values([c for c in ["line_index", "seed"] if c in cols]), use_container_width=True, hide_index=True)


def render_storyeval_artifacts(run: RunLayout) -> None:
    st.markdown("### Run artifacts")
    st.code("\n".join(str(root) for root in _run_member_roots(run)), language="text")

    metrics_files: list[Path] = []
    logs_files: list[Path] = []
    summary_files: list[Path] = []
    per_prompt_files: list[Path] = []
    for root in _run_member_roots(run):
        metrics_files.extend(sorted((root / "metrics").glob("*.json")) if (root / "metrics").exists() else [])
        logs_files.extend(sorted((root / "logs").glob("*")) if (root / "logs").exists() else [])
        summary_files.extend(sorted((root / "summary").glob("*")) if (root / "summary").exists() else [])
        per_prompt_files.extend(sorted((root / "per_prompt").glob("*.json")) if (root / "per_prompt").exists() else [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Metric JSON files", len(metrics_files))
    c2.metric("Log files", len(logs_files))
    c3.metric("Summary files", len(summary_files))
    c4.metric("Per-prompt records", len(per_prompt_files))

    with st.expander("Metrics files", expanded=False):
        st.write("\n".join(p.name for p in metrics_files) if metrics_files else "None")
    with st.expander("Summary files", expanded=False):
        st.write("\n".join(p.name for p in summary_files) if summary_files else "None")
    with st.expander("Log files", expanded=False):
        st.write("\n".join(p.name for p in logs_files) if logs_files else "None")

    summary = load_storyeval_summary(run)
    if summary:
        st.markdown("### StoryEval summaries by method")
        st.json(summary)


def render_header() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

        .stApp,
        [data-testid="stAppViewContainer"] {
            background: var(--background-color);
            color: var(--text-color);
            font-family: 'Manrope', sans-serif;
        }

        [data-testid="stAppViewContainer"] .main .block-container,
        [data-testid="stSidebar"] {
            color: var(--text-color);
            font-family: 'Manrope', sans-serif;
        }

        .hero {
            padding: 0.9rem 1.1rem;
            border-radius: 14px;
            border: 1px solid rgba(114, 128, 146, 0.24);
            background: linear-gradient(120deg, rgba(11, 70, 117, 0.95), rgba(24, 123, 102, 0.9));
            color: #f4f8ff;
            margin-bottom: 1rem;
        }

        .hero * {
            color: #f4f8ff !important;
        }

        .hero h1 {
            margin: 0;
            font-size: 1.5rem;
            letter-spacing: 0.01em;
        }

        .hero p {
            margin: 0.25rem 0 0 0;
            opacity: 0.92;
            font-size: 0.95rem;
        }

        .stat-card {
            border: 1px solid rgba(114, 128, 146, 0.24);
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
            background: var(--secondary-background-color);
        }

        .info-card,
        .recommendation-card {
            border: 1px solid rgba(114, 128, 146, 0.24);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: var(--secondary-background-color);
            height: 100%;
        }

        .recommendation-card h4,
        .info-card h4 {
            margin: 0 0 0.35rem 0;
        }

        .recommendation-card p,
        .info-card p,
        .recommendation-card li,
        .info-card li {
            margin: 0.2rem 0;
            line-height: 1.45;
        }

        .pill {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            background: rgba(37, 99, 235, 0.14);
            color: var(--text-color);
            margin-bottom: 0.55rem;
        }

        .mono {
            font-family: 'IBM Plex Mono', monospace;
            color: var(--text-color);
        }

        .section-heading {
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .tooltip-inline {
            display: inline-flex;
            align-items: center;
            gap: 0.38rem;
            flex-wrap: wrap;
        }

        .info-dot {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            position: relative;
            width: 1.05rem;
            height: 1.05rem;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.3);
            background: rgba(148, 163, 184, 0.08);
            color: #94a3b8;
            font-size: 0.72rem;
            font-weight: 700;
            line-height: 1;
            cursor: help;
        }

        .info-dot::after {
            content: attr(data-tooltip);
            position: absolute;
            left: 50%;
            bottom: calc(100% + 0.4rem);
            transform: translateX(-50%);
            min-width: 12rem;
            max-width: 18rem;
            padding: 0.45rem 0.55rem;
            border-radius: 10px;
            border: 1px solid rgba(203, 213, 225, 0.95);
            background: rgba(255, 255, 255, 0.98);
            color: #64748b;
            font-size: 0.74rem;
            font-weight: 500;
            line-height: 1.35;
            text-align: left;
            white-space: normal;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
            z-index: 1000;
        }

        .info-dot::before {
            content: "";
            position: absolute;
            left: 50%;
            bottom: calc(100% + 0.12rem);
            width: 0.45rem;
            height: 0.45rem;
            transform: translateX(-50%) rotate(45deg);
            background: rgba(255, 255, 255, 0.98);
            border-right: 1px solid rgba(203, 213, 225, 0.95);
            border-bottom: 1px solid rgba(203, 213, 225, 0.95);
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
            z-index: 999;
        }

        .info-dot:hover::after,
        .info-dot:hover::before,
        .info-dot:focus::after,
        .info-dot:focus::before {
            opacity: 1;
            visibility: visible;
        }

        .info-dot:focus {
            outline: 1px solid rgba(148, 163, 184, 0.25);
            outline-offset: 1px;
        }

        span.doinfo,
        [data-testid="stTooltipIcon"],
        [data-testid="stTooltipIcon"] span,
        button[aria-label*="Help"] span {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 1.05rem !important;
            height: 1.05rem !important;
            min-width: 1.05rem !important;
            min-height: 1.05rem !important;
            border-radius: 999px !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            background: rgba(148, 163, 184, 0.08) !important;
            color: #94a3b8 !important;
            fill: #94a3b8 !important;
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            line-height: 1 !important;
            box-shadow: none !important;
        }

        button[aria-label*="Help"] {
            position: relative !important;
            padding: 0 !important;
            min-height: 1.05rem !important;
            min-width: 1.05rem !important;
            width: 1.05rem !important;
            height: 1.05rem !important;
            border-radius: 999px !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            background: rgba(148, 163, 184, 0.08) !important;
            color: transparent !important;
            box-shadow: none !important;
        }

        button[aria-label*="Help"] svg {
            opacity: 0 !important;
            width: 0 !important;
            height: 0 !important;
        }

        button[aria-label*="Help"]::before {
            content: "i";
            position: absolute;
            inset: 0;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #94a3b8;
            font-size: 0.72rem;
            font-weight: 700;
            line-height: 1;
        }

        button[aria-label*="Help"]:hover,
        button[aria-label*="Help"]:focus,
        button[aria-label*="Help"]:focus-visible {
            border-color: rgba(148, 163, 184, 0.34) !important;
            background: rgba(148, 163, 184, 0.12) !important;
            box-shadow: none !important;
            outline: none !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.62rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.92rem !important;
        }
        </style>
        <div class="hero">
            <h1>KV-Cache Quantization Dashboard</h1>
            <p>Decision dashboard for selecting KV-cache methods for Self-Forcing Wan-1.3B across quality, stability, runtime, and memory trade-offs.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _describe_method(method_name: str) -> str:
    method = str(method_name)
    if method == "BF16":
        desc = "Uncompressed BF16 KV-cache baseline used as the quality and systems reference."
    elif method.startswith("RTN"):
        desc = "Round-to-nearest block quantization baseline that lowers precision without changing the overall cache policy."
    elif method.startswith("KIVI"):
        desc = "Asymmetric key/value quantization that uses different reduction schemes for keys and values."
    elif method.startswith("QUAROT_KV"):
        desc = "Hadamard-rotation-assisted KV quantization that spreads outliers before low-bit quantization."
    elif method.startswith("PRQ"):
        desc = "Progressive residual quantization that stores a coarse low-bit code plus a residual correction."
    elif method.startswith("QAQ"):
        desc = "Outlier-aware asymmetric quantization that preserves large activations more carefully than uniform rounding."
    elif method.startswith("AGE_TIER"):
        desc = "Recency-aware tiered quantization that keeps recent tokens at higher precision than older tokens."
    elif method.startswith("TPTQ"):
        desc = "Temporal progressive tiered quantization that combines recency-aware allocation with residual-style coding."
    elif method.startswith("FLOWCACHE_NATIVE_SOFT_PRUNE"):
        desc = "In-house FlowCache-inspired Wan2.1 adaptation that combines residual reuse with soft-prune KV compression."
    elif method.startswith("FLOWCACHE_NATIVE"):
        desc = "In-house FlowCache-inspired Wan2.1 adaptation that reuses cached denoising residuals when relative-L1 feature drift stays low."
    elif method.startswith("FLOWCACHE_SOFT_PRUNE"):
        desc = "In-house FlowCache-inspired soft-prune policy that keeps recent and important chunks while replacing evicted old chunks with pooled summaries."
    elif method.startswith("FLOWCACHE_PRUNE"):
        desc = "In-house FlowCache-inspired prune policy that keeps recent and important chunks while evicting the least important old chunks."
    elif method.startswith("FLOWCACHE_ADAPTIVE"):
        desc = "In-house FlowCache-inspired adaptive policy that allocates cache budget using temporal importance scores."
    elif method.startswith("FLOWCACHE_HYBRID"):
        desc = "In-house FlowCache-inspired hybrid policy that combines temporal chunking with layer-aware cache budgets."
    elif method.startswith("SPATIAL_MIXED"):
        desc = "Foreground/background mixed quantization that applies different quantizers to spatially distinct regions."
    else:
        desc = "Quantized KV-cache method included in the current comparison."

    notes: list[str] = []
    if "_REFRESH" in method:
        notes.append("Refresh variant periodically restores less-compressed state to limit accumulated error.")
    if "_RECENT2" in method:
        notes.append("Recent-context variant preserves the newest context more conservatively.")
    if "K2_V4" in method:
        notes.append("Key/value asymmetric-bit variant with lower-bit keys and higher-bit values.")
    if method.endswith("INT2"):
        notes.append("Uses a more aggressive 2-bit operating point.")
    elif method.endswith("INT4"):
        notes.append("Uses a milder 4-bit operating point.")
    return " ".join([desc, *notes]).strip()


def _method_glossary_rows(metric_df: pd.DataFrame) -> list[dict[str, str]]:
    if metric_df.empty:
        return []
    raw_col = "raw_method" if "raw_method" in metric_df.columns else "method"
    family_col = "method_family" if "method_family" in metric_df.columns else None
    display_cols = ["method"]
    if raw_col != "method":
        display_cols.append(raw_col)
    if family_col:
        display_cols.append(family_col)
    glossary_df = metric_df[display_cols].drop_duplicates().copy()
    glossary_df["__order"] = glossary_df[raw_col].astype(str).map(lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 9999)
    glossary_df = glossary_df.sort_values(["__order", "method"]).drop(columns="__order")

    rows: list[dict[str, str]] = []
    for row in glossary_df.itertuples(index=False):
        display_name = str(getattr(row, "method"))
        raw_method = str(getattr(row, raw_col))
        family = str(getattr(row, family_col)) if family_col else "-"
        if family.lower() == "nan":
            family = "-"
        rows.append(
            {
                "Method": display_name,
                "Family": family,
                "Explanation": _describe_method(raw_method),
            }
        )
    return rows


def _metric_glossary_rows(metric_df: pd.DataFrame) -> list[dict[str, str]]:
    explanations = {
        "videos": ("Videos", "Number of videos represented by the summary row.", "Context"),
        "logged_prompts": ("Logged prompts", "Number of prompt records contributing to the summary row.", "Context"),
        "psnr": ("PSNR", "Peak Signal-to-Noise Ratio relative to the BF16 reference. Higher is better.", "Higher"),
        "ssim": ("SSIM", "Structural Similarity Index relative to the BF16 reference. Higher is better.", "Higher"),
        "lpips": ("LPIPS", "Learned perceptual distance relative to the BF16 reference. Lower is better.", "Lower"),
        "background_consistency": ("Background consistency", "VBench score for background stability and scene coherence. Higher is better.", "Higher"),
        "imaging_quality": ("Imaging quality", "VBench score for overall visual quality and clarity. Higher is better.", "Higher"),
        "subject_consistency": ("Subject consistency", "VBench score for preserving the main subject through time. Higher is better.", "Higher"),
        "aesthetic_quality": ("Aesthetic quality", "VBench score for composition and visual appeal. Higher is better.", "Higher"),
        "compressed_kv_bytes_gb": ("Compressed KV size (GB)", "Estimated end-of-run KV-cache footprint after compression, in gigabytes. Lower is better.", "Lower"),
        "compressed_kv_bytes": ("Compressed KV size (bytes)", "Estimated end-of-run KV-cache footprint after compression, in raw bytes. Lower is better.", "Lower"),
        "compression_ratio": ("Compression ratio", "BF16 KV bytes divided by compressed KV bytes. Higher means stronger KV compression.", "Higher"),
        "avg_runtime_s_per_prompt": ("Avg runtime / prompt", "Average generation wall-clock time per prompt. Lower is better.", "Lower"),
        "runtime_overhead_pct_vs_bf16": ("Runtime overhead vs BF16", "Relative runtime increase or decrease compared with the BF16 baseline. Lower or negative is better.", "Lower"),
        "peak_vram_gb": ("Peak VRAM (GB)", "Maximum GPU memory observed during generation. Lower is better.", "Lower"),
        "drift_last_imaging_quality": ("Drift last imaging quality", "Last available imaging-quality point in the drift curve, used as a temporal-stability summary. Higher is better.", "Higher"),
    }
    preferred_order = [
        "videos",
        "logged_prompts",
        "psnr",
        "ssim",
        "lpips",
        "background_consistency",
        "imaging_quality",
        "subject_consistency",
        "aesthetic_quality",
        "compressed_kv_bytes_gb",
        "compressed_kv_bytes",
        "compression_ratio",
        "avg_runtime_s_per_prompt",
        "runtime_overhead_pct_vs_bf16",
        "peak_vram_gb",
        "drift_last_imaging_quality",
    ]
    rows: list[dict[str, str]] = []
    for key in preferred_order:
        if key not in metric_df.columns or key not in explanations:
            continue
        label, meaning, direction = explanations[key]
        rows.append({"Metric": label, "Meaning": meaning, "Better": direction})
    return rows


def render_overview_explainers(metric_df: pd.DataFrame) -> None:
    st.markdown("### Method glossary")
    st.caption("Brief descriptions for the methods currently visible in this overview.")
    method_rows = _method_glossary_rows(metric_df)
    if method_rows:
        st.dataframe(pd.DataFrame(method_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No method descriptions available for the current selection.")

    st.markdown("### Metric glossary")
    st.caption("Definitions for the quality, memory, runtime, and drift metrics shown above.")
    metric_rows = _metric_glossary_rows(metric_df)
    if metric_rows:
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No metric descriptions available for the current selection.")


def render_experiment_takeaways_moviegen(df: pd.DataFrame) -> None:
    methods = set(df["method"].dropna().astype(str).tolist()) if not df.empty else set()
    stable_methods = {
        "BF16",
        "RTN_INT4_REFRESH",
        "KIVI_INT4_REFRESH",
        "RTN_INT4_RECENT2",
        "QUAROT_KV_INT4_RECENT2",
    }
    if not stable_methods.issubset(methods):
        return

    st.markdown("### Experiment Takeaways")
    st.info(
        "\n".join(
            [
                "Best overall quantized method: `RTN_INT4_RECENT2`.",
                "Best high-compression option: `RTN_INT4_REFRESH`.",
                "`KIVI_INT4_REFRESH` completed, but its fidelity drop is materially larger than the RTN variants.",
                "`QUAROT_KV_INT4_RECENT2` is usable, but much slower than `RTN_INT4_RECENT2`.",
                "Quantized methods compress KV state, but in this implementation they do not beat BF16 on peak VRAM.",
                "See `EXPERIMENTS.md` for the full motivation, methodology, run registry, and per-benchmark analysis.",
            ]
        ),
        icon="📌",
    )


def render_overview(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No metrics are available for the selected run yet.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        best_psnr = df["psnr"].max(skipna=True)
        label = df.loc[df["psnr"].idxmax(), "method"] if df["psnr"].notna().any() else "-"
        st.metric(
            "Best PSNR",
            _format_psnr_value(best_psnr, is_bf16_reference=(label == "BF16")) if pd.notna(best_psnr) else "-",
            label,
            help="Peak Signal-to-Noise Ratio vs BF16 reference. Higher is better.",
        )
    with c2:
        best_lpips = df["lpips"].min(skipna=True)
        label = df.loc[df["lpips"].idxmin(), "method"] if df["lpips"].notna().any() else "-"
        st.metric(
            "Best LPIPS (lower)",
            f"{best_lpips:.4f}" if pd.notna(best_lpips) else "-",
            label,
            help="LPIPS vs BF16 reference. Lower is better.",
        )
    with c3:
        best_comp = df["compression_ratio"].max(skipna=True)
        label = df.loc[df["compression_ratio"].idxmax(), "method"] if df["compression_ratio"].notna().any() else "-"
        st.metric(
            "Best Compression",
            f"{best_comp:.2f}x" if pd.notna(best_comp) else "-",
            label,
            help="Estimated KV-cache compression ratio (BF16 bytes / compressed bytes). Higher is better.",
        )
    with c4:
        min_runtime = df["avg_runtime_s_per_prompt"].min(skipna=True)
        label = df.loc[df["avg_runtime_s_per_prompt"].idxmin(), "method"] if df["avg_runtime_s_per_prompt"].notna().any() else "-"
        st.metric(
            "Fastest / Prompt",
            _format_seconds(min_runtime) if pd.notna(min_runtime) else "-",
            label,
            help="Average generation runtime per prompt. Lower is better.",
        )

    render_experiment_takeaways_moviegen(df)

    st.markdown("### Unified method table")
    st.caption(KV_BYTES_NOTE)
    display_cols = [
        "method",
        "status",
        "source_run",
        "videos",
        "logged_prompts",
        "psnr",
        "ssim",
        "lpips",
        "background_consistency",
        "imaging_quality",
        "subject_consistency",
        "aesthetic_quality",
        "compressed_kv_bytes_gb",
        "compressed_kv_bytes",
        "compression_ratio",
        "avg_runtime_s_per_prompt",
        "runtime_overhead_pct_vs_bf16",
        "peak_vram_gb",
        "drift_last_imaging_quality",
    ]
    table_df = df[display_cols].copy()
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config=_metric_column_config(),
    )

    chart_row_1 = st.columns(2)
    with chart_row_1[0]:
        m = df.melt(id_vars=["method"], value_vars=["psnr", "ssim", "lpips"], var_name="metric", value_name="value")
        m = m.dropna(subset=["value"])
        if not m.empty:
            fig = px.bar(
                m,
                x="method",
                y="value",
                color="metric",
                barmode="group",
                title="Fidelity metrics",
                color_discrete_sequence=["#0f766e", "#1d4ed8", "#c2410c"],
            )
            fig.update_layout(height=360, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    with chart_row_1[1]:
        v = df.melt(
            id_vars=["method"],
            value_vars=["background_consistency", "imaging_quality", "subject_consistency", "aesthetic_quality"],
            var_name="metric",
            value_name="value",
        ).dropna(subset=["value"])
        if not v.empty:
            fig = px.bar(
                v,
                x="method",
                y="value",
                color="metric",
                barmode="group",
                title="VBench metrics",
                color_discrete_sequence=["#047857", "#2563eb", "#d97706", "#b45309"],
            )
            fig.update_layout(height=360, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

    drift_df = df.dropna(subset=["drift_last_imaging_quality"])
    if not drift_df.empty:
        st.markdown("### Drift summary")
        fig = px.bar(
            drift_df,
            x="method",
            y="drift_last_imaging_quality",
            color="method",
            title="Last available drift imaging_quality by method",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(height=340, xaxis_title=None, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Quality-efficiency tradeoff")
    scatter_df = df.dropna(subset=["compression_ratio", "avg_runtime_s_per_prompt"], how="any")
    if not scatter_df.empty:
        quality_col = "imaging_quality" if scatter_df["imaging_quality"].notna().any() else "psnr"
        fig = px.scatter(
            scatter_df,
            x="compression_ratio",
            y=quality_col,
            color="method",
            size="peak_vram_gb",
            hover_data=["avg_runtime_s_per_prompt", "lpips", "runtime_overhead_pct_vs_bf16"],
            title=f"Compression vs {quality_col}",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    render_overview_explainers(df)


def render_video_comparison(
    run: RunLayout,
    methods: list[str],
    prompts: dict[int, str],
    video_index: dict[str, dict[int, Path]],
) -> None:
    if not video_index:
        st.warning("No videos found for this run yet.")
        return

    available_prompt_ids: set[int] = set()
    for method_map in video_index.values():
        available_prompt_ids.update(method_map.keys())

    if not available_prompt_ids:
        st.warning("No prompt-indexed videos found.")
        return

    selected_prompt = st.selectbox("Prompt ID", sorted(available_prompt_ids), index=0)
    prompt_txt = prompts.get(int(selected_prompt), "Prompt text unavailable for this ID")
    st.markdown(f"**Prompt {selected_prompt}:** {prompt_txt}")

    default_methods = [m for m in ["BF16", "RTN_INT4", "KIVI_INT4", "QUAROT_KV_INT4"] if m in methods]
    selected_methods = st.multiselect(
        "Methods to display",
        methods,
        default=default_methods[:4] if default_methods else methods[:4],
    )

    if not selected_methods:
        st.info("Select at least one method to display videos.")
        return

    fidelity_maps = {m: load_fidelity_per_video(run, m) for m in selected_methods}

    cols = st.columns(min(4, max(1, len(selected_methods))))
    for idx, method in enumerate(selected_methods):
        col = cols[idx % len(cols)]
        with col:
            st.markdown(f"#### {method}")
            path = video_index.get(method, {}).get(int(selected_prompt))
            if path is None:
                st.warning("Missing video for this prompt")
                continue
            st.video(str(path))

            video_name = path.name
            metric = fidelity_maps.get(method, {}).get(video_name)
            if metric:
                ssim = metric.get("ssim")
                psnr = metric.get("psnr")
                lpips = metric.get("lpips")
                st.caption(
                    f"PSNR: {_format_psnr_value(psnr, is_bf16_reference=str(row.get('method', '')) == 'BF16')} | SSIM: {ssim:.4f} | LPIPS: {lpips:.4f}" if lpips is not None else f"PSNR: {_format_psnr_value(psnr, is_bf16_reference=str(row.get('method', '')) == 'BF16')} | SSIM: {ssim:.4f}"
                )


def render_prompt_analytics(run: RunLayout, methods: list[str]) -> None:
    records: list[dict[str, Any]] = []
    for method in methods:
        for row in load_generation_records(run, method):
            r = dict(row)
            r["method"] = method
            records.append(r)

    if not records:
        st.warning("No generation logs available for prompt-level analytics yet.")
        return

    df = pd.DataFrame(records)
    required = {"prompt_id", "wall_clock_runtime_s", "peak_vram_bytes", "method"}
    if not required.issubset(set(df.columns)):
        st.warning("Generation log schema is incomplete for prompt-level analytics.")
        return

    df["peak_vram_gb"] = df["peak_vram_bytes"].astype(float) / (1024**3)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            df.sort_values(["method", "prompt_id"]),
            x="prompt_id",
            y="wall_clock_runtime_s",
            color="method",
            markers=True,
            title="Per-prompt runtime",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            df,
            x="method",
            y="wall_clock_runtime_s",
            color="method",
            title="Runtime distribution by method",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### VRAM usage curves")
    trace_rows: list[dict[str, Any]] = []
    for method in methods:
        for rec in load_vram_trace_records(run, method):
            prompt_id = rec.get("prompt_id")
            if prompt_id is None:
                continue
            seed = rec.get("seed")
            for sample in rec.get("samples", []):
                allocated_bytes = sample.get("allocated_bytes")
                reserved_bytes = sample.get("reserved_bytes")
                t_s = sample.get("t_s")
                if allocated_bytes is None or reserved_bytes is None or t_s is None:
                    continue
                trace_rows.append(
                    {
                        "method": method,
                        "prompt_id": int(prompt_id),
                        "seed": int(seed) if seed is not None else None,
                        "t_s": float(t_s),
                        "allocated_gb": float(allocated_bytes) / (1024**3),
                        "reserved_gb": float(reserved_bytes) / (1024**3),
                        "bf16_kv_gb": float(sample.get("bf16_kv_bytes", 0)) / (1024**3),
                        "compressed_kv_gb": float(sample.get("compressed_kv_bytes", 0)) / (1024**3),
                    }
                )

    if trace_rows:
        trace_df = pd.DataFrame(trace_rows)
        prompt_ids = sorted(trace_df["prompt_id"].unique().tolist())
        c3, c4, c5, c6 = st.columns([1, 1, 1, 2])
        with c3:
            trace_prompt_id = st.selectbox("Trace prompt ID", prompt_ids, index=0)
        with c4:
            trace_metric = st.selectbox("Trace metric", ["allocated_gb", "reserved_gb"], index=0)
        with c5:
            kv_trace_metric = st.selectbox("KV metric", ["compressed_kv_gb", "bf16_kv_gb"], index=0)
        with c6:
            trace_methods = st.multiselect(
                "Trace methods",
                options=methods,
                default=[m for m in ["BF16", "RTN_INT4", "KIVI_INT4", "QUAROT_KV_INT4"] if m in methods] or methods,
            )
        filtered = trace_df[(trace_df["prompt_id"] == trace_prompt_id) & (trace_df["method"].isin(trace_methods))]
        if not filtered.empty:
            plot_cols = st.columns(2)
            with plot_cols[0]:
                fig = px.line(
                    filtered.sort_values(["method", "t_s"]),
                    x="t_s",
                    y=trace_metric,
                    color="method",
                    title=f"VRAM over time (prompt {trace_prompt_id})",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig.update_layout(height=380, xaxis_title="time (s)", yaxis_title=trace_metric.replace("_", " "))
                st.plotly_chart(fig, use_container_width=True)
            with plot_cols[1]:
                fig = px.line(
                    filtered.sort_values(["method", "t_s"]),
                    x="t_s",
                    y=kv_trace_metric,
                    color="method",
                    title=f"KV-cache size over time (prompt {trace_prompt_id})",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig.update_layout(height=380, xaxis_title="time (s)", yaxis_title=kv_trace_metric.replace("_", " "))
                st.plotly_chart(fig, use_container_width=True)
            peak_summary = (
                filtered.groupby("method", as_index=False)[[trace_metric, kv_trace_metric]]
                .max()
                .rename(
                    columns={
                        trace_metric: f"peak_{trace_metric}",
                        kv_trace_metric: f"peak_{kv_trace_metric}",
                    }
                )
                .sort_values(f"peak_{trace_metric}", ascending=False)
            )
            st.caption("KV-cache curve shows active cache bytes over time. Quantized methods can switch between compressed bytes and BF16-equivalent bytes.")
            st.dataframe(peak_summary, use_container_width=True, hide_index=True)
        else:
            st.info("No VRAM trace points found for selected prompt/method filters.")
    else:
        st.info("No VRAM/KV trace logs found in this run. New runs will include `logs/vram_trace_<method>.jsonl`.")

    st.markdown("### Prompt-level table")
    out_cols = [
        "method",
        "prompt_id",
        "seed",
        "wall_clock_runtime_s",
        "peak_vram_gb",
        "total_frames",
        "output_video",
        "prompt",
    ]
    existing = [c for c in out_cols if c in df.columns]
    st.dataframe(
        df[existing].sort_values(["prompt_id", "method"]),
        use_container_width=True,
        hide_index=True,
        column_config={
            "method": st.column_config.TextColumn("method", help="Quantization method name."),
            "prompt_id": st.column_config.NumberColumn("prompt_id", help="Prompt ID from prompt file.", format="%d"),
            "seed": st.column_config.NumberColumn("seed", help="Deterministic seed used for this prompt.", format="%d"),
            "wall_clock_runtime_s": st.column_config.NumberColumn(
                "wall_clock_runtime_s", help="End-to-end generation runtime for this prompt.", format="%.3f s"
            ),
            "peak_vram_gb": st.column_config.NumberColumn(
                "peak_vram_gb",
                help=f"Peak GPU memory for this prompt. {QUANT_VRAM_NOTE}",
                format="%.3f GB",
            ),
            "total_frames": st.column_config.NumberColumn("total_frames", help="Total generated frames.", format="%d"),
            "output_video": st.column_config.TextColumn("output_video", help="Relative output video path."),
            "prompt": st.column_config.TextColumn("prompt", help="Prompt text."),
        },
    )


def render_artifacts(run: RunLayout) -> None:
    st.markdown("### Run artifacts")
    st.code(str(run.root), language="text")

    metrics_files: list[Path] = []
    logs_files: list[Path] = []
    table_files: list[Path] = []

    for d in run.metric_dirs:
        metrics_files.extend(sorted(d.glob("*.json")))
    for d in run.log_dirs:
        logs_files.extend(sorted(d.glob("generation_*.jsonl")))
        logs_files.extend(sorted(d.glob("vram_trace_*.jsonl")))
    for d in run.table_dirs:
        table_files.extend(sorted(d.glob("baseline_summary.*")))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Metrics JSON**")
        if metrics_files:
            st.write("\n".join([p.name for p in metrics_files]))
        else:
            st.write("No metric JSON files found")

    with c2:
        st.markdown("**Generation logs**")
        if logs_files:
            st.write("\n".join([p.name for p in logs_files]))
        else:
            st.write("No generation logs found")

    with c3:
        st.markdown("**Summary tables**")
        if table_files:
            st.write("\n".join([p.name for p in table_files]))
        else:
            st.write("No summary table found")

    csv_path = _find_file(run.table_dirs, "baseline_summary.csv")
    if csv_path and csv_path.exists():
        st.markdown("### baseline_summary.csv")
        summary_df = pd.read_csv(csv_path)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download baseline_summary.csv",
            data=csv_path.read_bytes(),
            file_name=f"{run.label.replace('/', '_')}_baseline_summary.csv",
            mime="text/csv",
        )


def _coerce_path_value(value: Any) -> Path | None:
    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or raw.lower() == "nan":
        return None
    path = Path(raw)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _resolve_combined_video_path(row: pd.Series) -> Path | None:
    candidates: list[Path] = []

    def add_candidate(value: Any) -> None:
        path = _coerce_path_value(value)
        if path is not None and path not in candidates:
            candidates.append(path)

    add_candidate(row.get("video_path"))
    add_candidate(row.get("video_rel_path"))

    video_name = None
    for key in ("video_name", "video_rel_path", "video_path"):
        value = row.get(key)
        if isinstance(value, str) and value.strip() and value.strip().lower() != "nan":
            video_name = Path(value).name
            break

    run_root = _coerce_path_value(row.get("run_root"))
    if run_root is not None and video_name:
        for method_key in ("method", "method_display"):
            method_name = row.get(method_key)
            if isinstance(method_name, str) and method_name.strip() and method_name.strip().lower() != "nan":
                add_candidate(run_root / "videos" / method_name.strip() / video_name)
        add_candidate(run_root / "videos" / video_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def load_combined_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    numeric_cols = [
        "prompt_index",
        "seed",
        "resolution_h",
        "resolution_w",
        "num_frames",
        "fps",
        "duration_sec",
        "wall_time_sec",
        "peak_vram_bytes",
        "peak_vram_mb",
        "compression_ratio",
        "bf16_kv_bytes",
        "compressed_kv_bytes",
        "quantize_time_s",
        "dequantize_time_s",
        "total_runtime_s",
        "avg_runtime_s_per_prompt",
        "moviegen_fidelity_psnr",
        "moviegen_fidelity_ssim",
        "moviegen_fidelity_lpips",
        "moviegen_fidelity_psnr_agg",
        "moviegen_fidelity_ssim_agg",
        "moviegen_fidelity_lpips_agg",
        "moviegen_background_consistency",
        "moviegen_imaging_quality",
        "moviegen_subject_consistency",
        "moviegen_aesthetic_quality",
        "moviegen_background_consistency_agg",
        "moviegen_imaging_quality_agg",
        "moviegen_subject_consistency_agg",
        "moviegen_aesthetic_quality_agg",
        "moviegen_drift_points",
        "moviegen_drift_last_imaging_quality",
        "storyeval_background_consistency",
        "storyeval_imaging_quality",
        "storyeval_subject_consistency",
        "storyeval_aesthetic_quality",
        "storyeval_background_consistency_agg",
        "storyeval_imaging_quality_agg",
        "storyeval_subject_consistency_agg",
        "storyeval_aesthetic_quality_agg",
        "storyeval_avg_runtime_sec",
        "storyeval_avg_peak_vram_mb",
        "storyeval_max_peak_vram_mb",
        "storyeval_num_records",
        "storyeval_num_prompts",
        "storyeval_num_success",
        "storyeval_num_failed",
        "storyeval_drift_points",
        "storyeval_drift_last_imaging_quality",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_ten_second" in df.columns:
        df["is_ten_second"] = df["is_ten_second"].astype(str).str.lower().isin({"true", "1", "yes"})
    if "video_path" in df.columns or "video_rel_path" in df.columns:
        df["resolved_video_path"] = df.apply(lambda row: str(_resolve_combined_video_path(row) or ""), axis=1)
    return df


@st.cache_data(show_spinner=False)
def load_dashboard_workspace_cached(repo_root_str: str) -> dict[str, Any]:
    return load_dashboard_workspace(Path(repo_root_str))


@st.cache_data(show_spinner=False)
def load_combined_gaps(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload)


def _first_nonnull(series: pd.Series) -> Any:
    series = series.dropna()
    return None if series.empty else series.iloc[0]


def _dataset_method_defaults(methods: list[str]) -> list[str]:
    preferred_prefixes = [
        "BF16",
        "RTN_INT4",
        "QUAROT_KV_INT4",
        "AGE_TIER_INT4",
        "FLOWCACHE_SOFT_PRUNE_INT4",
        "FLOWCACHE_PRUNE_INT4",
        "PRQ_INT4",
        "QAQ_INT4",
    ]
    defaults = [
        method
        for pref in preferred_prefixes
        for method in methods
        if method == pref or method.startswith(f"{pref} [")
    ]
    if defaults:
        return defaults[: min(len(defaults), 8)]
    return methods[: min(len(methods), 8)]


def _dataset_series_defaults(series_labels: list[str]) -> list[str]:
    preferred_prefixes = [
        "BF16",
        "RTN_INT4",
        "QUAROT_KV_INT4",
        "AGE_TIER_INT4",
        "FLOWCACHE_SOFT_PRUNE_INT4",
        "FLOWCACHE_PRUNE_INT4",
        "PRQ_INT4",
        "QAQ_INT4",
    ]
    defaults = [
        label
        for pref in preferred_prefixes
        for label in series_labels
        if (
            (label.split(" / ", 1)[1] if " / " in label else label) == pref
            or (label.split(" / ", 1)[1] if " / " in label else label).startswith(f"{pref} [")
        )
    ]
    if defaults:
        return defaults[: min(len(defaults), 8)]
    return series_labels[: min(len(series_labels), 8)]


def build_dataset_trace_df(filtered_df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    if filtered_df.empty:
        return pd.DataFrame()
    required_cols = ["source_user", "run_name", "run_root", "method", "method_display"]
    if not set(required_cols).issubset(set(filtered_df.columns)):
        return pd.DataFrame()

    trace_rows: list[dict[str, Any]] = []
    grouped = filtered_df.groupby(required_cols, dropna=False)
    for keys, group in grouped:
        source_user, run_name, run_root, method, method_display = keys
        if pd.isna(run_root) or pd.isna(method):
            continue
        fallback_bf16_kv_bytes = _first_nonnull(group.get("bf16_kv_bytes", pd.Series(dtype=float)))
        fallback_compressed_kv_bytes = _first_nonnull(group.get("compressed_kv_bytes", pd.Series(dtype=float)))
        records = load_dataset_vram_trace_records(str(run_root), benchmark, str(method))
        for rec in records:
            rec_method = str(rec.get("method", method))
            if rec_method != str(method):
                continue
            prompt_id = rec.get("prompt_id")
            if prompt_id is None:
                continue
            seed = rec.get("seed")
            try:
                seed_value = int(seed) if seed is not None and not pd.isna(seed) else None
            except Exception:
                seed_value = None
            for sample in rec.get("samples", []):
                allocated_bytes = sample.get("allocated_bytes")
                reserved_bytes = sample.get("reserved_bytes")
                t_s = sample.get("t_s")
                if allocated_bytes is None or reserved_bytes is None or t_s is None:
                    continue
                sample_bf16_kv_bytes = sample.get("bf16_kv_bytes")
                if sample_bf16_kv_bytes in (None, 0, 0.0):
                    sample_bf16_kv_bytes = fallback_bf16_kv_bytes
                sample_compressed_kv_bytes = sample.get("compressed_kv_bytes")
                if sample_compressed_kv_bytes in (None, 0, 0.0):
                    sample_compressed_kv_bytes = fallback_compressed_kv_bytes
                trace_rows.append(
                    {
                        "source_user": str(source_user),
                        "run_name": str(run_name),
                        "method": str(method),
                        "method_display": str(method_display),
                        "series_label": f"{source_user} / {method_display}",
                        "prompt_id": str(prompt_id),
                        "seed": seed_value,
                        "t_s": float(t_s),
                        "allocated_gb": float(allocated_bytes) / (1024**3),
                        "reserved_gb": float(reserved_bytes) / (1024**3),
                        "bf16_kv_gb": float(sample_bf16_kv_bytes or 0) / (1024**3),
                        "compressed_kv_gb": float(sample_compressed_kv_bytes or 0) / (1024**3),
                    }
                )
    return pd.DataFrame(trace_rows)


def build_dataset_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [
        "source_user",
        "source_repo",
        "benchmark",
        "run_name",
        "run_root",
        "method_display",
        "method",
        "config_id",
        "method_family",
    ]
    for keys, group in df.groupby(group_cols, dropna=False):
        (
            source_user,
            source_repo,
            benchmark,
            run_name,
            run_root,
            method_display,
            method,
            config_id,
            method_family,
        ) = keys
        peak_vram_bytes = group["peak_vram_bytes"].max(skipna=True) if "peak_vram_bytes" in group else None
        if pd.isna(peak_vram_bytes):
            peak_vram_bytes = None
        avg_runtime = _first_nonnull(group.get("avg_runtime_s_per_prompt", pd.Series(dtype=float)))
        if avg_runtime is None and "storyeval_avg_runtime_sec" in group:
            avg_runtime = _first_nonnull(group["storyeval_avg_runtime_sec"])
        if avg_runtime is None and "wall_time_sec" in group:
            avg_runtime = group["wall_time_sec"].mean(skipna=True)
        background = _first_nonnull(group.get("storyeval_background_consistency_agg", pd.Series(dtype=float)))
        if background is None:
            background = _first_nonnull(group.get("moviegen_background_consistency_agg", pd.Series(dtype=float)))
        imaging = _first_nonnull(group.get("storyeval_imaging_quality_agg", pd.Series(dtype=float)))
        if imaging is None:
            imaging = _first_nonnull(group.get("moviegen_imaging_quality_agg", pd.Series(dtype=float)))
        subject = _first_nonnull(group.get("storyeval_subject_consistency_agg", pd.Series(dtype=float)))
        if subject is None:
            subject = _first_nonnull(group.get("moviegen_subject_consistency_agg", pd.Series(dtype=float)))
        aesthetic = _first_nonnull(group.get("storyeval_aesthetic_quality_agg", pd.Series(dtype=float)))
        if aesthetic is None:
            aesthetic = _first_nonnull(group.get("moviegen_aesthetic_quality_agg", pd.Series(dtype=float)))
        psnr = _first_nonnull(group.get("moviegen_fidelity_psnr_agg", pd.Series(dtype=float)))
        ssim = _first_nonnull(group.get("moviegen_fidelity_ssim_agg", pd.Series(dtype=float)))
        lpips = _first_nonnull(group.get("moviegen_fidelity_lpips_agg", pd.Series(dtype=float)))
        drift_last = _first_nonnull(group.get("storyeval_drift_last_imaging_quality", pd.Series(dtype=float)))
        if drift_last is None:
            drift_last = _first_nonnull(group.get("moviegen_drift_last_imaging_quality", pd.Series(dtype=float)))
        normalized_family = infer_method_family(str(method), str(method_family) if pd.notna(method_family) else None)
        rows.append(
            {
                "method": method_display,
                "status": "complete",
                "source_run": f"{source_user}:{run_name}",
                "source_user": source_user,
                "source_repo": source_repo,
                "benchmark": benchmark,
                "run_name": run_name,
                "run_root": run_root,
                "raw_method": method,
                "config_id": config_id,
                "method_family": normalized_family,
                "videos": int(len(group)),
                "logged_prompts": int(len(group)),
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "background_consistency": background,
                "imaging_quality": imaging,
                "subject_consistency": subject,
                "aesthetic_quality": aesthetic,
                "bf16_kv_bytes": _first_nonnull(group.get("bf16_kv_bytes", pd.Series(dtype=float))),
                "compressed_kv_bytes": _first_nonnull(group.get("compressed_kv_bytes", pd.Series(dtype=float))),
                "bf16_kv_bytes_gb": (
                    float(_first_nonnull(group["bf16_kv_bytes"])) / (1024**3)
                    if "bf16_kv_bytes" in group and _first_nonnull(group["bf16_kv_bytes"]) is not None
                    else None
                ),
                "compressed_kv_bytes_gb": (
                    float(_first_nonnull(group["compressed_kv_bytes"])) / (1024**3)
                    if "compressed_kv_bytes" in group and _first_nonnull(group["compressed_kv_bytes"]) is not None
                    else None
                ),
                "compression_ratio": _first_nonnull(group.get("compression_ratio", pd.Series(dtype=float))),
                "avg_runtime_s_per_prompt": avg_runtime,
                "runtime_overhead_pct_vs_bf16": None,
                "peak_vram_gb": (float(peak_vram_bytes) / (1024**3)) if peak_vram_bytes is not None else None,
                "drift_last_imaging_quality": drift_last,
            }
        )
    metric_df = pd.DataFrame(rows)
    if metric_df.empty:
        return metric_df
    for (benchmark, source_user), group in metric_df.groupby(["benchmark", "source_user"]):
        bf16_rows = group[group["raw_method"] == "BF16"]
        if bf16_rows.empty:
            continue
        bf16_runtime = bf16_rows.iloc[0]["avg_runtime_s_per_prompt"]
        if bf16_runtime is None or pd.isna(bf16_runtime) or float(bf16_runtime) == 0.0:
            continue
        idx = metric_df.index[(metric_df["benchmark"] == benchmark) & (metric_df["source_user"] == source_user)]
        metric_df.loc[idx, "runtime_overhead_pct_vs_bf16"] = (
            100.0 * (metric_df.loc[idx, "avg_runtime_s_per_prompt"] - bf16_runtime) / bf16_runtime
        )
    return metric_df.sort_values(["source_user", "method"], ignore_index=True)


def load_dataset_storyeval_drift(filtered_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if filtered_df.empty:
        return pd.DataFrame()
    for (run_root, method_display) in filtered_df[["run_root", "method_display"]].drop_duplicates().itertuples(index=False):
        path = Path(str(run_root)) / "metrics" / "drift_imaging_quality.json"
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        for point in payload.get("curve", []) or []:
            row = dict(point)
            row["method"] = method_display
            rows.append(row)
    return pd.DataFrame(rows)


def render_dataset_storyeval_overview(filtered_df: pd.DataFrame) -> None:
    metric_df = build_dataset_metric_table(filtered_df)
    if metric_df.empty:
        st.warning("No StoryEval rows match the current filters.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Methods", int(metric_df["method"].nunique()))
    c2.metric("Videos", int(filtered_df["video_name"].nunique()))
    c3.metric("Prompt records", int(len(filtered_df)))
    avg_runtime = metric_df["avg_runtime_s_per_prompt"].mean(skipna=True)
    c4.metric("Avg Runtime / Prompt", f"{float(avg_runtime):.2f}s" if pd.notna(avg_runtime) else "-")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Best Background", f"{metric_df['background_consistency'].max(skipna=True):.4f}" if metric_df["background_consistency"].notna().any() else "-")
    c6.metric("Best Imaging", f"{metric_df['imaging_quality'].max(skipna=True):.4f}" if metric_df["imaging_quality"].notna().any() else "-")
    c7.metric("Best Subject", f"{metric_df['subject_consistency'].max(skipna=True):.4f}" if metric_df["subject_consistency"].notna().any() else "-")
    c8.metric("Best Aesthetic", f"{metric_df['aesthetic_quality'].max(skipna=True):.4f}" if metric_df["aesthetic_quality"].notna().any() else "-")

    st.markdown("### Unified method table")
    st.dataframe(metric_df, use_container_width=True, hide_index=True, column_config=_metric_column_config())

    drift_df = load_dataset_storyeval_drift(filtered_df)
    if not drift_df.empty:
        st.markdown("### Long-Horizon Drift (Imaging Quality)")
        x_col = "seconds" if "seconds" in drift_df.columns else "frame_cap"
        fig = px.line(drift_df, x=x_col, y="imaging_quality", color="method", markers=True, title="StoryEval drift curve")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    melt_df = metric_df.melt(
        id_vars=["method"],
        value_vars=["background_consistency", "imaging_quality", "subject_consistency", "aesthetic_quality"],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if not melt_df.empty:
        fig = px.bar(melt_df, x="method", y="value", color="metric", barmode="group", title="StoryEval VBench metrics")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    render_overview_explainers(metric_df)


def render_dataset_moviegen_overview(filtered_df: pd.DataFrame) -> None:
    metric_df = build_dataset_metric_table(filtered_df)
    if metric_df.empty:
        st.warning("No MovieGen rows match the current filters.")
        return
    render_overview(metric_df)


def render_dataset_video_explorer(filtered_df: pd.DataFrame, benchmark: str) -> None:
    if filtered_df.empty:
        st.warning("No rows match the current dataset filters.")
        return
    prompt_col = "prompt_id"
    prompt_options = sorted([str(x) for x in filtered_df[prompt_col].dropna().unique().tolist()])
    if not prompt_options:
        st.warning("No prompt-indexed videos found.")
        return
    selected_prompt = st.selectbox("Prompt ID", prompt_options, index=0, key=f"dataset_prompt_{benchmark}")
    prompt_df = filtered_df[filtered_df[prompt_col].astype(str) == str(selected_prompt)].copy()
    prompt_text = _first_nonnull(prompt_df.get("prompt", pd.Series(dtype=str)))
    if prompt_text:
        st.markdown(f"**Prompt:** {prompt_text}")

    compare_options = [
        f"{row.source_user} / {row.method_display}"
        for row in prompt_df[["source_user", "method_display"]].drop_duplicates().itertuples(index=False)
    ]
    selected_compare = st.multiselect(
        "Methods to display",
        compare_options,
        default=compare_options[: min(len(compare_options), 4)],
        key=f"dataset_compare_{benchmark}_{selected_prompt}",
    )
    if not selected_compare:
        st.info("Select at least one method to display videos.")
        return

    cols = st.columns(min(4, max(1, len(selected_compare))))
    for idx, label in enumerate(selected_compare):
        source_user, method_display = label.split(" / ", 1)
        video_rows = prompt_df[
            (prompt_df["source_user"] == source_user) & (prompt_df["method_display"] == method_display)
        ].sort_values("seed")
        col = cols[idx % len(cols)]
        with col:
            st.markdown(f"#### {label}")
            if video_rows.empty:
                st.warning("Missing prompt for this method")
                continue
            if len(video_rows) > 1:
                seed_options = [int(x) for x in video_rows["seed"].dropna().unique().tolist()]
                selected_seed = st.selectbox(
                    "Seed",
                    seed_options,
                    index=0,
                    key=f"dataset_seed_{benchmark}_{selected_prompt}_{label}",
                )
                row = video_rows[video_rows["seed"] == selected_seed].iloc[0]
            else:
                row = video_rows.iloc[0]
            video_path = _coerce_path_value(row.get("resolved_video_path")) or _resolve_combined_video_path(row)
            if video_path and video_path.exists():
                st.video(str(video_path))
            else:
                st.warning("Video file missing")
            if benchmark == "storyeval":
                captions = [
                    f"bg: {row['storyeval_background_consistency']:.4f}" if pd.notna(row.get("storyeval_background_consistency")) else None,
                    f"img: {row['storyeval_imaging_quality']:.4f}" if pd.notna(row.get("storyeval_imaging_quality")) else None,
                    f"subj: {row['storyeval_subject_consistency']:.4f}" if pd.notna(row.get("storyeval_subject_consistency")) else None,
                    f"aes: {row['storyeval_aesthetic_quality']:.4f}" if pd.notna(row.get("storyeval_aesthetic_quality")) else None,
                ]
            else:
                captions = [
                    f"PSNR: {_format_psnr_value(row.get('moviegen_fidelity_psnr'), is_bf16_reference=str(row.get('method_display', row.get('method', ''))) == 'BF16')}" if pd.notna(row.get("moviegen_fidelity_psnr")) else None,
                    f"SSIM: {row['moviegen_fidelity_ssim']:.4f}" if pd.notna(row.get("moviegen_fidelity_ssim")) else None,
                    f"LPIPS: {row['moviegen_fidelity_lpips']:.4f}" if pd.notna(row.get("moviegen_fidelity_lpips")) else None,
                ]
            captions = [item for item in captions if item is not None]
            if captions:
                st.caption(" | ".join(captions))

    keep_cols = [
        "source_user",
        "method_display",
        "seed",
        "wall_time_sec",
        "peak_vram_mb",
        "duration_sec",
        "video_path",
    ]
    if benchmark == "storyeval":
        keep_cols.extend(
            [
                "storyeval_background_consistency",
                "storyeval_imaging_quality",
                "storyeval_subject_consistency",
                "storyeval_aesthetic_quality",
            ]
        )
    else:
        keep_cols.extend(["moviegen_fidelity_psnr", "moviegen_fidelity_ssim", "moviegen_fidelity_lpips"])
    st.markdown("### Prompt/seed records")
    prompt_table = _prepare_psnr_display_df(prompt_df[keep_cols], method_column="method_display" if "method_display" in prompt_df.columns else "method")
    st.dataframe(prompt_table, use_container_width=True, hide_index=True)


def render_dataset_prompt_analytics(filtered_df: pd.DataFrame, benchmark: str) -> None:
    if filtered_df.empty:
        st.warning("No prompt-level rows match the current filters.")
        return
    df = filtered_df.copy()
    if "prompt_index" not in df.columns or df["prompt_index"].notna().sum() == 0:
        st.warning("Prompt ordering is unavailable for the current selection.")
        return
    df["series_label"] = df["source_user"].astype(str) + " / " + df["method_display"].astype(str)

    st.markdown("### Runtime and VRAM trends")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            df.sort_values("prompt_index"),
            x="prompt_index",
            y="wall_time_sec",
            color="series_label",
            markers=True,
            title="Per-prompt runtime",
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(
            df.sort_values("prompt_index"),
            x="prompt_index",
            y="peak_vram_mb",
            color="series_label",
            markers=True,
            title="Per-prompt peak VRAM (MB)",
        )
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    trace_df = build_dataset_trace_df(filtered_df, benchmark)
    if not trace_df.empty:
        prompt_opts = sorted([str(x) for x in trace_df["prompt_id"].dropna().unique().tolist()])
        if prompt_opts:
            st.markdown("### VRAM and KV-cache traces")
            tc1, tc2, tc3, tc4 = st.columns([1, 1, 1, 2])
            with tc1:
                selected_prompt = st.selectbox("Trace prompt", prompt_opts, index=0, key=f"dataset_trace_prompt_{benchmark}")
            filtered_prompt = trace_df[trace_df["prompt_id"].astype(str) == selected_prompt]
            seed_opts = sorted([int(x) for x in filtered_prompt["seed"].dropna().unique().tolist()])
            selected_seed = None
            with tc2:
                if seed_opts:
                    selected_seed = st.selectbox(
                        "Trace seed",
                        seed_opts,
                        index=0,
                        key=f"dataset_trace_seed_{benchmark}",
                    )
                else:
                    st.caption("Trace seed: unavailable")
            with tc3:
                vram_metric = st.selectbox(
                    "Trace VRAM metric",
                    ["allocated_gb", "reserved_gb"],
                    index=0,
                    key=f"dataset_trace_vram_{benchmark}",
                )
            series_options = sorted(filtered_prompt["series_label"].dropna().unique().tolist())
            trace_series = st.multiselect(
                "Trace series",
                options=series_options,
                default=_dataset_series_defaults(series_options),
                key=f"dataset_trace_series_{benchmark}",
            )
            with tc4:
                kv_metric = st.selectbox(
                    "Trace KV metric",
                    ["compressed_kv_gb", "bf16_kv_gb"],
                    index=0,
                    key=f"dataset_trace_kv_{benchmark}",
                )
            filtered = filtered_prompt[filtered_prompt["series_label"].isin(trace_series)]
            if selected_seed is not None:
                filtered = filtered[filtered["seed"] == selected_seed]
            if not filtered.empty:
                plot_cols = st.columns(2)
                with plot_cols[0]:
                    fig = px.line(
                        filtered.sort_values(["series_label", "t_s"]),
                        x="t_s",
                        y=vram_metric,
                        color="series_label",
                        title=f"VRAM over time ({selected_prompt}" + (f", seed {selected_seed})" if selected_seed is not None else ")"),
                    )
                    fig.update_layout(height=360, xaxis_title="time (s)", yaxis_title=vram_metric.replace("_", " "))
                    st.plotly_chart(fig, use_container_width=True)
                with plot_cols[1]:
                    fig = px.line(
                        filtered.sort_values(["series_label", "t_s"]),
                        x="t_s",
                        y=kv_metric,
                        color="series_label",
                        title=f"KV-cache size over time ({selected_prompt}" + (f", seed {selected_seed})" if selected_seed is not None else ")"),
                    )
                    fig.update_layout(height=360, xaxis_title="time (s)", yaxis_title=kv_metric.replace("_", " "))
                    st.plotly_chart(fig, use_container_width=True)
                peak_summary = (
                    filtered.groupby("series_label", as_index=False)[[vram_metric, kv_metric]]
                    .max()
                    .rename(columns={vram_metric: f"peak_{vram_metric}", kv_metric: f"peak_{kv_metric}"})
                    .sort_values(f"peak_{vram_metric}", ascending=False)
                )
                st.caption("KV-cache curve shows active cache bytes over time. Quantized methods can switch between compressed bytes and BF16-equivalent bytes.")
                st.dataframe(peak_summary, use_container_width=True, hide_index=True)
            else:
                st.info("No VRAM trace points found for the selected prompt/seed/series filters.")
    else:
        st.info("No VRAM/KV trace logs found for the current dataset selection.")

    metric_cols = [
        "prompt_index",
        "prompt_id",
        "source_user",
        "method_display",
        "wall_time_sec",
        "peak_vram_mb",
    ]
    if benchmark == "storyeval":
        metric_cols.extend(["storyeval_imaging_quality", "storyeval_subject_consistency"])
    else:
        metric_cols.extend(["moviegen_fidelity_psnr", "moviegen_imaging_quality", "moviegen_subject_consistency"])
    st.markdown("### Prompt-level table")
    st.dataframe(df[metric_cols], use_container_width=True, hide_index=True)


def render_dataset_artifacts(filtered_df: pd.DataFrame, gaps_df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", int(len(filtered_df)))
    c2.metric("Runs", int(filtered_df["run_root"].nunique()) if not filtered_df.empty else 0)
    c3.metric("Videos", int(filtered_df["video_name"].nunique()) if not filtered_df.empty else 0)
    c4.metric("Open gaps", int(len(gaps_df)) if gaps_df is not None and not gaps_df.empty else 0)

    st.download_button(
        "Download combined dataset CSV",
        data=COMBINED_DATASET_PATH.read_bytes() if COMBINED_DATASET_PATH.exists() else b"",
        file_name=COMBINED_DATASET_PATH.name,
        mime="text/csv",
    )
    if COMBINED_GAPS_PATH.exists():
        st.download_button(
            "Download gap report JSON",
            data=COMBINED_GAPS_PATH.read_bytes(),
            file_name=COMBINED_GAPS_PATH.name,
            mime="application/json",
        )

    st.markdown("### Active gap report")
    if gaps_df is None or gaps_df.empty:
        st.success("No remaining dataset gaps.")
    else:
        st.dataframe(gaps_df, use_container_width=True, hide_index=True)

    st.markdown("### Selected row sample")
    if filtered_df.empty:
        st.info("No rows selected.")
    else:
        sample_cols = [
            "source_user",
            "benchmark",
            "run_name",
            "method_display",
            "prompt_id",
            "seed",
            "resolved_video_path" if "resolved_video_path" in filtered_df.columns else "video_path",
            "wall_time_sec",
            "peak_vram_mb",
        ]
        st.dataframe(filtered_df[sample_cols].head(200), use_container_width=True, hide_index=True)


def _render_html_card(title: str, body: str, css_class: str = "info-card", title_tooltip: str | None = None) -> None:
    heading = _tooltip_inline_html(title, title_tooltip) if title_tooltip else escape(title)
    st.markdown(
        f"""
        <div class="{css_class}">
            <h4>{heading}</h4>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _primary_source_note(analysis: DecisionAnalysis) -> str:
    primary = analysis.primary_source_path or "the discovered primary dataset"
    return (
        f"Primary analysis source: `{primary}`. Supporting CSV exports and registries are retained in the source catalog "
        "for provenance, but recommendations are computed from the most complete merged comparison table."
    )


def render_dataset_decision_header(analysis: DecisionAnalysis) -> None:
    _render_heading(2, "Research decision layer", _tooltip_text("research_decision_layer"))

    _render_heading(3, "Method recommendations", "Headline candidate cards for the main presentation narrative.")
    st.caption(
        f"{_tooltip_text('recommendation_focus')} Current recommendation focus: `{analysis.recommendation_focus}`. "
        f"{analysis.recommendation_focus_description}"
    )
    card_order = [
        "default_practical",
        "aggressive_compression",
        "fastest",
        "quality_first",
        "bf16_reference",
    ]
    card_columns = st.columns(len(card_order))
    for idx, key in enumerate(card_order):
        payload = analysis.recommendations.get(key)
        with card_columns[idx]:
            if not payload:
                if key == "default_practical":
                    st.info("No non-BF16 method meets the active practical caps and quality tolerances.")
                else:
                    st.info("No recommendation available under the current filters.")
                continue
            row = payload["row"]
            psnr_delta_text = _psnr_delta_tooltip_text(row)
            body = (
                f"<div class='pill'>{escape(payload['label'])}</div>"
                f"<p>{escape(payload['reason'])}</p>"
                f"{_card_metric_line('Compression', _format_metric_value(row.get('compression_ratio'), 2, 'x'), 'compression_ratio', 'Higher means stronger KV-cache compression.')}"
                f"{_card_metric_line('Runtime', _format_metric_value(row.get('avg_runtime_s_per_prompt'), 1, 's'), 'runtime', 'Average generation wall-clock time per prompt.')}"
                f"{_card_metric_line('Peak VRAM', _format_metric_value(row.get('peak_vram_gb'), 2, ' GB'), 'peak_vram', 'Maximum GPU memory observed during generation.')}"
                f"<p><strong>{_tooltip_inline_html('SSIM Δ vs BF16', 'Difference in SSIM relative to BF16 for this method.') }:</strong> {_format_metric_for_tooltip(row.get('ssim_delta_vs_bf16'))}</p>"
                f"<p><strong>{_tooltip_inline_html('LPIPS Δ vs BF16', 'Difference in LPIPS relative to BF16 for this method. More negative is better.') }:</strong> {_format_metric_for_tooltip(row.get('lpips_delta_vs_bf16'))}</p>"
                f"{_card_metric_line('Drift Δ vs BF16', _format_signed_metric(row.get('drift_last_imaging_quality_delta_vs_bf16'), 3), 'drift_delta_vs_bf16', 'Difference in drift-last imaging quality relative to BF16.')}"
            )
            if psnr_delta_text is not None:
                body += f"<p><strong>{_tooltip_inline_html('PSNR Δ vs BF16', 'Difference in PSNR relative to BF16 for this method.') }:</strong> {psnr_delta_text}</p>"
            if payload.get("caution"):
                body += f"<p><strong>{_tooltip_inline_html('Caution', _tooltip_text('caution_label'))}:</strong> {escape(payload['caution'])}</p>"
            _render_html_card(payload["method"], body, css_class="recommendation-card", title_tooltip=_describe_method(payload["method"]))

    st.caption(_primary_source_note(analysis))


def render_executive_summary_tab(analysis: DecisionAnalysis) -> None:
    method_df = analysis.method_summary
    if method_df.empty:
        st.warning("No benchmark-level methods are available for the current filters.")
        return

    summary_cols = st.columns(5)
    summary_cols[0].metric("Methods in scope", int(method_df["method"].nunique()), help=_metric_help("methods_in_scope", "Number of visible methods after filtering."))
    summary_cols[1].metric(
        "Source users",
        int(analysis.run_summary["source_user"].nunique()) if "source_user" in analysis.run_summary.columns else 0,
        help=_metric_help("source_users", "Distinct users contributing visible rows."),
    )
    summary_cols[2].metric(
        "Balanced frontier",
        len(analysis.frontier_members.get("balanced_practical", [])),
        help=_metric_help("balanced_frontier", "Methods that survive the balanced practical frontier."),
    )
    summary_cols[3].metric(
        "Best VRAM reduction",
        _format_metric_value(method_df["peak_vram_reduction_vs_bf16_pct"].max(skipna=True), 1, "%"),
        help=_metric_help("best_vram_reduction", "Largest peak-VRAM reduction relative to BF16."),
    )
    summary_cols[4].metric("Primary benchmark", analysis.benchmark, help=_metric_help("primary_benchmark", "Benchmark currently driving the recommendation layer."))
    st.caption(f"{_tooltip_text('recommendation_focus')} Recommendation focus: `{analysis.recommendation_focus}`. {analysis.recommendation_focus_description}")

    _render_heading(3, "Experiment takeaways", "Short narrative summary of the main empirical conclusions for the selected benchmark.")
    takeaway_md = "\n".join(f"- {takeaway}" for takeaway in analysis.takeaways)
    st.markdown(takeaway_md)

    _render_heading(3, "Candidate comparison", _tooltip_text("candidate_comparison"))
    st.caption("Normalized chart: higher is better on every bar. Runtime and VRAM are inverted into efficiency scores.")
    st.plotly_chart(plot_top_candidate_profile(method_df, analysis.recommendations), use_container_width=True)

    _render_heading(3, "Family-level pattern summary", _tooltip_text("family_summary"))
    st.plotly_chart(plot_family_summary(method_df), use_container_width=True)

    frontier_rows = []
    for frontier_key, methods in analysis.frontier_members.items():
        frontier_rows.append(
            {
                "frontier": FRONTIER_DEFINITIONS[frontier_key]["label"],
                "methods": ", ".join(methods) if methods else "-",
            }
        )
    _render_heading(3, "Pareto frontier members", _tooltip_text("pareto_frontier"))
    frontier_df = pd.DataFrame(frontier_rows)
    st.dataframe(
        frontier_df,
        use_container_width=True,
        hide_index=True,
        column_config=_decision_column_config(frontier_df.columns.tolist()),
    )


def render_pareto_analysis_tab(analysis: DecisionAnalysis) -> None:
    method_df = analysis.method_summary
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_compression_vs_quality(method_df, analysis.recommendations), use_container_width=True)
    with c2:
        st.plotly_chart(plot_compression_vs_drift(method_df, analysis.recommendations), use_container_width=True)

    _render_heading(3, "Frontier membership table", _tooltip_text("frontier_membership_table"))
    frontier_cols = [
        "method",
        "method_family",
        "pareto_balanced_practical",
        "pareto_quality_preserving_compression",
        "pareto_systems_efficiency",
        "pareto_quality_first",
        "dominated_by_balanced_practical_count",
        "dominated_by_balanced_practical",
        "pareto_balanced_practical_explanation",
    ]
    frontier_table = _project_sorted_table(
        method_df,
        frontier_cols,
        ["pareto_balanced_practical", "peak_vram_gb", "ssim_drop_vs_bf16", "lpips_delta_vs_bf16", "compression_ratio"],
        [False, True, True, True, False],
    )
    st.dataframe(
        frontier_table,
        use_container_width=True,
        hide_index=True,
        column_config=_decision_column_config(frontier_table.columns.tolist()),
    )


def render_constraint_rankings_tab(analysis: DecisionAnalysis) -> None:
    defaults = _calibrated_constraint_defaults(analysis.method_summary, analysis.recommendation_focus)
    st.caption(
        f"{_tooltip_text('constraint_rankings')} Defaults below are calibrated from the current top 3 non-BF16 methods "
        f"under `{analysis.recommendation_focus}`: {defaults['source_methods']}."
    )
    c1, c2, c3 = st.columns(3)
    runtime_max = c1.number_input(
        "Runtime cap (s / prompt)",
        min_value=0.0,
        value=float(defaults["runtime_max"]),
        step=1.0,
        key=f"constraint_runtime_cap_{analysis.benchmark}",
    )
    vram_max = c2.number_input(
        "Peak VRAM cap (GB)",
        min_value=0.0,
        value=float(defaults["vram_max"]),
        step=0.1,
        key=f"constraint_vram_cap_{analysis.benchmark}",
    )
    drift_drop_max = c3.number_input(
        "Drift drop cap vs BF16",
        min_value=0.0,
        value=float(defaults["drift_drop_max"]),
        step=0.005,
        format="%.3f",
        key=f"constraint_drift_cap_{analysis.benchmark}",
    )
    c4, c5, c6 = st.columns(3)
    ssim_drop_max = c4.number_input(
        "SSIM drop cap vs BF16",
        min_value=0.0,
        value=float(defaults["ssim_drop_max"]),
        step=0.005,
        format="%.3f",
        key=f"constraint_ssim_cap_{analysis.benchmark}",
    )
    psnr_min = c5.number_input(
        "PSNR minimum",
        min_value=0.0,
        value=float(defaults["psnr_min"]),
        step=0.5,
        key=f"constraint_psnr_min_{analysis.benchmark}",
    )
    lpips_increase_max = c6.number_input(
        "LPIPS increase cap vs BF16",
        min_value=0.0,
        value=float(defaults["lpips_increase_max"]),
        step=0.005,
        format="%.3f",
        key=f"constraint_lpips_cap_{analysis.benchmark}",
    )

    live_tables = _build_live_constraint_rankings(
        analysis.method_summary,
        analysis.recommendation_focus,
        {
            "runtime_max": float(runtime_max),
            "vram_max": float(vram_max),
            "ssim_drop_max": float(ssim_drop_max),
            "psnr_min": float(psnr_min),
            "lpips_increase_max": float(lpips_increase_max),
            "drift_drop_max": float(drift_drop_max),
        },
    )

    ranking_tabs = st.tabs(list(live_tables.keys()))
    for tab, (title, table) in zip(ranking_tabs, live_tables.items()):
        with tab:
            if table.empty:
                st.info("No methods satisfy this ranking under the current thresholds.")
            else:
                preview = table.head(12)
                st.dataframe(
                    preview,
                    use_container_width=True,
                    hide_index=True,
                    column_config=_decision_column_config(preview.columns.tolist()),
                )


def render_method_explorer_tab(analysis: DecisionAnalysis) -> None:
    method_df = analysis.method_summary
    ordered_methods = _order_methods(set(method_df["method"].dropna().astype(str).tolist()))
    selected_method = st.selectbox(
        "Method",
        ordered_methods,
        index=0,
        key=f"decision_method_{analysis.benchmark}",
        help="Pick one method to see its interpretation, frontier status, provenance, and derived explanation fields.",
    )
    selected_row = method_df[method_df["method"] == selected_method].iloc[0]

    metric_cols = st.columns(3)
    metric_cols[0].metric("Compression", _format_metric_value(selected_row.get("compression_ratio"), 2, "x"), help=_metric_help("compression_ratio", "KV-cache compression ratio relative to BF16."))
    metric_cols[1].metric("Peak VRAM", _format_metric_value(selected_row.get("peak_vram_gb"), 2, " GB"), help=_metric_help("peak_vram", "Maximum GPU memory observed during generation."))
    metric_cols[2].metric("Runtime / prompt", _format_metric_value(selected_row.get("avg_runtime_s_per_prompt"), 1, "s"), help=_metric_help("runtime", "Average end-to-end generation time per prompt."))

    detail_metric_cols = st.columns(4)
    detail_metric_cols[0].metric("PSNR Δ vs BF16", _psnr_delta_tooltip_text(selected_row), help="Difference in PSNR relative to BF16 for the selected method.")
    detail_metric_cols[1].metric("SSIM Δ vs BF16", _format_metric_for_tooltip(selected_row.get("ssim_delta_vs_bf16")), help="Difference in SSIM relative to BF16 for the selected method.")
    detail_metric_cols[2].metric("LPIPS Δ vs BF16", _format_metric_for_tooltip(selected_row.get("lpips_delta_vs_bf16")), help="Difference in LPIPS relative to BF16 for the selected method. More negative is better.")
    detail_metric_cols[3].metric("Drift Δ vs BF16", _format_signed_metric(selected_row.get("drift_last_imaging_quality_delta_vs_bf16"), 3), help=_metric_help("drift_delta_vs_bf16", "Difference in drift-last imaging quality relative to BF16."))

    detail_cols = st.columns(2)
    with detail_cols[0]:
        _render_heading(4, "Method interpretation", "Presentation summary of what this method is, where it fits, and what to watch out for.")
        st.markdown(f"- **Family:** {selected_row.get('method_family', '-')}")
        st.caption(_tooltip_text("method_family"))
        st.markdown(f"- **Bit-width / mode:** {selected_row.get('bit_width_label', '-')}")
        st.caption(_tooltip_text("bit_width_mode"))
        st.markdown(f"- **Quantization details:** {selected_row.get('quantization_details', '-')}")
        st.caption(_tooltip_text("quantization_details", "Short summary of the actual cache policy used by the method."))
        st.markdown(f"- **Recommended for:** {selected_row.get('recommended_for', '-') or '-'}")
        st.caption(_tooltip_text("recommended_for"))
        st.markdown(f"- **Caution:** {selected_row.get('caution_label', '-') or '-'}")
        st.caption(_tooltip_text("caution_label"))
    with detail_cols[1]:
        _render_heading(4, "Pareto position", _tooltip_text("pareto_status"))
        st.caption("Orange diamond = selected method. Green = frontier members. Gray = off-frontier methods. Black guide lines mark BF16.")
        plot_pairs = [
            ("balanced_practical", "quality_preserving_compression"),
            ("systems_efficiency", "quality_first"),
        ]
        for left_key, right_key in plot_pairs:
            c_left, c_right = st.columns(2)
            with c_left:
                st.plotly_chart(
                    plot_frontier_position(method_df, left_key, selected_method),
                    use_container_width=True,
                    key=f"frontier_position_{analysis.benchmark}_{selected_method}_{left_key}",
                )
            with c_right:
                st.plotly_chart(
                    plot_frontier_position(method_df, right_key, selected_method),
                    use_container_width=True,
                    key=f"frontier_position_{analysis.benchmark}_{selected_method}_{right_key}",
                )

    _render_heading(4, "Run-level provenance", _tooltip_text("run_provenance"))
    run_rows = analysis.run_summary[analysis.run_summary["method"] == selected_method].copy()
    run_rows = run_rows.sort_values(["source_user", "run_name"])
    st.dataframe(
        run_rows,
        use_container_width=True,
        hide_index=True,
        column_config=_decision_column_config(run_rows.columns.tolist()),
    )

    _render_heading(4, "Explainability table", _tooltip_text("explainability_table"))
    explain_df = analysis.explainability_table
    st.dataframe(
        explain_df,
        use_container_width=True,
        hide_index=True,
        column_config=_decision_column_config(explain_df.columns.tolist()),
    )


def render_systems_trace_preview(filtered_df: pd.DataFrame, benchmark: str, analysis: DecisionAnalysis) -> None:
    trace_df = build_dataset_trace_df(filtered_df, benchmark)
    if trace_df.empty:
        st.info("No trace logs are available for the current selection.")
        return

    prompt_options = sorted([str(x) for x in trace_df["prompt_id"].dropna().unique().tolist()])
    default_prompt = prompt_options[0]
    selected_prompt = st.selectbox("Trace preview prompt", prompt_options, index=0, key=f"systems_trace_prompt_{benchmark}")
    preview_df = trace_df[trace_df["prompt_id"].astype(str) == selected_prompt].copy()
    if preview_df.empty:
        st.info("No trace samples match the selected prompt.")
        return

    preferred_series = []
    for payload in analysis.recommendations.values():
        if payload["method"] in preview_df["method"].astype(str).tolist():
            matches = preview_df[preview_df["method"].astype(str) == payload["method"]]["series_label"].dropna().unique().tolist()
            preferred_series.extend(matches)
    preferred_series = preferred_series[: min(len(set(preferred_series)), 5)]
    series_options = sorted(preview_df["series_label"].dropna().unique().tolist())
    selected_series = st.multiselect(
        "Trace series",
        series_options,
        default=preferred_series or _dataset_series_defaults(series_options),
        key=f"systems_trace_series_{benchmark}",
    )
    preview_df = preview_df[preview_df["series_label"].isin(selected_series)]
    if preview_df.empty:
        st.info("No trace series selected.")
        return

    plot_cols = st.columns(2)
    with plot_cols[0]:
        fig = px.line(
            preview_df.sort_values(["series_label", "t_s"]),
            x="t_s",
            y="allocated_gb",
            color="series_label",
            title=f"Allocated VRAM over time ({selected_prompt})",
        )
        fig.update_layout(height=340, xaxis_title="time (s)", yaxis_title="allocated VRAM (GB)")
        st.plotly_chart(fig, use_container_width=True)
    with plot_cols[1]:
        fig = px.line(
            preview_df.sort_values(["series_label", "t_s"]),
            x="t_s",
            y="compressed_kv_gb",
            color="series_label",
            title=f"Compressed KV size over time ({selected_prompt})",
        )
        fig.update_layout(height=340, xaxis_title="time (s)", yaxis_title="compressed KV (GB)")
        st.plotly_chart(fig, use_container_width=True)
    st.caption(QUANT_VRAM_NOTE)


def render_systems_analysis_tab(filtered_df: pd.DataFrame, benchmark: str, analysis: DecisionAnalysis) -> None:
    _render_heading(3, "Systems trade-offs", _tooltip_text("systems_tradeoffs"))
    st.caption(
        "These plots separate nominal KV compression from realized runtime and peak-VRAM behavior. "
        "This is where methods that look efficient on paper but fail in the current integration become obvious."
    )
    top_row = st.columns(2)
    with top_row[0]:
        st.plotly_chart(plot_peak_vram_vs_quality(analysis.method_summary, analysis.recommendations), use_container_width=True)
    with top_row[1]:
        st.plotly_chart(plot_vram_vs_runtime(analysis.method_summary), use_container_width=True)

    bottom_row = st.columns(2)
    with bottom_row[0]:
        st.plotly_chart(plot_runtime_vs_quality(analysis.method_summary), use_container_width=True)
    with bottom_row[1]:
        st.plotly_chart(plot_compression_vs_peak_vram(analysis.method_summary), use_container_width=True)

    _render_heading(3, "Trace preview", _tooltip_text("trace_preview"))
    st.caption("Trace curves are shown here as a systems sanity check rather than as isolated artifacts.")
    render_systems_trace_preview(filtered_df, benchmark, analysis)


def render_quality_drift_tab(filtered_df: pd.DataFrame, analysis: DecisionAnalysis) -> None:
    method_df = analysis.method_summary.copy()
    delta_plot_df = method_df.melt(
        id_vars=["method"],
        value_vars=[
            "psnr_delta_vs_bf16",
            "drift_last_imaging_quality_delta_vs_bf16",
            "ssim_delta_vs_bf16",
            "lpips_delta_vs_bf16",
        ],
        var_name="metric",
        value_name="delta",
    ).dropna(subset=["delta"])
    if not delta_plot_df.empty:
        delta_plot_df["metric"] = delta_plot_df["metric"].map(
            {
                "psnr_delta_vs_bf16": "PSNR Δ vs BF16",
                "drift_last_imaging_quality_delta_vs_bf16": "Drift Δ vs BF16",
                "ssim_delta_vs_bf16": "SSIM Δ vs BF16",
                "lpips_delta_vs_bf16": "LPIPS Δ vs BF16",
            }
        ).fillna(delta_plot_df["metric"])
        fig = px.bar(
            delta_plot_df,
            x="method",
            y="delta",
            color="metric",
            barmode="group",
            title="BF16-relative quality and drift deltas",
        )
        fig.update_layout(height=420, xaxis_title=None, yaxis_title="Delta vs BF16")
        st.plotly_chart(fig, use_container_width=True)

    quality_cols = [
        "method",
        "method_family",
        "imaging_quality",
        "drift_last_imaging_quality",
        "psnr",
        "ssim",
        "lpips",
        "psnr_delta_vs_bf16",
        "drift_last_imaging_quality_delta_vs_bf16",
        "ssim_delta_vs_bf16",
        "lpips_delta_vs_bf16",
        "auto_explanation",
    ]
    _render_heading(3, "Quality and stability table", _tooltip_text("quality_stability_table"))
    quality_table = _project_sorted_table(
        method_df,
        quality_cols,
        ["ssim_drop_vs_bf16", "lpips_delta_vs_bf16", "psnr", "drift_last_imaging_quality_drop_vs_bf16"],
        [True, True, False, True],
    )
    quality_table = _prepare_psnr_display_df(quality_table)
    st.dataframe(
        quality_table,
        use_container_width=True,
        hide_index=True,
        column_config=_decision_column_config(quality_table.columns.tolist()),
    )

    if analysis.benchmark == "storyeval":
        drift_df = load_dataset_storyeval_drift(filtered_df)
        if not drift_df.empty:
            _render_heading(3, "StoryEval drift curves", _tooltip_text("storyeval_drift_curves"))
            x_col = "seconds" if "seconds" in drift_df.columns else "frame_cap"
            fig = px.line(drift_df, x=x_col, y="imaging_quality", color="method", markers=True, title="StoryEval drift curve")
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("StoryEval-style drift curves are only available when the benchmark filter is set to StoryEval.")


def render_raw_method_table_tab(analysis: DecisionAnalysis) -> None:
    method_df = analysis.method_summary.copy()
    raw_cols = [
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
    raw_sort_columns, raw_sort_ascending = get_recommendation_sort(analysis.recommendation_focus)
    ordered = _project_sorted_table(
        method_df,
        raw_cols,
        raw_sort_columns,
        raw_sort_ascending,
    )
    ordered = _prepare_psnr_display_df(ordered)
    _render_heading(3, "Raw method table", _tooltip_text("raw_method_table"))
    st.dataframe(
        ordered,
        use_container_width=True,
        hide_index=True,
        column_config=_decision_column_config(ordered.columns.tolist()),
    )
    st.download_button(
        "Download derived method table",
        ordered.to_csv(index=False).encode("utf-8"),
        file_name=f"{analysis.benchmark}_derived_method_table.csv",
        mime="text/csv",
    )


def render_presentation_page(filtered_df: pd.DataFrame, benchmark: str, analysis: DecisionAnalysis) -> None:
    method_options = _order_methods(set(filtered_df["method_display"].dropna().astype(str).tolist()))
    if not method_options:
        st.warning("No methods are available for the current presentation filters.")
        return

    _render_heading(3, "Presentation page", _tooltip_text("presentation_page"))
    control_cols = st.columns([2, 2, 1])
    default_methods = _resolve_presentation_methods(method_options)
    with control_cols[0]:
        focus_methods = st.multiselect(
            "Presentation methods",
            options=method_options,
            default=default_methods,
            key=f"presentation_methods_{benchmark}",
            help=_tooltip_text("presentation_methods"),
        )

    if not focus_methods:
        st.info("Select at least one presentation method.")
        return

    prompt_options = sorted([str(x) for x in filtered_df["prompt_id"].dropna().unique().tolist()])
    if not prompt_options:
        st.warning("No prompt-indexed rows are available for the current selection.")
        return

    prompt_labels: dict[str, str] = {}
    for prompt_id in prompt_options:
        prompt_rows = filtered_df[filtered_df["prompt_id"].astype(str) == prompt_id]
        prompt_text = _first_nonnull(prompt_rows.get("prompt", pd.Series(dtype=str)))
        prompt_preview = str(prompt_text).replace("\n", " ").strip() if prompt_text is not None else ""
        if len(prompt_preview) > 80:
            prompt_preview = prompt_preview[:77].rstrip() + "..."
        prompt_labels[prompt_id] = f"{prompt_id} - {prompt_preview}" if prompt_preview else prompt_id

    with control_cols[1]:
        selected_prompt = st.selectbox(
            "Input / prompt",
            options=prompt_options,
            index=0,
            format_func=lambda prompt_id: prompt_labels.get(str(prompt_id), str(prompt_id)),
            key=f"presentation_prompt_{benchmark}",
            help=_tooltip_text("presentation_input"),
        )

    prompt_df = filtered_df[filtered_df["prompt_id"].astype(str) == str(selected_prompt)].copy()
    prompt_text = _first_nonnull(prompt_df.get("prompt", pd.Series(dtype=str)))
    with control_cols[2]:
        st.metric(
            "Methods shown",
            len(focus_methods),
            help=_metric_help("presentation_methods", "Number of methods currently pinned to the presentation page."),
        )
    if prompt_text:
        st.caption(f"Prompt text: {prompt_text}")

    focus_summary = _ordered_focus_rows(analysis.method_summary, focus_methods)
    if focus_summary.empty:
        st.warning("The selected presentation methods are not available in the current benchmark slice.")
        return

    missing_methods = [method for method in focus_methods if method not in focus_summary["method"].astype(str).tolist()]
    if missing_methods:
        st.warning(f"Missing in the current benchmark slice: {', '.join(missing_methods)}")

    _render_heading(3, "Video comparison", _tooltip_text("presentation_videos"))
    st.caption("Videos are shown first for the selected prompt. Each card keeps the main systems and fidelity metrics beside the clip.")
    _render_video_sync_controls(f"presentation_{benchmark}")
    video_cols = st.columns(min(3, max(1, len(focus_methods))))
    focus_prompt_rows = prompt_df[prompt_df["method_display"].astype(str).isin(focus_methods)].copy()
    focus_prompt_rows = _ordered_focus_rows(focus_prompt_rows, focus_methods, method_column="method_display")
    for idx, method in enumerate(focus_methods):
        method_rows = focus_prompt_rows[focus_prompt_rows["method_display"].astype(str) == method].copy()
        col = video_cols[idx % len(video_cols)]
        with col:
            st.markdown(f"#### {_tooltip_inline_html(method, _describe_method(method))}", unsafe_allow_html=True)
            if method_rows.empty:
                st.warning("No video for this prompt")
                continue
            method_rows = method_rows.sort_values(["source_user", "run_name", "seed"], na_position="last")
            row = method_rows.iloc[0]
            st.caption(f"{row.get('source_user', '-')} / {row.get('run_name', '-')}")
            video_path = _coerce_path_value(row.get("resolved_video_path")) or _resolve_combined_video_path(row)
            if video_path and video_path.exists():
                st.video(str(video_path))
            else:
                st.warning("Video file missing")

            summary_row = focus_summary[focus_summary["method"].astype(str) == method]
            summary = summary_row.iloc[0] if not summary_row.empty else None
            top_metrics = st.columns(4)
            top_metrics[0].metric(
                "Compression",
                _format_metric_value(summary.get("compression_ratio"), 2, "x") if summary is not None else "-",
                help=_metric_help("compression_ratio", "KV-cache compression ratio relative to BF16."),
            )
            top_metrics[1].metric(
                "Peak VRAM",
                _format_metric_value(summary.get("peak_vram_gb"), 2, " GB") if summary is not None else "-",
                help=_metric_help("peak_vram", "Maximum GPU memory observed during generation."),
            )
            top_metrics[2].metric(
                "Runtime",
                _format_metric_value(summary.get("avg_runtime_s_per_prompt"), 1, "s") if summary is not None else "-",
                help=_metric_help("runtime", "Average end-to-end generation time per prompt."),
            )
            top_metrics[3].metric(
                "Imaging quality",
                _format_metric_value(summary.get("imaging_quality"), 3) if summary is not None else "-",
                help="Aggregate VBench imaging-quality score. Higher is better.",
            )
            quality_metrics = st.columns(4)
            quality_metrics[0].metric(
                "PSNR",
                _format_psnr_value(summary.get("psnr"), is_bf16_reference=bool(summary.get("is_bf16_reference"))) if summary is not None else "-",
                _psnr_delta_tooltip_text(summary) if summary is not None else None,
                help="Peak Signal-to-Noise Ratio relative to BF16. Higher is better.",
            )
            quality_metrics[1].metric(
                "SSIM",
                _format_metric_value(summary.get("ssim"), 4) if summary is not None else "-",
                _format_metric_for_tooltip(summary.get("ssim_delta_vs_bf16")) if summary is not None else None,
                help="Structural Similarity Index relative to BF16. Higher is better.",
            )
            quality_metrics[2].metric(
                "LPIPS",
                _format_metric_value(summary.get("lpips"), 4) if summary is not None else "-",
                _format_metric_for_tooltip(summary.get("lpips_delta_vs_bf16")) if summary is not None else None,
                help="Learned perceptual image distance relative to BF16. Lower is better.",
            )
            quality_metrics[3].metric(
                "Last drift quality",
                _format_metric_value(summary.get("drift_last_imaging_quality"), 3) if summary is not None else "-",
                _format_signed_metric(summary.get("drift_last_imaging_quality_delta_vs_bf16"), 3) if summary is not None else None,
                help="Last available imaging-quality point from the drift curve. Higher is better for temporal stability.",
            )

    _render_heading(3, "Focused comparison table", _tooltip_text("presentation_focus_table"))
    focus_cols = [
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
        "psnr_delta_vs_bf16",
        "ssim_delta_vs_bf16",
        "lpips_delta_vs_bf16",
        "drift_last_imaging_quality_delta_vs_bf16",
        "pareto_balanced_practical",
        "pareto_quality_preserving_compression",
        "pareto_systems_efficiency",
        "pareto_quality_first",
        "recommended_for",
        "caution_label",
    ]
    focus_table = focus_summary[[column for column in focus_cols if column in focus_summary.columns]]
    focus_table = _prepare_psnr_display_df(focus_table)
    st.dataframe(
        focus_table,
        use_container_width=True,
        hide_index=True,
        column_config=_decision_column_config(focus_cols),
    )

    _render_heading(3, "Core trade-off plots", _tooltip_text("presentation_graphs"))
    st.caption("Orange diamonds mark the current presentation methods directly on the same plots used elsewhere in the dashboard.")
    plot_pairs = [
        (
            plot_compression_vs_quality(analysis.method_summary, analysis.recommendations),
            "compression_ratio",
            "ssim",
        ),
        (
            plot_compression_vs_drift(analysis.method_summary, analysis.recommendations),
            "compression_ratio",
            "drift_last_imaging_quality",
        ),
        (
            plot_peak_vram_vs_quality(analysis.method_summary, analysis.recommendations),
            "peak_vram_gb",
            "ssim",
        ),
        (
            plot_vram_vs_runtime(analysis.method_summary),
            "peak_vram_gb",
            "avg_runtime_s_per_prompt",
        ),
        (
            plot_runtime_vs_quality(analysis.method_summary),
            "avg_runtime_s_per_prompt",
            "ssim",
        ),
        (
            plot_compression_vs_peak_vram(analysis.method_summary),
            "compression_ratio",
            "peak_vram_gb",
        ),
    ]
    for left_idx in range(0, len(plot_pairs), 2):
        row_cols = st.columns(2)
        for col, (fig, x_col, y_col) in zip(row_cols, plot_pairs[left_idx : left_idx + 2]):
            with col:
                st.plotly_chart(
                    _highlight_focus_methods(fig, analysis.method_summary, focus_methods, x_col, y_col),
                    use_container_width=True,
                )

    trace_df = build_dataset_trace_df(filtered_df, benchmark)
    if not trace_df.empty:
        focus_trace_df = trace_df[
            (trace_df["prompt_id"].astype(str) == str(selected_prompt))
            & (trace_df["method_display"].astype(str).isin(focus_methods))
        ].copy()
        focus_trace_df = _ordered_focus_rows(focus_trace_df, focus_methods, method_column="method_display")
        if not focus_trace_df.empty:
            _render_heading(3, "Systems traces", _tooltip_text("presentation_traces"))
            trace_cols = st.columns(2)
            with trace_cols[0]:
                fig = px.line(
                    focus_trace_df.sort_values(["method_display", "t_s"]),
                    x="t_s",
                    y="allocated_gb",
                    color="method_display",
                    markers=False,
                    title=f"Allocated VRAM over time ({selected_prompt})",
                )
                fig.update_layout(height=340, xaxis_title="time (s)", yaxis_title="allocated VRAM (GB)")
                st.plotly_chart(fig, use_container_width=True)
            with trace_cols[1]:
                fig = px.line(
                    focus_trace_df.sort_values(["method_display", "t_s"]),
                    x="t_s",
                    y="compressed_kv_gb",
                    color="method_display",
                    markers=False,
                    title=f"Compressed KV over time ({selected_prompt})",
                )
                fig.update_layout(height=340, xaxis_title="time (s)", yaxis_title="compressed KV (GB)")
                st.plotly_chart(fig, use_container_width=True)

    _render_heading(3, "Prompt-level records", _tooltip_text("presentation_prompt_records"))
    prompt_table_cols = [
        "source_user",
        "run_name",
        "method_display",
        "prompt_id",
        "seed",
        "wall_time_sec",
        "peak_vram_mb",
        "moviegen_imaging_quality",
        "moviegen_fidelity_psnr",
        "moviegen_fidelity_ssim",
        "moviegen_fidelity_lpips",
        "storyeval_imaging_quality",
        "storyeval_subject_consistency",
    ]
    prompt_table = _ordered_focus_rows(focus_prompt_rows, focus_methods, method_column="method_display")
    displayed_prompt_cols = [column for column in prompt_table_cols if column in prompt_table.columns]
    st.dataframe(
        prompt_table[displayed_prompt_cols],
        use_container_width=True,
        hide_index=True,
        column_config=_presentation_prompt_column_config(displayed_prompt_cols),
    )

    _render_heading(3, "Run provenance", _tooltip_text("presentation_provenance"))
    run_rows = _ordered_focus_rows(analysis.run_summary, focus_methods)
    st.dataframe(
        run_rows,
        use_container_width=True,
        hide_index=True,
        column_config=_decision_column_config(run_rows.columns.tolist()),
    )

    _render_heading(3, "Decision tree", "Toggle open for a presentation-ready method-selection tree based on the current benchmark.")
    show_decision_tree = st.toggle(
        "Show decision tree for method selection",
        value=False,
        key=f"presentation_decision_tree_{benchmark}",
        help="Displays the slide-ready decision tree and summary picks for the current benchmark.",
    )
    if show_decision_tree:
        render_presentation_decision_tree(analysis)


def render_notes_and_caveats_tab(filtered_df: pd.DataFrame, gaps_df: pd.DataFrame, analysis: DecisionAnalysis) -> None:
    _render_heading(3, "Honest caveats", "Limitations and interpretation guardrails for the current dataset and integration.")
    st.markdown(
        """
        - Current runs are proxy evaluations rather than definitive longer-horizon validation.
        - Lower KV bytes do not always imply lower peak VRAM, because dequantization scratch buffers and temporary allocations still matter in the current stack.
        - Some quality and runtime behavior depends on the current integration details, not just on the abstract quantization algorithm.
        - Recommendations should therefore be read as “best under the current stack and current runs,” not as universal claims.
        """
    )

    _render_heading(3, "Source catalog", _tooltip_text("source_catalog"))
    catalog_cols = ["path", "kind", "analysis_role", "selected_as_primary", "rows", "column_count", "note"]
    present_catalog_cols = [column for column in catalog_cols if column in analysis.source_catalog.columns]
    source_catalog = analysis.source_catalog[present_catalog_cols]
    st.dataframe(
        source_catalog,
        use_container_width=True,
        hide_index=True,
        column_config=_decision_column_config(source_catalog.columns.tolist()),
    )

    _render_heading(3, "Metric and method glossary", _tooltip_text("metric_glossary"))
    render_overview_explainers(analysis.method_summary)

    st.markdown("### Dataset artifacts")
    render_dataset_artifacts(filtered_df, gaps_df)


def render_dataset_dashboard(
    df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    source_catalog: pd.DataFrame,
    primary_source_path: str | None,
) -> None:
    st.sidebar.markdown("## Comparison dataset")
    benchmark_options = sorted([str(x) for x in df["benchmark"].dropna().unique().tolist()])
    selected_benchmark = st.sidebar.selectbox(
        "Benchmark",
        benchmark_options,
        index=0,
        key="dataset_benchmark",
        help=_tooltip_text("benchmark"),
    )
    filtered = df[df["benchmark"] == selected_benchmark].copy()

    source_options = sorted([str(x) for x in filtered["source_user"].dropna().unique().tolist()])
    selected_sources = st.sidebar.multiselect(
        "Source users",
        source_options,
        default=source_options,
        key="dataset_sources",
        help=_tooltip_text("source_users"),
    )
    if selected_sources:
        filtered = filtered[filtered["source_user"].isin(selected_sources)]
    else:
        filtered = filtered.iloc[0:0]

    run_options = sorted([str(x) for x in filtered["run_name"].dropna().unique().tolist()])
    selected_runs = st.sidebar.multiselect(
        "Runs",
        run_options,
        default=run_options,
        key=f"dataset_runs_{selected_benchmark}",
        help=_tooltip_text("runs"),
    )
    if selected_runs:
        filtered = filtered[filtered["run_name"].isin(selected_runs)]
    else:
        filtered = filtered.iloc[0:0]

    method_options = _order_methods(set(filtered["method_display"].dropna().astype(str).tolist()))
    selected_methods = st.sidebar.multiselect(
        "Methods",
        method_options,
        default=method_options,
        key=f"dataset_methods_{selected_benchmark}",
        help=_tooltip_text("methods"),
    )
    if selected_methods:
        filtered = filtered[filtered["method_display"].isin(selected_methods)]
    else:
        filtered = filtered.iloc[0:0]

    st.sidebar.metric("Rows", int(len(filtered)))
    st.sidebar.metric("Runs", int(filtered["run_root"].nunique()) if not filtered.empty else 0)
    st.sidebar.metric("Methods", int(filtered["method_display"].nunique()) if not filtered.empty else 0)
    if filtered.empty:
        st.warning("No rows match the current comparison filters.")
        return

    preview_analysis = build_dashboard_analysis(
        filtered,
        source_catalog=source_catalog,
        benchmark=selected_benchmark,
        recommendation_focus="Single-GPU practical",
        thresholds={
            **DEFAULT_THRESHOLDS,
            "runtime_max": None,
            "vram_max": None,
        },
        primary_source_path=primary_source_path,
    )
    if preview_analysis.method_summary.empty:
        st.warning("No method-level summaries are available for the current filters.")
        return

    method_df = preview_analysis.method_summary
    runtime_values = method_df["avg_runtime_s_per_prompt"].dropna()
    vram_values = method_df["peak_vram_gb"].dropna()
    compression_values = method_df["compression_ratio"].dropna()
    bf16_rows = method_df[method_df["method"] == "BF16"]
    bf16_runtime = float(bf16_rows.iloc[0]["avg_runtime_s_per_prompt"]) if not bf16_rows.empty and pd.notna(bf16_rows.iloc[0]["avg_runtime_s_per_prompt"]) else None
    bf16_vram = float(bf16_rows.iloc[0]["peak_vram_gb"]) if not bf16_rows.empty and pd.notna(bf16_rows.iloc[0]["peak_vram_gb"]) else None

    runtime_max_cap = float(runtime_values.max()) if not runtime_values.empty else 1.0
    vram_max_cap = float(vram_values.max()) if not vram_values.empty else 1.0
    compression_max_cap = float(compression_values.max()) if not compression_values.empty else 1.0

    st.sidebar.markdown("## Decision controls")
    recommendation_focus = st.sidebar.selectbox(
        "Recommendation focus",
        list(RECOMMENDATION_FOCUS_PRESETS.keys()),
        index=0,
        key=f"decision_focus_{selected_benchmark}",
        help=_tooltip_text("recommendation_focus"),
    )
    st.sidebar.caption(RECOMMENDATION_FOCUS_PRESETS[recommendation_focus]["description"])
    calibrated_defaults = _calibrated_constraint_defaults(method_df, recommendation_focus)
    runtime_default = min(runtime_max_cap, float(calibrated_defaults["runtime_max"]))
    vram_default = min(vram_max_cap, float(calibrated_defaults["vram_max"]))
    ssim_default = min(0.30, float(calibrated_defaults["ssim_drop_max"]))
    lpips_default = min(0.30, float(calibrated_defaults["lpips_increase_max"]))
    drift_default = min(0.20, float(calibrated_defaults["drift_drop_max"]))
    min_compression_default = min(max(compression_max_cap, 1.0), max(1.0, float(calibrated_defaults["min_compression"])))
    with st.sidebar.expander("Constraint thresholds", expanded=True):
        runtime_max = st.slider(
            "Runtime max (s / prompt)",
            min_value=0.0,
            max_value=max(runtime_max_cap, 1.0),
            value=max(runtime_default, 0.0),
            step=1.0,
            key=f"decision_runtime_{selected_benchmark}",
            help=_tooltip_text("runtime"),
        )
        vram_max = st.slider(
            "Peak VRAM max (GB)",
            min_value=0.0,
            max_value=max(vram_max_cap, 1.0),
            value=max(vram_default, 0.0),
            step=0.1,
            key=f"decision_vram_{selected_benchmark}",
            help=_tooltip_text("peak_vram"),
        )
        acceptable_ssim_drop = st.slider(
            "Acceptable SSIM drop vs BF16",
            min_value=0.0,
            max_value=0.30,
            value=ssim_default,
            step=0.005,
            key=f"decision_ssim_drop_{selected_benchmark}",
            help="Maximum allowed SSIM drop relative to BF16 for methods considered practically acceptable.",
        )
        acceptable_lpips_increase = st.slider(
            "Acceptable LPIPS increase vs BF16",
            min_value=0.0,
            max_value=0.30,
            value=lpips_default,
            step=0.005,
            key=f"decision_lpips_increase_{selected_benchmark}",
            help="Maximum allowed LPIPS increase relative to BF16 for methods considered practically acceptable.",
        )
        acceptable_drift_drop = st.slider(
            "Acceptable drift drop vs BF16",
            min_value=0.0,
            max_value=0.20,
            value=drift_default,
            step=0.005,
            key=f"decision_drift_drop_{selected_benchmark}",
            help=_tooltip_text("drift_delta_vs_bf16"),
        )
        min_compression = st.slider(
            "Minimum compression ratio",
            min_value=1.0,
            max_value=max(compression_max_cap, 1.0),
            value=min_compression_default,
            step=0.05,
            key=f"decision_min_comp_{selected_benchmark}",
            help=_tooltip_text("compression_ratio"),
        )

    analysis = build_dashboard_analysis(
        filtered,
        source_catalog=source_catalog,
        benchmark=selected_benchmark,
        recommendation_focus=recommendation_focus,
        thresholds={
            "runtime_max": runtime_max,
            "vram_max": vram_max,
            "acceptable_ssim_drop": acceptable_ssim_drop,
            "acceptable_lpips_increase": acceptable_lpips_increase,
            "acceptable_drift_drop": acceptable_drift_drop,
            "min_compression": min_compression,
        },
        primary_source_path=primary_source_path,
    )

    tab_labels = [
        "Presentation Page",
        "Overview",
        "Pareto Analysis",
        "Constraint Rankings",
        "Detailed Method Explorer",
        "Systems Analysis",
        "Quality / Drift Analysis",
        "Video Explorer",
        "Prompt Analytics",
        "Raw Method Table",
        "Notes / Caveats",
    ]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(tab_labels)
    with tab1:
        render_presentation_page(filtered, selected_benchmark, analysis)
    with tab2:
        render_executive_summary_tab(analysis)
    with tab3:
        render_pareto_analysis_tab(analysis)
    with tab4:
        render_constraint_rankings_tab(analysis)
    with tab5:
        render_method_explorer_tab(analysis)
    with tab6:
        render_systems_analysis_tab(filtered, selected_benchmark, analysis)
    with tab7:
        render_quality_drift_tab(filtered, analysis)
    with tab8:
        render_dataset_video_explorer(filtered, selected_benchmark)
    with tab9:
        render_dataset_prompt_analytics(filtered, selected_benchmark)
    with tab10:
        render_raw_method_table_tab(analysis)
    with tab11:
        benchmark_gaps = gaps_df[gaps_df["benchmark"] == selected_benchmark] if gaps_df is not None and not gaps_df.empty and "benchmark" in gaps_df.columns else gaps_df
        render_notes_and_caveats_tab(filtered, benchmark_gaps if isinstance(benchmark_gaps, pd.DataFrame) else pd.DataFrame(), analysis)


def main() -> None:
    st.set_page_config(page_title="KV-Cache Quantization Dashboard", layout="wide")
    render_header()

    workspace = load_dashboard_workspace_cached(str(REPO_ROOT))
    primary_source_path = workspace.get("primary_path")
    combined_df = load_combined_dataset(REPO_ROOT / primary_source_path) if primary_source_path else workspace.get("primary_df", pd.DataFrame())
    if not combined_df.empty:
        combined_gaps = load_combined_gaps(COMBINED_GAPS_PATH)
        render_dataset_dashboard(
            combined_df,
            combined_gaps,
            workspace.get("source_catalog", pd.DataFrame()),
            primary_source_path,
        )
        return

    runs = discover_runs(RESULTS_ROOT)
    if not runs:
        st.error("No runs found under results/. Generate data first.")
        return

    st.sidebar.markdown("## Run selection")
    benchmark_by_label = {run.label: _infer_run_benchmark(run) for run in runs}
    benchmark_options = ["all"] + sorted(set(benchmark_by_label.values()))
    selected_benchmark = st.sidebar.selectbox("Benchmark", benchmark_options, index=0, key="selected_benchmark")

    filtered_runs = runs
    if selected_benchmark != "all":
        filtered_runs = [r for r in runs if benchmark_by_label.get(r.label) == selected_benchmark]

    if not filtered_runs:
        st.warning("No runs match the selected benchmark filter.")
        return

    run_options = {run.label: run for run in filtered_runs}
    labels = list(run_options.keys())
    latest_run = max(filtered_runs, key=_extract_run_unix_ts)

    if "selected_run_label" not in st.session_state or st.session_state["selected_run_label"] not in run_options:
        st.session_state["selected_run_label"] = latest_run.label

    selected_label = st.sidebar.selectbox("Choose run", labels, key="selected_run_label")
    selected_run = run_options[selected_label]
    st.sidebar.caption(f"{len(filtered_runs)} run(s) match filter.")

    with st.sidebar.expander("Run management"):
        if _is_deletable_run(selected_run):
            if st.button("Delete selected run", type="secondary"):
                st.session_state["delete_pending_label"] = selected_run.label
                st.rerun()
        else:
            st.caption("Current run is protected and cannot be deleted from the dashboard.")

    pending_label = st.session_state.get("delete_pending_label")
    if pending_label:
        @st.dialog("Confirm Run Deletion")
        def confirm_delete_dialog() -> None:
            st.warning(f"Delete run `{pending_label}`? This action cannot be undone.")
            pending_run = run_options.get(pending_label)
            if pending_run is not None:
                st.code(str(pending_run.root), language="text")
            else:
                st.caption("Run no longer exists in the current index.")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Cancel", key="delete_run_cancel_btn"):
                    st.session_state.pop("delete_pending_label", None)
                    st.rerun()
            with c2:
                if st.button("Delete", type="primary", key="delete_run_confirm_btn"):
                    if pending_run is None:
                        st.session_state.pop("delete_pending_label", None)
                        st.cache_data.clear()
                        st.rerun()
                    ok, _ = _delete_run_directory(pending_run)
                    st.session_state.pop("delete_pending_label", None)
                    st.cache_data.clear()
                    updated_runs = discover_runs(RESULTS_ROOT)
                    if updated_runs:
                        newest = max(updated_runs, key=_extract_run_unix_ts)
                        st.session_state["selected_run_label"] = newest.label
                    else:
                        st.session_state.pop("selected_run_label", None)
                    st.rerun()

        confirm_delete_dialog()

    st.sidebar.markdown("## Run snapshot")
    st.sidebar.markdown(f"`{selected_run.label}`")
    run_meta = load_run_meta(selected_run)
    if run_meta:
        if selected_run.benchmark == "storyeval":
            st.sidebar.caption(
                f"run_id={run_meta.get('run_id', selected_run.root.name)}, created={run_meta.get('created_utc', '-')}"
            )
        else:
            st.sidebar.caption(
                f"run_name={run_meta.get('run_name', '-')}, ts={run_meta.get('run_timestamp_unix', '-')}"
            )
    st.sidebar.metric("Benchmark", selected_run.benchmark)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Video Explorer", "Prompt Analytics", "Artifacts"])

    if selected_run.benchmark == "storyeval":
        methods = list_methods(selected_run)
        selected_methods = st.sidebar.multiselect("Methods", methods, default=methods, key=f"storyeval_methods_{selected_run.label}")
        if not selected_methods:
            st.warning("Select at least one method.")
            return
        records = load_storyeval_records(selected_run)
        st.sidebar.metric("Videos found", len([r for r in records if not r.get("error")]))
        st.sidebar.metric("Prompt records", len(records))
        with tab1:
            render_storyeval_overview(selected_run, selected_methods)
        with tab2:
            render_storyeval_video_explorer(selected_run, selected_methods)
        with tab3:
            render_storyeval_prompt_analytics(selected_run, selected_methods)
        with tab4:
            render_storyeval_artifacts(selected_run)
        return

    methods = list_methods(selected_run)
    if not methods:
        st.warning("Run found, but no methods discovered yet.")
        return

    selected_methods = st.sidebar.multiselect("Methods", methods, default=methods)
    if not selected_methods:
        st.warning("Select at least one method.")
        return

    prompts_path = Path(
        st.sidebar.text_input("Prompt file", value=str(DEFAULT_PROMPTS_FILE), help="Used to show prompt text by prompt_id")
    )
    prompts = load_prompts(prompts_path)

    metric_df = build_metric_table(selected_run, selected_methods)
    video_index = build_video_index(selected_run, selected_methods)

    st.sidebar.metric("Methods", len(selected_methods))
    total_videos = int(metric_df["videos"].sum()) if not metric_df.empty else 0
    st.sidebar.metric("Videos found", total_videos)
    st.sidebar.metric("Logged prompts", int(metric_df["logged_prompts"].max()) if not metric_df.empty else 0)

    with tab1:
        render_overview(metric_df)
    with tab2:
        render_video_comparison(selected_run, selected_methods, prompts, video_index)
    with tab3:
        render_prompt_analytics(selected_run, selected_methods)
    with tab4:
        render_artifacts(selected_run)


if __name__ == "__main__":
    main()
