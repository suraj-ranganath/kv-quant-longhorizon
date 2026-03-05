#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_PROMPTS_FILE = REPO_ROOT / "prompts" / "MovieGenVideoBench_extended.txt"

METHOD_ORDER = [
    "BF16",
    "RTN_INT4",
    "RTN_INT2",
    "KIVI_INT4",
    "KIVI_INT2",
    "QUAROT_KV_INT4",
    "QUAROT_KV_INT2",
]

VIDEO_RE = re.compile(r"prompt_(\d+)_seed_(\d+)\.mp4$")

QUANT_VRAM_NOTE = (
    "Why quantized methods can show higher peak VRAM in this dashboard: "
    "the current Self-Forcing hook keeps full BF16 cache tensors (`k`/`v`) allocated, "
    "then adds quantized `quant_state`, and also allocates transient dequantized/work buffers "
    "during cache read/write. So this is not yet a pure in-place KV memory replacement."
)


@dataclass
class RunLayout:
    label: str
    root: Path
    metric_dirs: list[Path]
    log_dirs: list[Path]
    video_dirs: list[Path]
    table_dirs: list[Path]


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


def _order_methods(methods: set[str]) -> list[str]:
    ordered = [m for m in METHOD_ORDER if m in methods]
    extras = sorted(m for m in methods if m not in METHOD_ORDER)
    return ordered + extras


def _metric_column_config() -> dict[str, Any]:
    return {
        "method": st.column_config.TextColumn("method", help="Quantization method name."),
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
        "psnr": st.column_config.NumberColumn(
            "psnr",
            help="Peak Signal-to-Noise Ratio vs BF16 reference. Higher is better.",
            format="%.4f",
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
            help="Estimated BF16 KV bytes divided by estimated compressed KV bytes. Higher is better for compression.",
            format="%.4f",
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
    }


@st.cache_data(show_spinner=False)
def discover_runs(results_root: Path) -> list[RunLayout]:
    runs: list[RunLayout] = []

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
            RunLayout(
                label="live/current",
                root=results_root,
                metric_dirs=[current_metrics],
                log_dirs=[current_logs],
                video_dirs=[current_videos],
                table_dirs=[current_tables],
            )
        )

    archive_root = results_root / "archive"
    if archive_root.exists():
        for d in sorted([p for p in archive_root.iterdir() if p.is_dir()], reverse=True):
            metric_dirs = [d / "metrics", d]
            log_dirs = [d / "logs", d]
            video_dirs = [d / "videos", d]
            table_dirs = [d / "tables", d]
            runs.append(
                RunLayout(
                    label=f"archive/{d.name}",
                    root=d,
                    metric_dirs=[p for p in metric_dirs if p.exists()],
                    log_dirs=[p for p in log_dirs if p.exists()],
                    video_dirs=[p for p in video_dirs if p.exists()],
                    table_dirs=[p for p in table_dirs if p.exists()],
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
def load_generation_records(run: RunLayout, method: str) -> list[dict[str, Any]]:
    path = _find_file(run.log_dirs, f"generation_{method}.jsonl")
    if path is None:
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
def load_metric_payload(run: RunLayout, prefix: str, method: str) -> dict[str, Any] | None:
    path = _find_file(run.metric_dirs, f"{prefix}_{method}.json")
    if path is None:
        return None
    return _read_json(path)


@st.cache_data(show_spinner=False)
def build_metric_table(run: RunLayout, methods: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for method in methods:
        efficiency = load_metric_payload(run, "efficiency", method) or {}
        fidelity = load_metric_payload(run, "fidelity", method) or {}
        vbench = load_metric_payload(run, "vbench", method) or {}

        records = load_generation_records(run, method)
        num_videos = 0
        for base in run.video_dirs:
            method_dir = base / method
            if method_dir.exists():
                num_videos = max(num_videos, len(list(method_dir.glob("prompt_*_seed_*.mp4"))))

        fidelity_agg = fidelity.get("aggregate", {}) if isinstance(fidelity, dict) else {}

        row: dict[str, Any] = {
            "method": method,
            "videos": num_videos,
            "logged_prompts": len(records),
            "psnr": fidelity_agg.get("psnr"),
            "ssim": fidelity_agg.get("ssim"),
            "lpips": fidelity_agg.get("lpips"),
            "background_consistency": _extract_vbench_scalar(vbench.get("background_consistency") if isinstance(vbench, dict) else None),
            "imaging_quality": _extract_vbench_scalar(vbench.get("imaging_quality") if isinstance(vbench, dict) else None),
            "subject_consistency": _extract_vbench_scalar(vbench.get("subject_consistency") if isinstance(vbench, dict) else None),
            "aesthetic_quality": _extract_vbench_scalar(vbench.get("aesthetic_quality") if isinstance(vbench, dict) else None),
            "compression_ratio": efficiency.get("compression_ratio"),
            "total_runtime_s": efficiency.get("total_runtime_s"),
            "avg_runtime_s_per_prompt": efficiency.get("avg_runtime_s_per_prompt"),
            "peak_vram_gb": (float(efficiency["peak_vram_bytes"]) / (1024**3)) if efficiency.get("peak_vram_bytes") is not None else None,
            "quantize_time_s": efficiency.get("quantize_time_s"),
            "dequantize_time_s": efficiency.get("dequantize_time_s"),
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


def render_header() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

        .stApp {
            background: radial-gradient(circle at 12% 15%, #f4f9f1 0%, #eef4ff 42%, #f9f6ef 100%);
            font-family: 'Manrope', sans-serif;
        }

        .hero {
            padding: 0.9rem 1.1rem;
            border-radius: 14px;
            border: 1px solid rgba(20, 44, 73, 0.14);
            background: linear-gradient(120deg, rgba(11, 70, 117, 0.95), rgba(24, 123, 102, 0.9));
            color: #f4f8ff;
            margin-bottom: 1rem;
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
            border: 1px solid rgba(13, 62, 97, 0.12);
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
            background: rgba(255, 255, 255, 0.8);
        }

        .mono {
            font-family: 'IBM Plex Mono', monospace;
        }
        </style>
        <div class="hero">
            <h1>QVG Baseline Replication Dashboard</h1>
            <p>Presentation workspace for Self-Forcing-Wan-1.3B runs: videos, fidelity, VBench, and systems metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
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
            f"{best_psnr:.3f}" if pd.notna(best_psnr) else "-",
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

    st.markdown("### Unified method table")
    st.info(QUANT_VRAM_NOTE, icon="ℹ️")
    display_cols = [
        "method",
        "videos",
        "logged_prompts",
        "psnr",
        "ssim",
        "lpips",
        "background_consistency",
        "imaging_quality",
        "subject_consistency",
        "aesthetic_quality",
        "compression_ratio",
        "avg_runtime_s_per_prompt",
        "runtime_overhead_pct_vs_bf16",
        "peak_vram_gb",
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
                    f"PSNR: {psnr:.3f} | SSIM: {ssim:.4f} | LPIPS: {lpips:.4f}" if lpips is not None else f"PSNR: {psnr:.3f} | SSIM: {ssim:.4f}"
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


def main() -> None:
    st.set_page_config(page_title="QVG Baseline Dashboard", layout="wide")
    render_header()

    runs = discover_runs(RESULTS_ROOT)
    if not runs:
        st.error("No runs found under results/. Generate data first.")
        return

    run_options = {run.label: run for run in runs}
    st.sidebar.markdown("## Run selection")
    selected_label = st.sidebar.selectbox("Choose run", list(run_options.keys()), index=0)
    selected_run = run_options[selected_label]

    if st.sidebar.button("Refresh run index"):
        st.cache_data.clear()
        st.rerun()

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

    st.sidebar.markdown("## Run snapshot")
    st.sidebar.markdown(f"`{selected_run.label}`")
    st.sidebar.metric("Methods", len(selected_methods))
    total_videos = int(metric_df["videos"].sum()) if not metric_df.empty else 0
    st.sidebar.metric("Videos found", total_videos)
    st.sidebar.metric("Logged prompts", int(metric_df["logged_prompts"].max()) if not metric_df.empty else 0)

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Video Explorer", "Prompt Analytics", "Artifacts"])

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
