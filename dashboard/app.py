#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_PROMPTS_FILE = REPO_ROOT / "prompts" / "MovieGenVideoBench_extended.txt"
PBENCH_RESULTS_ROOT = RESULTS_ROOT / "pbench"
PBENCH_CACHE_ROOT = REPO_ROOT / "data_cache" / "pbench"

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


@dataclass
class RunLayout:
    label: str
    root: Path
    metric_dirs: list[Path]
    log_dirs: list[Path]
    video_dirs: list[Path]
    table_dirs: list[Path]


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
    m_prefix = re.match(r"^runs/(\d+)_", run.label)
    if m_prefix:
        return int(m_prefix.group(1))
    m_suffix = re.match(r"^runs/.+_(\d+)$", run.label)
    if m_suffix:
        return int(m_suffix.group(1))
    if run.label.startswith("archive/"):
        return _parse_archive_timestamp(run.label.split("/", 1)[1])
    try:
        return int(run.root.stat().st_mtime)
    except Exception:
        return 0


def _is_deletable_run(run: RunLayout) -> bool:
    if run.root.resolve() == RESULTS_ROOT.resolve():
        return False
    return run.label.startswith("runs/") or run.label.startswith("archive/")


def _delete_run_directory(run: RunLayout) -> tuple[bool, str]:
    if not _is_deletable_run(run):
        return False, "Only runs under runs/ or archive/ can be deleted from the dashboard."
    if not run.root.exists():
        return False, f"Run path does not exist: {run.root}"
    try:
        resolved = run.root.resolve()
        if resolved == RESULTS_ROOT.resolve():
            return False, "Refusing to delete results root."
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
    }


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
                    "root": str(d),
                    "metric_dirs": [str(p) for p in metric_dirs if p.exists()],
                    "log_dirs": [str(p) for p in log_dirs if p.exists()],
                    "video_dirs": [str(p) for p in video_dirs if p.exists()],
                    "table_dirs": [str(p) for p in table_dirs if p.exists()],
                }
            )

    return runs


def discover_runs(results_root: Path) -> list[RunLayout]:
    payloads = discover_runs_payload(str(results_root))
    runs: list[RunLayout] = []
    for p in payloads:
        runs.append(
            RunLayout(
                label=p["label"],
                root=Path(p["root"]),
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
def load_vram_trace_records(run: RunLayout, method: str) -> list[dict[str, Any]]:
    path = _find_file(run.log_dirs, f"vram_trace_{method}.jsonl")
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
    meta_path = run.root / "run_meta.json"
    if meta_path.exists():
        payload = _read_json(meta_path)
        if isinstance(payload, dict):
            return payload
    return {}


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
    st.info(QUANT_VRAM_NOTE, icon="ℹ️")


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
    st.caption(KV_BYTES_NOTE)
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
        "compressed_kv_bytes_gb",
        "compressed_kv_bytes",
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
                    }
                )

    if trace_rows:
        trace_df = pd.DataFrame(trace_rows)
        prompt_ids = sorted(trace_df["prompt_id"].unique().tolist())
        c3, c4, c5 = st.columns([1, 1, 2])
        with c3:
            trace_prompt_id = st.selectbox("Trace prompt ID", prompt_ids, index=0)
        with c4:
            trace_metric = st.selectbox("Trace metric", ["allocated_gb", "reserved_gb"], index=0)
        with c5:
            trace_methods = st.multiselect(
                "Trace methods",
                options=methods,
                default=[m for m in ["BF16", "RTN_INT4", "KIVI_INT4", "QUAROT_KV_INT4"] if m in methods] or methods,
            )
        filtered = trace_df[(trace_df["prompt_id"] == trace_prompt_id) & (trace_df["method"].isin(trace_methods))]
        if not filtered.empty:
            fig = px.line(
                filtered.sort_values(["method", "t_s"]),
                x="t_s",
                y=trace_metric,
                color="method",
                title=f"Per-process VRAM over time (prompt {trace_prompt_id})",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(height=380, xaxis_title="time (s)", yaxis_title=trace_metric.replace("_", " "))
            st.plotly_chart(fig, use_container_width=True)
            peak_summary = (
                filtered.groupby("method", as_index=False)[trace_metric]
                .max()
                .rename(columns={trace_metric: f"peak_{trace_metric}"})
                .sort_values(f"peak_{trace_metric}", ascending=False)
            )
            st.dataframe(peak_summary, use_container_width=True, hide_index=True)
        else:
            st.info("No VRAM trace points found for selected prompt/method filters.")
    else:
        st.info("No VRAM trace logs found in this run. New runs will include `logs/vram_trace_<method>.jsonl`.")

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


def load_pbench_cached_samples(cache_file: str) -> list[dict[str, Any]]:
    path = Path(cache_file)
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
            except Exception:
                continue
    return rows


def discover_pbench_runs_payload(pbench_root_str: str) -> list[dict[str, Any]]:
    pbench_root = Path(pbench_root_str)
    runs: list[dict[str, Any]] = []
    if not pbench_root.exists():
        return runs
    for d in sorted([p for p in pbench_root.iterdir() if p.is_dir()], reverse=True):
        summary_path = d / "summary.json"
        config_path = d / "config.json"
        per_sample_dir = d / "per_sample"
        videos_dir = d / "videos"
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path) or {}
        config = _read_json(config_path) or {}
        runs.append(
            {
                "run_id": d.name,
                "root": str(d),
                "summary": summary,
                "config": config,
                "summary_path": str(summary_path),
                "config_path": str(config_path),
                "per_sample_dir": str(per_sample_dir),
                "videos_dir": str(videos_dir),
            }
        )
    return runs


def load_pbench_per_sample_records(per_sample_dir_str: str) -> list[dict[str, Any]]:
    per_sample_dir = Path(per_sample_dir_str)
    rows: list[dict[str, Any]] = []
    if not per_sample_dir.exists():
        return rows
    for path in sorted(per_sample_dir.glob("*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        payload["__path"] = str(path)
        rows.append(payload)
    return rows


def _pbench_sample_status(rec: dict[str, Any]) -> str:
    if rec.get("errors"):
        return "failed"
    qa = rec.get("qa_results", [])
    if not qa:
        return "no_qa"
    if any(bool(q.get("pending", False)) for q in qa):
        return "pending"
    answered = [q for q in qa if isinstance(q.get("pred_answer"), bool)]
    if not answered:
        return "pending"
    if all(bool(q.get("correct", False)) for q in answered) and len(answered) == len(qa):
        return "correct"
    if any(q.get("correct") is False for q in answered):
        return "incorrect"
    return "mixed"


def _summarize_pbench_records(records: list[dict[str, Any]], run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    completed = 0
    failed = 0
    total_runtime = 0.0
    total_questions = 0
    answered_questions = 0
    correct_answers = 0
    pending_answers = 0
    by_domain: dict[str, dict[str, int]] = {}

    for rec in records:
        errors = rec.get("errors", [])
        if errors:
            failed += 1
        else:
            completed += 1
            runtime = rec.get("runtime_s")
            if runtime is None:
                runtime = rec.get("runtime")
            if isinstance(runtime, (int, float)):
                total_runtime += float(runtime)

        meta = rec.get("meta", {}) if isinstance(rec.get("meta"), dict) else {}
        domain = meta.get("domain") or meta.get("task") or meta.get("type") or "unknown"
        dom = by_domain.setdefault(str(domain), {"total": 0, "answered": 0, "correct": 0, "pending": 0})

        for qa in rec.get("qa_results", []):
            total_questions += 1
            dom["total"] += 1
            if bool(qa.get("pending", False)):
                pending_answers += 1
                dom["pending"] += 1
                continue
            pred = qa.get("pred_answer")
            if isinstance(pred, bool):
                answered_questions += 1
                dom["answered"] += 1
                if bool(qa.get("correct", False)):
                    correct_answers += 1
                    dom["correct"] += 1

    accuracy_overall = (correct_answers / answered_questions) if answered_questions > 0 else 0.0
    accuracy_by_domain = {
        k: ((v["correct"] / v["answered"]) if v["answered"] > 0 else 0.0) for k, v in by_domain.items()
    }
    avg_runtime_s = total_runtime / completed if completed > 0 else None

    return {
        "run_id": run_id,
        "method": "self_forcing_wan_1.3b",
        "config": config,
        "counts": {
            "records": len(records),
            "completed": completed,
            "failed": failed,
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "pending_answers": pending_answers,
            "correct_answers": correct_answers,
        },
        "accuracy_overall": accuracy_overall,
        "accuracy_by_domain": accuracy_by_domain,
        "avg_runtime_s": avg_runtime_s,
    }


def render_pbench_tab() -> None:
    st.markdown("### PBench Dataset Browser")
    cache_files = sorted(PBENCH_CACHE_ROOT.glob("normalized_*.jsonl"), reverse=True)
    if not cache_files:
        st.info("No normalized PBench cache found. Run `scripts/run_pbench.py` to populate `data_cache/pbench`.")
    else:
        cache_map = {p.name: p for p in cache_files}
        selected_cache = st.selectbox("Cached split", list(cache_map.keys()), index=0, key="pbench_cache_file")
        samples = load_pbench_cached_samples(str(cache_map[selected_cache]))
        if samples:
            sample_ids = [str(s.get("sample_id", f"sample_{i}")) for i, s in enumerate(samples)]
            selected_id = st.selectbox("Sample ID", sample_ids, index=0, key="pbench_sample_id")
            selected = next((s for s in samples if str(s.get("sample_id")) == selected_id), samples[0])

            st.markdown(f"**Prompt**: {selected.get('prompt', '')}")
            cond_image_path = selected.get("cond_image_path")
            if isinstance(cond_image_path, str) and Path(cond_image_path).exists():
                st.image(cond_image_path, caption="Conditioning image", use_container_width=True)
            qa_pairs = selected.get("qa_pairs", [])
            if qa_pairs:
                st.dataframe(pd.DataFrame(qa_pairs), use_container_width=True, hide_index=True)
            else:
                st.caption("No QA pairs found for this sample.")
        else:
            st.caption("Selected cache is empty.")

    st.markdown("### PBench Runs")
    runs = discover_pbench_runs_payload(str(PBENCH_RESULTS_ROOT))
    if not runs:
        st.info("No PBench run summaries found under `results/pbench/*/summary.json`.")
        return

    run_rows: list[dict[str, Any]] = []
    for r in runs:
        summary = r.get("summary", {})
        config = r.get("config", {})
        counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
        run_rows.append(
            {
                "run_id": r["run_id"],
                "evaluator": config.get("evaluator"),
                "split": config.get("split"),
                "max_samples": config.get("max_samples"),
                "completed": counts.get("completed"),
                "failed": counts.get("failed"),
                "accuracy_overall": summary.get("accuracy_overall"),
                "avg_runtime_s": summary.get("avg_runtime_s"),
            }
        )
    run_df = pd.DataFrame(run_rows)
    st.dataframe(run_df, use_container_width=True, hide_index=True)

    run_map = {r["run_id"]: r for r in runs}
    selected_run_id = st.selectbox("Run ID", list(run_map.keys()), index=0, key="pbench_run_id")
    selected_run = run_map[selected_run_id]
    summary = selected_run.get("summary", {})
    config = selected_run.get("config", {})
    counts = summary.get("counts", {}) if isinstance(summary, dict) else {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accuracy", f"{float(summary.get('accuracy_overall', 0.0)):.3f}")
    with c2:
        st.metric("Completed", int(counts.get("completed", 0)))
    with c3:
        st.metric("Failed", int(counts.get("failed", 0)))
    with c4:
        pending = int(counts.get("pending_answers", 0))
        st.metric("Pending Answers", pending)

    st.caption(f"evaluator={config.get('evaluator')} split={config.get('split')} max_samples={config.get('max_samples')}")

    records = load_pbench_per_sample_records(str(selected_run["per_sample_dir"]))
    if not records:
        st.info("No per-sample records found for this run.")
        return

    filter_opts = ["all", "correct", "incorrect", "pending", "failed"]
    selected_filter = st.selectbox("Sample filter", filter_opts, index=0, key="pbench_filter")
    filtered_records = [r for r in records if selected_filter == "all" or _pbench_sample_status(r) == selected_filter]
    if not filtered_records:
        st.info("No samples match this filter.")
        return

    selector_labels = [
        f"{r.get('sample_id')} | seed={r.get('seed')} | status={_pbench_sample_status(r)}" for r in filtered_records
    ]
    picked_label = st.selectbox("Per-sample record", selector_labels, index=0, key="pbench_record_label")
    picked_idx = selector_labels.index(picked_label)
    rec = filtered_records[picked_idx]

    st.markdown(f"**Sample**: `{rec.get('sample_id')}` | **Seed**: `{rec.get('seed')}`")
    st.markdown(f"**Prompt**: {rec.get('prompt', '')}")
    cond_image_path = rec.get("cond_image_path")
    if isinstance(cond_image_path, str) and Path(cond_image_path).exists():
        st.image(cond_image_path, caption="Conditioning image", use_container_width=True)

    video_path = rec.get("generated_video_path")
    if isinstance(video_path, str) and Path(video_path).exists():
        st.video(video_path)
    elif isinstance(video_path, str):
        alt = Path(selected_run["root"]) / "videos" / Path(video_path).name
        if alt.exists():
            st.video(str(alt))
        else:
            st.caption("Video file not found.")

    qa_rows = rec.get("qa_results", [])
    if qa_rows:
        st.dataframe(pd.DataFrame(qa_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No QA rows in this sample record.")

    if config.get("evaluator") == "manual" and qa_rows:
        st.markdown("#### Manual QA Update")
        with st.form(f"manual_update_{selected_run_id}_{rec.get('sample_id')}_{rec.get('seed')}"):
            updates: dict[int, str] = {}
            for idx, qa in enumerate(qa_rows):
                q = str(qa.get("question", f"question_{idx}"))
                updates[idx] = st.selectbox(
                    f"Q{idx + 1}: {q}",
                    options=["Keep", "True", "False", "Unset (Pending)"],
                    index=0,
                    key=f"manual_choice_{selected_run_id}_{rec.get('sample_id')}_{rec.get('seed')}_{idx}",
                )
            submitted = st.form_submit_button("Save Manual Predictions")

        if submitted:
            changed = False
            for idx, choice in updates.items():
                if choice == "Keep":
                    continue
                qa = qa_rows[idx]
                if choice == "Unset (Pending)":
                    qa["pred_answer"] = None
                    qa["correct"] = None
                    qa["pending"] = True
                else:
                    pred = choice == "True"
                    qa["pred_answer"] = pred
                    qa["correct"] = pred == bool(qa.get("gt_answer", False))
                    qa["pending"] = False
                changed = True

            if changed:
                rec_path = Path(rec["__path"])
                rec_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
                updated_records = load_pbench_per_sample_records(str(selected_run["per_sample_dir"]))
                updated_summary = _summarize_pbench_records(updated_records, selected_run_id, config)
                Path(selected_run["summary_path"]).write_text(
                    json.dumps(updated_summary, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                st.cache_data.clear()
                st.success("Manual predictions saved and summary updated.")
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="QVG Baseline Dashboard", layout="wide")
    render_header()

    runs = discover_runs(RESULTS_ROOT)
    if not runs:
        st.error("No runs found under results/. Generate data first.")
        return

    run_options = {run.label: run for run in runs}
    st.sidebar.markdown("## Run selection")
    labels = list(run_options.keys())
    latest_run = max(runs, key=_extract_run_unix_ts)

    if "selected_run_label" not in st.session_state or st.session_state["selected_run_label"] not in run_options:
        st.session_state["selected_run_label"] = latest_run.label

    selected_label = st.sidebar.selectbox("Choose run", labels, key="selected_run_label")
    selected_run = run_options[selected_label]

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

    methods = list_methods(selected_run)
    selected_methods: list[str] = []
    if methods:
        selected_methods = st.sidebar.multiselect("Methods", methods, default=methods)
    else:
        st.sidebar.caption("No baseline methods discovered for the selected run.")

    prompts_path = Path(
        st.sidebar.text_input("Prompt file", value=str(DEFAULT_PROMPTS_FILE), help="Used to show prompt text by prompt_id")
    )
    prompts = load_prompts(prompts_path)

    metric_df = build_metric_table(selected_run, selected_methods) if selected_methods else pd.DataFrame()
    video_index = build_video_index(selected_run, selected_methods) if selected_methods else {}

    st.sidebar.markdown("## Run snapshot")
    st.sidebar.markdown(f"`{selected_run.label}`")
    run_meta = load_run_meta(selected_run)
    if run_meta:
        st.sidebar.caption(
            f"run_name={run_meta.get('run_name', '-')}, ts={run_meta.get('run_timestamp_unix', '-')}"
        )
    st.sidebar.metric("Methods", len(selected_methods))
    total_videos = int(metric_df["videos"].sum()) if not metric_df.empty and "videos" in metric_df.columns else 0
    st.sidebar.metric("Videos found", total_videos)
    st.sidebar.metric(
        "Logged prompts",
        int(metric_df["logged_prompts"].max()) if not metric_df.empty and "logged_prompts" in metric_df.columns else 0,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Video Explorer", "Prompt Analytics", "Artifacts", "Embodied (PBench)"]
    )

    with tab1:
        if selected_methods:
            render_overview(metric_df)
        else:
            st.info("No baseline methods selected for this run.")
    with tab2:
        if selected_methods:
            render_video_comparison(selected_run, selected_methods, prompts, video_index)
        else:
            st.info("No videos available for baseline comparison in this run.")
    with tab3:
        if selected_methods:
            render_prompt_analytics(selected_run, selected_methods)
        else:
            st.info("No prompt-level baseline analytics available for this run.")
    with tab4:
        render_artifacts(selected_run)
    with tab5:
        render_pbench_tab()


if __name__ == "__main__":
    main()
