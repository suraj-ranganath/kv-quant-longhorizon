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
STORYEVAL_RESULTS_ROOT = RESULTS_ROOT / "benchmarks" / "storyeval"
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
def load_storyeval_vram_trace_records(run: RunLayout) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in _run_member_roots(run):
        path = root / "logs" / "vram_trace_storyeval.jsonl"
        method = _storyeval_method_name(root)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload["method"] = payload.get("method", method)
                rows.append(payload)
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
        }
        rows.append(row)
    return pd.DataFrame(rows)


def render_experiment_takeaways_storyeval(metric_df: pd.DataFrame) -> None:
    methods = set(metric_df["method"].dropna().astype(str).tolist()) if not metric_df.empty else set()
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
                "On StoryEval, the ranking matches MovieGen: `RTN_INT4_RECENT2` is the strongest completed quantized method.",
                "`RTN_INT4_REFRESH` is the simplest strong baseline when higher compression matters more than absolute quality.",
                "`KIVI_INT4_REFRESH` underperforms the RTN variants on imaging quality, subject consistency, and drift.",
                "`QUAROT_KV_INT4_RECENT2` is competitive, but its runtime cost is much higher.",
                "See `EXPERIMENTS.md` for run paths, full metrics, and method-by-method analysis.",
            ]
        ),
        icon="📌",
    )


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

    render_experiment_takeaways_storyeval(metric_df)

    if not metric_df.empty:
        st.markdown("### Unified method table")
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

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

    render_experiment_takeaways_moviegen(df)

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


def main() -> None:
    st.set_page_config(page_title="QVG Baseline Dashboard", layout="wide")
    render_header()

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
