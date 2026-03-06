#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
STORYEVAL_ROOT = REPO_ROOT / "results" / "benchmarks" / "storyeval"
VIDEO_NAME_RE = re.compile(r"^(?P<prompt_id>.+)_seed(?P<seed>\d+)\.mp4$")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iso_from_epoch(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "-"


@st.cache_data(show_spinner=False)
def discover_storyeval_runs(root_str: str) -> list[dict[str, Any]]:
    root = Path(root_str)
    if not root.exists():
        return []

    runs: list[dict[str, Any]] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
        summary_json = _read_json(d / "summary" / "summary.json") or {}
        vbench_json = _read_json(d / "metrics" / "vbench.json") or {}
        vbench_agg = vbench_json.get("aggregate", {}) if isinstance(vbench_json.get("aggregate"), dict) else {}
        per_prompt_files = list((d / "per_prompt").glob("*.json")) if (d / "per_prompt").exists() else []
        runs.append(
            {
                "run_id": d.name,
                "run_dir": str(d),
                "updated_utc": _iso_from_epoch(d.stat().st_mtime),
                "num_records": int(summary_json.get("num_records", len(per_prompt_files))),
                "num_prompts": int(summary_json.get("num_prompts", 0)),
                "num_success": int(summary_json.get("num_success", 0)),
                "num_failed": int(summary_json.get("num_failed", 0)),
                "avg_runtime_sec": summary_json.get("avg_runtime_sec"),
                "background_consistency": vbench_agg.get("background_consistency"),
                "imaging_quality": vbench_agg.get("imaging_quality"),
                "subject_consistency": vbench_agg.get("subject_consistency"),
                "aesthetic_quality": vbench_agg.get("aesthetic_quality"),
            }
        )
    return runs


@st.cache_data(show_spinner=False)
def load_run_payload(run_dir_str: str) -> dict[str, Any]:
    run_dir = Path(run_dir_str)
    per_prompt_records: list[dict[str, Any]] = []
    per_prompt_dir = run_dir / "per_prompt"
    if per_prompt_dir.exists():
        for p in sorted(per_prompt_dir.glob("*.json")):
            try:
                per_prompt_records.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue

    vbench = _read_json(run_dir / "metrics" / "vbench.json") or {}
    drift = _read_json(run_dir / "metrics" / "drift_imaging_quality.json") or {}
    summary = _read_json(run_dir / "summary" / "summary.json") or {}
    cfg = _read_json(run_dir / "summary" / "config.json") or {}

    return {
        "run_dir": str(run_dir),
        "records": per_prompt_records,
        "vbench": vbench,
        "drift": drift,
        "summary": summary,
        "config": cfg,
        "drift_plot_path": str(run_dir / "plots" / "drift_imaging_quality.png"),
    }


def _parse_video_name(path_str: str) -> tuple[str | None, int | None]:
    name = Path(path_str).name
    m = VIDEO_NAME_RE.match(name)
    if not m:
        return None, None
    return m.group("prompt_id"), int(m.group("seed"))


def _build_per_video_metric_map(vbench: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    per_video = vbench.get("per_video", {})
    if isinstance(per_video, dict):
        for video_name, rec in per_video.items():
            if isinstance(rec, dict):
                out[video_name] = rec
    return out


def render() -> None:
    st.set_page_config(page_title="StoryEval", layout="wide")
    st.title("StoryEval Benchmark (T2V, default 10s)")
    st.caption("Long-horizon benchmark runs under `results/benchmarks/storyeval/*`.")

    if not STORYEVAL_ROOT.exists():
        st.info(f"No StoryEval runs found. Expected root: `{STORYEVAL_ROOT}`")
        return

    runs = discover_storyeval_runs(str(STORYEVAL_ROOT))
    if not runs:
        st.info(f"No StoryEval runs found in `{STORYEVAL_ROOT}`")
        return

    st.subheader("Runs Browser")
    runs_df = pd.DataFrame(runs)
    st.dataframe(
        runs_df[
            [
                "run_id",
                "updated_utc",
                "num_prompts",
                "num_records",
                "num_success",
                "num_failed",
                "avg_runtime_sec",
                "background_consistency",
                "imaging_quality",
                "subject_consistency",
                "aesthetic_quality",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    run_ids = [r["run_id"] for r in runs]
    selected_run = st.selectbox("Choose StoryEval run", run_ids, index=0)
    run_meta = next(r for r in runs if r["run_id"] == selected_run)
    payload = load_run_payload(run_meta["run_dir"])

    summary = payload["summary"]
    vbench = payload["vbench"]
    drift = payload["drift"]
    records = payload["records"]
    config = payload["config"]
    per_video_metrics = _build_per_video_metric_map(vbench)

    st.subheader("Run Detail")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prompts", int(summary.get("num_prompts", 0)))
    c2.metric("Success", int(summary.get("num_success", 0)))
    c3.metric("Failed", int(summary.get("num_failed", 0)))
    c4.metric("Avg Runtime (s)", f"{float(summary.get('avg_runtime_sec', 0.0)):.2f}" if summary.get("avg_runtime_sec") is not None else "-")

    c5, c6, c7, c8 = st.columns(4)
    agg = vbench.get("aggregate", {}) if isinstance(vbench.get("aggregate"), dict) else {}
    c5.metric("VBench Background", f"{float(agg.get('background_consistency', 0.0)):.4f}" if agg.get("background_consistency") is not None else "-")
    c6.metric("VBench Imaging", f"{float(agg.get('imaging_quality', 0.0)):.4f}" if agg.get("imaging_quality") is not None else "-")
    c7.metric("VBench Subject", f"{float(agg.get('subject_consistency', 0.0)):.4f}" if agg.get("subject_consistency") is not None else "-")
    c8.metric("VBench Aesthetic", f"{float(agg.get('aesthetic_quality', 0.0)):.4f}" if agg.get("aesthetic_quality") is not None else "-")

    with st.expander("Run Config", expanded=False):
        st.json(config)

    st.subheader("Drift Curve")
    drift_curve = drift.get("curve", []) if isinstance(drift.get("curve"), list) else []
    drift_plot_path = Path(payload["drift_plot_path"])
    if drift_plot_path.exists():
        st.image(str(drift_plot_path), caption="drift_imaging_quality.png", use_container_width=True)
    elif drift_curve:
        df = pd.DataFrame(drift_curve)
        if "seconds" in df.columns and "imaging_quality" in df.columns:
            fig = px.line(df, x="seconds", y="imaging_quality", markers=True, title="Imaging Quality vs Prefix Duration")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drift artifacts found for this run.")

    st.subheader("Prompt Explorer")
    if not records:
        st.warning("No per_prompt records found for this run.")
        return

    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        pid = rec.get("prompt_id")
        if not isinstance(pid, str):
            video_rel = rec.get("generated_video_path", "")
            pid, _ = _parse_video_name(video_rel if isinstance(video_rel, str) else "")
        if not isinstance(pid, str):
            continue
        by_prompt.setdefault(pid, []).append(rec)

    prompt_ids = sorted(by_prompt.keys())
    selected_prompt_id = st.selectbox("Prompt ID", prompt_ids, index=0)
    prompt_records = sorted(by_prompt[selected_prompt_id], key=lambda r: int(r.get("seed", 0)))
    prompt_text = str(prompt_records[0].get("prompt", ""))
    st.markdown("**Prompt Text**")
    st.write(prompt_text)

    video_rows: list[dict[str, Any]] = []
    for rec in prompt_records:
        video_rel = rec.get("generated_video_path")
        if not isinstance(video_rel, str):
            continue
        video_abs = (REPO_ROOT / video_rel).resolve()
        if not video_abs.exists():
            continue
        st.markdown(f"**Seed {rec.get('seed')}**")
        st.video(str(video_abs))
        metrics = per_video_metrics.get(video_abs.name, {})
        video_rows.append(
            {
                "video": video_abs.name,
                "seed": rec.get("seed"),
                "runtime_sec": rec.get("wall_time_sec"),
                "peak_vram_mb": rec.get("peak_vram_mb"),
                "background_consistency": metrics.get("background_consistency"),
                "imaging_quality": metrics.get("imaging_quality"),
                "subject_consistency": metrics.get("subject_consistency"),
                "aesthetic_quality": metrics.get("aesthetic_quality"),
            }
        )

    if video_rows:
        st.markdown("**Per-Video Metrics**")
        st.dataframe(pd.DataFrame(video_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render()
