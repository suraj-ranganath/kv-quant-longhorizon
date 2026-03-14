#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / 'results'
DOC_ASSETS = REPO_ROOT / 'docs' / 'experiment_assets'

METHOD_ORDER = [
    'BF16',
    'RTN_INT4',
    'RTN_INT2',
    'KIVI_INT4',
    'KIVI_INT2',
    'QUAROT_KV_INT4',
    'QUAROT_KV_INT2',
    'RTN_INT4_REFRESH',
    'KIVI_INT4_REFRESH',
    'QUAROT_KV_INT4_REFRESH',
    'RTN_K2_V4',
    'KIVI_K2_V4',
    'RTN_INT4_RECENT2',
    'QUAROT_KV_INT4_RECENT2',
]

MOVIEGEN_SOURCES = {
    'BF16': {'run': 'results/runs/1773038789_newideas10s_10prompts', 'status': 'complete', 'note': 'BF16 source chosen from the new-method run because it includes drift metrics and full traces.'},
    'RTN_INT4': {'run': 'results/runs/1772751420_baseline10s_10prompts_v3', 'status': 'complete'},
    'RTN_INT2': {'run': 'results/runs/1772751420_baseline10s_10prompts_v3', 'status': 'complete'},
    'KIVI_INT4': {'run': 'results/runs/1772751420_baseline10s_10prompts_v3', 'status': 'complete'},
    'KIVI_INT2': {'run': 'results/runs/1772751420_baseline10s_10prompts_v3', 'status': 'complete'},
    'QUAROT_KV_INT4': {'run': 'results/runs/1772751420_baseline10s_10prompts_v3', 'status': 'complete'},
    'QUAROT_KV_INT2': {'run': 'results/runs/1772751420_baseline10s_10prompts_v3', 'status': 'complete'},
    'RTN_INT4_REFRESH': {'run': 'results/runs/1773038789_newideas10s_10prompts', 'status': 'complete'},
    'KIVI_INT4_REFRESH': {'run': 'results/runs/1773038789_newideas10s_10prompts', 'status': 'complete'},
    'QUAROT_KV_INT4_REFRESH': {'run': 'results/runs/1773037963_newideas10s_10prompts', 'status': 'partial_failed', 'note': 'Failed after 2/10 prompts with CUDA OOM during refresh-only writeback.'},
    'RTN_K2_V4': {'run': 'results/runs/1773038789_newideas10s_10prompts', 'status': 'partial_failed', 'note': 'Failed after 5/10 prompts with CUDA OOM.'},
    'KIVI_K2_V4': {'run': 'results/runs/1773038789_newideas10s_10prompts', 'status': 'partial_failed', 'note': 'Failed after 3/10 prompts with CUDA OOM.'},
    'RTN_INT4_RECENT2': {'run': 'results/runs/1773038789_newideas10s_10prompts', 'status': 'complete'},
    'QUAROT_KV_INT4_RECENT2': {'run': 'results/runs/1773038789_newideas10s_10prompts', 'status': 'complete'},
}

STORYEVAL_SOURCES = {
    'BF16': {'run': 'results/benchmarks/storyeval/storyeval_BF16_10prompts_10s_1773038789', 'status': 'complete', 'note': 'BF16 source chosen from the new-method StoryEval run for consistency with drift traces.'},
    'RTN_INT4': {'run': 'results/benchmarks/storyeval/storyeval_RTN_INT4_10prompts_10s_1772778648', 'status': 'complete'},
    'RTN_INT2': {'run': 'results/benchmarks/storyeval/storyeval_RTN_INT2_10prompts_10s_1772778648', 'status': 'complete'},
    'KIVI_INT4': {'run': 'results/benchmarks/storyeval/storyeval_KIVI_INT4_10prompts_10s_1772778648', 'status': 'complete'},
    'KIVI_INT2': {'run': 'results/benchmarks/storyeval/storyeval_KIVI_INT2_10prompts_10s_1772778648', 'status': 'complete'},
    'QUAROT_KV_INT4': {'run': 'results/benchmarks/storyeval/storyeval_QUAROT_KV_INT4_10prompts_10s_1772778648', 'status': 'complete'},
    'QUAROT_KV_INT2': {'run': 'results/benchmarks/storyeval/storyeval_QUAROT_KV_INT2_10prompts_10s_1772778648', 'status': 'complete'},
    'RTN_INT4_REFRESH': {'run': 'results/benchmarks/storyeval/storyeval_RTN_INT4_REFRESH_10prompts_10s_1773038789', 'status': 'complete'},
    'KIVI_INT4_REFRESH': {'run': 'results/benchmarks/storyeval/storyeval_KIVI_INT4_REFRESH_10prompts_10s_1773038789', 'status': 'complete'},
    'QUAROT_KV_INT4_REFRESH': {'run': None, 'status': 'not_run', 'note': 'MovieGen-only partial attempt; no StoryEval production run exists.'},
    'RTN_K2_V4': {'run': None, 'status': 'not_run', 'note': 'MovieGen-only partial attempt; no StoryEval production run exists.'},
    'KIVI_K2_V4': {'run': None, 'status': 'not_run', 'note': 'MovieGen-only partial attempt; no StoryEval production run exists.'},
    'RTN_INT4_RECENT2': {'run': 'results/benchmarks/storyeval/storyeval_RTN_INT4_RECENT2_10prompts_10s_1773038789', 'status': 'complete'},
    'QUAROT_KV_INT4_RECENT2': {'run': 'results/benchmarks/storyeval/storyeval_QUAROT_KV_INT4_RECENT2_10prompts_10s_1773038789', 'status': 'complete'},
}


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _prepare_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _rel_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        _safe_unlink(link_path)
    rel_target = os.path.relpath(target, start=link_path.parent)
    link_path.symlink_to(rel_target, target_is_directory=target.is_dir())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _load_moviegen_row(run_root: Path, method: str, status: str, note: str | None) -> dict[str, Any]:
    efficiency = _read_json(run_root / 'metrics' / f'efficiency_{method}.json') or {}
    fidelity = _read_json(run_root / 'metrics' / f'fidelity_{method}.json') or {}
    vbench = _read_json(run_root / 'metrics' / f'vbench_{method}.json') or {}
    drift = _read_json(run_root / 'metrics' / f'drift_{method}.json') or {}
    gen_log = run_root / 'logs' / f'generation_{method}.jsonl'
    log_rows = []
    if gen_log.exists():
        with gen_log.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        log_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    video_dir = run_root / 'videos' / method
    num_videos = len(list(video_dir.glob('prompt_*_seed_*.mp4'))) if video_dir.exists() else 0
    fidelity_agg = fidelity.get('aggregate', {}) if isinstance(fidelity.get('aggregate'), dict) else {}
    drift_curve = drift.get('curve', []) if isinstance(drift, dict) else []

    def _drift_scalar(point: dict[str, Any]) -> float | None:
        val = point.get('imaging_quality')
        if isinstance(val, list) and val:
            return float(val[0])
        if isinstance(val, (int, float)):
            return float(val)
        return None

    row = {
        'method': method,
        'status': status,
        'source_run': str(run_root.relative_to(REPO_ROOT)),
        'videos': num_videos,
        'logged_prompts': len(log_rows),
        'psnr': fidelity_agg.get('psnr'),
        'ssim': fidelity_agg.get('ssim'),
        'lpips': fidelity_agg.get('lpips'),
        'background_consistency': (vbench.get('background_consistency') or [None])[0] if isinstance(vbench.get('background_consistency'), list) else vbench.get('background_consistency'),
        'imaging_quality': (vbench.get('imaging_quality') or [None])[0] if isinstance(vbench.get('imaging_quality'), list) else vbench.get('imaging_quality'),
        'subject_consistency': (vbench.get('subject_consistency') or [None])[0] if isinstance(vbench.get('subject_consistency'), list) else vbench.get('subject_consistency'),
        'aesthetic_quality': (vbench.get('aesthetic_quality') or [None])[0] if isinstance(vbench.get('aesthetic_quality'), list) else vbench.get('aesthetic_quality'),
        'bf16_kv_bytes': efficiency.get('bf16_kv_bytes'),
        'compressed_kv_bytes': efficiency.get('compressed_kv_bytes'),
        'compression_ratio': efficiency.get('compression_ratio'),
        'total_runtime_s': efficiency.get('total_runtime_s'),
        'avg_runtime_s_per_prompt': efficiency.get('avg_runtime_s_per_prompt'),
        'peak_vram_bytes': efficiency.get('peak_vram_bytes'),
        'peak_vram_gb': (float(efficiency['peak_vram_bytes']) / (1024**3)) if efficiency.get('peak_vram_bytes') is not None else None,
        'quantize_time_s': efficiency.get('quantize_time_s'),
        'dequantize_time_s': efficiency.get('dequantize_time_s'),
        'quantize_calls': efficiency.get('quantize_calls'),
        'dequantize_calls': efficiency.get('dequantize_calls'),
        'drift_points': len(drift_curve),
        'drift_last_imaging_quality': _drift_scalar(drift_curve[-1]) if drift_curve else None,
        'note': note,
    }
    return row


def _load_storyeval_row(root: Path | None, method: str, status: str, note: str | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if root is None:
        row = {
            'method': method,
            'status': status,
            'source_run': '-',
            'videos': 0,
            'logged_prompts': 0,
            'background_consistency': None,
            'imaging_quality': None,
            'subject_consistency': None,
            'aesthetic_quality': None,
            'avg_runtime_s_per_prompt': None,
            'peak_vram_gb': None,
            'drift_points': 0,
            'drift_last_imaging_quality': None,
            'note': note,
        }
        return row, {}, {}

    summary = _read_json(root / 'summary' / 'summary.json') or {}
    vbench = _read_json(root / 'metrics' / 'vbench.json') or {}
    drift = _read_json(root / 'metrics' / 'drift_imaging_quality.json') or {}
    agg = vbench.get('aggregate', {}) if isinstance(vbench.get('aggregate'), dict) else {}
    curve = drift.get('curve', []) if isinstance(drift, dict) else []
    row = {
        'method': method,
        'status': status,
        'source_run': str(root.relative_to(REPO_ROOT)),
        'videos': summary.get('num_success', summary.get('counts', {}).get('completed')),
        'logged_prompts': summary.get('num_records'),
        'background_consistency': agg.get('background_consistency', summary.get('vbench_background_consistency')),
        'imaging_quality': agg.get('imaging_quality', summary.get('vbench_imaging_quality')),
        'subject_consistency': agg.get('subject_consistency', summary.get('vbench_subject_consistency')),
        'aesthetic_quality': agg.get('aesthetic_quality', summary.get('vbench_aesthetic_quality')),
        'avg_runtime_s_per_prompt': summary.get('avg_runtime_sec'),
        'peak_vram_gb': (float(summary['avg_peak_vram_mb']) / 1024.0) if summary.get('avg_peak_vram_mb') is not None else None,
        'drift_points': len(curve),
        'drift_last_imaging_quality': curve[-1].get('imaging_quality') if curve else summary.get('drift_last_imaging_quality'),
        'note': note,
    }
    return row, summary, _read_json(root / 'summary' / 'config.json') or {}


def _moviegen_drift_frames(payload: dict[str, Any]) -> list[dict[str, float]]:
    out = []
    for point in payload.get('curve', []) if isinstance(payload, dict) else []:
        val = point.get('imaging_quality')
        if isinstance(val, list) and val:
            val = val[0]
        if isinstance(val, (int, float)):
            out.append({'frame_cap': float(point.get('frame_cap', 0)), 'imaging_quality': float(val)})
    return out


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _value_to_markdown(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown_table(path: Path, df: pd.DataFrame) -> None:
    cols = [str(c) for c in df.columns.tolist()]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_value_to_markdown(row[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _palette() -> list[str]:
    return [
        "#0f766e",
        "#1d4ed8",
        "#c2410c",
        "#7c3aed",
        "#b45309",
        "#059669",
        "#be123c",
        "#4f46e5",
    ]


def _pick_color(idx: int) -> str:
    colors = _palette()
    return colors[idx % len(colors)]


def _write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                *body,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def _save_grouped_bar_svg(df: pd.DataFrame, metrics: list[str], title: str, out_path: Path) -> None:
    plot_df = df[["method"] + metrics].copy()
    plot_df = plot_df.melt(id_vars=["method"], var_name="metric", value_name="value").dropna(subset=["value"])
    if plot_df.empty:
        return

    methods = [m for m in METHOD_ORDER if m in plot_df["method"].astype(str).unique().tolist()]
    metric_names = [m for m in metrics if m in plot_df["metric"].unique().tolist()]
    if not methods or not metric_names:
        return

    width = 1500
    height = 700
    left = 110
    right = 60
    top = 80
    bottom = 170
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_val = max(float(v) for v in plot_df["value"].tolist())
    max_val = max(max_val, 1e-6)
    group_w = chart_w / max(len(methods), 1)
    bar_w = max(group_w / (len(metric_names) + 1), 6)

    body = [
        f'<text x="{width/2}" y="34" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{_svg_escape(title)}</text>'
    ]

    for i in range(6):
        y = top + chart_h * i / 5
        val = max_val * (1 - i / 5)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+chart_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        body.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="Arial" fill="#475569">{val:.3f}</text>')

    for mi, method in enumerate(methods):
        method_vals = plot_df[plot_df["method"] == method].set_index("metric")["value"].to_dict()
        base_x = left + mi * group_w + (group_w - len(metric_names) * bar_w) / 2
        for mj, metric in enumerate(metric_names):
            val = method_vals.get(metric)
            if pd.isna(val):
                continue
            bar_h = chart_h * float(val) / max_val
            x = base_x + mj * bar_w
            y = top + chart_h - bar_h
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-2:.1f}" height="{bar_h:.1f}" fill="{_pick_color(mj)}" rx="2"/>'
            )
        label_x = left + mi * group_w + group_w / 2
        body.append(
            f'<text x="{label_x:.1f}" y="{top + chart_h + 18}" text-anchor="end" transform="rotate(-35 {label_x:.1f},{top + chart_h + 18})" font-size="11" font-family="Arial" fill="#0f172a">{_svg_escape(method)}</text>'
        )

    legend_x = left
    legend_y = height - 30
    for idx, metric in enumerate(metric_names):
        x = legend_x + idx * 180
        body.append(f'<rect x="{x}" y="{legend_y-12}" width="14" height="14" fill="{_pick_color(idx)}"/>')
        body.append(f'<text x="{x+20}" y="{legend_y}" font-size="12" font-family="Arial" fill="#0f172a">{_svg_escape(metric)}</text>')

    _write_svg(out_path, width, height, body)


def _save_scatter_svg(df: pd.DataFrame, x: str, y: str, size: str | None, title: str, out_path: Path) -> None:
    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        return
    width = 1100
    height = 700
    left = 90
    right = 50
    top = 80
    bottom = 80
    chart_w = width - left - right
    chart_h = height - top - bottom
    min_x, max_x = float(plot_df[x].min()), float(plot_df[x].max())
    min_y, max_y = float(plot_df[y].min()), float(plot_df[y].max())
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0

    def sx(val: float) -> float:
        return left + (val - min_x) / (max_x - min_x) * chart_w

    def sy(val: float) -> float:
        return top + chart_h - (val - min_y) / (max_y - min_y) * chart_h

    body = [
        f'<text x="{width/2}" y="34" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{_svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#334155" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#334155" stroke-width="2"/>',
        f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-size="13" font-family="Arial">{_svg_escape(x)}</text>',
        f'<text x="24" y="{height/2}" text-anchor="middle" font-size="13" font-family="Arial" transform="rotate(-90 24,{height/2})">{_svg_escape(y)}</text>',
    ]
    for i in range(6):
        xv = min_x + (max_x - min_x) * i / 5
        yv = min_y + (max_y - min_y) * i / 5
        xpx = sx(xv)
        ypx = sy(yv)
        body.append(f'<line x1="{xpx:.1f}" y1="{top}" x2="{xpx:.1f}" y2="{top+chart_h}" stroke="#e5e7eb" stroke-width="1"/>')
        body.append(f'<line x1="{left}" y1="{ypx:.1f}" x2="{left+chart_w}" y2="{ypx:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        body.append(f'<text x="{xpx:.1f}" y="{top+chart_h+18}" text-anchor="middle" font-size="11" font-family="Arial">{xv:.2f}</text>')
        body.append(f'<text x="{left-10}" y="{ypx+4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{yv:.3f}</text>')

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        radius = 8
        if size and pd.notna(row.get(size)):
            radius = max(6, min(18, float(row[size]) * 1.8))
        xpx = sx(float(row[x]))
        ypx = sy(float(row[y]))
        color = _pick_color(idx)
        body.append(f'<circle cx="{xpx:.1f}" cy="{ypx:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.7" stroke="#0f172a" stroke-width="1"/>')
        body.append(f'<text x="{xpx+8:.1f}" y="{ypx-8:.1f}" font-size="11" font-family="Arial">{_svg_escape(str(row["method"]))}</text>')

    _write_svg(out_path, width, height, body)


def _save_line_svg(curves: dict[str, list[dict[str, float]]], title: str, x_key: str, x_label: str, y_label: str, out_path: Path) -> None:
    series = {m: pts for m, pts in curves.items() if pts}
    if not series:
        return
    width = 1200
    height = 700
    left = 90
    right = 50
    top = 80
    bottom = 80
    chart_w = width - left - right
    chart_h = height - top - bottom
    all_x = [float(p[x_key]) for pts in series.values() for p in pts]
    all_y = [float(p["imaging_quality"]) for pts in series.values() for p in pts]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0

    def sx(val: float) -> float:
        return left + (val - min_x) / (max_x - min_x) * chart_w

    def sy(val: float) -> float:
        return top + chart_h - (val - min_y) / (max_y - min_y) * chart_h

    body = [
        f'<text x="{width/2}" y="34" text-anchor="middle" font-size="22" font-family="Arial" font-weight="700">{_svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#334155" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#334155" stroke-width="2"/>',
        f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-size="13" font-family="Arial">{_svg_escape(x_label)}</text>',
        f'<text x="24" y="{height/2}" text-anchor="middle" font-size="13" font-family="Arial" transform="rotate(-90 24,{height/2})">{_svg_escape(y_label)}</text>',
    ]
    for i in range(6):
        xv = min_x + (max_x - min_x) * i / 5
        yv = min_y + (max_y - min_y) * i / 5
        xpx = sx(xv)
        ypx = sy(yv)
        body.append(f'<line x1="{xpx:.1f}" y1="{top}" x2="{xpx:.1f}" y2="{top+chart_h}" stroke="#e5e7eb" stroke-width="1"/>')
        body.append(f'<line x1="{left}" y1="{ypx:.1f}" x2="{left+chart_w}" y2="{ypx:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        body.append(f'<text x="{xpx:.1f}" y="{top+chart_h+18}" text-anchor="middle" font-size="11" font-family="Arial">{xv:.2f}</text>')
        body.append(f'<text x="{left-10}" y="{ypx+4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{yv:.3f}</text>')

    legend_y = top + 18
    legend_x = left
    legend_i = 0
    for method in METHOD_ORDER:
        pts = series.get(method)
        if not pts:
            continue
        color = _pick_color(legend_i)
        points = " ".join(f"{sx(float(p[x_key])):.1f},{sy(float(p['imaging_quality'])):.1f}" for p in pts)
        body.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
        for p in pts:
            body.append(f'<circle cx="{sx(float(p[x_key])):.1f}" cy="{sy(float(p["imaging_quality"])):.1f}" r="3.5" fill="{color}"/>')
        lx = legend_x + (legend_i % 4) * 250
        ly = legend_y + (legend_i // 4) * 20
        body.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+18}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{lx+24}" y="{ly+4}" font-size="12" font-family="Arial">{_svg_escape(method)}</text>')
        legend_i += 1

    _write_svg(out_path, width, height, body)


def build_moviegen_presentation(ts: int) -> tuple[Path, pd.DataFrame]:
    run_dir = RESULTS_ROOT / 'runs' / f'{ts}_presentation_moviegen_fullmatrix'
    _prepare_dir(run_dir)
    for sub in ['metrics', 'logs', 'videos', 'tables']:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {}
    rows = []
    drift_curves: dict[str, list[dict[str, float]]] = {}

    for method in METHOD_ORDER:
        spec = MOVIEGEN_SOURCES[method]
        source_root = REPO_ROOT / spec['run']
        note = spec.get('note')
        manifest[method] = {
            'status': spec['status'],
            'source_run': spec['run'],
            'note': note,
        }

        for kind, rel in [
            ('metrics', source_root / 'metrics' / f'efficiency_{method}.json'),
            ('metrics', source_root / 'metrics' / f'fidelity_{method}.json'),
            ('metrics', source_root / 'metrics' / f'vbench_{method}.json'),
            ('metrics', source_root / 'metrics' / f'drift_{method}.json'),
            ('logs', source_root / 'logs' / f'generation_{method}.jsonl'),
            ('logs', source_root / 'logs' / f'vram_trace_{method}.jsonl'),
        ]:
            if rel.exists():
                _rel_symlink(rel, run_dir / kind / rel.name)

        video_src = source_root / 'videos' / method
        if video_src.exists():
            _rel_symlink(video_src, run_dir / 'videos' / method)

        row = _load_moviegen_row(source_root, method, spec['status'], note)
        rows.append(row)

        drift_path = source_root / 'metrics' / f'drift_{method}.json'
        if drift_path.exists():
            drift_curves[method] = _moviegen_drift_frames(_read_json(drift_path) or {})

    df = pd.DataFrame(rows)
    df['method'] = pd.Categorical(df['method'], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values('method').reset_index(drop=True)
    _write_csv(run_dir / 'tables' / 'baseline_summary.csv', df)
    _write_markdown_table(run_dir / 'tables' / 'baseline_summary.md', df)
    (run_dir / 'tables' / 'method_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    run_meta = {
        'benchmark': 'moviegen',
        'run_name': 'presentation_moviegen_fullmatrix',
        'run_timestamp_unix': ts,
        'created_utc': pd.Timestamp.utcnow().isoformat(),
        'presentation_run': True,
        'description': 'Synthetic collated comparison run spanning baseline and new-method MovieGen experiments.',
        'method_manifest': manifest,
    }
    (run_dir / 'run_meta.json').write_text(json.dumps(run_meta, indent=2), encoding='utf-8')

    _write_csv(DOC_ASSETS / 'moviegen_fullmatrix_summary.csv', df)
    _save_grouped_bar_svg(df, ['psnr', 'ssim', 'lpips'], 'MovieGen fidelity metrics (all tried methods)', DOC_ASSETS / 'moviegen_fidelity_fullmatrix.svg')
    _save_grouped_bar_svg(df, ['background_consistency', 'imaging_quality', 'subject_consistency', 'aesthetic_quality'], 'MovieGen VBench metrics (all tried methods)', DOC_ASSETS / 'moviegen_vbench_fullmatrix.svg')
    _save_scatter_svg(df, 'compression_ratio', 'imaging_quality', 'peak_vram_gb', 'MovieGen compression vs imaging quality', DOC_ASSETS / 'moviegen_quality_efficiency_fullmatrix.svg')
    _save_line_svg(drift_curves, 'MovieGen drift curves (available methods only)', 'frame_cap', 'frame_cap', 'imaging_quality', DOC_ASSETS / 'moviegen_drift_fullmatrix.svg')
    return run_dir, df


def build_storyeval_presentation(ts: int) -> tuple[str, pd.DataFrame]:
    suffix = f'presentation_fullmatrix_{ts}'
    storyeval_root = RESULTS_ROOT / 'benchmarks' / 'storyeval'
    rows = []
    drift_curves: dict[str, list[dict[str, float]]] = {}

    for method in METHOD_ORDER:
        spec = STORYEVAL_SOURCES[method]
        root = storyeval_root / f'storyeval_{method}_{suffix}'
        _prepare_dir(root)
        for sub in ['videos', 'per_prompt', 'metrics', 'logs', 'summary']:
            (root / sub).mkdir(parents=True, exist_ok=True)

        source_run = spec['run']
        source_root = REPO_ROOT / source_run if source_run else None
        note = spec.get('note')

        config = {
            'benchmark': 'storyeval',
            'method': method,
            'run_id': root.name,
            'presentation_group': f'storyeval/{suffix}',
            'created_utc': pd.Timestamp.utcnow().isoformat(),
            'source_run': source_run,
            'status': spec['status'],
            'note': note,
        }
        summary_payload: dict[str, Any]

        if source_root is not None:
            for sub in ['videos', 'per_prompt', 'metrics', 'logs']:
                src = source_root / sub
                if src.exists():
                    _safe_unlink(root / sub)
                    _rel_symlink(src, root / sub)
            src_cfg = _read_json(source_root / 'summary' / 'config.json') or {}
            src_summary = _read_json(source_root / 'summary' / 'summary.json') or {}
            config = {**src_cfg, **config}
            summary_payload = dict(src_summary)
            summary_payload.update({
                'presentation_group': f'storyeval/{suffix}',
                'source_run': source_run,
                'status': spec['status'],
                'note': note,
            })
            drift = _read_json(source_root / 'metrics' / 'drift_imaging_quality.json') or {}
            curve = drift.get('curve', []) if isinstance(drift, dict) else []
            if curve:
                drift_curves[method] = [{'seconds': float(p.get('seconds', 0.0)), 'imaging_quality': float(p['imaging_quality'])} for p in curve if p.get('imaging_quality') is not None]
        else:
            summary_payload = {
                'benchmark': 'storyeval',
                'run_id': root.name,
                'num_records': 0,
                'num_prompts': 0,
                'num_success': 0,
                'num_failed': 0,
                'avg_runtime_sec': None,
                'avg_peak_vram_mb': None,
                'drift_points': 0,
                'drift_last_imaging_quality': None,
                'source_run': source_run,
                'status': spec['status'],
                'note': note,
            }
        (root / 'summary' / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (root / 'summary' / 'summary.json').write_text(json.dumps(summary_payload, indent=2), encoding='utf-8')

        row, _summary, _cfg = _load_storyeval_row(source_root, method, spec['status'], note)
        rows.append(row)

    df = pd.DataFrame(rows)
    df['method'] = pd.Categorical(df['method'], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values('method').reset_index(drop=True)
    _write_csv(DOC_ASSETS / 'storyeval_fullmatrix_summary.csv', df)
    _save_grouped_bar_svg(df, ['background_consistency', 'imaging_quality', 'subject_consistency', 'aesthetic_quality'], 'StoryEval VBench metrics (all tried methods)', DOC_ASSETS / 'storyeval_vbench_fullmatrix.svg')
    _save_grouped_bar_svg(df, ['avg_runtime_s_per_prompt', 'peak_vram_gb'], 'StoryEval runtime / VRAM comparison', DOC_ASSETS / 'storyeval_runtime_vram_fullmatrix.svg')
    _save_line_svg(drift_curves, 'StoryEval drift curves (available methods only)', 'seconds', 'seconds', 'imaging_quality', DOC_ASSETS / 'storyeval_drift_fullmatrix.svg')
    return suffix, df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamp', type=int, default=int(time.time()))
    args = parser.parse_args()
    DOC_ASSETS.mkdir(parents=True, exist_ok=True)
    moviegen_run, moviegen_df = build_moviegen_presentation(args.timestamp)
    storyeval_group, storyeval_df = build_storyeval_presentation(args.timestamp)
    print(json.dumps({
        'moviegen_run': str(moviegen_run.relative_to(REPO_ROOT)),
        'storyeval_group': f'storyeval/{storyeval_group}',
        'doc_assets': str(DOC_ASSETS.relative_to(REPO_ROOT)),
        'moviegen_methods': moviegen_df['method'].astype(str).tolist(),
        'storyeval_methods': storyeval_df['method'].astype(str).tolist(),
    }, indent=2))


if __name__ == '__main__':
    main()
