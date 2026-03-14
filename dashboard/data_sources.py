from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

MAX_INLINE_SCAN_BYTES = 25 * 1024 * 1024
PRIMARY_SCORE_COLUMNS = {
    "comparison_key": 80,
    "benchmark": 50,
    "source_user": 25,
    "run_name": 20,
    "run_root": 20,
    "method": 40,
    "method_display": 15,
    "compression_ratio": 20,
    "peak_vram_bytes": 15,
    "avg_runtime_s_per_prompt": 15,
    "moviegen_imaging_quality_agg": 25,
    "storyeval_imaging_quality_agg": 25,
    "moviegen_drift_last_imaging_quality": 25,
    "storyeval_drift_last_imaging_quality": 25,
}
PRIMARY_KIND_BONUS = {
    "combined_dataset": 200,
    "methods_quality": 120,
    "method_export": 90,
    "run_summary": 40,
    "registry": 10,
    "gap_report": 10,
    "other": 0,
}


def _categorize_csv_source(path: Path) -> str:
    lower_name = path.name.lower()
    lower_path = str(path).lower()
    if lower_name == "combined_comparison_dataset.csv":
        return "combined_dataset"
    if lower_name == "methods_quality.csv":
        return "methods_quality"
    if "representative_metrics" in lower_name or "export" in lower_name:
        return "method_export"
    if lower_name in {"summary.csv", "baseline_summary.csv"}:
        return "run_summary"
    if "registry" in lower_path or "manifest" in lower_name:
        return "registry"
    if "gaps" in lower_name:
        return "gap_report"
    return "other"


def _safe_read_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return None, str(exc)


def _score_primary_candidate(path: Path, columns: list[str], row_count: int) -> int:
    score = PRIMARY_KIND_BONUS.get(_categorize_csv_source(path), 0)
    for column, weight in PRIMARY_SCORE_COLUMNS.items():
        if column in columns:
            score += weight
    if row_count >= 500:
        score += 40
    elif row_count >= 100:
        score += 20
    elif row_count > 0:
        score += 5
    return score


def discover_csv_sources(repo_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in sorted(repo_root.rglob("*.csv")):
        if any(part.startswith(".") for part in path.parts):
            continue
        if "__pycache__" in path.parts:
            continue
        size_bytes = path.stat().st_size
        kind = _categorize_csv_source(path)
        note = ""
        frame: pd.DataFrame | None = None
        if size_bytes <= MAX_INLINE_SCAN_BYTES:
            frame, error = _safe_read_csv(path)
            if error:
                note = f"Read failed: {error}"
        else:
            note = "Skipped inline read because the file is larger than the scan budget."
        columns = frame.columns.tolist() if frame is not None else []
        row_count = int(len(frame)) if frame is not None else 0
        records.append(
            {
                "path": str(path.relative_to(repo_root)),
                "kind": kind,
                "size_bytes": int(size_bytes),
                "rows": row_count,
                "column_count": int(len(columns)),
                "columns": columns,
                "primary_score": _score_primary_candidate(path, columns, row_count),
                "selected_as_primary": False,
                "analysis_role": "supporting",
                "note": note,
            }
        )
    catalog = pd.DataFrame(records)
    if catalog.empty:
        return catalog

    primary_mask = catalog["kind"].isin({"combined_dataset", "methods_quality", "method_export", "run_summary", "other"})
    candidates = catalog[primary_mask].copy()
    if not candidates.empty:
        primary_idx = candidates.sort_values(["primary_score", "rows", "size_bytes"], ascending=[False, False, False]).index[0]
        catalog.loc[primary_idx, "selected_as_primary"] = True
        catalog.loc[primary_idx, "analysis_role"] = "primary"
        supplementary_mask = catalog["kind"].isin({"methods_quality", "method_export", "run_summary", "registry", "gap_report"}) & ~catalog["selected_as_primary"]
        catalog.loc[supplementary_mask, "analysis_role"] = "supplementary"
    return catalog.sort_values(["selected_as_primary", "primary_score", "rows", "path"], ascending=[False, False, False, True]).reset_index(drop=True)


def load_dashboard_workspace(repo_root: Path) -> dict[str, Any]:
    catalog = discover_csv_sources(repo_root)
    if catalog.empty:
        return {
            "primary_df": pd.DataFrame(),
            "primary_path": None,
            "source_catalog": catalog,
            "supplementary_tables": [],
        }

    primary_rows = catalog[catalog["selected_as_primary"]]
    primary_path: Path | None = None
    primary_df = pd.DataFrame()
    if not primary_rows.empty:
        primary_rel = primary_rows.iloc[0]["path"]
        primary_path = repo_root / str(primary_rel)
        primary_df, _ = _safe_read_csv(primary_path)
        if primary_df is None:
            primary_df = pd.DataFrame()

    supplementary_tables: list[dict[str, Any]] = []
    for row in catalog[catalog["analysis_role"] == "supplementary"].itertuples(index=False):
        rel_path = Path(str(row.path))
        abs_path = repo_root / rel_path
        frame, error = _safe_read_csv(abs_path)
        supplementary_tables.append(
            {
                "path": str(rel_path),
                "kind": str(row.kind),
                "note": str(row.note) if row.note else (f"Read failed: {error}" if error else "Loaded successfully."),
                "df": frame if frame is not None else pd.DataFrame(),
            }
        )

    if primary_path is not None and not primary_df.empty:
        primary_df = primary_df.copy()
        primary_df["dataset_provenance_path"] = str(primary_path.relative_to(repo_root))

    return {
        "primary_df": primary_df,
        "primary_path": str(primary_path.relative_to(repo_root)) if primary_path is not None else None,
        "source_catalog": catalog,
        "supplementary_tables": supplementary_tables,
    }
