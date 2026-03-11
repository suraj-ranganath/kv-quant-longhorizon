#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backfill_paths import BACKFILL_ROOT, metric_backfill_aux_dir, metric_backfill_path

DEFAULT_DATASET_CSV = REPO_ROOT / "results" / "combined" / "combined_comparison_dataset.csv"
DEFAULT_FIDELITY_PYTHON = Path("/home/suraj/miniforge3/envs/qvg_sf_infer/bin/python")
DEFAULT_EVAL_PYTHON = Path("/home/suraj/miniforge3/envs/qvg_sf_eval/bin/python")
DEFAULT_MOVIEGEN_PROMPTS = REPO_ROOT / "prompts" / "MovieGenVideoBench_extended.txt"
DEFAULT_SUMMARY_JSON = BACKFILL_ROOT / "missing_evaluations_summary.json"
SURAJ_STORYEVAL_ROOT = Path("/data/suraj/kv-quant-longhorizon/results/benchmarks/storyeval")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def has_metric_value(row: dict[str, str], key: str) -> bool:
    return row.get(key) not in (None, "")


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$", shlex.join(cmd))
    subprocess.check_call(cmd, cwd=str(REPO_ROOT), env=env)


def storyeval_method_from_root(root: Path) -> str | None:
    for candidate in (root / "summary" / "config.json", root / "run_meta.json"):
        if candidate.exists():
            payload = load_json(candidate)
            method = payload.get("method")
            if isinstance(method, str) and method:
                return method.upper()
    match = re.match(r"^storyeval_(.+?)(?:_[0-9a-f]{12})?_10prompts_10s.*$", root.name)
    if match:
        return match.group(1).upper()
    return None


def storyeval_bf16_rank(source_user: str, root: Path) -> tuple[int, str]:
    run_name = root.name.lower()
    if source_user == "suraj":
        if "presentation_fullmatrix" in run_name:
            return 0, run_name
        if "10prompts_10s" in run_name:
            return 1, run_name
        return 2, run_name
    if "10prompts_10s" in run_name:
        return 0, run_name
    if "smoke" in run_name:
        return 2, run_name
    return 1, run_name


def discover_storyeval_bf16_reference(source_user: str) -> str | None:
    root = SURAJ_STORYEVAL_ROOT if source_user == "suraj" else REPO_ROOT / "results" / "benchmarks" / "storyeval"
    if not root.exists():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and storyeval_method_from_root(path) == "BF16"]
    if not candidates:
        return None
    chosen = sorted(candidates, key=lambda path: storyeval_bf16_rank(source_user, path))[0]
    return str(chosen.resolve())


def collect_targets(dataset_csv: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    bf16_storyeval_by_source: dict[str, str] = {}
    bf16_moviegen_by_source: dict[str, str] = {}
    bf16_moviegen_by_source_run: dict[tuple[str, str], str] = {}

    with dataset_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            benchmark = row.get("benchmark")
            source_user = row.get("source_user")
            run_root = row.get("run_root")
            method = row.get("method")
            if not benchmark or not source_user or not run_root or not method:
                continue
            key = (benchmark, source_user, run_root, method)
            group = grouped.setdefault(
                key,
                {
                    "benchmark": benchmark,
                    "source_user": source_user,
                    "run_root": run_root,
                    "method": method,
                    "rows": [],
                    "missing_moviegen_fidelity": False,
                    "missing_moviegen_vbench": False,
                    "missing_storyeval_fidelity": False,
                },
            )
            group["rows"].append(row)
            if benchmark == "moviegen":
                if not has_metric_value(row, "moviegen_fidelity_psnr_agg"):
                    group["missing_moviegen_fidelity"] = True
                if not has_metric_value(row, "moviegen_background_consistency_agg"):
                    group["missing_moviegen_vbench"] = True
                if method == "BF16":
                    bf16_moviegen_by_source[source_user] = run_root
                    bf16_moviegen_by_source_run[(source_user, run_root)] = run_root
            elif benchmark == "storyeval":
                if not has_metric_value(row, "storyeval_fidelity_psnr_agg"):
                    group["missing_storyeval_fidelity"] = True
                if method == "BF16":
                    bf16_storyeval_by_source[source_user] = run_root

    return {
        "groups": list(grouped.values()),
        "bf16_storyeval_by_source": bf16_storyeval_by_source,
        "bf16_moviegen_by_source": bf16_moviegen_by_source,
        "bf16_moviegen_by_source_run": bf16_moviegen_by_source_run,
    }


def count_mp4s(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("*.mp4"))


def backfill_moviegen_vbench(
    *,
    group: dict[str, Any],
    eval_python: Path,
    prompt_file: Path,
) -> dict[str, Any]:
    run_root = Path(group["run_root"]).resolve()
    method = str(group["method"])
    videos_dir = run_root / "videos" / method
    output_json = metric_backfill_path(
        benchmark="moviegen",
        metric_kind="vbench",
        run_root=run_root,
        method=method,
    )
    out_dir = metric_backfill_aux_dir(
        benchmark="moviegen",
        metric_kind="vbench",
        run_root=run_root,
        method=method,
    )
    env = os.environ.copy()
    env["PYTHON_BIN"] = str(eval_python)
    cmd = [
        "bash",
        str(REPO_ROOT / "scripts" / "03_eval_vbench.sh"),
        method,
        str(videos_dir),
        str(prompt_file),
        str(out_dir),
        str(output_json),
    ]
    run_cmd(cmd, env=env)
    return {
        "task": "moviegen_vbench",
        "source_user": group["source_user"],
        "run_root": str(run_root),
        "method": method,
        "output_json": str(output_json),
        "num_videos": count_mp4s(videos_dir),
    }


def resolve_moviegen_bf16_dir(
    *,
    group: dict[str, Any],
    bf16_moviegen_by_source: dict[str, str],
    bf16_moviegen_by_source_run: dict[tuple[str, str], str],
) -> Path:
    run_root = Path(group["run_root"]).resolve()
    source_user = str(group["source_user"])
    same_run_root = bf16_moviegen_by_source_run.get((source_user, str(run_root)))
    if same_run_root:
        return Path(same_run_root).resolve() / "videos" / "BF16"
    fallback = bf16_moviegen_by_source.get(source_user)
    if fallback:
        return Path(fallback).resolve() / "videos" / "BF16"
    raise RuntimeError(f"No MovieGen BF16 reference found for {source_user}:{run_root}")


def backfill_fidelity(
    *,
    benchmark: str,
    group: dict[str, Any],
    bf16_dir: Path,
    candidate_dir: Path,
    fidelity_python: Path,
    device: str,
    match_intersection: bool,
) -> dict[str, Any]:
    run_root = Path(group["run_root"]).resolve()
    method = str(group["method"])
    output_json = metric_backfill_path(
        benchmark=benchmark,
        metric_kind="fidelity",
        run_root=run_root,
        method=method,
    )
    cmd = [
        str(fidelity_python),
        str(REPO_ROOT / "scripts" / "02_eval_fidelity.py"),
        "--bf16-dir",
        str(bf16_dir),
        "--candidate-dir",
        str(candidate_dir),
        "--output",
        str(output_json),
        "--device",
        device,
    ]
    if match_intersection:
        cmd.append("--match-intersection")
    run_cmd(cmd)
    return {
        "task": f"{benchmark}_fidelity",
        "source_user": group["source_user"],
        "run_root": str(run_root),
        "method": method,
        "bf16_dir": str(bf16_dir),
        "candidate_dir": str(candidate_dir),
        "output_json": str(output_json),
        "match_intersection": match_intersection,
        "num_candidate_videos": count_mp4s(candidate_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill missing MovieGen/StoryEval evaluation metrics from existing videos.")
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--fidelity-python", type=Path, default=DEFAULT_FIDELITY_PYTHON)
    parser.add_argument("--eval-python", type=Path, default=DEFAULT_EVAL_PYTHON)
    parser.add_argument("--moviegen-prompts", type=Path, default=DEFAULT_MOVIEGEN_PROMPTS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=("moviegen", "storyeval"),
        help="Restrict backfills to one or more benchmarks.",
    )
    parser.add_argument(
        "--method",
        action="append",
        help="Restrict backfills to one or more methods.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = collect_targets(args.dataset_csv)
    selected_benchmarks = set(args.benchmark or ["moviegen", "storyeval"])
    selected_methods = set(args.method or [])
    groups = [
        group
        for group in targets["groups"]
        if group["benchmark"] in selected_benchmarks and (not selected_methods or group["method"] in selected_methods)
    ]

    task_rows: list[dict[str, Any]] = []

    moviegen_vbench_groups = [
        group for group in groups if group["benchmark"] == "moviegen" and group["missing_moviegen_vbench"]
    ]
    for group in sorted(moviegen_vbench_groups, key=lambda row: (row["source_user"], row["method"], row["run_root"])):
        task_rows.append(
            backfill_moviegen_vbench(
                group=group,
                eval_python=args.eval_python,
                prompt_file=args.moviegen_prompts,
            )
        )

    moviegen_fidelity_groups = [
        group for group in groups if group["benchmark"] == "moviegen" and group["missing_moviegen_fidelity"]
    ]
    for group in sorted(moviegen_fidelity_groups, key=lambda row: (row["source_user"], row["method"], row["run_root"])):
        candidate_dir = Path(group["run_root"]).resolve() / "videos" / str(group["method"])
        bf16_dir = resolve_moviegen_bf16_dir(
            group=group,
            bf16_moviegen_by_source=targets["bf16_moviegen_by_source"],
            bf16_moviegen_by_source_run=targets["bf16_moviegen_by_source_run"],
        )
        match_intersection = count_mp4s(candidate_dir) != count_mp4s(bf16_dir)
        task_rows.append(
            backfill_fidelity(
                benchmark="moviegen",
                group=group,
                bf16_dir=bf16_dir,
                candidate_dir=candidate_dir,
                fidelity_python=args.fidelity_python,
                device=args.device,
                match_intersection=match_intersection,
            )
        )

    storyeval_fidelity_groups = [
        group for group in groups if group["benchmark"] == "storyeval" and group["missing_storyeval_fidelity"]
    ]
    for group in sorted(storyeval_fidelity_groups, key=lambda row: (row["source_user"], row["method"], row["run_root"])):
        source_user = str(group["source_user"])
        bf16_root = targets["bf16_storyeval_by_source"].get(source_user)
        if not bf16_root:
            bf16_root = discover_storyeval_bf16_reference(source_user)
            if bf16_root:
                targets["bf16_storyeval_by_source"][source_user] = bf16_root
        if not bf16_root:
            raise RuntimeError(f"No StoryEval BF16 reference found for {group['source_user']}")
        bf16_dir = Path(bf16_root).resolve() / "videos"
        candidate_dir = Path(group["run_root"]).resolve() / "videos"
        task_rows.append(
            backfill_fidelity(
                benchmark="storyeval",
                group=group,
                bf16_dir=bf16_dir,
                candidate_dir=candidate_dir,
                fidelity_python=args.fidelity_python,
                device=args.device,
                match_intersection=False,
            )
        )

    task_counts = defaultdict(int)
    for row in task_rows:
        task_counts[str(row["task"])] += 1

    write_json(
        args.summary_json,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "task_counts": dict(task_counts),
            "tasks": task_rows,
        },
    )
    print(f"[missing-evals] summary {args.summary_json}")


if __name__ == "__main__":
    main()
