#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "results" / "combined" / "registry" / "storyeval_manifest.json"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def matches_filters(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.method and row["method"] not in set(args.method):
        return False
    if args.method_family and row["method_family"] not in set(args.method_family):
        return False
    if args.config_id:
        config_ids = {str(x) for x in row.get("config_ids") or []}
        if row.get("primary_config_id"):
            config_ids.add(str(row["primary_config_id"]))
        if not config_ids.intersection(set(args.config_id)):
            return False
    return True


def run_one(command: list[str], env: dict[str, str], *, dry_run: bool) -> int:
    print("$", shlex.join(command))
    if dry_run:
        return 0
    return subprocess.call(command, cwd=str(REPO_ROOT), env=env)


def insert_runner_flag(command: list[str], flag: str) -> None:
    try:
        idx = command.index("--")
    except ValueError:
        idx = len(command)
    command.insert(idx, flag)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StoryEval parity commands from the combined-storyeval manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gpu-id", type=str, required=True, help="CUDA_VISIBLE_DEVICES value to reserve for the batch.")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--method", action="append", default=[], help="Only run the specified method(s). Can be repeated.")
    parser.add_argument("--method-family", action="append", default=[], help="Only run the specified method family/families.")
    parser.add_argument("--config-id", action="append", default=[], help="Only run manifest rows containing these config ids.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-vbench", action="store_true")
    parser.add_argument("--skip-drift", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--alloc-conf",
        type=str,
        default="expandable_segments:True",
        help="PYTORCH_CUDA_ALLOC_CONF value applied to every run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = [row for row in load_manifest(args.manifest) if matches_filters(row, args)]
    rows = rows[args.start_index :]
    if args.max_items is not None:
        rows = rows[: args.max_items]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["PYTORCH_CUDA_ALLOC_CONF"] = args.alloc_conf

    failures: list[tuple[str, int]] = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        print(f"[{idx}/{total}] {row['run_name']} :: {row['method']}")
        command = list(row["command"])
        if args.resume:
            insert_runner_flag(command, "--resume")
        if args.force:
            insert_runner_flag(command, "--force")
        if args.skip_vbench:
            insert_runner_flag(command, "--skip-vbench")
        if args.skip_drift:
            insert_runner_flag(command, "--skip-drift")
        if args.skip_summary:
            insert_runner_flag(command, "--skip-summary")
        code = run_one(command, env, dry_run=args.dry_run)
        if code != 0:
            failures.append((row["run_name"], code))
            print(f"[failed] {row['run_name']} exit={code}")
            if not args.continue_on_error:
                break

    if failures:
        raise SystemExit(
            "StoryEval manifest completed with failures: "
            + ", ".join(f"{run_name} (exit {code})" for run_name, code in failures)
        )
    print(f"StoryEval manifest completed: {total} run(s) processed.")


if __name__ == "__main__":
    main()
