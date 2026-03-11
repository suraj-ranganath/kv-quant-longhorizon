#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKFILL_MANIFEST = REPO_ROOT / "results" / "combined" / "registry" / "backfill_manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "combined" / "registry"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def build_storyeval_record(row: dict[str, Any]) -> dict[str, Any]:
    run_name = f"storyeval_{slugify(row['method'])}_{row['primary_config_id']}_10prompts_10s"
    record = {
        "primary_config_id": row["primary_config_id"],
        "config_ids": list(row.get("config_ids") or [row["primary_config_id"]]),
        "method": row["method"],
        "method_family": row["method_family"],
        "sources": list(row.get("sources") or []),
        "run_labels": list(row.get("run_labels") or []),
        "run_name": run_name,
        "run_root": f"results/benchmarks/storyeval/{run_name}",
        "source_moviegen_run_root": row.get("run_root"),
        "passthrough_args": list(row.get("passthrough_args") or []),
        "profile_flowcache": bool(row.get("profile_flowcache")),
        "flowcache_profile_recent_ratio": row.get("flowcache_profile_recent_ratio"),
        "flowcache_profile_min_scale": row.get("flowcache_profile_min_scale"),
        "flowcache_profile_max_scale": row.get("flowcache_profile_max_scale"),
        "dashboard_ready_runs": int(row.get("dashboard_ready_runs") or 0),
        "notes": list(row.get("notes") or []),
    }
    record["command"] = build_command(record)
    return record


def build_bf16_record() -> dict[str, Any]:
    record = {
        "primary_config_id": None,
        "config_ids": [],
        "method": "BF16",
        "method_family": "BF16",
        "sources": ["combined"],
        "run_labels": [],
        "run_name": "storyeval_bf16_10prompts_10s",
        "run_root": "results/benchmarks/storyeval/storyeval_bf16_10prompts_10s",
        "source_moviegen_run_root": None,
        "passthrough_args": ["--device", "cuda:0", "--use-ema"],
        "profile_flowcache": False,
        "flowcache_profile_recent_ratio": None,
        "flowcache_profile_min_scale": None,
        "flowcache_profile_max_scale": None,
        "dashboard_ready_runs": 1,
        "notes": ["Synthetic StoryEval baseline row added for BF16 parity."],
    }
    record["command"] = build_command(record)
    return record


def build_command(record: dict[str, Any]) -> list[str]:
    cmd = [
        "python3",
        "scripts/27_run_storyeval_backfill.py",
        "--run-root",
        record["run_root"],
        "--run-name",
        record["run_name"],
        "--method",
        record["method"],
    ]
    if record.get("primary_config_id"):
        cmd.extend(["--config-id", str(record["primary_config_id"])])
    if record.get("source_moviegen_run_root"):
        cmd.extend(["--source-moviegen-run-root", str(record["source_moviegen_run_root"])])
    if record.get("profile_flowcache"):
        cmd.extend(
            [
                "--profile-flowcache",
                "--flowcache-profile-recent-ratio",
                str(record["flowcache_profile_recent_ratio"]),
                "--flowcache-profile-min-scale",
                str(record["flowcache_profile_min_scale"]),
                "--flowcache-profile-max-scale",
                str(record["flowcache_profile_max_scale"]),
            ]
        )
    cmd.append("--")
    cmd.extend(record["passthrough_args"])
    return cmd


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_name",
        "run_root",
        "method",
        "method_family",
        "primary_config_id",
        "config_ids",
        "source_moviegen_run_root",
        "profile_flowcache",
        "flowcache_profile_recent_ratio",
        "flowcache_profile_min_scale",
        "flowcache_profile_max_scale",
        "dashboard_ready_runs",
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_name": row["run_name"],
                    "run_root": row["run_root"],
                    "method": row["method"],
                    "method_family": row["method_family"],
                    "primary_config_id": row["primary_config_id"],
                    "config_ids": ",".join(str(x) for x in row["config_ids"]),
                    "source_moviegen_run_root": row["source_moviegen_run_root"],
                    "profile_flowcache": row["profile_flowcache"],
                    "flowcache_profile_recent_ratio": row["flowcache_profile_recent_ratio"],
                    "flowcache_profile_min_scale": row["flowcache_profile_min_scale"],
                    "flowcache_profile_max_scale": row["flowcache_profile_max_scale"],
                    "dashboard_ready_runs": row["dashboard_ready_runs"],
                    "command": shlex.join(build_command(row)),
                }
            )


def write_shell_script(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for row in rows:
        label = row["primary_config_id"] or "baseline"
        lines.append(f"# {row['method']} :: {label}")
        lines.append(shlex.join(build_command(row)))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build StoryEval parity commands for combined-workspace methods.")
    parser.add_argument("--backfill-manifest", type=Path, default=DEFAULT_BACKFILL_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backfill_rows = load_json(args.backfill_manifest)
    rows = [build_bf16_record()]
    rows.extend(build_storyeval_record(row) for row in backfill_rows)
    rows = sorted(rows, key=lambda row: (row["method_family"], row["method"], str(row["primary_config_id"] or "")))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = args.output_dir / "storyeval_manifest.json"
    manifest_csv = args.output_dir / "storyeval_manifest.csv"
    manifest_sh = args.output_dir / "storyeval_commands.sh"
    write_json(manifest_json, rows)
    write_csv(manifest_csv, rows)
    write_shell_script(manifest_sh, rows)
    print(f"StoryEval manifest rows: {len(rows)}")
    print(f"Wrote {manifest_json}")
    print(f"Wrote {manifest_csv}")
    print(f"Wrote {manifest_sh}")


if __name__ == "__main__":
    main()
