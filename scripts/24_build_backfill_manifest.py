#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "results" / "combined" / "registry" / "unique_configurations.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "combined" / "registry"

GENERATION_DEFAULTS: dict[str, Any] = {
    "prq_residual_bits": 4,
    "qaq_outlier_threshold": 6.0,
    "age_tier_recent_ratio": 0.3,
    "age_tier_recent_bits": 4,
    "age_tier_recent_method": "RTN",
    "age_tier_old_method": "RTN",
    "tptq_recent_ratio": 0.3,
    "tptq_recent_bits": 4,
    "tptq_recent_method": "RTN",
    "tptq_residual_bits": 2,
    "tptq_outlier_threshold": 6.0,
    "tptq_outlier_max_ratio": 0.005,
    "flowcache_recent_ratio": 0.25,
    "flowcache_recent_bits": 4,
    "flowcache_recent_method": "RTN",
    "flowcache_old_method": "RTN",
    "flowcache_min_layer_budget_scale": 0.75,
    "flowcache_max_layer_budget_scale": 1.25,
    "flowcache_profile_min_scale": 0.70,
    "flowcache_profile_max_scale": 1.30,
    "flowcache_important_old_ratio": 0.20,
    "flowcache_importance_alpha": 0.7,
    "flowcache_importance_beta": 0.3,
    "flowcache_prune_retained_old_ratio": 0.30,
    "flowcache_prune_refresh_gap_chunks": 1,
    "flowcache_native_rel_l1_thresh": 1.50,
    "flowcache_native_warmup_steps": 0,
    "spatial_fg_method": "RTN",
    "spatial_fg_bits": 4,
    "spatial_bg_method": "RTN",
    "spatial_bg_bits": 2,
    "spatial_mask_policy": "hybrid",
    "spatial_variance_threshold": 0.02,
    "spatial_min_foreground_ratio": 0.45,
    "spatial_max_foreground_ratio": 0.85,
    "spatial_target_foreground_ratio": 0.65,
}

PROFILE_REQUIRED_FAMILIES = {
    "FLOWCACHE_ADAPTIVE",
    "FLOWCACHE_PRUNE",
    "FLOWCACHE_SOFT_PRUNE",
    "FLOWCACHE_NATIVE",
    "FLOWCACHE_NATIVE_SOFT_PRUNE",
}

FLOWCACHE_PROFILE_DEFAULT_MIN = 0.70
FLOWCACHE_PROFILE_DEFAULT_MAX = 1.30


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_config_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unsupported config payload type: {type(value)!r}")


def coalesce(meta: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in meta and meta[key] is not None:
            return meta[key]
    return default


def as_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    return int(float(value))


def as_float(value: Any) -> float:
    return float(value)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def split_labels(run_labels: str | None) -> list[str]:
    if not run_labels:
        return []
    return [label for label in (item.strip() for item in run_labels.split(",")) if label]


def parse_spatial_method(method: str) -> tuple[str, int, str, int]:
    match = re.fullmatch(
        r"SPATIAL_MIXED_FG_(RTN|KIVI|QUAROT_KV)_INT(2|4)_BG_(RTN|KIVI|QUAROT_KV)_INT(2|4)",
        method,
    )
    if not match:
        raise ValueError(f"Unsupported spatial mixed method name: {method}")
    return match.group(1), int(match.group(2)), match.group(3), int(match.group(4))


def build_spatial_args(method: str, meta: dict[str, Any]) -> list[str]:
    fg_method, fg_bits, bg_method, bg_bits = parse_spatial_method(method)
    return [
        "--spatial-fg-method",
        str(meta.get("spatial_fg_method", fg_method)),
        "--spatial-fg-bits",
        str(as_int(meta.get("spatial_fg_bits", fg_bits))),
        "--spatial-bg-method",
        str(meta.get("spatial_bg_method", bg_method)),
        "--spatial-bg-bits",
        str(as_int(meta.get("spatial_bg_bits", bg_bits))),
        "--spatial-mask-policy",
        str(meta.get("spatial_mask_policy", GENERATION_DEFAULTS["spatial_mask_policy"])),
        "--spatial-variance-threshold",
        str(as_float(meta.get("spatial_variance_threshold", GENERATION_DEFAULTS["spatial_variance_threshold"]))),
        "--spatial-min-foreground-ratio",
        str(as_float(meta.get("spatial_min_foreground_ratio", GENERATION_DEFAULTS["spatial_min_foreground_ratio"]))),
        "--spatial-max-foreground-ratio",
        str(as_float(meta.get("spatial_max_foreground_ratio", GENERATION_DEFAULTS["spatial_max_foreground_ratio"]))),
        "--spatial-target-foreground-ratio",
        str(as_float(meta.get("spatial_target_foreground_ratio", GENERATION_DEFAULTS["spatial_target_foreground_ratio"]))),
    ]


def build_flowcache_spec(method_family: str, meta: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
    recent_ratio = as_float(
        coalesce(
            meta,
            "flowcache_config_chunk_recent_ratio",
            "flowcache_adaptive_config_chunk_recent_ratio",
            "flowcache_prune_config_chunk_recent_ratio",
            "flowcache_soft_prune_config_chunk_recent_ratio",
            default=GENERATION_DEFAULTS["flowcache_recent_ratio"],
        )
    )
    recent_bits = as_int(
        coalesce(
            meta,
            "flowcache_recent_bits",
            "flowcache_adaptive_recent_bits",
            "flowcache_prune_recent_bits",
            "flowcache_soft_prune_recent_bits",
            default=GENERATION_DEFAULTS["flowcache_recent_bits"],
        )
    )
    recent_method = str(meta.get("flowcache_recent_method", GENERATION_DEFAULTS["flowcache_recent_method"]))
    old_method = str(meta.get("flowcache_old_method", GENERATION_DEFAULTS["flowcache_old_method"]))
    important_old_ratio = as_float(
        coalesce(
            meta,
            "flowcache_important_old_ratio",
            "flowcache_adaptive_config_important_old_ratio",
            "flowcache_prune_config_important_old_ratio",
            "flowcache_soft_prune_config_important_old_ratio",
            default=GENERATION_DEFAULTS["flowcache_important_old_ratio"],
        )
    )
    importance_alpha = as_float(meta.get("flowcache_importance_alpha", GENERATION_DEFAULTS["flowcache_importance_alpha"]))
    importance_beta = as_float(meta.get("flowcache_importance_beta", GENERATION_DEFAULTS["flowcache_importance_beta"]))
    retained_old_ratio = as_float(
        coalesce(
            meta,
            "flowcache_prune_retained_old_ratio",
            "flowcache_prune_config_retained_old_ratio",
            "flowcache_soft_prune_config_retained_old_ratio",
            default=GENERATION_DEFAULTS["flowcache_prune_retained_old_ratio"],
        )
    )
    refresh_gap_chunks = as_int(
        coalesce(
            meta,
            "flowcache_prune_refresh_gap_chunks",
            "flowcache_soft_prune_refresh_gap_chunks",
            default=GENERATION_DEFAULTS["flowcache_prune_refresh_gap_chunks"],
        )
    )
    native_rel_l1_thresh = as_float(
        meta.get("flowcache_native_rel_l1_thresh", GENERATION_DEFAULTS["flowcache_native_rel_l1_thresh"])
    )
    native_warmup_steps = as_int(
        meta.get("flowcache_native_warmup_steps", GENERATION_DEFAULTS["flowcache_native_warmup_steps"])
    )

    profile_flowcache = method_family in PROFILE_REQUIRED_FAMILIES or any(
        key in meta for key in ("flowcache_profile_min_scale", "flowcache_profile_max_scale")
    )

    if method_family == "FLOWCACHE_HYBRID":
        min_layer_budget_scale = as_float(
            meta.get("flowcache_min_layer_budget_scale", GENERATION_DEFAULTS["flowcache_min_layer_budget_scale"])
        )
        max_layer_budget_scale = as_float(
            meta.get("flowcache_max_layer_budget_scale", GENERATION_DEFAULTS["flowcache_max_layer_budget_scale"])
        )
    else:
        min_layer_budget_scale = as_float(
            meta.get("flowcache_min_layer_budget_scale", FLOWCACHE_PROFILE_DEFAULT_MIN)
        )
        max_layer_budget_scale = as_float(
            meta.get("flowcache_max_layer_budget_scale", FLOWCACHE_PROFILE_DEFAULT_MAX)
        )

    profile_min_scale = as_float(meta.get("flowcache_profile_min_scale", FLOWCACHE_PROFILE_DEFAULT_MIN))
    profile_max_scale = as_float(meta.get("flowcache_profile_max_scale", FLOWCACHE_PROFILE_DEFAULT_MAX))

    args: list[str] = []
    if method_family != "FLOWCACHE_NATIVE":
        args.extend(
            [
                "--flowcache-recent-ratio",
                str(recent_ratio),
                "--flowcache-recent-bits",
                str(recent_bits),
                "--flowcache-recent-method",
                recent_method,
                "--flowcache-old-method",
                old_method,
                "--flowcache-min-layer-budget-scale",
                str(min_layer_budget_scale),
                "--flowcache-max-layer-budget-scale",
                str(max_layer_budget_scale),
            ]
        )

    if method_family in {"FLOWCACHE_ADAPTIVE", "FLOWCACHE_PRUNE", "FLOWCACHE_SOFT_PRUNE", "FLOWCACHE_NATIVE_SOFT_PRUNE"}:
        args.extend(
            [
                "--flowcache-important-old-ratio",
                str(important_old_ratio),
                "--flowcache-importance-alpha",
                str(importance_alpha),
                "--flowcache-importance-beta",
                str(importance_beta),
            ]
        )

    if method_family in {"FLOWCACHE_PRUNE", "FLOWCACHE_SOFT_PRUNE", "FLOWCACHE_NATIVE_SOFT_PRUNE"}:
        args.extend(
            [
                "--flowcache-prune-retained-old-ratio",
                str(retained_old_ratio),
                "--flowcache-prune-refresh-gap-chunks",
                str(refresh_gap_chunks),
            ]
        )

    if method_family in {"FLOWCACHE_NATIVE", "FLOWCACHE_NATIVE_SOFT_PRUNE"}:
        args.extend(
            [
                "--flowcache-native-rel-l1-thresh",
                str(native_rel_l1_thresh),
                "--flowcache-native-warmup-steps",
                str(native_warmup_steps),
            ]
        )

    profile_spec = None
    if profile_flowcache:
        profile_spec = {
            "recent_ratio": recent_ratio,
            "min_scale": profile_min_scale,
            "max_scale": profile_max_scale,
        }
    return args, profile_spec


def build_passthrough(method: str, method_family: str, meta: dict[str, Any]) -> tuple[list[str], dict[str, Any] | None]:
    args: list[str] = ["--device", "cuda:0", "--use-ema"]
    profile_spec: dict[str, Any] | None = None

    if method_family == "PRQ":
        args.extend(
            [
                "--prq-residual-bits",
                str(as_int(meta.get("prq_residual_bits", GENERATION_DEFAULTS["prq_residual_bits"]))),
            ]
        )
    elif method_family == "QAQ":
        args.extend(
            [
                "--qaq-outlier-threshold",
                str(as_float(meta.get("qaq_outlier_threshold", GENERATION_DEFAULTS["qaq_outlier_threshold"]))),
            ]
        )
    elif method_family == "AGE_TIER":
        args.extend(
            [
                "--age-tier-recent-ratio",
                str(as_float(meta.get("age_tier_config_recent_ratio", GENERATION_DEFAULTS["age_tier_recent_ratio"]))),
                "--age-tier-recent-bits",
                str(as_int(meta.get("age_tier_recent_bits", GENERATION_DEFAULTS["age_tier_recent_bits"]))),
                "--age-tier-recent-method",
                str(meta.get("age_tier_recent_method", GENERATION_DEFAULTS["age_tier_recent_method"])),
                "--age-tier-old-method",
                str(meta.get("age_tier_old_method", GENERATION_DEFAULTS["age_tier_old_method"])),
            ]
        )
    elif method_family == "TPTQ":
        args.extend(
            [
                "--tptq-recent-ratio",
                str(as_float(meta.get("tptq_config_recent_ratio", GENERATION_DEFAULTS["tptq_recent_ratio"]))),
                "--tptq-recent-bits",
                str(as_int(meta.get("tptq_recent_bits", GENERATION_DEFAULTS["tptq_recent_bits"]))),
                "--tptq-recent-method",
                str(meta.get("tptq_recent_method", GENERATION_DEFAULTS["tptq_recent_method"])),
                "--tptq-residual-bits",
                str(as_int(meta.get("tptq_residual_bits", GENERATION_DEFAULTS["tptq_residual_bits"]))),
                "--tptq-outlier-threshold",
                str(as_float(meta.get("tptq_outlier_threshold", GENERATION_DEFAULTS["tptq_outlier_threshold"]))),
                "--tptq-outlier-max-ratio",
                str(as_float(meta.get("tptq_outlier_max_ratio", GENERATION_DEFAULTS["tptq_outlier_max_ratio"]))),
            ]
        )
    elif method_family == "SPATIAL_MIXED":
        args.extend(build_spatial_args(method, meta))
    elif method_family.startswith("FLOWCACHE_"):
        flowcache_args, profile_spec = build_flowcache_spec(method_family, meta)
        args.extend(flowcache_args)
    return args, profile_spec


def build_manifest_record(row: dict[str, Any]) -> dict[str, Any]:
    payload = parse_config_payload(row["config_payload"])
    meta = payload.get("quant_meta", {}) or {}
    passthrough_args, profile_spec = build_passthrough(row["method"], row["method_family"], meta)
    run_labels = split_labels(row.get("run_labels"))
    if row["method_family"] == "FLOWCACHE_HYBRID" and not profile_spec:
        if any(
            token in label
            for label in run_labels
            for token in (
                "flowcache_adaptive",
                "flowcache_prune",
                "flowcache_soft_prune",
                "flowcache_native",
            )
        ):
            profile_spec = {
                "recent_ratio": as_float(
                    meta.get("flowcache_config_chunk_recent_ratio", GENERATION_DEFAULTS["flowcache_recent_ratio"])
                ),
                "min_scale": FLOWCACHE_PROFILE_DEFAULT_MIN,
                "max_scale": FLOWCACHE_PROFILE_DEFAULT_MAX,
            }
    method_slug = slugify(row["method"])
    run_name = f"backfill_{method_slug}_{row['config_id']}_10s"
    return {
        "primary_config_id": row["config_id"],
        "config_ids": [row["config_id"]],
        "method": row["method"],
        "method_family": row["method_family"],
        "sources": sorted({item.strip() for item in str(row.get("sources", "")).split(",") if item.strip()}),
        "run_labels": run_labels,
        "run_name": run_name,
        "run_root": f"results/runs/{run_name}",
        "passthrough_args": passthrough_args,
        "profile_flowcache": bool(profile_spec),
        "flowcache_profile_recent_ratio": profile_spec["recent_ratio"] if profile_spec else None,
        "flowcache_profile_min_scale": profile_spec["min_scale"] if profile_spec else None,
        "flowcache_profile_max_scale": profile_spec["max_scale"] if profile_spec else None,
        "dashboard_ready_runs": int(row.get("dashboard_ready_runs") or 0),
        "notes": [],
    }


def signature_for_record(record: dict[str, Any]) -> str:
    payload = {
        "method": record["method"],
        "method_family": record["method_family"],
        "passthrough_args": record["passthrough_args"],
        "profile_flowcache": record["profile_flowcache"],
        "flowcache_profile_recent_ratio": record["flowcache_profile_recent_ratio"],
        "flowcache_profile_min_scale": record["flowcache_profile_min_scale"],
        "flowcache_profile_max_scale": record["flowcache_profile_max_scale"],
    }
    return json.dumps(payload, sort_keys=True)


def build_command(record: dict[str, Any]) -> list[str]:
    cmd = [
        "python3",
        "scripts/23_run_moviegen_backfill.py",
        "--run-root",
        record["run_root"],
        "--run-name",
        record["run_name"],
        "--method",
        record["method"],
        "--config-id",
        record["primary_config_id"],
    ]
    if record["profile_flowcache"]:
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


def should_keep(row: dict[str, Any]) -> tuple[bool, str | None]:
    if row.get("benchmark") != "moviegen":
        return False, "non-moviegen benchmark"
    if not row.get("needs_moviegen_10s_backfill"):
        return False, "already has 10s coverage"
    if int(row.get("dashboard_ready_runs") or 0) <= 0:
        return False, "no dashboard-ready short baseline"
    return True, None


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
        "sources",
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
                    "config_ids": ",".join(row["config_ids"]),
                    "sources": ",".join(row["sources"]),
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
        lines.append(f"# {row['method']} :: {','.join(row['config_ids'])}")
        lines.append(shlex.join(build_command(row)))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deduplicated 10-second MovieGen backfill manifest.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_json(args.registry)
    manifest_by_signature: dict[str, dict[str, Any]] = {}
    skipped: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        keep, reason = should_keep(row)
        if not keep:
            skipped[reason or "skipped"].append(row["config_id"])
            continue
        record = build_manifest_record(row)
        signature = signature_for_record(record)
        if signature in manifest_by_signature:
            existing = manifest_by_signature[signature]
            existing["config_ids"].append(row["config_id"])
            existing["config_ids"] = sorted(set(existing["config_ids"]))
            existing["run_labels"] = sorted(set(existing["run_labels"] + record["run_labels"]))
            existing["sources"] = sorted(set(existing["sources"] + record["sources"]))
            existing["dashboard_ready_runs"] = max(existing["dashboard_ready_runs"], record["dashboard_ready_runs"])
            existing["notes"].append(f"deduped {row['config_id']} into {existing['primary_config_id']}")
            continue
        manifest_by_signature[signature] = record

    manifest_rows = sorted(
        manifest_by_signature.values(),
        key=lambda row: (row["method_family"], row["method"], row["primary_config_id"]),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = output_dir / "backfill_manifest.json"
    manifest_csv = output_dir / "backfill_manifest.csv"
    manifest_sh = output_dir / "backfill_commands.sh"
    skipped_json = output_dir / "backfill_skipped.json"

    for row in manifest_rows:
        row["command"] = build_command(row)

    write_json(manifest_json, manifest_rows)
    write_csv(manifest_csv, manifest_rows)
    write_shell_script(manifest_sh, manifest_rows)
    write_json(skipped_json, skipped)

    print(f"Wrote {manifest_json}")
    print(f"Wrote {manifest_csv}")
    print(f"Wrote {manifest_sh}")
    print(f"Wrote {skipped_json}")
    print(f"Backfill commands: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
