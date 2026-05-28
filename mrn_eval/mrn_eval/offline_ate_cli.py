#!/usr/bin/env python3
"""CLI wrapper around ``mrn_eval.offline_ate`` for post-hoc bag analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from mrn_eval.offline_ate import (
    AlignmentResult,
    ErrorStats,
    compute_ate,
    compute_rpe,
    load_trajectory_csv,
    stats_to_dict,
    time_align,
)


def _format_float(value: float | None) -> str:
    if value is None:
        return "nan"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.4f}"


def _build_metrics(
    *,
    estimated_path: Path,
    truth_path: Path,
    estimated_label: str,
    truth_label: str,
    alignment: AlignmentResult,
    ate: ErrorStats,
    rpe_results: dict[float, ErrorStats],
    max_offset_sec: float,
) -> dict:
    return {
        "estimated": {
            "label": estimated_label,
            "path": str(estimated_path),
            "sample_count": alignment.estimated_count,
        },
        "truth": {
            "label": truth_label,
            "path": str(truth_path),
            "sample_count": alignment.truth_count,
        },
        "alignment": {
            "max_offset_tolerance_sec": max_offset_sec,
            "matched_count": alignment.matched_count,
            "dropped_count": alignment.dropped_count,
            "mean_time_offset_sec": alignment.mean_time_offset_sec,
            "max_time_offset_sec": alignment.max_time_offset_sec,
        },
        "ate": stats_to_dict(ate),
        "rpe": {
            f"{delta:g}s": stats_to_dict(stats)
            for delta, stats in sorted(rpe_results.items())
        },
    }


def _build_report(metrics: dict) -> str:
    lines = ["# Offline ATE Report", ""]
    est = metrics["estimated"]
    truth = metrics["truth"]
    align = metrics["alignment"]
    lines.append(
        f"- Estimated: `{est['label']}` "
        f"({est['sample_count']} samples) — `{est['path']}`"
    )
    lines.append(
        f"- Truth: `{truth['label']}` "
        f"({truth['sample_count']} samples) — `{truth['path']}`"
    )
    lines.append(
        f"- Alignment: matched={align['matched_count']}, "
        f"dropped={align['dropped_count']}, "
        f"mean_offset={_format_float(align['mean_time_offset_sec'])}s, "
        f"max_offset={_format_float(align['max_time_offset_sec'])}s "
        f"(tolerance {_format_float(align['max_offset_tolerance_sec'])}s)"
    )
    lines.append("")
    lines.append("## Absolute Trajectory Error")
    lines.append("")
    lines.append("| Metric | Value [m] |")
    lines.append("| --- | ---: |")
    ate = metrics["ate"]
    for key in ("rmse", "mean", "stddev", "max"):
        lines.append(f"| {key} | {_format_float(ate.get(key))} |")
    lines.append(f"| count | {ate['count']} |")
    lines.append("")
    if metrics["rpe"]:
        lines.append("## Relative Pose Error")
        lines.append("")
        lines.append("| Delta | RMSE [m] | Mean [m] | Stddev [m] | Max [m] | Pairs |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for delta_key, stats in metrics["rpe"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        delta_key,
                        _format_float(stats["rmse"]),
                        _format_float(stats["mean"]),
                        _format_float(stats["stddev"]),
                        _format_float(stats["max"]),
                        str(stats["count"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mrn_eval_offline_ate",
        description=(
            "Compute ATE and RPE between an estimated trajectory and a truth "
            "trajectory (both CSV with stamp_sec,x,y[,z])."
        ),
    )
    parser.add_argument("--estimated", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument(
        "--max-offset-sec",
        type=float,
        default=0.05,
        help="Maximum |estimated.stamp - truth.stamp| during alignment (default 0.05).",
    )
    parser.add_argument(
        "--rpe-delta-sec",
        type=float,
        action="append",
        default=None,
        help="Time delta(s) for RPE evaluation (repeatable). Default: [1.0].",
    )
    parser.add_argument("--estimated-label", default="estimated")
    parser.add_argument("--truth-label", default="truth")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write metrics.json and report.md. Stdout-only when omitted.",
    )
    args = parser.parse_args(argv)

    try:
        estimated = load_trajectory_csv(args.estimated)
        truth = load_trajectory_csv(args.truth)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not estimated:
        print(f"error: {args.estimated} contains no samples", file=sys.stderr)
        return 2
    if not truth:
        print(f"error: {args.truth} contains no samples", file=sys.stderr)
        return 2

    alignment = time_align(estimated, truth, max_offset_sec=args.max_offset_sec)
    if alignment.matched_count == 0:
        print(
            "error: no samples matched within "
            f"max-offset-sec={args.max_offset_sec} "
            f"(estimated={len(estimated)}, truth={len(truth)})",
            file=sys.stderr,
        )
        return 3

    ate = compute_ate(alignment.pairs)
    deltas = args.rpe_delta_sec if args.rpe_delta_sec else [1.0]
    rpe_results: dict[float, ErrorStats] = {}
    for delta in deltas:
        rpe_results[delta] = compute_rpe(alignment.pairs, delta_sec=delta)

    metrics = _build_metrics(
        estimated_path=args.estimated,
        truth_path=args.truth,
        estimated_label=args.estimated_label,
        truth_label=args.truth_label,
        alignment=alignment,
        ate=ate,
        rpe_results=rpe_results,
        max_offset_sec=args.max_offset_sec,
    )
    report = _build_report(metrics)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
