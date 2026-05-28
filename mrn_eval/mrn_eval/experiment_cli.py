#!/usr/bin/env python3
"""Run replayable MRN experiments from YAML configs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.util
import json
import os
import platform
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

import yaml


@dataclass(frozen=True)
class MethodPlan:
    name: str
    config_path: Path | None
    graph_executable: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config_path": str(self.config_path) if self.config_path else None,
            "graph_executable": self.graph_executable,
        }


@dataclass(frozen=True)
class SweepCase:
    name: str
    parameter: str
    value: Any
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameter": self.parameter,
            "value": self.value,
            "overrides": self.overrides,
        }


@dataclass(frozen=True)
class BagReplayPlan:
    directory: Path
    manifest_path: Path | None
    play_rate: float
    storage: str
    agent_ids: tuple[str, ...]
    enable_online_ate: bool
    extra_play_args: tuple[str, ...]
    validation_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "play_rate": self.play_rate,
            "storage": self.storage,
            "agent_ids": list(self.agent_ids),
            "enable_online_ate": self.enable_online_ate,
            "extra_play_args": list(self.extra_play_args),
            "validation_summary": dict(self.validation_summary),
        }

    def launch_args(self) -> list[str]:
        args = [
            f"bag_dir:={self.directory}",
            f"play_rate:={self.play_rate}",
            f"storage:={self.storage}",
            f"agent_ids:={','.join(self.agent_ids)}",
            f"enable_online_ate:={'true' if self.enable_online_ate else 'false'}",
        ]
        if self.extra_play_args:
            args.append(f"extra_play_args:={' '.join(self.extra_play_args)}")
        return args


@dataclass(frozen=True)
class ExperimentPlan:
    config_path: Path
    name: str
    seed: int | None
    launch_file: str
    scenario: str
    scenario_path: Path
    network_profile: Path | None
    methods: list[MethodPlan]
    sweep_cases: list[SweepCase]
    output_dir: Path
    report_path: Path
    metrics_path: Path
    acceptance_path: Path
    provenance_path: Path
    duration_sec: float
    ros_domain_id: int | None
    bag_replay: BagReplayPlan | None = None

    @property
    def launch_args(self) -> list[str]:
        if self.bag_replay is not None:
            return self.bag_replay.launch_args()
        args = [f"scenario:={self.scenario}"]
        if self.network_profile is not None:
            args.append(f"network_profile:={self.network_profile}")
        return args

    def launch_args_for(
        self,
        method: MethodPlan | None = None,
        scenario_path: Path | None = None,
    ) -> list[str]:
        args = list(self.launch_args)
        if self.bag_replay is None and scenario_path is not None:
            args.append(f"scenario_path:={scenario_path}")
        if method is not None and method.graph_executable:
            args.append(f"graph_executable:={method.graph_executable}")
        return args

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "name": self.name,
            "seed": self.seed,
            "launch_file": self.launch_file,
            "launch_args": self.launch_args,
            "scenario": self.scenario,
            "scenario_path": str(self.scenario_path),
            "network_profile": str(self.network_profile) if self.network_profile else None,
            "methods": [method.to_dict() for method in self.methods],
            "sweep_cases": [case.to_dict() for case in self.sweep_cases],
            "output_dir": str(self.output_dir),
            "report_path": str(self.report_path),
            "metrics_path": str(self.metrics_path),
            "acceptance_path": str(self.acceptance_path),
            "provenance_path": str(self.provenance_path),
            "duration_sec": self.duration_sec,
            "ros_domain_id": self.ros_domain_id,
            "bag_replay": self.bag_replay.to_dict() if self.bag_replay else None,
        }


def load_experiment_plan(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    duration_sec: float = 45.0,
    ros_domain_id: int | None = None,
    sweep_case_names: list[str] | None = None,
) -> ExperimentPlan:
    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")

    path = _resolve_existing_path(config_path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"experiment config must be a mapping: {path}")

    experiment = _mapping(config.get("experiment", {}), "experiment")
    name = str(experiment.get("name") or path.stem)
    seed = experiment.get("seed")
    seed = int(seed) if seed is not None else None

    input_config = _mapping(config.get("input", {}), "input")
    scenario_value = str(input_config.get("scenario", "gnss_outage_3robots.yaml"))
    scenario = Path(scenario_value).name
    scenario_path = _resolve_scenario_path(scenario_value, base_dir=path.parent)

    network_config = config.get("network", {})
    network_profile = None
    if network_config:
        network_profile = _resolve_path(
            str(_mapping(network_config, "network").get("profile", "")),
            base_dir=path.parent,
        )

    bag_replay = _load_bag_replay_plan(config, base_dir=path.parent)

    if bag_replay is not None:
        if network_profile is not None:
            raise ValueError(
                "experiment cannot combine 'bag' replay with 'network' profile injection"
            )
        launch_file = "bag_replay.launch.py"
    elif network_profile is not None:
        launch_file = "gnss_outage_packet_loss.launch.py"
    else:
        launch_file = "cooperative_localization.launch.py"
    methods = _load_method_plans(config, base_dir=path.parent)
    sweep_cases = _filter_sweep_cases(_load_sweep_cases(config), sweep_case_names)
    if bag_replay is not None and sweep_cases:
        raise ValueError(
            "experiment cannot combine 'bag' replay with sweep cases in this release"
        )
    out_dir = Path(output_dir) if output_dir is not None else Path("out") / "experiments" / name
    out_dir = out_dir.resolve()

    return ExperimentPlan(
        config_path=path,
        name=name,
        seed=seed,
        launch_file=launch_file,
        scenario=scenario,
        scenario_path=scenario_path,
        network_profile=network_profile,
        methods=methods,
        sweep_cases=sweep_cases,
        output_dir=out_dir,
        report_path=out_dir / "report.md",
        metrics_path=out_dir / "metrics.json",
        acceptance_path=out_dir / "acceptance.json",
        provenance_path=out_dir / "provenance.json",
        duration_sec=duration_sec,
        ros_domain_id=ros_domain_id,
        bag_replay=bag_replay,
    )


def _load_bag_replay_plan(
    config: dict[str, Any], base_dir: Path
) -> BagReplayPlan | None:
    raw = config.get("bag")
    if raw is None:
        return None
    bag = _mapping(raw, "bag")
    directory_value = str(bag.get("directory") or bag.get("path") or "").strip()
    if not directory_value:
        raise ValueError("bag.directory is required when 'bag' block is present")
    directory = _resolve_existing_dir(directory_value, base_dir=base_dir)

    manifest_value = bag.get("manifest")
    manifest_path: Path | None = None
    if manifest_value:
        manifest_path = _resolve_path(str(manifest_value), base_dir=base_dir)
        if manifest_path is None or not manifest_path.is_file():
            raise ValueError(f"bag.manifest does not exist: {manifest_value}")

    play_rate = float(bag.get("play_rate", 1.0))
    if play_rate <= 0.0:
        raise ValueError("bag.play_rate must be positive")
    storage = str(bag.get("storage", "mcap")).strip() or "mcap"

    raw_agents = bag.get("agent_ids", ["robot_1", "robot_2"])
    if isinstance(raw_agents, str):
        raw_agents = [item.strip() for item in raw_agents.split(",") if item.strip()]
    if not isinstance(raw_agents, list) or not raw_agents:
        raise ValueError("bag.agent_ids must be a non-empty list")
    agent_ids = tuple(str(item) for item in raw_agents)

    enable_online_ate = bool(bag.get("enable_online_ate", False))

    raw_extra = bag.get("extra_play_args", [])
    if isinstance(raw_extra, str):
        raw_extra = raw_extra.split()
    if not isinstance(raw_extra, list):
        raise ValueError("bag.extra_play_args must be a list or string")
    extra_play_args = tuple(str(item) for item in raw_extra)

    validation_summary: dict[str, Any] = {}
    if manifest_path is not None:
        validation_summary = _validate_bag_directory(directory, manifest_path)

    return BagReplayPlan(
        directory=directory,
        manifest_path=manifest_path,
        play_rate=play_rate,
        storage=storage,
        agent_ids=agent_ids,
        enable_online_ate=enable_online_ate,
        extra_play_args=extra_play_args,
        validation_summary=validation_summary,
    )


def _validate_bag_directory(directory: Path, manifest_path: Path) -> dict[str, Any]:
    validator = _load_validate_bag_module()
    if validator is None:
        return {"skipped": "validate_bag module unavailable"}
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"failed to load manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a mapping: {manifest_path}")
    try:
        return validator.validate_bag(directory, manifest)
    except validator.BagValidationError as error:
        raise ValueError(f"bag validation failed for {directory}: {error}") from error


def _load_validate_bag_module():
    candidates = [
        Path.cwd() / "tools" / "validate_bag.py",
        Path(__file__).resolve().parents[2] / "tools" / "validate_bag.py",
        Path(__file__).resolve().parents[3] / "tools" / "validate_bag.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "_mrn_validate_bag", candidate
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


def _resolve_existing_dir(value: str, base_dir: Path) -> Path:
    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([
            Path.cwd() / path,
            base_dir / path,
            base_dir.parent / path,
        ])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise ValueError(f"bag.directory does not exist: {value}")


def run_experiment(plan: ExperimentPlan) -> None:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    (plan.output_dir / "plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(plan.config_path, plan.output_dir / "experiment.yaml")

    env = os.environ.copy()
    if plan.ros_domain_id is not None:
        env["ROS_DOMAIN_ID"] = str(plan.ros_domain_id)

    methods = plan.methods or [MethodPlan(name="default", config_path=None, graph_executable=None)]
    method_results = []
    if plan.sweep_cases:
        for sweep_index, sweep_case in enumerate(plan.sweep_cases):
            scenario_path = _write_sweep_scenario(plan, sweep_case)
            for method_index, method in enumerate(methods):
                domain_index = sweep_index * len(methods) + method_index
                method_results.append(
                    _run_method(
                        plan,
                        method,
                        index=domain_index,
                        env=env,
                        sweep_case=sweep_case,
                        scenario_path=scenario_path,
                    )
                )
    else:
        for index, method in enumerate(methods):
            method_results.append(_run_method(plan, method, index=index, env=env))

    metrics = _aggregate_method_metrics(plan, method_results)
    config = _load_yaml_mapping(plan.config_path, "experiment config")
    acceptance = evaluate_acceptance(config, metrics)
    plan.metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan.acceptance_path.write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan.report_path.write_text(
        _format_aggregate_report(plan, method_results, metrics, acceptance=acceptance),
        encoding="utf-8",
    )
    _write_provenance_outputs(plan, metrics, acceptance)

    if not plan.report_path.exists() or not plan.metrics_path.exists():
        raise RuntimeError(f"experiment did not produce report outputs in {plan.output_dir}")

    if not acceptance["passed"]:
        raise RuntimeError(f"experiment acceptance failed: {plan.acceptance_path}")


def evaluate_acceptance(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    acceptance = _mapping(config.get("acceptance", {}), "acceptance")
    checks: list[dict[str, Any]] = []
    if not acceptance:
        return {"passed": True, "checks": checks}

    localization_checks = acceptance.get("localization", [])
    if localization_checks is None:
        localization_checks = []
    if not isinstance(localization_checks, list):
        raise ValueError("acceptance.localization must be a list")
    rows = metrics.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("metrics.rows must be a list")
    for index, rule in enumerate(localization_checks):
        rule = _mapping(rule, f"acceptance.localization[{index}]")
        agent_id = str(rule.get("agent_id", ""))
        method = str(rule.get("method", ""))
        method_run = str(rule.get("method_run", ""))
        candidate_rows = metrics.get("method_rows", rows) if method_run else rows
        row = _find_metric_row(candidate_rows, agent_id=agent_id, method=method, method_run=method_run)
        prefix = (
            f"localization[{method_run}:{agent_id}/{method}]"
            if method_run
            else f"localization[{agent_id}/{method}]"
        )
        if row is None:
            checks.append(_check_result(f"{prefix}.exists", False, None, "row exists"))
            continue
        checks.append(_check_result(f"{prefix}.exists", True, True, "row exists"))
        if "min_improvement_vs_local" in rule:
            checks.append(
                _threshold_check(
                    f"{prefix}.improvement_vs_local",
                    row.get("improvement_vs_local"),
                    float(rule["min_improvement_vs_local"]),
                    ">=",
                )
            )
        if "max_ate_rmse" in rule:
            checks.append(
                _threshold_check(
                    f"{prefix}.ate_rmse",
                    row.get("ate_rmse"),
                    float(rule["max_ate_rmse"]),
                    "<=",
                )
            )
        if "min_availability" in rule:
            checks.append(
                _threshold_check(
                    f"{prefix}.localization_availability",
                    row.get("localization_availability"),
                    float(rule["min_availability"]),
                    ">=",
                )
            )
        if "max_ate_rmse_ratio_vs_method" in rule:
            vs_method_run = str(rule.get("vs_method_run", ""))
            other_row = _find_metric_row(
                candidate_rows,
                agent_id=agent_id,
                method=method,
                method_run=vs_method_run,
            )
            ratio_limit = float(rule["max_ate_rmse_ratio_vs_method"])
            this_ate = row.get("ate_rmse")
            other_ate = other_row.get("ate_rmse") if other_row else None
            actual_ratio: float | None = None
            if (
                this_ate is not None
                and other_ate is not None
                and float(other_ate) > 0.0
            ):
                actual_ratio = float(this_ate) / float(other_ate)
            checks.append(
                _threshold_check(
                    f"{prefix}.ate_rmse_ratio_vs[{vs_method_run}]",
                    actual_ratio,
                    ratio_limit,
                    "<=",
                )
            )

    selected_sweep_cases = _selected_sweep_case_names(metrics)
    network_rules = acceptance.get("network")
    if network_rules is not None:
        if isinstance(network_rules, dict):
            network_rules = [network_rules]
        if not isinstance(network_rules, list):
            raise ValueError("acceptance.network must be a mapping or list")
        primary_network_rows = metrics.get("network_rows", [])
        method_network_rows = metrics.get("method_network_rows", primary_network_rows)
        if not isinstance(primary_network_rows, list):
            raise ValueError("metrics.network_rows must be a list")
        if not isinstance(method_network_rows, list):
            raise ValueError("metrics.method_network_rows must be a list")
        for index, rule_value in enumerate(network_rules):
            network_rule = _mapping(rule_value, f"acceptance.network[{index}]")
            method_run = str(network_rule.get("method_run", ""))
            sweep_case = str(network_rule.get("sweep_case", ""))
            qos_profile_name = str(network_rule.get("qos_profile_name", ""))
            if sweep_case and selected_sweep_cases and sweep_case not in selected_sweep_cases:
                continue
            candidate_rows = (
                method_network_rows
                if method_run or sweep_case
                else primary_network_rows
            )
            network_rows = _filter_network_rows(
                candidate_rows,
                method_run=method_run,
                sweep_case=sweep_case,
                qos_profile_name=qos_profile_name,
            )
            prefix = _network_acceptance_prefix(
                method_run=method_run,
                sweep_case=sweep_case,
                qos_profile_name=qos_profile_name,
            )
            if "min_rows" in network_rule:
                checks.append(
                    _threshold_check(
                        f"{prefix}.rows",
                        len(network_rows),
                        int(network_rule["min_rows"]),
                        ">=",
                    )
                )
            if "min_observed_loss_rate" in network_rule:
                max_loss_rate = max(
                    (float(row.get("loss_rate") or 0.0) for row in network_rows),
                    default=0.0,
                )
                checks.append(
                    _threshold_check(
                        f"{prefix}.max_observed_loss_rate",
                        max_loss_rate,
                        float(network_rule["min_observed_loss_rate"]),
                        ">=",
                    )
                )
            if "max_observed_loss_rate" in network_rule:
                max_loss_rate = max(
                    (float(row.get("loss_rate") or 0.0) for row in network_rows),
                    default=0.0,
                )
                checks.append(
                    _threshold_check(
                        f"{prefix}.max_observed_loss_rate",
                        max_loss_rate,
                        float(network_rule["max_observed_loss_rate"]),
                        "<=",
                    )
                )
            if "max_mean_latency_sec" in network_rule:
                max_mean_latency = max(
                    (float(row.get("latency_mean_sec") or 0.0) for row in network_rows),
                    default=0.0,
                )
                checks.append(
                    _threshold_check(
                        f"{prefix}.max_mean_latency_sec",
                        max_mean_latency,
                        float(network_rule["max_mean_latency_sec"]),
                        "<=",
                    )
                )
            if "min_mean_latency_sec" in network_rule:
                min_mean_latency = min(
                    (float(row.get("latency_mean_sec") or 0.0) for row in network_rows),
                    default=0.0,
                )
                checks.append(
                    _threshold_check(
                        f"{prefix}.min_mean_latency_sec",
                        min_mean_latency,
                        float(network_rule["min_mean_latency_sec"]),
                        ">=",
                    )
                )

    graph_checks = acceptance.get("graph", [])
    if graph_checks is None:
        graph_checks = []
    if not isinstance(graph_checks, list):
        raise ValueError("acceptance.graph must be a list")
    graph_rows = metrics.get("graph_rows", [])
    method_graph_rows = metrics.get("method_graph_rows", graph_rows)
    if not isinstance(graph_rows, list):
        raise ValueError("metrics.graph_rows must be a list")
    if not isinstance(method_graph_rows, list):
        raise ValueError("metrics.method_graph_rows must be a list")
    for index, rule in enumerate(graph_checks):
        rule = _mapping(rule, f"acceptance.graph[{index}]")
        method_run = str(rule.get("method_run", ""))
        sweep_case = str(rule.get("sweep_case", ""))
        if sweep_case and selected_sweep_cases and sweep_case not in selected_sweep_cases:
            continue
        backend = str(rule.get("backend", rule.get("backend_name", "")))
        candidate_rows = method_graph_rows if method_run or sweep_case else graph_rows
        row = _find_graph_row(
            candidate_rows,
            method_run=method_run,
            sweep_case=sweep_case,
            backend=backend,
        )
        prefix = _graph_acceptance_prefix(method_run=method_run, sweep_case=sweep_case, backend=backend)
        if row is None:
            checks.append(_check_result(f"{prefix}.exists", False, None, "row exists"))
            continue
        checks.append(_check_result(f"{prefix}.exists", True, True, "row exists"))
        if "min_accepted_constraints" in rule:
            checks.append(
                _threshold_check(
                    f"{prefix}.accepted_constraint_count",
                    row.get("accepted_constraint_count"),
                    int(rule["min_accepted_constraints"]),
                    ">=",
                )
            )
        if "min_rejected_constraints" in rule:
            checks.append(
                _threshold_check(
                    f"{prefix}.rejected_constraint_count",
                    row.get("rejected_constraint_count"),
                    int(rule["min_rejected_constraints"]),
                    ">=",
                )
            )
        if "max_rejected_constraints" in rule:
            checks.append(
                _threshold_check(
                    f"{prefix}.rejected_constraint_count",
                    row.get("rejected_constraint_count"),
                    int(rule["max_rejected_constraints"]),
                    "<=",
                )
            )
        if "max_stale_constraints" in rule:
            checks.append(
                _threshold_check(
                    f"{prefix}.stale_constraint_count",
                    row.get("stale_constraint_count"),
                    int(rule["max_stale_constraints"]),
                    "<=",
                )
            )
        min_rejection_reasons = rule.get("min_rejection_reasons", {})
        if min_rejection_reasons is None:
            min_rejection_reasons = {}
        if not isinstance(min_rejection_reasons, dict):
            raise ValueError(f"acceptance.graph[{index}].min_rejection_reasons must be a mapping")
        rejection_reasons = row.get("rejection_reasons", {})
        if not isinstance(rejection_reasons, dict):
            rejection_reasons = {}
        for reason, expected_count in min_rejection_reasons.items():
            checks.append(
                _threshold_check(
                    f"{prefix}.rejection_reasons[{reason}]",
                    rejection_reasons.get(reason, 0),
                    int(expected_count),
                    ">=",
                )
            )

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mrn_experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["plan", "run"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("config")
        sub.add_argument("--duration", type=float, default=45.0)
        sub.add_argument("--output-dir", default="")
        sub.add_argument("--ros-domain-id", type=int, default=None)
        sub.add_argument(
            "--sweep-case",
            action="append",
            default=None,
            help="Run only the named sweep case. Repeat to select multiple cases.",
        )

    args = parser.parse_args(argv)
    plan = load_experiment_plan(
        args.config,
        output_dir=args.output_dir or None,
        duration_sec=args.duration,
        ros_domain_id=args.ros_domain_id,
        sweep_case_names=args.sweep_case,
    )
    if args.command == "plan":
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return

    run_experiment(plan)
    print(f"wrote {plan.report_path}")
    print(f"wrote {plan.metrics_path}")
    print(f"wrote {plan.acceptance_path}")
    print(f"wrote {plan.provenance_path}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load_method_plans(config: dict[str, Any], base_dir: Path) -> list[MethodPlan]:
    method_entries = config.get("methods", [])
    if method_entries is None:
        return []
    if not isinstance(method_entries, list):
        raise ValueError("methods must be a list")
    methods = []
    for index, entry in enumerate(method_entries):
        entry = _mapping(entry, f"methods[{index}]")
        name = str(entry.get("name") or f"method_{index}")
        config_path = _resolve_path(str(entry.get("config", "")), base_dir=base_dir)
        graph_executable = entry.get("graph_executable")
        if graph_executable is None:
            graph_executable = _graph_executable_from_config(config_path)
        methods.append(
            MethodPlan(
                name=name,
                config_path=config_path,
                graph_executable=str(graph_executable) if graph_executable else None,
            )
        )
    return methods


def _load_sweep_cases(config: dict[str, Any]) -> list[SweepCase]:
    sweep_entries = config.get("sweeps")
    if sweep_entries is None and "sweep" in config:
        sweep_entries = [config["sweep"]]
    if sweep_entries is None and "clock_drift_ms" in config:
        sweep_entries = [
            {
                "name": "clock_drift_ms",
                "parameter": "faults.clock_drift_ms",
                "values": config["clock_drift_ms"],
            }
        ]
    if sweep_entries is None:
        return []
    if not isinstance(sweep_entries, list):
        raise ValueError("sweeps must be a list")

    cases = []
    for index, entry in enumerate(sweep_entries):
        entry = _mapping(entry, f"sweeps[{index}]")
        case_entries = entry.get("cases")
        if case_entries is not None:
            if not isinstance(case_entries, list) or not case_entries:
                raise ValueError(f"sweeps[{index}].cases must be a non-empty list")
            prefix = str(entry.get("name") or "case")
            parameter = str(entry.get("parameter") or entry.get("path") or prefix)
            for case_index, case_entry in enumerate(case_entries):
                case_entry = _mapping(case_entry, f"sweeps[{index}].cases[{case_index}]")
                overrides = case_entry.get("values", case_entry.get("overrides"))
                if not isinstance(overrides, dict) or not overrides:
                    raise ValueError(
                        f"sweeps[{index}].cases[{case_index}].values must be a non-empty mapping"
                    )
                label = str(case_entry.get("name") or case_entry.get("value") or case_index)
                slug = _slug_value(label)
                case_name = label if label.startswith(f"{prefix}_") else f"{prefix}_{slug}"
                cases.append(
                    SweepCase(
                        name=case_name,
                        parameter=parameter,
                        value=case_entry.get("value", label),
                        overrides=dict(overrides),
                    )
                )
            continue

        parameter = str(entry.get("parameter") or entry.get("path") or "")
        if not parameter:
            raise ValueError(f"sweeps[{index}].parameter is required")
        values = entry.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweeps[{index}].values must be a non-empty list")
        prefix = str(entry.get("name") or parameter.replace(".", "_"))
        for value in values:
            case_name = f"{prefix}_{_slug_value(value)}"
            cases.append(
                SweepCase(
                    name=case_name,
                    parameter=parameter,
                    value=value,
                    overrides={parameter: value},
                )
            )
    return cases


def _filter_sweep_cases(
    sweep_cases: list[SweepCase],
    selected_names: list[str] | None,
) -> list[SweepCase]:
    if not selected_names:
        return sweep_cases
    if not sweep_cases:
        raise ValueError("--sweep-case was provided, but experiment defines no sweep cases")
    selected = set(selected_names)
    available = {case.name for case in sweep_cases}
    unknown = sorted(selected - available)
    if unknown:
        available_text = ", ".join(sorted(available))
        raise ValueError(
            f"unknown sweep case(s): {', '.join(unknown)}; available: {available_text}"
        )
    return [case for case in sweep_cases if case.name in selected]


def _graph_executable_from_config(config_path: Path | None) -> str | None:
    if config_path is None or not config_path.exists():
        return None
    config = _load_yaml_mapping(config_path, f"method config {config_path}")
    backend = _mapping(config.get("backend", {}), f"backend in {config_path}")
    backend_name = str(backend.get("name", ""))
    if backend_name == "dummy":
        return "dummy_graph_node.py"
    if backend_name in {"relative_anchor", "gtsam_fixed_lag"}:
        return "relative_anchor_graph_node.py"
    return None


def _run_method(
    plan: ExperimentPlan,
    method: MethodPlan,
    index: int,
    env: dict[str, str],
    sweep_case: SweepCase | None = None,
    scenario_path: Path | None = None,
) -> dict[str, Any]:
    if sweep_case is not None:
        method_dir = plan.output_dir / "sweeps" / sweep_case.name / "methods" / method.name
    else:
        method_dir = plan.output_dir / "methods" / method.name
    method_dir.mkdir(parents=True, exist_ok=True)
    report_path = method_dir / "report.md"
    metrics_path = method_dir / "metrics.json"
    launch_log_path = method_dir / "launch.log"
    method_plan_path = method_dir / "method.json"
    method_plan_path.write_text(
        json.dumps(method.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    method_env = dict(env)
    if plan.ros_domain_id is not None:
        method_env["ROS_DOMAIN_ID"] = str(plan.ros_domain_id + index)

    with launch_log_path.open("w", encoding="utf-8") as launch_log:
        launch_cmd = [
            "ros2",
            "launch",
            "mrn_demos",
            plan.launch_file,
            *plan.launch_args_for(method, scenario_path=scenario_path),
        ]
        launch_process = subprocess.Popen(
            launch_cmd,
            stdout=launch_log,
            stderr=subprocess.STDOUT,
            env=method_env,
            start_new_session=True,
        )
        try:
            time.sleep(3.0)
            report_cmd = [
                "ros2",
                "run",
                "mrn_eval",
                "mrn_report",
                "--duration",
                f"{plan.duration_sec}",
                "--output",
                str(report_path),
                "--json-output",
                str(metrics_path),
                "--title",
                f"MRN Benchmark Report: {method.name}",
            ]
            subprocess.run(report_cmd, check=True, env=method_env)
        finally:
            _terminate_process_group(launch_process)

    if not report_path.exists() or not metrics_path.exists():
        raise RuntimeError(f"method did not produce report outputs: {method.name}")

    result = {
        "name": method.name,
        "config_path": str(method.config_path) if method.config_path else None,
        "graph_executable": method.graph_executable,
        "report_path": str(report_path),
        "metrics_path": str(metrics_path),
        "launch_log_path": str(launch_log_path),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
    }
    if sweep_case is not None:
        result["sweep_case"] = sweep_case.to_dict()
        result["scenario_path"] = str(scenario_path) if scenario_path else None
    return result


def _aggregate_method_metrics(
    plan: ExperimentPlan,
    method_results: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = _primary_method_result(method_results)
    primary_metrics = primary["metrics"] if primary else {}
    method_rows = []
    method_network_rows = []
    method_graph_rows = []
    for result in method_results:
        metrics = result["metrics"]
        sweep_metadata = result.get("sweep_case")
        for row in metrics.get("rows", []):
            row = dict(row)
            row["method_run"] = result["name"]
            if sweep_metadata:
                row.update(_sweep_row_metadata(sweep_metadata))
            method_rows.append(row)
        for row in metrics.get("network_rows", []):
            row = dict(row)
            row["method_run"] = result["name"]
            if sweep_metadata:
                row.update(_sweep_row_metadata(sweep_metadata))
            method_network_rows.append(row)
        for row in metrics.get("graph_rows", []):
            row = dict(row)
            row["method_run"] = result["name"]
            if sweep_metadata:
                row.update(_sweep_row_metadata(sweep_metadata))
            method_graph_rows.append(row)

    return {
        "title": "MRN Experiment Metrics",
        "experiment": plan.name,
        "duration_sec": plan.duration_sec,
        "primary_method": primary["name"] if primary else None,
        "sweep_cases": [case.to_dict() for case in plan.sweep_cases],
        "experiments": sorted(
            {
                experiment
                for result in method_results
                for experiment in result["metrics"].get("experiments", [])
            }
        ),
        "methods": [
            {
                key: value
                for key, value in result.items()
                if key != "metrics"
            }
            for result in method_results
        ],
        "rows": primary_metrics.get("rows", []),
        "network_rows": primary_metrics.get("network_rows", []),
        "graph_rows": primary_metrics.get("graph_rows", []),
        "method_rows": method_rows,
        "method_network_rows": method_network_rows,
        "method_graph_rows": method_graph_rows,
    }


def _format_aggregate_report(
    plan: ExperimentPlan,
    method_results: list[dict[str, Any]],
    metrics: dict[str, Any],
    acceptance: dict[str, Any] | None = None,
) -> str:
    has_sweeps = any(result.get("sweep_case") for result in method_results)
    lines = [
        "# MRN Experiment Report",
        "",
        f"Experiment: `{plan.name}`",
        f"Duration: `{plan.duration_sec:.1f}s`",
        f"Primary method: `{metrics.get('primary_method')}`",
        "",
    ]
    if acceptance is not None:
        lines.extend(_format_acceptance_section(acceptance))
        lines.append("")

    lines.extend([
        "## Method Runs",
        "",
    ])
    if has_sweeps:
        lines.extend(
            [
                "| Sweep Case | Method | Graph Executable | Report | Metrics |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
    else:
        lines.extend(
            [
                "| Method | Graph Executable | Report | Metrics |",
                "| --- | --- | --- | --- |",
            ]
        )
    for result in method_results:
        row_values = [
            result["name"],
            result.get("graph_executable") or "-",
            result["report_path"],
            result["metrics_path"],
        ]
        if has_sweeps:
            row_values.insert(0, _result_sweep_name(result))
        lines.append(
            "| "
            + " | ".join(row_values)
            + " |"
        )
    lines.append("")

    if plan.sweep_cases:
        lines.extend(
            [
                "## Sweep Cases",
                "",
                "| Case | Parameter | Value |",
                "| --- | --- | ---: |",
            ]
        )
        for sweep_case in plan.sweep_cases:
            lines.append(
                "| "
                + " | ".join(
                    [
                        sweep_case.name,
                        sweep_case.parameter,
                        _format_value(sweep_case.value),
                    ]
                )
                + " |"
            )
        lines.append("")

    method_rows = metrics.get("method_rows", [])
    if method_rows:
        lines.extend(["## Method Comparison", ""])
        if has_sweeps:
            lines.extend(
                [
                    "| Sweep Case | Sweep Value | Method Run | Agent | Output | ATE RMSE [m] | Improvement vs Local [m] | Availability |",
                    "| --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
                ]
            )
        else:
            lines.extend(
                [
                    "| Method Run | Agent | Output | ATE RMSE [m] | Improvement vs Local [m] | Availability |",
                    "| --- | --- | --- | ---: | ---: | ---: |",
                ]
            )
        for row in sorted(
            method_rows,
            key=lambda value: (
                _sort_value(value.get("sweep_value")),
                str(value.get("sweep_case", "")),
                str(value.get("method_run", "")),
                str(value.get("agent_id", "")),
                str(value.get("method", "")),
            ),
        ):
            row_values = [
                str(row.get("method_run", "-")),
                str(row.get("agent_id", "-")),
                str(row.get("method", "-")),
                _format_number(row.get("ate_rmse")),
                _format_number(row.get("improvement_vs_local")),
                _format_number(row.get("localization_availability")),
            ]
            if has_sweeps:
                row_values.insert(0, str(row.get("sweep_case", "-")))
                row_values.insert(1, _format_value(row.get("sweep_value")))
            lines.append("| " + " | ".join(row_values) + " |")
        lines.append("")

    method_network_rows = metrics.get("method_network_rows", [])
    if method_network_rows:
        lines.extend(["## Network Comparison", ""])
        if has_sweeps:
            lines.extend(
                [
                    "| Sweep Case | Sweep Value | Method Run | Link | Loss Rate | Latency Mean [ms] | Jitter [ms] | Max Latency [ms] | Received | Lost | QoS | Transport |",
                    "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
                ]
            )
        else:
            lines.extend(
                [
                    "| Method Run | Link | Loss Rate | Latency Mean [ms] | Jitter [ms] | Max Latency [ms] | Received | Lost | QoS | Transport |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
                ]
            )
        for row in sorted(
            method_network_rows,
            key=lambda value: (
                _sort_value(value.get("sweep_value")),
                str(value.get("sweep_case", "")),
                str(value.get("method_run", "")),
                str(value.get("local_agent_id", "")),
                str(value.get("remote_agent_id", "")),
                str(value.get("link_name", "")),
            ),
        ):
            row_values = [
                str(row.get("method_run", "-")),
                _network_link_name(row),
                _format_number(row.get("loss_rate")),
                _format_milliseconds(row.get("latency_mean_sec")),
                _format_milliseconds(row.get("latency_stddev_sec")),
                _format_milliseconds(row.get("max_latency_sec")),
                _format_count(row.get("received_count")),
                _format_count(row.get("lost_count")),
                str(row.get("qos_profile_name") or "-"),
                str(row.get("transport_name") or "-"),
            ]
            if has_sweeps:
                row_values.insert(0, str(row.get("sweep_case", "-")))
                row_values.insert(1, _format_value(row.get("sweep_value")))
            lines.append("| " + " | ".join(row_values) + " |")
        lines.append("")

    method_graph_rows = metrics.get("method_graph_rows", [])
    if method_graph_rows:
        lines.extend(["## Graph Status Comparison", ""])
        if has_sweeps:
            lines.extend(
                [
                    "| Sweep Case | Sweep Value | Method Run | Backend | Accepted | Rejected | Stale | Last Rejection | Top Rejection Reasons | Messages |",
                    "| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: |",
                ]
            )
        else:
            lines.extend(
                [
                    "| Method Run | Backend | Accepted | Rejected | Stale | Last Rejection | Top Rejection Reasons | Messages |",
                    "| --- | --- | ---: | ---: | ---: | --- | --- | ---: |",
                ]
            )
        for row in sorted(
            method_graph_rows,
            key=lambda value: (
                _sort_value(value.get("sweep_value")),
                str(value.get("sweep_case", "")),
                str(value.get("method_run", "")),
                str(value.get("backend_name", "")),
            ),
        ):
            row_values = [
                str(row.get("method_run", "-")),
                str(row.get("backend_name", "-")),
                _format_count(row.get("accepted_constraint_count")),
                _format_count(row.get("rejected_constraint_count")),
                _format_count(row.get("stale_constraint_count")),
                str(row.get("last_rejection_reason") or "-"),
                _format_rejection_reasons(row.get("rejection_reasons")),
                _format_count(row.get("messages_seen")),
            ]
            if has_sweeps:
                row_values.insert(0, str(row.get("sweep_case", "-")))
                row_values.insert(1, _format_value(row.get("sweep_value")))
            lines.append("| " + " | ".join(row_values) + " |")
        lines.append("")

    primary_rows = metrics.get("rows", [])
    if primary_rows:
        lines.extend(
            [
                "## Primary Metrics",
                "",
                "| Agent | Method | ATE RMSE [m] | Improvement vs Local [m] | Availability |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in primary_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("agent_id", "-")),
                        str(row.get("method", "-")),
                        _format_number(row.get("ate_rmse")),
                        _format_number(row.get("improvement_vs_local")),
                        _format_number(row.get("localization_availability")),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def _format_acceptance_section(acceptance: dict[str, Any]) -> list[str]:
    passed = bool(acceptance.get("passed", False))
    checks = acceptance.get("checks", [])
    lines = [
        "## Acceptance",
        "",
        f"Status: `{'passed' if passed else 'failed'}`",
        "",
    ]
    if not isinstance(checks, list) or not checks:
        lines.append("No acceptance checks configured.")
        return lines
    lines.extend(
        [
            "| Check | Result | Actual | Expected |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for check in checks:
        if not isinstance(check, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(check.get("name", "-")),
                    "PASS" if check.get("passed") else "FAIL",
                    _format_check_value(check.get("actual")),
                    _format_check_value(check.get("expected")),
                ]
            )
            + " |"
        )
    return lines


def _write_provenance_outputs(
    plan: ExperimentPlan,
    metrics: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    provenance = _collect_provenance(plan, metrics, acceptance)
    plan.provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (plan.output_dir / "command.txt").write_text(
        provenance["command"] + "\n",
        encoding="utf-8",
    )
    (plan.output_dir / "git_info.txt").write_text(
        _format_git_info(provenance.get("git", {})),
        encoding="utf-8",
    )
    (plan.output_dir / "ros_distro.txt").write_text(
        _format_ros_info(provenance.get("ros", {})),
        encoding="utf-8",
    )
    (plan.output_dir / "dependency_versions.txt").write_text(
        _format_dependency_versions(provenance.get("dependencies", {})),
        encoding="utf-8",
    )
    (plan.output_dir / "environment.json").write_text(
        json.dumps(provenance.get("environment", {}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _collect_provenance(
    plan: ExperimentPlan,
    metrics: dict[str, Any],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    git_info = _git_info()
    ros_info = _ros_info()
    dependencies = _dependency_versions()
    environment = _environment_snapshot()
    return {
        "schema_version": "0.1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": _reconstruct_command(plan),
        "cwd": str(Path.cwd()),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "git": git_info,
        "ros": ros_info,
        "dependencies": dependencies,
        "environment": environment,
        "experiment": {
            "name": plan.name,
            "config_path": str(plan.config_path),
            "scenario_path": str(plan.scenario_path),
            "network_profile": str(plan.network_profile) if plan.network_profile else None,
            "duration_sec": plan.duration_sec,
            "ros_domain_id": plan.ros_domain_id,
            "methods": [method.to_dict() for method in plan.methods],
            "sweep_cases": [case.to_dict() for case in plan.sweep_cases],
            "primary_method": metrics.get("primary_method"),
            "acceptance_passed": acceptance.get("passed"),
            "acceptance_check_count": len(acceptance.get("checks", []))
            if isinstance(acceptance.get("checks", []), list)
            else 0,
            "bag_replay": plan.bag_replay.to_dict() if plan.bag_replay else None,
        },
        "artifacts": {
            "plan": str(plan.output_dir / "plan.json"),
            "experiment": str(plan.output_dir / "experiment.yaml"),
            "report": str(plan.report_path),
            "metrics": str(plan.metrics_path),
            "acceptance": str(plan.acceptance_path),
            "provenance": str(plan.provenance_path),
            "command": str(plan.output_dir / "command.txt"),
            "git_info": str(plan.output_dir / "git_info.txt"),
            "ros_distro": str(plan.output_dir / "ros_distro.txt"),
            "dependency_versions": str(plan.output_dir / "dependency_versions.txt"),
            "environment": str(plan.output_dir / "environment.json"),
        },
    }


def _reconstruct_command(plan: ExperimentPlan) -> str:
    args = [
        "ros2",
        "run",
        "mrn_eval",
        "mrn_experiment",
        "run",
        str(plan.config_path),
        "--duration",
        f"{plan.duration_sec:g}",
        "--output-dir",
        str(plan.output_dir),
    ]
    if plan.ros_domain_id is not None:
        args.extend(["--ros-domain-id", str(plan.ros_domain_id)])
    for sweep_case in plan.sweep_cases:
        args.extend(["--sweep-case", sweep_case.name])
    return shlex.join(args)


def _git_info() -> dict[str, Any]:
    root = _run_capture(["git", "rev-parse", "--show-toplevel"])
    commit = _run_capture(["git", "rev-parse", "--verify", "HEAD"])
    branch = _run_capture(["git", "branch", "--show-current"])
    status = _run_capture(["git", "status", "--short"])
    diff_stat = _run_capture(["git", "diff", "--stat"])
    return {
        "root": root["stdout"],
        "commit": commit["stdout"],
        "branch": branch["stdout"],
        "status_short": status["stdout"],
        "diff_stat": diff_stat["stdout"],
        "dirty": bool(status["stdout"].strip() or diff_stat["stdout"].strip()),
        "errors": {
            key: value["stderr"]
            for key, value in {
                "root": root,
                "commit": commit,
                "branch": branch,
                "status_short": status,
                "diff_stat": diff_stat,
            }.items()
            if value["returncode"] != 0
        },
    }


def _ros_info() -> dict[str, Any]:
    keys = [
        "ROS_DISTRO",
        "ROS_VERSION",
        "ROS_PYTHON_VERSION",
        "ROS_DOMAIN_ID",
        "RMW_IMPLEMENTATION",
    ]
    package_prefix = _run_capture(["ros2", "pkg", "prefix", "mrn_eval"])
    return {
        "environment": {key: os.environ.get(key, "") for key in keys},
        "mrn_eval_prefix": package_prefix["stdout"],
        "mrn_eval_prefix_error": package_prefix["stderr"]
        if package_prefix["returncode"] != 0
        else "",
    }


def _dependency_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    versions["pyyaml"] = getattr(yaml, "__version__", "unknown")
    for module_name in ["rclpy", "launch", "launch_ros"]:
        versions[module_name] = _module_version(module_name)
    for package_name in ["colcon-core", "setuptools", "pip"]:
        versions[package_name] = _distribution_version(package_name)
    return versions


def _distribution_version(package_name: str) -> str:
    try:
        from importlib import metadata

        return metadata.version(package_name)
    except Exception as error:  # pragma: no cover - defensive diagnostics
        return f"unavailable: {error}"


def _module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception as error:  # pragma: no cover - defensive diagnostics
        return f"unavailable: {error}"
    return str(getattr(module, "__version__", "installed"))


def _environment_snapshot() -> dict[str, str]:
    keys = [
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "CYCLONEDDS_URI",
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "PYTHONPATH",
        "RMW_IMPLEMENTATION",
        "ROS_DISTRO",
        "ROS_DOMAIN_ID",
        "ROS_PYTHON_VERSION",
        "ROS_VERSION",
    ]
    return {key: os.environ.get(key, "") for key in keys if os.environ.get(key) is not None}


def _format_git_info(git_info: dict[str, Any]) -> str:
    lines = [
        f"root: {git_info.get('root', '')}",
        f"commit: {git_info.get('commit', '')}",
        f"branch: {git_info.get('branch', '')}",
        f"dirty: {git_info.get('dirty', False)}",
        "",
        "status:",
        git_info.get("status_short", "") or "(clean)",
        "",
        "diff_stat:",
        git_info.get("diff_stat", "") or "(none)",
    ]
    errors = git_info.get("errors", {})
    if errors:
        lines.extend(["", "errors:", json.dumps(errors, indent=2, sort_keys=True)])
    return "\n".join(lines) + "\n"


def _format_ros_info(ros_info: dict[str, Any]) -> str:
    lines = []
    for key, value in ros_info.get("environment", {}).items():
        lines.append(f"{key}: {value}")
    lines.append(f"mrn_eval_prefix: {ros_info.get('mrn_eval_prefix', '')}")
    if ros_info.get("mrn_eval_prefix_error"):
        lines.append(f"mrn_eval_prefix_error: {ros_info['mrn_eval_prefix_error']}")
    return "\n".join(lines) + "\n"


def _format_dependency_versions(dependencies: dict[str, Any]) -> str:
    return "\n".join(
        f"{key}: {value}" for key, value in sorted(dependencies.items())
    ) + "\n"


def _run_capture(command: list[str], timeout_sec: float = 3.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except Exception as error:  # pragma: no cover - defensive diagnostics
        return {"returncode": 1, "stdout": "", "stderr": str(error)}


def _primary_method_result(method_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not method_results:
        return None
    for result in method_results:
        if result["name"] in {"coop_graph", "cooperative", "relative_anchor"}:
            return result
    return method_results[-1]


def _write_sweep_scenario(plan: ExperimentPlan, sweep_case: SweepCase) -> Path:
    scenario = _load_yaml_mapping(plan.scenario_path, f"scenario {plan.scenario_path}")
    for parameter, value in sweep_case.overrides.items():
        _set_dotted_value(scenario, parameter, value)
    scenario_section = scenario.setdefault("scenario", {})
    if isinstance(scenario_section, dict):
        scenario_section["name"] = f"{plan.name}_{sweep_case.name}"

    scenario_dir = plan.output_dir / "sweeps" / sweep_case.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = scenario_dir / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump(scenario, sort_keys=False),
        encoding="utf-8",
    )
    return scenario_path


def _set_dotted_value(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = [key for key in dotted_path.split(".") if key]
    if not keys:
        raise ValueError("override path must not be empty")
    current: dict[str, Any] = mapping
    for key in keys[:-1]:
        next_value = current.setdefault(key, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"cannot set {dotted_path}: {key} is not a mapping")
        current = next_value
    current[keys[-1]] = value


def _sweep_row_metadata(sweep_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "sweep_case": sweep_metadata.get("name"),
        "sweep_parameter": sweep_metadata.get("parameter"),
        "sweep_value": sweep_metadata.get("value"),
    }


def _result_sweep_name(result: dict[str, Any]) -> str:
    sweep_case = result.get("sweep_case")
    if isinstance(sweep_case, dict):
        return str(sweep_case.get("name") or "-")
    return "-"


def _format_number(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_check_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_milliseconds(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 1000.0:.3f}"


def _format_count(value: Any) -> str:
    if value is None:
        return ""
    return str(int(value))


def _network_link_name(row: dict[str, Any]) -> str:
    link_name = row.get("link_name")
    if link_name:
        return str(link_name)
    local_agent_id = row.get("local_agent_id")
    remote_agent_id = row.get("remote_agent_id")
    if local_agent_id or remote_agent_id:
        return f"{local_agent_id or '-'}->{remote_agent_id or '-'}"
    return "-"


def _format_rejection_reasons(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    items = sorted(
        ((str(reason), int(count)) for reason, count in value.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return ", ".join(f"{reason}:{count}" for reason, count in items[:4])


def _sort_value(value: Any) -> tuple[int, float | str]:
    if value is None:
        return (0, "")
    try:
        return (1, float(value))
    except (TypeError, ValueError):
        return (2, str(value))


def _slug_value(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "minus_").replace(".", "p")
    slug = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in slug.split("_") if part) or "value"


def _load_yaml_mapping(path: Path, name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return _mapping(value, name)


def _find_metric_row(
    rows: list[Any],
    agent_id: str,
    method: str,
    method_run: str = "",
) -> dict[str, Any] | None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("agent_id") != agent_id or row.get("method") != method:
            continue
        if method_run and row.get("method_run") != method_run:
            continue
        return row
    return None


def _find_graph_row(
    rows: list[Any],
    method_run: str = "",
    sweep_case: str = "",
    backend: str = "",
) -> dict[str, Any] | None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        if method_run and row.get("method_run") != method_run:
            continue
        if sweep_case and row.get("sweep_case") != sweep_case:
            continue
        if backend and row.get("backend_name") != backend:
            continue
        return row
    return None


def _filter_network_rows(
    rows: list[Any],
    method_run: str = "",
    sweep_case: str = "",
    qos_profile_name: str = "",
) -> list[dict[str, Any]]:
    filtered = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if method_run and row.get("method_run") != method_run:
            continue
        if sweep_case and row.get("sweep_case") != sweep_case:
            continue
        if qos_profile_name and row.get("qos_profile_name") != qos_profile_name:
            continue
        filtered.append(row)
    return filtered


def _selected_sweep_case_names(metrics: dict[str, Any]) -> set[str]:
    sweep_cases = metrics.get("sweep_cases", [])
    if not isinstance(sweep_cases, list) or not sweep_cases:
        return set()
    names = set()
    for entry in sweep_cases:
        if isinstance(entry, dict) and entry.get("name"):
            names.add(str(entry["name"]))
    return names


def _network_acceptance_prefix(
    method_run: str = "",
    sweep_case: str = "",
    qos_profile_name: str = "",
) -> str:
    parts = []
    if sweep_case:
        parts.append(sweep_case)
    if method_run:
        parts.append(method_run)
    if qos_profile_name:
        parts.append(qos_profile_name)
    return f"network[{':'.join(parts)}]" if parts else "network"


def _graph_acceptance_prefix(method_run: str = "", sweep_case: str = "", backend: str = "") -> str:
    parts = []
    if sweep_case:
        parts.append(sweep_case)
    if method_run:
        parts.append(method_run)
    if backend:
        parts.append(backend)
    return f"graph[{':'.join(parts)}]" if parts else "graph"


def _threshold_check(name: str, actual: Any, expected: float, operator: str) -> dict[str, Any]:
    if actual is None:
        return _check_result(name, False, actual, f"{operator} {expected}")
    actual_float = float(actual)
    if operator == ">=":
        passed = actual_float >= expected
    elif operator == "<=":
        passed = actual_float <= expected
    else:
        raise ValueError(f"unsupported threshold operator: {operator}")
    return _check_result(name, passed, actual_float, f"{operator} {expected}")


def _check_result(name: str, passed: bool, actual: Any, expected: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def _resolve_path(value: str, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        base_dir / path,
        base_dir.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _resolve_scenario_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                Path.cwd() / path,
                base_dir / path,
                base_dir.parent / path,
                Path.cwd() / "mrn_demos" / "config" / "scenarios" / path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / "mrn_demos" / "config" / "scenarios" / path.name).resolve()


def _resolve_existing_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        Path.cwd().parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=2.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


if __name__ == "__main__":
    main(sys.argv[1:])
