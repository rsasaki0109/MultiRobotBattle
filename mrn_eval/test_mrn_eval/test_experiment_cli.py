import json
from pathlib import Path
import tempfile
import unittest

import yaml

from mrn_eval.experiment_cli import (
    _format_aggregate_report,
    _write_provenance_outputs,
    _write_sweep_scenario,
    evaluate_acceptance,
    load_experiment_plan,
    main,
)


class TestExperimentCli(unittest.TestCase):
    def test_loads_gnss_packet_loss_plan(self):
        plan = load_experiment_plan(
            "experiments/gnss_outage_packet_loss.yaml",
            output_dir="out/test_exp",
            duration_sec=12.0,
            ros_domain_id=88,
        )
        self.assertEqual(plan.name, "gnss_outage_packet_loss")
        self.assertEqual(plan.seed, 42)
        self.assertEqual(plan.launch_file, "gnss_outage_packet_loss.launch.py")
        self.assertEqual(plan.scenario, "gnss_outage_3robots.yaml")
        self.assertTrue(str(plan.network_profile).endswith("mrn_netem/config/loss20_delay80.yaml"))
        self.assertIn("network_profile:=", plan.launch_args[1])
        self.assertEqual([method.name for method in plan.methods], ["local_only", "coop_graph"])
        self.assertEqual(plan.methods[0].graph_executable, "dummy_graph_node.py")
        self.assertEqual(plan.methods[1].graph_executable, "relative_anchor_graph_node.py")
        self.assertIn("graph_executable:=relative_anchor_graph_node.py", plan.launch_args_for(plan.methods[1]))
        self.assertEqual(plan.duration_sec, 12.0)
        self.assertEqual(plan.ros_domain_id, 88)
        self.assertTrue(str(plan.acceptance_path).endswith("out/test_exp/acceptance.json"))
        self.assertTrue(str(plan.provenance_path).endswith("out/test_exp/provenance.json"))

    def test_loads_clock_drift_sweep_plan(self):
        plan = load_experiment_plan(
            "experiments/clock_drift_sensitivity.yaml",
            output_dir="out/test_sweep",
            duration_sec=6.0,
            ros_domain_id=90,
        )
        self.assertEqual(plan.name, "clock_drift_sensitivity")
        self.assertEqual(plan.launch_file, "gnss_outage_packet_loss.launch.py")
        self.assertEqual([method.name for method in plan.methods], ["coop_graph"])
        self.assertTrue(str(plan.scenario_path).endswith("mrn_demos/config/scenarios/gnss_outage_3robots.yaml"))
        self.assertEqual([case.value for case in plan.sweep_cases], [0, 10, 30, 50, 100])
        self.assertEqual(plan.sweep_cases[2].name, "clock_drift_ms_30")
        self.assertEqual(plan.sweep_cases[2].overrides, {"faults.clock_drift_ms": 30})

    def test_filters_clock_drift_sweep_plan(self):
        plan = load_experiment_plan(
            "experiments/clock_drift_sensitivity.yaml",
            output_dir="out/test_sweep",
            duration_sec=6.0,
            sweep_case_names=["clock_drift_ms_50", "clock_drift_ms_100"],
        )
        self.assertEqual([case.name for case in plan.sweep_cases], ["clock_drift_ms_50", "clock_drift_ms_100"])

    def test_loads_qos_case_sweep_plan(self):
        plan = load_experiment_plan(
            "experiments/qos_best_effort_vs_reliable.yaml",
            output_dir="out/test_qos",
            duration_sec=6.0,
        )
        self.assertEqual(plan.name, "qos_best_effort_vs_reliable")
        self.assertEqual(plan.launch_file, "cooperative_localization.launch.py")
        self.assertEqual([method.name for method in plan.methods], ["coop_graph"])
        self.assertIsNone(plan.network_profile)
        self.assertEqual(
            [case.name for case in plan.sweep_cases],
            ["qos_profile_best_effort_fast", "qos_profile_reliable_constraints"],
        )
        self.assertEqual(plan.sweep_cases[0].parameter, "faults.qos_profile_name")
        self.assertEqual(plan.sweep_cases[0].value, "agent_state_fast")
        self.assertEqual(
            plan.sweep_cases[0].overrides,
            {
                "faults.qos_profile_name": "agent_state_fast",
                "faults.packet_loss_percent": 30,
                "faults.latency_ms_mean": 40,
                "faults.latency_ms_stddev": 12,
            },
        )

    def test_rejects_unknown_sweep_case(self):
        with self.assertRaisesRegex(ValueError, "unknown sweep case"):
            load_experiment_plan(
                "experiments/clock_drift_sensitivity.yaml",
                output_dir="out/test_sweep",
                duration_sec=6.0,
                sweep_case_names=["missing_case"],
            )

    def test_sweep_scenario_overrides_base_scenario(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = load_experiment_plan(
                "experiments/clock_drift_sensitivity.yaml",
                output_dir=Path(directory) / "out",
                duration_sec=6.0,
            )
            scenario_path = _write_sweep_scenario(plan, plan.sweep_cases[-1])
            scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        self.assertEqual(scenario["faults"]["clock_drift_ms"], 100)
        self.assertEqual(
            scenario["scenario"]["name"],
            "clock_drift_sensitivity_clock_drift_ms_100",
        )

    def test_case_sweep_scenario_applies_multiple_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = load_experiment_plan(
                "experiments/qos_best_effort_vs_reliable.yaml",
                output_dir=Path(directory) / "out",
                duration_sec=6.0,
            )
            scenario_path = _write_sweep_scenario(plan, plan.sweep_cases[0])
            scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        self.assertEqual(scenario["faults"]["qos_profile_name"], "agent_state_fast")
        self.assertEqual(scenario["faults"]["packet_loss_percent"], 30)
        self.assertEqual(scenario["faults"]["latency_ms_mean"], 40)
        self.assertEqual(
            scenario["scenario"]["name"],
            "qos_best_effort_vs_reliable_qos_profile_best_effort_fast",
        )

    def test_writes_provenance_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = load_experiment_plan(
                "experiments/clock_drift_sensitivity.yaml",
                output_dir=Path(directory) / "out",
                duration_sec=6.0,
                sweep_case_names=["clock_drift_ms_100"],
            )
            plan.output_dir.mkdir(parents=True, exist_ok=True)
            metrics = {"primary_method": "coop_graph"}
            acceptance = {"passed": True, "checks": [{"name": "demo", "passed": True}]}
            _write_provenance_outputs(plan, metrics, acceptance)
            provenance = json.loads(plan.provenance_path.read_text(encoding="utf-8"))
            output_dir = plan.output_dir
            self.assertTrue((output_dir / "command.txt").exists())
            self.assertTrue((output_dir / "git_info.txt").exists())
            self.assertTrue((output_dir / "ros_distro.txt").exists())
            self.assertTrue((output_dir / "dependency_versions.txt").exists())
            self.assertTrue((output_dir / "environment.json").exists())
        self.assertEqual(provenance["schema_version"], "0.1.0")
        self.assertEqual(provenance["experiment"]["name"], "clock_drift_sensitivity")
        self.assertEqual(provenance["experiment"]["acceptance_passed"], True)
        self.assertIn("--sweep-case clock_drift_ms_100", provenance["command"])
        self.assertIn("git", provenance)
        self.assertIn("ros", provenance)

    def test_defaults_to_cooperative_launch_without_network_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exp.yaml"
            path.write_text("experiment:\n  name: local\n", encoding="utf-8")
            plan = load_experiment_plan(path, output_dir=Path(directory) / "out")
        self.assertEqual(plan.launch_file, "cooperative_localization.launch.py")
        self.assertIsNone(plan.network_profile)

    def test_plan_command_prints_json(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "exp.yaml"
            config.write_text("experiment:\n  name: demo\n", encoding="utf-8")
            output = Path(directory) / "plan.json"
            # Exercise the parser path by temporarily redirecting stdout.
            import contextlib
            import io

            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                main(["plan", str(config), "--output-dir", str(output.parent), "--duration", "5"])
            data = json.loads(stream.getvalue())
        self.assertEqual(data["name"], "demo")
        self.assertEqual(data["duration_sec"], 5.0)
        self.assertTrue(data["acceptance_path"].endswith("acceptance.json"))

    def test_plan_command_filters_sweep_cases(self):
        import contextlib
        import io

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            main([
                "plan",
                "experiments/clock_drift_sensitivity.yaml",
                "--duration",
                "5",
                "--sweep-case",
                "clock_drift_ms_100",
            ])
        data = json.loads(stream.getvalue())
        self.assertEqual(
            [case["name"] for case in data["sweep_cases"]],
            ["clock_drift_ms_100"],
        )

    def test_rejects_invalid_duration(self):
        with self.assertRaises(ValueError):
            load_experiment_plan("experiments/gnss_outage_packet_loss.yaml", duration_sec=0.0)

    def test_acceptance_passes_expected_metrics(self):
        config = {
            "acceptance": {
                "localization": [
                    {
                        "agent_id": "robot_2",
                        "method_run": "coop_graph",
                        "method": "cooperative",
                        "min_improvement_vs_local": 0.2,
                        "max_ate_rmse": 0.3,
                        "min_availability": 0.95,
                    }
                ],
                "network": {
                    "method_run": "coop_graph",
                    "min_rows": 2,
                    "min_observed_loss_rate": 0.1,
                    "max_mean_latency_sec": 0.2,
                },
            }
        }
        metrics = {
            "method_rows": [
                {
                    "method_run": "coop_graph",
                    "agent_id": "robot_2",
                    "method": "cooperative",
                    "improvement_vs_local": 0.5,
                    "ate_rmse": 0.05,
                    "localization_availability": 1.0,
                }
            ],
            "method_network_rows": [
                {"method_run": "coop_graph", "loss_rate": 0.2, "latency_mean_sec": 0.08},
                {"method_run": "coop_graph", "loss_rate": 0.15, "latency_mean_sec": 0.09},
                {"method_run": "local_only", "loss_rate": 0.01, "latency_mean_sec": 0.01},
            ],
        }
        result = evaluate_acceptance(config, metrics)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 7)

    def test_acceptance_passes_network_sweep_rules(self):
        config = {
            "acceptance": {
                "network": [
                    {
                        "method_run": "coop_graph",
                        "sweep_case": "qos_profile_best_effort_fast",
                        "qos_profile_name": "agent_state_fast",
                        "min_rows": 2,
                        "min_observed_loss_rate": 0.2,
                        "max_mean_latency_sec": 0.08,
                    },
                    {
                        "method_run": "coop_graph",
                        "sweep_case": "qos_profile_reliable_constraints",
                        "qos_profile_name": "relative_constraint",
                        "min_rows": 2,
                        "max_observed_loss_rate": 0.12,
                        "min_mean_latency_sec": 0.08,
                    },
                ]
            }
        }
        metrics = {
            "method_network_rows": [
                {
                    "method_run": "coop_graph",
                    "sweep_case": "qos_profile_best_effort_fast",
                    "qos_profile_name": "agent_state_fast",
                    "loss_rate": 0.31,
                    "latency_mean_sec": 0.04,
                },
                {
                    "method_run": "coop_graph",
                    "sweep_case": "qos_profile_best_effort_fast",
                    "qos_profile_name": "agent_state_fast",
                    "loss_rate": 0.27,
                    "latency_mean_sec": 0.05,
                },
                {
                    "method_run": "coop_graph",
                    "sweep_case": "qos_profile_reliable_constraints",
                    "qos_profile_name": "relative_constraint",
                    "loss_rate": 0.04,
                    "latency_mean_sec": 0.10,
                },
                {
                    "method_run": "coop_graph",
                    "sweep_case": "qos_profile_reliable_constraints",
                    "qos_profile_name": "relative_constraint",
                    "loss_rate": 0.06,
                    "latency_mean_sec": 0.11,
                },
            ],
        }
        result = evaluate_acceptance(config, metrics)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 6)

    def test_acceptance_fails_bad_metrics(self):
        config = {
            "acceptance": {
                "localization": [
                    {
                        "agent_id": "robot_2",
                        "method_run": "coop_graph",
                        "method": "cooperative",
                        "min_improvement_vs_local": 0.2,
                    }
                ],
                "network": {"method_run": "coop_graph", "min_rows": 3},
            }
        }
        metrics = {
            "method_rows": [
                {
                    "method_run": "coop_graph",
                    "agent_id": "robot_2",
                    "method": "cooperative",
                    "improvement_vs_local": 0.05,
                }
            ],
            "method_network_rows": [{"method_run": "coop_graph", "loss_rate": 0.2}],
        }
        result = evaluate_acceptance(config, metrics)
        self.assertFalse(result["passed"])
        failed = [check["name"] for check in result["checks"] if not check["passed"]]
        self.assertIn("localization[coop_graph:robot_2/cooperative].improvement_vs_local", failed)
        self.assertIn("network[coop_graph].rows", failed)

    def test_acceptance_passes_graph_sweep_rules(self):
        config = {
            "acceptance": {
                "graph": [
                    {
                        "method_run": "coop_graph",
                        "sweep_case": "clock_drift_ms_50",
                        "backend": "relative_anchor",
                        "max_rejected_constraints": 0,
                        "max_stale_constraints": 0,
                    },
                    {
                        "method_run": "coop_graph",
                        "sweep_case": "clock_drift_ms_100",
                        "backend": "relative_anchor",
                        "min_accepted_constraints": 1,
                        "min_rejected_constraints": 1,
                        "min_rejection_reasons": {
                            "clock_offset_too_large": 1,
                        },
                    },
                ]
            }
        }
        metrics = {
            "method_graph_rows": [
                {
                    "method_run": "coop_graph",
                    "sweep_case": "clock_drift_ms_50",
                    "backend_name": "relative_anchor",
                    "accepted_constraint_count": 105,
                    "rejected_constraint_count": 0,
                    "stale_constraint_count": 0,
                    "rejection_reasons": {},
                },
                {
                    "method_run": "coop_graph",
                    "sweep_case": "clock_drift_ms_100",
                    "backend_name": "relative_anchor",
                    "accepted_constraint_count": 36,
                    "rejected_constraint_count": 69,
                    "stale_constraint_count": 0,
                    "rejection_reasons": {"clock_offset_too_large": 69},
                },
            ]
        }
        result = evaluate_acceptance(config, metrics)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 7)

    def test_acceptance_fails_graph_sweep_rules(self):
        config = {
            "acceptance": {
                "graph": [
                    {
                        "method_run": "coop_graph",
                        "sweep_case": "clock_drift_ms_100",
                        "backend": "relative_anchor",
                        "min_rejected_constraints": 1,
                        "min_rejection_reasons": {
                            "clock_offset_too_large": 1,
                        },
                    }
                ]
            }
        }
        metrics = {
            "method_graph_rows": [
                {
                    "method_run": "coop_graph",
                    "sweep_case": "clock_drift_ms_100",
                    "backend_name": "relative_anchor",
                    "accepted_constraint_count": 105,
                    "rejected_constraint_count": 0,
                    "stale_constraint_count": 0,
                    "rejection_reasons": {},
                }
            ]
        }
        result = evaluate_acceptance(config, metrics)
        self.assertFalse(result["passed"])
        failed = [check["name"] for check in result["checks"] if not check["passed"]]
        self.assertIn(
            "graph[clock_drift_ms_100:coop_graph:relative_anchor].rejected_constraint_count",
            failed,
        )
        self.assertIn(
            "graph[clock_drift_ms_100:coop_graph:relative_anchor].rejection_reasons[clock_offset_too_large]",
            failed,
        )

    def test_acceptance_skips_graph_rules_for_unselected_sweeps(self):
        config = {
            "acceptance": {
                "graph": [
                    {
                        "method_run": "coop_graph",
                        "sweep_case": "clock_drift_ms_50",
                        "backend": "relative_anchor",
                        "max_rejected_constraints": 0,
                    },
                    {
                        "method_run": "coop_graph",
                        "sweep_case": "clock_drift_ms_100",
                        "backend": "relative_anchor",
                        "min_rejected_constraints": 1,
                    },
                ]
            }
        }
        metrics = {
            "sweep_cases": [
                {"name": "clock_drift_ms_100"},
            ],
            "method_graph_rows": [
                {
                    "method_run": "coop_graph",
                    "sweep_case": "clock_drift_ms_100",
                    "backend_name": "relative_anchor",
                    "accepted_constraint_count": 36,
                    "rejected_constraint_count": 69,
                    "stale_constraint_count": 0,
                    "rejection_reasons": {"clock_offset_too_large": 69},
                }
            ],
        }
        result = evaluate_acceptance(config, metrics)
        self.assertTrue(result["passed"])
        self.assertEqual(
            [check["name"] for check in result["checks"]],
            [
                "graph[clock_drift_ms_100:coop_graph:relative_anchor].exists",
                "graph[clock_drift_ms_100:coop_graph:relative_anchor].rejected_constraint_count",
            ],
        )

    def test_acceptance_without_rules_passes(self):
        result = evaluate_acceptance({}, {"rows": [], "network_rows": []})
        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"], [])

    def test_aggregate_report_includes_network_comparison(self):
        plan = load_experiment_plan(
            "experiments/gnss_outage_packet_loss.yaml",
            output_dir="out/test_exp",
            duration_sec=12.0,
            ros_domain_id=88,
        )
        method_results = [
            {
                "name": "local_only",
                "graph_executable": "dummy_graph_node.py",
                "report_path": "/tmp/local/report.md",
                "metrics_path": "/tmp/local/metrics.json",
            },
            {
                "name": "coop_graph",
                "graph_executable": "relative_anchor_graph_node.py",
                "report_path": "/tmp/coop/report.md",
                "metrics_path": "/tmp/coop/metrics.json",
            },
        ]
        metrics = {
            "primary_method": "coop_graph",
            "method_network_rows": [
                {
                    "method_run": "coop_graph",
                    "local_agent_id": "robot_1",
                    "remote_agent_id": "robot_2",
                    "link_name": "robot_1->robot_2",
                    "loss_rate": 0.2,
                    "latency_mean_sec": 0.08,
                    "latency_stddev_sec": 0.01,
                    "max_latency_sec": 0.12,
                    "received_count": 80,
                    "lost_count": 20,
                    "qos_profile_name": "relative_constraint",
                    "transport_name": "loopback_netem",
                }
            ],
        }
        report = _format_aggregate_report(plan, method_results, metrics)
        self.assertIn("## Network Comparison", report)
        self.assertIn("| coop_graph | robot_1->robot_2 | 0.200 | 80.000 | 10.000 | 120.000 | 80 | 20 |", report)

    def test_aggregate_report_includes_acceptance_summary(self):
        plan = load_experiment_plan(
            "experiments/clock_drift_sensitivity.yaml",
            output_dir="out/test_sweep",
            duration_sec=6.0,
            sweep_case_names=["clock_drift_ms_100"],
        )
        method_results = [
            {
                "name": "coop_graph",
                "graph_executable": "relative_anchor_graph_node.py",
                "report_path": "/tmp/clock_100/report.md",
                "metrics_path": "/tmp/clock_100/metrics.json",
                "sweep_case": plan.sweep_cases[0].to_dict(),
            }
        ]
        metrics = {"primary_method": "coop_graph"}
        acceptance = {
            "passed": True,
            "checks": [
                {
                    "name": "graph[clock_drift_ms_100:coop_graph:relative_anchor].rejected_constraint_count",
                    "passed": True,
                    "actual": 69.0,
                    "expected": ">= 1",
                }
            ],
        }
        report = _format_aggregate_report(plan, method_results, metrics, acceptance=acceptance)
        self.assertIn("## Acceptance", report)
        self.assertIn("Status: `passed`", report)
        self.assertIn(
            "| graph[clock_drift_ms_100:coop_graph:relative_anchor].rejected_constraint_count | PASS | 69.000 | >= 1 |",
            report,
        )

    def test_aggregate_report_includes_sweep_columns(self):
        plan = load_experiment_plan(
            "experiments/clock_drift_sensitivity.yaml",
            output_dir="out/test_sweep",
            duration_sec=6.0,
            ros_domain_id=90,
        )
        method_results = [
            {
                "name": "coop_graph",
                "graph_executable": "relative_anchor_graph_node.py",
                "report_path": "/tmp/clock_30/report.md",
                "metrics_path": "/tmp/clock_30/metrics.json",
                "sweep_case": plan.sweep_cases[2].to_dict(),
            }
        ]
        metrics = {
            "primary_method": "coop_graph",
            "method_rows": [
                {
                    "sweep_case": "clock_drift_ms_30",
                    "sweep_parameter": "faults.clock_drift_ms",
                    "sweep_value": 30,
                    "method_run": "coop_graph",
                    "agent_id": "robot_2",
                    "method": "cooperative",
                    "ate_rmse": 0.05,
                    "improvement_vs_local": 0.48,
                    "localization_availability": 1.0,
                }
            ],
            "method_graph_rows": [
                {
                    "sweep_case": "clock_drift_ms_30",
                    "sweep_parameter": "faults.clock_drift_ms",
                    "sweep_value": 30,
                    "method_run": "coop_graph",
                    "backend_name": "relative_anchor",
                    "accepted_constraint_count": 10,
                    "rejected_constraint_count": 4,
                    "stale_constraint_count": 0,
                    "last_rejection_reason": "clock_offset_exceeds_threshold",
                    "rejection_reasons": {"clock_offset_exceeds_threshold": 4},
                    "messages_seen": 3,
                }
            ],
        }
        report = _format_aggregate_report(plan, method_results, metrics)
        self.assertIn("## Sweep Cases", report)
        self.assertIn("| clock_drift_ms_30 | faults.clock_drift_ms | 30 |", report)
        self.assertIn("| Sweep Case | Sweep Value | Method Run | Agent | Output |", report)
        self.assertIn("| clock_drift_ms_30 | 30 | coop_graph | robot_2 | cooperative |", report)
        self.assertIn("## Graph Status Comparison", report)
        self.assertIn(
            "| clock_drift_ms_30 | 30 | coop_graph | relative_anchor | 10 | 4 | 0 |",
            report,
        )


    def test_loads_bag_replay_plan(self):
        plan = load_experiment_plan(
            "experiments/bag_replay_smoke.yaml",
            output_dir="out/test_bag_replay",
            duration_sec=8.0,
            ros_domain_id=77,
        )
        self.assertEqual(plan.name, "bag_replay_smoke")
        self.assertEqual(plan.launch_file, "bag_replay.launch.py")
        self.assertIsNone(plan.network_profile)
        self.assertIsNotNone(plan.bag_replay)
        bag = plan.bag_replay
        self.assertTrue(str(bag.directory).endswith("experiments/fixtures/synthetic_bag"))
        self.assertEqual(bag.play_rate, 1.0)
        self.assertEqual(bag.agent_ids, ("robot_1", "robot_2"))
        self.assertFalse(bag.enable_online_ate)
        self.assertEqual(bag.validation_summary.get("storage"), "mcap")
        self.assertEqual(bag.validation_summary.get("required_topic_count"), 4)
        launch_args = plan.launch_args
        self.assertTrue(any(arg.startswith("bag_dir:=") for arg in launch_args))
        self.assertIn("play_rate:=1.0", launch_args)
        self.assertIn("agent_ids:=robot_1,robot_2", launch_args)
        self.assertIn("enable_online_ate:=false", launch_args)
        method_args = plan.launch_args_for(plan.methods[0])
        self.assertIn("graph_executable:=relative_anchor_graph_node.py", method_args)
        # scenario_path is ignored for bag replay even if provided.
        self.assertNotIn(
            "scenario_path:=" + str(plan.scenario_path),
            plan.launch_args_for(plan.methods[0], scenario_path=plan.scenario_path),
        )

    def test_bag_replay_rejects_network_profile_combination(self):
        with tempfile.TemporaryDirectory() as directory:
            bag_dir = Path(directory) / "stub_bag"
            bag_dir.mkdir()
            netem_path = Path(directory) / "netem.yaml"
            netem_path.write_text("loss_percent: 5\n", encoding="utf-8")
            config_path = Path(directory) / "exp.yaml"
            config_path.write_text(
                "experiment:\n  name: bad\n"
                f"bag:\n  directory: {bag_dir}\n"
                f"network:\n  profile: {netem_path}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cannot combine"):
                load_experiment_plan(config_path, output_dir=Path(directory) / "out")

    def test_bag_replay_missing_directory_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "exp.yaml"
            config_path.write_text(
                "experiment:\n  name: bad\n"
                "bag:\n  directory: experiments/fixtures/does_not_exist\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "bag.directory does not exist"):
                load_experiment_plan(config_path, output_dir=Path(directory) / "out")

    def test_bag_replay_validation_failure_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            bag_dir = Path(directory) / "broken_bag"
            bag_dir.mkdir()
            metadata = {
                "rosbag2_bagfile_information": {
                    "version": 6,
                    "storage_identifier": "mcap",
                    "duration": {"nanoseconds": 1_000_000_000},
                    "starting_time": {"nanoseconds_since_epoch": 0},
                    "message_count": 10,
                    "topics_with_message_count": [
                        {
                            "topic_metadata": {
                                "name": "/robot_1/mrn/agent_state",
                                "type": "geometry_msgs/msg/PoseStamped",
                                "serialization_format": "cdr",
                                "offered_qos_profiles": "",
                            },
                            "message_count": 10,
                        }
                    ],
                }
            }
            (bag_dir / "metadata.yaml").write_text(
                yaml.safe_dump(metadata), encoding="utf-8"
            )
            manifest_path = Path(directory) / "manifest.yaml"
            manifest_path.write_text(
                yaml.safe_dump(
                    {
                        "dataset": {"storage": "mcap"},
                        "topics": [
                            {
                                "name": "/robot_1/mrn/agent_state",
                                "type": "mrn_msgs/msg/AgentState",
                                "required": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config_path = Path(directory) / "exp.yaml"
            config_path.write_text(
                "experiment:\n  name: bad\n"
                f"bag:\n  directory: {bag_dir}\n  manifest: {manifest_path}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "bag validation failed"):
                load_experiment_plan(config_path, output_dir=Path(directory) / "out")

    def test_bag_replay_provenance_includes_bag_block(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = load_experiment_plan(
                "experiments/bag_replay_smoke.yaml",
                output_dir=Path(directory) / "out",
                duration_sec=6.0,
            )
            plan.output_dir.mkdir(parents=True, exist_ok=True)
            metrics = {"primary_method": "coop_graph"}
            acceptance = {"passed": True, "checks": []}
            _write_provenance_outputs(plan, metrics, acceptance)
            provenance = json.loads(plan.provenance_path.read_text(encoding="utf-8"))
        bag_block = provenance["experiment"]["bag_replay"]
        self.assertIsNotNone(bag_block)
        self.assertEqual(bag_block["agent_ids"], ["robot_1", "robot_2"])
        self.assertEqual(bag_block["validation_summary"]["required_topic_count"], 4)


if __name__ == "__main__":
    unittest.main()
