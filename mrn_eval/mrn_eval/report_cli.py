#!/usr/bin/env python3
"""Collect EvaluationSummary messages and write a Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import rclpy
from mrn_eval.report import EvaluationReport
from mrn_msgs.msg import CommStatus, ConstraintGraph, EvaluationSummary
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy._rclpy_pybind11 import RCLError


class ReportCollector(Node):
    def __init__(
        self,
        topic: str,
        agent_ids: list[str],
        comm_topic_template: str,
        graph_topic: str,
    ) -> None:
        super().__init__("mrn_report")
        self.topic = topic
        self.comm_topic_template = comm_topic_template
        self.graph_topic = graph_topic
        self.report = EvaluationReport()
        self.subscription = self.create_subscription(
            EvaluationSummary,
            topic,
            self._summary_callback,
            10,
        )
        self.comm_subscriptions = [
            self.create_subscription(
                CommStatus,
                comm_topic_template.format(agent_id=agent_id),
                self._comm_callback,
                10,
            )
            for agent_id in agent_ids
        ]
        self.graph_subscription = self.create_subscription(
            ConstraintGraph,
            graph_topic,
            self._graph_callback,
            10,
        )

    def _summary_callback(self, msg: EvaluationSummary) -> None:
        self.report.update(
            experiment_name=msg.experiment_name,
            method_name=msg.method_name,
            ate_rmse=float(msg.ate_rmse),
            localization_availability=float(msg.localization_availability),
        )

    def _comm_callback(self, msg: CommStatus) -> None:
        self.report.update_network(
            local_agent_id=msg.local_agent_id,
            remote_agent_id=msg.remote_agent_id,
            loss_rate=float(msg.loss_rate),
            latency_mean_sec=_duration_to_sec(msg.latency_mean),
            latency_stddev_sec=_duration_to_sec(msg.latency_stddev),
            max_latency_sec=_duration_to_sec(msg.max_latency),
            received_count=int(msg.received_count),
            lost_count=int(msg.lost_count),
            qos_profile_name=msg.qos_profile_name,
            transport_name=msg.transport_name,
        )

    def _graph_callback(self, msg: ConstraintGraph) -> None:
        self.report.update_graph_status(
            backend_name=msg.backend_name,
            accepted_constraint_count=int(msg.accepted_constraint_count),
            rejected_constraint_count=int(msg.rejected_constraint_count),
            stale_constraint_count=int(msg.stale_constraint_count),
            rejection_reasons={
                reason: int(count)
                for reason, count in zip(msg.rejection_reasons, msg.rejection_reason_counts)
            },
            last_rejection_reason=msg.last_rejection_reason,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mrn_report")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--topic", default="/mrn/eval/summary")
    parser.add_argument("--output", "-o", default="-")
    parser.add_argument("--json-output", default="")
    parser.add_argument("--title", default="MRN Benchmark Report")
    parser.add_argument(
        "--agent-ids",
        nargs="+",
        default=["robot_1", "robot_2", "robot_3"],
        help="Agent ids whose CommStatus topics should be collected.",
    )
    parser.add_argument(
        "--comm-topic-template",
        default="/{agent_id}/mrn/comm_status",
        help="Python format string with {agent_id}.",
    )
    parser.add_argument("--graph-topic", default="/mrn/graph/status")
    args = parser.parse_args(argv)

    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")

    rclpy.init()
    node = ReportCollector(args.topic, args.agent_ids, args.comm_topic_template, args.graph_topic)
    deadline = time.monotonic() + args.duration
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        markdown = node.report.to_markdown(
            title=args.title,
            duration_sec=args.duration,
            topic=args.topic,
            comm_topic=args.comm_topic_template,
            graph_topic=args.graph_topic,
        )
        if args.output == "-":
            print(markdown)
        else:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
            print(f"wrote {output_path}")
        if args.json_output:
            json_path = Path(args.json_output)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                node.report.to_json(
                    title=args.title,
                    duration_sec=args.duration,
                    topic=args.topic,
                    comm_topic=args.comm_topic_template,
                    graph_topic=args.graph_topic,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"wrote {json_path}")
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

    if not node.report.rows and not node.report.network_rows and not node.report.graph_rows:
        raise SystemExit(2)


def _duration_to_sec(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


if __name__ == "__main__":
    main(sys.argv[1:])
