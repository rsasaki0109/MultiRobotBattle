"""Report helpers for MRN evaluation summaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math


@dataclass
class SummaryRow:
    experiment_name: str
    method_name: str
    ate_rmse: float
    localization_availability: float
    messages_seen: int = 0

    @property
    def agent_id(self) -> str:
        if "/" not in self.method_name:
            return ""
        return self.method_name.split("/", 1)[0]

    @property
    def method(self) -> str:
        if "/" not in self.method_name:
            return self.method_name
        return self.method_name.split("/", 1)[1]


@dataclass
class NetworkRow:
    local_agent_id: str
    remote_agent_id: str
    loss_rate: float
    latency_mean_sec: float
    latency_stddev_sec: float
    max_latency_sec: float
    received_count: int
    lost_count: int
    qos_profile_name: str = ""
    transport_name: str = ""
    messages_seen: int = 0

    @property
    def link_name(self) -> str:
        return f"{self.local_agent_id}->{self.remote_agent_id}"


@dataclass
class GraphStatusRow:
    backend_name: str
    accepted_constraint_count: int
    rejected_constraint_count: int
    stale_constraint_count: int
    rejection_reasons: dict[str, int]
    last_rejection_reason: str = ""
    messages_seen: int = 0

    @property
    def total_constraint_count(self) -> int:
        return (
            self.accepted_constraint_count
            + self.rejected_constraint_count
            + self.stale_constraint_count
        )

    @property
    def rejection_rate(self) -> float:
        total = self.total_constraint_count
        if total == 0:
            return 0.0
        return self.rejected_constraint_count / total

    @property
    def top_rejection_reasons(self) -> str:
        if not self.rejection_reasons:
            return "-"
        items = sorted(
            self.rejection_reasons.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return ", ".join(f"{reason}:{count}" for reason, count in items[:4])


class EvaluationReport:
    def __init__(self) -> None:
        self._rows: dict[str, SummaryRow] = {}
        self._network_rows: dict[tuple[str, str], NetworkRow] = {}
        self._graph_rows: dict[str, GraphStatusRow] = {}

    def update(
        self,
        experiment_name: str,
        method_name: str,
        ate_rmse: float,
        localization_availability: float,
    ) -> None:
        existing = self._rows.get(method_name)
        messages_seen = 1 if existing is None else existing.messages_seen + 1
        self._rows[method_name] = SummaryRow(
            experiment_name=experiment_name,
            method_name=method_name,
            ate_rmse=ate_rmse,
            localization_availability=localization_availability,
            messages_seen=messages_seen,
        )

    def update_network(
        self,
        local_agent_id: str,
        remote_agent_id: str,
        loss_rate: float,
        latency_mean_sec: float,
        latency_stddev_sec: float,
        max_latency_sec: float,
        received_count: int,
        lost_count: int,
        qos_profile_name: str = "",
        transport_name: str = "",
    ) -> None:
        key = (local_agent_id, remote_agent_id)
        existing = self._network_rows.get(key)
        messages_seen = 1 if existing is None else existing.messages_seen + 1
        self._network_rows[key] = NetworkRow(
            local_agent_id=local_agent_id,
            remote_agent_id=remote_agent_id,
            loss_rate=loss_rate,
            latency_mean_sec=latency_mean_sec,
            latency_stddev_sec=latency_stddev_sec,
            max_latency_sec=max_latency_sec,
            received_count=int(received_count),
            lost_count=int(lost_count),
            qos_profile_name=qos_profile_name,
            transport_name=transport_name,
            messages_seen=messages_seen,
        )

    def update_graph_status(
        self,
        backend_name: str,
        accepted_constraint_count: int,
        rejected_constraint_count: int,
        stale_constraint_count: int,
        rejection_reasons: dict[str, int],
        last_rejection_reason: str = "",
    ) -> None:
        key = backend_name or "unknown"
        existing = self._graph_rows.get(key)
        messages_seen = 1 if existing is None else existing.messages_seen + 1
        self._graph_rows[key] = GraphStatusRow(
            backend_name=key,
            accepted_constraint_count=int(accepted_constraint_count),
            rejected_constraint_count=int(rejected_constraint_count),
            stale_constraint_count=int(stale_constraint_count),
            rejection_reasons=dict(rejection_reasons),
            last_rejection_reason=last_rejection_reason,
            messages_seen=messages_seen,
        )

    @property
    def rows(self) -> list[SummaryRow]:
        return sorted(self._rows.values(), key=lambda row: (row.agent_id, row.method))

    @property
    def network_rows(self) -> list[NetworkRow]:
        return sorted(
            self._network_rows.values(),
            key=lambda row: (row.local_agent_id, row.remote_agent_id),
        )

    @property
    def graph_rows(self) -> list[GraphStatusRow]:
        return sorted(self._graph_rows.values(), key=lambda row: row.backend_name)

    def to_markdown(
        self,
        title: str = "MRN Benchmark Report",
        duration_sec: float | None = None,
        topic: str = "/mrn/eval/summary",
        comm_topic: str = "/robot_i/mrn/comm_status",
        graph_topic: str = "/mrn/graph/status",
    ) -> str:
        return format_markdown_report(
            self.rows,
            network_rows=self.network_rows,
            graph_rows=self.graph_rows,
            title=title,
            duration_sec=duration_sec,
            topic=topic,
            comm_topic=comm_topic,
            graph_topic=graph_topic,
        )

    def to_metrics_dict(
        self,
        title: str = "MRN Benchmark Report",
        duration_sec: float | None = None,
        topic: str = "/mrn/eval/summary",
        comm_topic: str = "/robot_i/mrn/comm_status",
        graph_topic: str = "/mrn/graph/status",
    ) -> dict:
        return format_metrics_dict(
            self.rows,
            network_rows=self.network_rows,
            graph_rows=self.graph_rows,
            title=title,
            duration_sec=duration_sec,
            topic=topic,
            comm_topic=comm_topic,
            graph_topic=graph_topic,
        )

    def to_json(
        self,
        title: str = "MRN Benchmark Report",
        duration_sec: float | None = None,
        topic: str = "/mrn/eval/summary",
        comm_topic: str = "/robot_i/mrn/comm_status",
        graph_topic: str = "/mrn/graph/status",
    ) -> str:
        return json.dumps(
            self.to_metrics_dict(
                title=title,
                duration_sec=duration_sec,
                topic=topic,
                comm_topic=comm_topic,
                graph_topic=graph_topic,
            ),
            indent=2,
            sort_keys=True,
        )


def format_markdown_report(
    rows: list[SummaryRow],
    network_rows: list[NetworkRow] | None = None,
    graph_rows: list[GraphStatusRow] | None = None,
    title: str = "MRN Benchmark Report",
    duration_sec: float | None = None,
    topic: str = "/mrn/eval/summary",
    comm_topic: str = "/robot_i/mrn/comm_status",
    graph_topic: str = "/mrn/graph/status",
) -> str:
    lines = [f"# {title}", "", f"Source topic: `{topic}`"]
    lines.append(f"Network topic: `{comm_topic}`")
    lines.append(f"Graph topic: `{graph_topic}`")
    if duration_sec is not None:
        lines.append(f"Collection duration: `{duration_sec:.1f}s`")
    lines.append("")

    network_rows = network_rows or []
    graph_rows = graph_rows or []
    if not rows and not network_rows and not graph_rows:
        lines.append("No evaluation, network, or graph summaries were received.")
        lines.append("")
        return "\n".join(lines)

    experiment_names = sorted({row.experiment_name for row in rows if row.experiment_name})
    if experiment_names:
        lines.append(f"Experiment: `{', '.join(experiment_names)}`")
        lines.append("")

    lines.extend(
        [
            "| Agent | Method | ATE RMSE [m] | Improvement vs Local [m] | Availability | Messages |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    local_by_agent = {
        row.agent_id: row.ate_rmse
        for row in rows
        if row.agent_id and row.method == "local_only"
    }
    for row in rows:
        improvement = ""
        local_ate = local_by_agent.get(row.agent_id)
        if local_ate is not None and row.method != "local_only":
            improvement = _format_float(local_ate - row.ate_rmse)
        lines.append(
            "| "
            + " | ".join(
                [
                    row.agent_id or "-",
                    row.method,
                    _format_float(row.ate_rmse),
                    improvement,
                    _format_float(row.localization_availability),
                    str(row.messages_seen),
                ]
            )
            + " |"
        )
    lines.append("")

    if network_rows:
        lines.append("## Network Diagnostics")
        lines.append("")
        lines.extend(
            [
                "| Link | Loss Rate | Latency Mean [ms] | Jitter [ms] | Max Latency [ms] | Received | Lost | QoS | Transport | Messages |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for row in network_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.link_name,
                        _format_float(row.loss_rate),
                        _format_float(row.latency_mean_sec * 1000.0),
                        _format_float(row.latency_stddev_sec * 1000.0),
                        _format_float(row.max_latency_sec * 1000.0),
                        str(row.received_count),
                        str(row.lost_count),
                        row.qos_profile_name or "-",
                        row.transport_name or "-",
                        str(row.messages_seen),
                    ]
                )
                + " |"
            )
        lines.append("")

    if graph_rows:
        lines.append("## Graph Status")
        lines.append("")
        lines.extend(
            [
                "| Backend | Accepted | Rejected | Stale | Reject Rate | Last Rejection | Top Rejection Reasons | Messages |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
            ]
        )
        for row in graph_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.backend_name or "-",
                        str(row.accepted_constraint_count),
                        str(row.rejected_constraint_count),
                        str(row.stale_constraint_count),
                        _format_float(row.rejection_rate),
                        row.last_rejection_reason or "-",
                        row.top_rejection_reasons,
                        str(row.messages_seen),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def format_metrics_dict(
    rows: list[SummaryRow],
    network_rows: list[NetworkRow] | None = None,
    graph_rows: list[GraphStatusRow] | None = None,
    title: str = "MRN Benchmark Report",
    duration_sec: float | None = None,
    topic: str = "/mrn/eval/summary",
    comm_topic: str = "/robot_i/mrn/comm_status",
    graph_topic: str = "/mrn/graph/status",
) -> dict:
    local_by_agent = {
        row.agent_id: row.ate_rmse
        for row in rows
        if row.agent_id and row.method == "local_only"
    }
    experiments = sorted({row.experiment_name for row in rows if row.experiment_name})
    return {
        "title": title,
        "topic": topic,
        "network_topic": comm_topic,
        "graph_topic": graph_topic,
        "duration_sec": duration_sec,
        "experiments": experiments,
        "rows": [
            {
                "agent_id": row.agent_id,
                "method": row.method,
                "method_name": row.method_name,
                "experiment_name": row.experiment_name,
                "ate_rmse": _json_float(row.ate_rmse),
                "improvement_vs_local": _json_float(
                    local_by_agent[row.agent_id] - row.ate_rmse
                )
                if row.agent_id in local_by_agent and row.method != "local_only"
                else None,
                "localization_availability": _json_float(row.localization_availability),
                "messages_seen": row.messages_seen,
            }
            for row in sorted(rows, key=lambda row: (row.agent_id, row.method))
        ],
        "network_rows": [
            {
                "local_agent_id": row.local_agent_id,
                "remote_agent_id": row.remote_agent_id,
                "link_name": row.link_name,
                "loss_rate": _json_float(row.loss_rate),
                "latency_mean_sec": _json_float(row.latency_mean_sec),
                "latency_stddev_sec": _json_float(row.latency_stddev_sec),
                "max_latency_sec": _json_float(row.max_latency_sec),
                "received_count": row.received_count,
                "lost_count": row.lost_count,
                "qos_profile_name": row.qos_profile_name,
                "transport_name": row.transport_name,
                "messages_seen": row.messages_seen,
            }
            for row in sorted(
                network_rows or [],
                key=lambda row: (row.local_agent_id, row.remote_agent_id),
            )
        ],
        "graph_rows": [
            {
                "backend_name": row.backend_name,
                "accepted_constraint_count": row.accepted_constraint_count,
                "rejected_constraint_count": row.rejected_constraint_count,
                "stale_constraint_count": row.stale_constraint_count,
                "total_constraint_count": row.total_constraint_count,
                "rejection_rate": _json_float(row.rejection_rate),
                "rejection_reasons": dict(row.rejection_reasons),
                "last_rejection_reason": row.last_rejection_reason,
                "messages_seen": row.messages_seen,
            }
            for row in sorted(graph_rows or [], key=lambda row: row.backend_name)
        ],
    }


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return f"{value:.3f}"


def _json_float(value: float) -> float | None:
    if math.isnan(value) or math.isinf(value):
        return None
    return value
