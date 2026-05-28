import json
import unittest

from mrn_eval.report import (
    EvaluationReport,
    GraphStatusRow,
    NetworkRow,
    SummaryRow,
    format_markdown_report,
)


class TestReport(unittest.TestCase):
    def test_report_formats_improvement(self):
        markdown = format_markdown_report(
            [
                SummaryRow(
                    experiment_name="demo",
                    method_name="robot_2/local_only",
                    ate_rmse=1.6,
                    localization_availability=1.0,
                    messages_seen=3,
                ),
                SummaryRow(
                    experiment_name="demo",
                    method_name="robot_2/cooperative",
                    ate_rmse=0.3,
                    localization_availability=1.0,
                    messages_seen=3,
                ),
            ],
            duration_sec=5.0,
        )
        self.assertIn("| robot_2 | cooperative | 0.300 | 1.300 | 1.000 | 3 |", markdown)

    def test_report_formats_network_diagnostics(self):
        markdown = format_markdown_report(
            [],
            network_rows=[
                NetworkRow(
                    local_agent_id="robot_1",
                    remote_agent_id="robot_2",
                    loss_rate=0.2,
                    latency_mean_sec=0.08,
                    latency_stddev_sec=0.015,
                    max_latency_sec=0.12,
                    received_count=8,
                    lost_count=2,
                    qos_profile_name="relative_constraint",
                    transport_name="synthetic_loopback",
                    messages_seen=4,
                )
            ],
        )
        self.assertIn("## Network Diagnostics", markdown)
        self.assertIn(
            "| robot_1->robot_2 | 0.200 | 80.000 | 15.000 | 120.000 | 8 | 2 |",
            markdown,
        )

    def test_report_formats_graph_status(self):
        markdown = format_markdown_report(
            [],
            graph_rows=[
                GraphStatusRow(
                    backend_name="relative_anchor",
                    accepted_constraint_count=10,
                    rejected_constraint_count=4,
                    stale_constraint_count=1,
                    rejection_reasons={"clock_offset_exceeds_threshold": 3, "stale": 1},
                    last_rejection_reason="clock_offset_exceeds_threshold",
                    messages_seen=5,
                )
            ],
        )
        self.assertIn("## Graph Status", markdown)
        self.assertIn("Reject Rate", markdown)
        self.assertIn(
            "| relative_anchor | 10 | 4 | 1 | 0.267 | clock_offset_exceeds_threshold |",
            markdown,
        )
        self.assertIn("clock_offset_exceeds_threshold:3", markdown)

    def test_graph_status_row_rejection_rate(self):
        row = GraphStatusRow(
            backend_name="relative_anchor",
            accepted_constraint_count=8,
            rejected_constraint_count=1,
            stale_constraint_count=1,
            rejection_reasons={"stale": 1},
        )
        self.assertEqual(row.total_constraint_count, 10)
        self.assertAlmostEqual(row.rejection_rate, 0.1)

    def test_graph_status_row_rejection_rate_zero_total(self):
        row = GraphStatusRow(
            backend_name="dummy",
            accepted_constraint_count=0,
            rejected_constraint_count=0,
            stale_constraint_count=0,
            rejection_reasons={},
        )
        self.assertEqual(row.rejection_rate, 0.0)

    def test_report_updates_message_count(self):
        report = EvaluationReport()
        report.update("demo", "robot_1/local_only", 0.2, 1.0)
        report.update("demo", "robot_1/local_only", 0.1, 1.0)
        self.assertEqual(report.rows[0].messages_seen, 2)
        self.assertIn("0.100", report.to_markdown())

    def test_report_json_includes_improvement(self):
        report = EvaluationReport()
        report.update("demo", "robot_2/local_only", 1.6, 1.0)
        report.update("demo", "robot_2/cooperative", 0.3, 1.0)
        data = json.loads(report.to_json(duration_sec=5.0))
        cooperative = [
            row for row in data["rows"]
            if row["agent_id"] == "robot_2" and row["method"] == "cooperative"
        ][0]
        self.assertAlmostEqual(cooperative["improvement_vs_local"], 1.3)
        self.assertEqual(data["duration_sec"], 5.0)

    def test_report_json_includes_network_rows(self):
        report = EvaluationReport()
        report.update_network(
            "robot_1",
            "robot_2",
            loss_rate=0.2,
            latency_mean_sec=0.08,
            latency_stddev_sec=0.015,
            max_latency_sec=0.12,
            received_count=8,
            lost_count=2,
            qos_profile_name="relative_constraint",
            transport_name="synthetic_loopback",
        )
        report.update_network(
            "robot_1",
            "robot_2",
            loss_rate=0.25,
            latency_mean_sec=0.09,
            latency_stddev_sec=0.020,
            max_latency_sec=0.16,
            received_count=9,
            lost_count=3,
        )
        data = json.loads(report.to_json())
        row = data["network_rows"][0]
        self.assertEqual(row["link_name"], "robot_1->robot_2")
        self.assertAlmostEqual(row["loss_rate"], 0.25)
        self.assertEqual(row["messages_seen"], 2)

    def test_report_json_includes_graph_rows(self):
        report = EvaluationReport()
        report.update_graph_status(
            backend_name="relative_anchor",
            accepted_constraint_count=10,
            rejected_constraint_count=4,
            stale_constraint_count=1,
            rejection_reasons={"clock_offset_exceeds_threshold": 4},
            last_rejection_reason="clock_offset_exceeds_threshold",
        )
        report.update_graph_status(
            backend_name="relative_anchor",
            accepted_constraint_count=11,
            rejected_constraint_count=5,
            stale_constraint_count=1,
            rejection_reasons={"clock_offset_exceeds_threshold": 5},
            last_rejection_reason="clock_offset_exceeds_threshold",
        )
        data = json.loads(report.to_json())
        row = data["graph_rows"][0]
        self.assertEqual(row["backend_name"], "relative_anchor")
        self.assertEqual(row["accepted_constraint_count"], 11)
        self.assertEqual(row["rejected_constraint_count"], 5)
        self.assertEqual(row["messages_seen"], 2)
        self.assertEqual(row["total_constraint_count"], 17)
        self.assertAlmostEqual(row["rejection_rate"], 5 / 17)


if __name__ == "__main__":
    unittest.main()
