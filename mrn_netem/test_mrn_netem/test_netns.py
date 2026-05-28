import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mrn_netem.netns import (
    build_netem_args,
    build_setup_commands,
    build_spec,
    build_teardown_commands,
)
from mrn_netem.netns_cli import main as netns_main
from mrn_netem.profile import NetworkFaultProfile


def _basic_profile() -> NetworkFaultProfile:
    return NetworkFaultProfile(
        packet_loss_percent=20.0,
        latency_ms_mean=80.0,
        jitter_ms=15.0,
        duplicate_percent=1.0,
        corrupt_percent=0.5,
    )


class TestBuildNetemArgs(unittest.TestCase):
    def test_identity_profile_emits_no_args(self):
        self.assertEqual(build_netem_args(NetworkFaultProfile()), [])

    def test_loss_only(self):
        profile = NetworkFaultProfile(packet_loss_percent=12.5)
        self.assertEqual(build_netem_args(profile), ["loss", "12.5%"])

    def test_delay_with_jitter(self):
        profile = NetworkFaultProfile(latency_ms_mean=80.0, jitter_ms=15.0)
        self.assertEqual(build_netem_args(profile), ["delay", "80ms", "15ms"])

    def test_delay_uses_stddev_when_jitter_absent(self):
        profile = NetworkFaultProfile(latency_ms_mean=50.0, latency_ms_stddev=10.0)
        self.assertEqual(build_netem_args(profile), ["delay", "50ms", "10ms"])

    def test_full_profile(self):
        args = build_netem_args(_basic_profile())
        self.assertEqual(
            args,
            [
                "loss", "20%",
                "delay", "80ms", "15ms",
                "duplicate", "1%",
                "corrupt", "0.5%",
            ],
        )


class TestBuildSpec(unittest.TestCase):
    def test_two_agents_get_distinct_namespaces_and_ips(self):
        spec = build_spec(["robot_1", "robot_2"], NetworkFaultProfile())
        self.assertEqual(spec.bridge, "mrn_br0")
        self.assertEqual(spec.bridge_cidr, "10.42.0.1/24")
        self.assertEqual(spec.agents[0].namespace, "mrn_ns_robot_1")
        self.assertEqual(spec.agents[0].veth_host, "veth_robot_1_h")
        self.assertEqual(spec.agents[0].veth_ns, "veth_robot_1_n")
        self.assertEqual(spec.agents[0].ip_cidr, "10.42.0.10/24")
        self.assertEqual(spec.agents[1].ip_cidr, "10.42.0.11/24")
        self.assertNotEqual(spec.agents[0].veth_host, spec.agents[1].veth_host)

    def test_rejects_empty_agent_list(self):
        with self.assertRaisesRegex(ValueError, "at least one agent_id"):
            build_spec([], NetworkFaultProfile())

    def test_rejects_invalid_agent_chars(self):
        with self.assertRaisesRegex(ValueError, "alphanumerics"):
            build_spec(["robot 1"], NetworkFaultProfile())

    def test_rejects_long_agent_id(self):
        with self.assertRaisesRegex(ValueError, "exceeds 15 chars"):
            build_spec(["robot_id_12345"], NetworkFaultProfile())

    def test_rejects_duplicate_agent_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate agent_id"):
            build_spec(["robot_1", "robot_1"], NetworkFaultProfile())

    def test_subnet_third_octet_respected(self):
        spec = build_spec(
            ["robot_1"],
            NetworkFaultProfile(),
            subnet="172.30.5.0/24",
            first_agent_index=20,
        )
        self.assertEqual(spec.bridge_cidr, "172.30.5.1/24")
        self.assertEqual(spec.agents[0].ip_cidr, "172.30.5.20/24")


class TestBuildSetupCommands(unittest.TestCase):
    def test_brings_up_bridge_then_agents_in_order(self):
        spec = build_spec(["robot_1", "robot_2"], _basic_profile())
        cmds = build_setup_commands(spec)
        self.assertEqual(cmds[0], ["ip", "link", "add", "mrn_br0", "type", "bridge"])
        self.assertEqual(cmds[1], ["ip", "addr", "add", "10.42.0.1/24", "dev", "mrn_br0"])
        self.assertEqual(cmds[2], ["ip", "link", "set", "mrn_br0", "up"])
        self.assertIn(
            ["ip", "netns", "add", "mrn_ns_robot_1"],
            cmds,
        )
        self.assertIn(
            [
                "ip",
                "link",
                "add",
                "veth_robot_1_h",
                "type",
                "veth",
                "peer",
                "name",
                "veth_robot_1_n",
            ],
            cmds,
        )
        # tc netem applied on both veth ends with all profile fields.
        tc_host = [
            "tc",
            "qdisc",
            "add",
            "dev",
            "veth_robot_1_h",
            "root",
            "netem",
            "loss",
            "20%",
            "delay",
            "80ms",
            "15ms",
            "duplicate",
            "1%",
            "corrupt",
            "0.5%",
        ]
        tc_ns = [
            "ip",
            "netns",
            "exec",
            "mrn_ns_robot_1",
            "tc",
            "qdisc",
            "add",
            "dev",
            "veth_robot_1_n",
            "root",
            "netem",
            "loss",
            "20%",
            "delay",
            "80ms",
            "15ms",
            "duplicate",
            "1%",
            "corrupt",
            "0.5%",
        ]
        self.assertIn(tc_host, cmds)
        self.assertIn(tc_ns, cmds)

    def test_identity_profile_skips_tc_qdisc_add(self):
        spec = build_spec(["robot_1"], NetworkFaultProfile())
        cmds = build_setup_commands(spec)
        self.assertFalse(any(cmd[0] == "tc" for cmd in cmds))
        self.assertFalse(
            any(cmd[:3] == ["ip", "netns", "exec"] and "tc" in cmd for cmd in cmds)
        )


class TestBuildTeardownCommands(unittest.TestCase):
    def test_teardown_removes_qdisc_then_links_then_bridge(self):
        spec = build_spec(["robot_1"], _basic_profile())
        cmds = build_teardown_commands(spec)
        order = [cmd[0] + ":" + cmd[2] if len(cmd) > 2 else cmd[0] for cmd in cmds]
        # Last command must remove the bridge.
        self.assertEqual(cmds[-1], ["ip", "link", "del", "mrn_br0"])
        # qdisc deletion must precede link deletion for the same veth.
        qdisc_idx = next(
            i for i, c in enumerate(cmds) if c[:5] == ["tc", "qdisc", "del", "dev", "veth_robot_1_h"]
        )
        link_idx = next(
            i for i, c in enumerate(cmds) if c == ["ip", "link", "del", "veth_robot_1_h"]
        )
        self.assertLess(qdisc_idx, link_idx)


class TestCli(unittest.TestCase):
    def test_plan_emits_complete_json(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            rc = netns_main(["plan", "--agents", "robot_1,robot_2"])
        self.assertEqual(rc, 0)
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["bridge"], "mrn_br0")
        self.assertEqual(payload["agents"][0]["namespace"], "mrn_ns_robot_1")
        self.assertIn("setup_commands", payload)
        self.assertIn("teardown_commands", payload)
        self.assertIsInstance(payload["setup_commands"], list)
        self.assertTrue(all(isinstance(cmd, list) for cmd in payload["setup_commands"]))

    def test_plan_with_profile_includes_netem_args(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.yaml"
            profile_path.write_text(
                "network:\n  packet_loss_percent: 5\n  latency_ms_mean: 30\n",
                encoding="utf-8",
            )
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                rc = netns_main([
                    "plan",
                    "--agents",
                    "robot_1",
                    "--profile",
                    str(profile_path),
                ])
        self.assertEqual(rc, 0)
        payload = json.loads(stream.getvalue())
        flattened = [tok for cmd in payload["setup_commands"] for tok in cmd]
        self.assertIn("loss", flattened)
        self.assertIn("5%", flattened)
        self.assertIn("30ms", flattened)
        self.assertAlmostEqual(payload["profile"]["packet_loss_percent"], 5.0)

    def test_up_dry_run_prints_commands_without_executing(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            rc = netns_main(["up", "--agents", "robot_1", "--dry-run"])
        self.assertEqual(rc, 0)
        text = stream.getvalue()
        self.assertIn("ip link add mrn_br0 type bridge", text)
        self.assertIn("ip netns add mrn_ns_robot_1", text)

    def test_down_dry_run_prints_teardown(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            rc = netns_main(["down", "--agents", "robot_1", "--dry-run"])
        self.assertEqual(rc, 0)
        text = stream.getvalue()
        self.assertIn("ip link del mrn_br0", text)
        self.assertIn("ip netns del mrn_ns_robot_1", text)

    def test_up_without_root_errors(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), mock.patch(
            "mrn_netem.netns_cli._is_root", return_value=False
        ):
            rc = netns_main(["up", "--agents", "robot_1"])
        self.assertEqual(rc, 2)
        self.assertIn("requires root", err.getvalue())

    def test_up_as_root_invokes_subprocess(self):
        recorded: list[list[str]] = []

        class _Result:
            returncode = 0

        def _fake_run(cmd, check=False):
            recorded.append(list(cmd))
            return _Result()

        with mock.patch(
            "mrn_netem.netns_cli._is_root", return_value=True
        ), mock.patch("mrn_netem.netns_cli.subprocess.run", side_effect=_fake_run):
            rc = netns_main(["up", "--agents", "robot_1"])
        self.assertEqual(rc, 0)
        self.assertGreater(len(recorded), 5)
        self.assertEqual(recorded[0][:4], ["ip", "link", "add", "mrn_br0"])

    def test_ignore_errors_continues_on_failure(self):
        class _FailResult:
            returncode = 1

        recorded: list[list[str]] = []

        def _fake_run(cmd, check=False):
            recorded.append(list(cmd))
            return _FailResult()

        with mock.patch(
            "mrn_netem.netns_cli._is_root", return_value=True
        ), mock.patch("mrn_netem.netns_cli.subprocess.run", side_effect=_fake_run):
            rc = netns_main([
                "down",
                "--agents",
                "robot_1",
                "--ignore-errors",
            ])
        self.assertEqual(rc, 1)
        # ensure every command was attempted, not just the first.
        self.assertGreater(len(recorded), 3)


if __name__ == "__main__":
    unittest.main()
