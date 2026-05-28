"""CLI for the netns wrapper.

``mrn_netem_netns plan`` prints the planned argv lists as JSON so callers can
review what *would* run. ``up`` / ``down`` execute the same commands via
``subprocess.run`` and require root unless ``--dry-run`` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from mrn_netem.netns import (
    NetnsSpec,
    build_setup_commands,
    build_spec,
    build_teardown_commands,
)
from mrn_netem.profile import NetworkFaultProfile, load_network_profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mrn_netem_netns")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["plan", "up", "down"]:
        sp = sub.add_parser(name)
        sp.add_argument(
            "--profile",
            default="",
            help="YAML network profile (only meaningful for plan/up).",
        )
        sp.add_argument(
            "--agents",
            required=True,
            help="Comma-separated agent ids (e.g. robot_1,robot_2).",
        )
        sp.add_argument("--bridge", default="mrn_br0")
        sp.add_argument("--subnet", default="10.42.0.0/24")
        sp.add_argument(
            "--bridge-host-index",
            type=int,
            default=1,
            help="Bridge gateway /24 host octet.",
        )
        sp.add_argument(
            "--first-agent-index",
            type=int,
            default=10,
            help="First /24 host octet assigned to agent_ids in order.",
        )
        sp.add_argument(
            "--dry-run",
            action="store_true",
            help="Print commands instead of executing them.",
        )
        sp.add_argument(
            "--ignore-errors",
            action="store_true",
            help="Continue when an individual command exits non-zero (teardown is the typical use).",
        )

    args = parser.parse_args(list(argv) if argv is not None else None)

    profile = _load_profile(args.profile)
    spec = build_spec(
        agent_ids=_split_agents(args.agents),
        profile=profile,
        bridge=args.bridge,
        subnet=args.subnet,
        bridge_host_index=args.bridge_host_index,
        first_agent_index=args.first_agent_index,
    )

    if args.command == "plan":
        return _emit_plan(spec)
    if args.command == "up":
        commands = build_setup_commands(spec)
    else:
        commands = build_teardown_commands(spec)

    if args.dry_run:
        _print_commands(commands)
        return 0

    if not _is_root():
        print(
            "mrn_netem_netns requires root for execution. "
            "Run via sudo or pass --dry-run.",
            file=sys.stderr,
        )
        return 2

    return _execute(commands, ignore_errors=args.ignore_errors)


def _emit_plan(spec: NetnsSpec) -> int:
    payload = {
        "bridge": spec.bridge,
        "bridge_cidr": spec.bridge_cidr,
        "agents": [
            {
                "agent_id": agent.agent_id,
                "namespace": agent.namespace,
                "veth_host": agent.veth_host,
                "veth_ns": agent.veth_ns,
                "ip_cidr": agent.ip_cidr,
            }
            for agent in spec.agents
        ],
        "profile": spec.profile.to_dict(),
        "setup_commands": [list(cmd) for cmd in build_setup_commands(spec)],
        "teardown_commands": [list(cmd) for cmd in build_teardown_commands(spec)],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _print_commands(commands: list[list[str]]) -> None:
    for cmd in commands:
        print(shlex.join(cmd))


def _execute(commands: list[list[str]], ignore_errors: bool) -> int:
    rc = 0
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=False)
        except FileNotFoundError as error:
            print(f"command not found: {error}", file=sys.stderr)
            if not ignore_errors:
                return 1
            rc = 1
            continue
        if result.returncode != 0:
            message = f"command failed ({result.returncode}): {shlex.join(cmd)}"
            print(message, file=sys.stderr)
            if not ignore_errors:
                return result.returncode
            rc = result.returncode
    return rc


def _load_profile(path: str) -> NetworkFaultProfile:
    if not path:
        return NetworkFaultProfile()
    return load_network_profile(Path(path))


def _split_agents(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _is_root() -> bool:
    geteuid = getattr(os, "geteuid", None)
    return bool(geteuid and geteuid() == 0)


if __name__ == "__main__":
    raise SystemExit(main())
