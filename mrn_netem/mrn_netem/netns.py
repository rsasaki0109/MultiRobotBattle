"""Linux network namespace wrapper for the mrn_netem fault profiles.

The functions in this module translate a ``NetworkFaultProfile`` plus a list of
agent ids into ordered ``ip``/``tc`` argv lists. They are deliberately pure so
that the translation can be unit-tested without root or netns privileges.
Actual execution lives in ``mrn_netem.netns_cli``; this module only describes
what *would* happen.

Frame contract:

- One Linux bridge in the root namespace (default ``mrn_br0``).
- One network namespace per agent (``mrn_ns_<agent_id>``).
- One veth pair per agent. The host side is attached to the bridge, the
  namespace side carries an address in a configurable /24 (default
  ``10.42.0.0/24``).
- ``tc qdisc add ... root netem`` is applied on **both** veth ends so traffic in
  either direction sees the configured loss/delay/jitter.

The wrapper intentionally does not provide IPv6 or multicast routing — the
target use case is DDS unicast discovery via ``ROS_DISCOVERY_SERVER`` or
``CYCLONEDDS_URI`` pointing at the bridge gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mrn_netem.profile import NetworkFaultProfile


_MAX_IFNAME = 15  # Linux IFNAMSIZ - 1


@dataclass(frozen=True)
class NetnsAgent:
    agent_id: str
    namespace: str
    veth_host: str
    veth_ns: str
    ip_cidr: str

    @property
    def ip_address(self) -> str:
        return self.ip_cidr.split("/", 1)[0]


@dataclass(frozen=True)
class NetnsSpec:
    bridge: str
    bridge_cidr: str
    agents: tuple[NetnsAgent, ...]
    profile: NetworkFaultProfile

    @property
    def bridge_ip(self) -> str:
        return self.bridge_cidr.split("/", 1)[0]


def build_spec(
    agent_ids: Iterable[str],
    profile: NetworkFaultProfile,
    bridge: str = "mrn_br0",
    subnet: str = "10.42.0.0/24",
    bridge_host_index: int = 1,
    first_agent_index: int = 10,
) -> NetnsSpec:
    """Translate an ordered agent list into a concrete ``NetnsSpec``.

    Agent ids are used verbatim in the derived interface names so that two
    agents cannot collide. The veth host side ends up as ``veth_<agent>_h`` and
    the namespace side as ``veth_<agent>_n``; either name exceeding the 15-char
    IFNAMSIZ limit raises ``ValueError`` rather than truncating silently.
    """
    network, mask = subnet.split("/", 1)
    base_octets = network.split(".")
    if len(base_octets) != 4:
        raise ValueError(f"subnet must be IPv4 a.b.c.d/N: {subnet}")
    base_third = base_octets[2]
    seen_namespaces: set[str] = set()
    seen_interfaces: set[str] = set()
    agents = []
    for index, agent_id in enumerate(agent_ids):
        cleaned = _validate_agent_id(agent_id)
        namespace = f"mrn_ns_{cleaned}"
        veth_host = f"veth_{cleaned}_h"
        veth_ns = f"veth_{cleaned}_n"
        for name, kind in (
            (veth_host, "host veth"),
            (veth_ns, "namespace veth"),
        ):
            if len(name) > _MAX_IFNAME:
                raise ValueError(
                    f"derived {kind} name {name!r} exceeds {_MAX_IFNAME} chars"
                )
        if namespace in seen_namespaces:
            raise ValueError(f"duplicate agent_id derives namespace {namespace}")
        if veth_host in seen_interfaces or veth_ns in seen_interfaces:
            raise ValueError(
                f"duplicate agent_id derives veth interface for {agent_id}"
            )
        seen_namespaces.add(namespace)
        seen_interfaces.add(veth_host)
        seen_interfaces.add(veth_ns)
        host_octet = first_agent_index + index
        if host_octet > 254 or host_octet == bridge_host_index:
            raise ValueError(
                f"agent {agent_id} maps to invalid /24 host octet {host_octet}"
            )
        ip_cidr = f"{base_octets[0]}.{base_octets[1]}.{base_third}.{host_octet}/{mask}"
        agents.append(
            NetnsAgent(
                agent_id=cleaned,
                namespace=namespace,
                veth_host=veth_host,
                veth_ns=veth_ns,
                ip_cidr=ip_cidr,
            )
        )
    if not agents:
        raise ValueError("at least one agent_id is required")
    bridge_cidr = f"{base_octets[0]}.{base_octets[1]}.{base_third}.{bridge_host_index}/{mask}"
    return NetnsSpec(
        bridge=bridge,
        bridge_cidr=bridge_cidr,
        agents=tuple(agents),
        profile=profile,
    )


def build_setup_commands(spec: NetnsSpec) -> list[list[str]]:
    """Return the ordered argv lists that bring the topology up."""
    commands: list[list[str]] = []
    commands.append(["ip", "link", "add", spec.bridge, "type", "bridge"])
    commands.append(["ip", "addr", "add", spec.bridge_cidr, "dev", spec.bridge])
    commands.append(["ip", "link", "set", spec.bridge, "up"])
    for agent in spec.agents:
        commands.append(["ip", "netns", "add", agent.namespace])
        commands.append(
            [
                "ip",
                "link",
                "add",
                agent.veth_host,
                "type",
                "veth",
                "peer",
                "name",
                agent.veth_ns,
            ]
        )
        commands.append(
            ["ip", "link", "set", agent.veth_ns, "netns", agent.namespace]
        )
        commands.append(
            ["ip", "link", "set", agent.veth_host, "master", spec.bridge]
        )
        commands.append(["ip", "link", "set", agent.veth_host, "up"])
        commands.append(
            ["ip", "-n", agent.namespace, "link", "set", "lo", "up"]
        )
        commands.append(
            ["ip", "-n", agent.namespace, "link", "set", agent.veth_ns, "up"]
        )
        commands.append(
            [
                "ip",
                "-n",
                agent.namespace,
                "addr",
                "add",
                agent.ip_cidr,
                "dev",
                agent.veth_ns,
            ]
        )
        commands.append(
            [
                "ip",
                "-n",
                agent.namespace,
                "route",
                "add",
                "default",
                "via",
                spec.bridge_ip,
            ]
        )

    netem_args = build_netem_args(spec.profile)
    if netem_args:
        for agent in spec.agents:
            commands.append(
                [
                    "tc",
                    "qdisc",
                    "add",
                    "dev",
                    agent.veth_host,
                    "root",
                    "netem",
                    *netem_args,
                ]
            )
            commands.append(
                [
                    "ip",
                    "netns",
                    "exec",
                    agent.namespace,
                    "tc",
                    "qdisc",
                    "add",
                    "dev",
                    agent.veth_ns,
                    "root",
                    "netem",
                    *netem_args,
                ]
            )
    return commands


def build_teardown_commands(spec: NetnsSpec) -> list[list[str]]:
    """Return ordered argv lists that tear the topology down.

    Each command is safe to ignore on failure: ``ip``/``tc`` return non-zero if
    the resource is already missing, but the wrapper is expected to keep going.
    Commands are listed in dependency-safe order (qdisc → veth → namespace →
    bridge).
    """
    commands: list[list[str]] = []
    for agent in spec.agents:
        commands.append(
            ["tc", "qdisc", "del", "dev", agent.veth_host, "root"]
        )
        commands.append(
            [
                "ip",
                "netns",
                "exec",
                agent.namespace,
                "tc",
                "qdisc",
                "del",
                "dev",
                agent.veth_ns,
                "root",
            ]
        )
        commands.append(["ip", "link", "del", agent.veth_host])
        commands.append(["ip", "netns", "del", agent.namespace])
    commands.append(["ip", "link", "del", spec.bridge])
    return commands


def build_netem_args(profile: NetworkFaultProfile) -> list[str]:
    """Translate the in-process profile into tc netem operands.

    An empty list means the profile is the identity transform: callers may then
    skip the ``tc qdisc add`` step entirely.
    """
    args: list[str] = []
    if profile.packet_loss_percent > 0.0:
        args += ["loss", f"{profile.packet_loss_percent:g}%"]
    if profile.latency_ms_mean > 0.0 or profile.jitter_ms > 0.0:
        jitter = profile.jitter_ms or profile.latency_ms_stddev
        if jitter > 0.0:
            args += [
                "delay",
                f"{profile.latency_ms_mean:g}ms",
                f"{jitter:g}ms",
            ]
        else:
            args += ["delay", f"{profile.latency_ms_mean:g}ms"]
    if profile.duplicate_percent > 0.0:
        args += ["duplicate", f"{profile.duplicate_percent:g}%"]
    if profile.corrupt_percent > 0.0:
        args += ["corrupt", f"{profile.corrupt_percent:g}%"]
    return args


def _validate_agent_id(agent_id: str) -> str:
    cleaned = str(agent_id).strip()
    if not cleaned:
        raise ValueError("agent_id must be non-empty")
    if not all(char.isalnum() or char == "_" for char in cleaned):
        raise ValueError(
            f"agent_id {agent_id!r} must contain only alphanumerics or '_'"
        )
    return cleaned
