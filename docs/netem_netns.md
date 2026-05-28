# Linux Network Namespace Wrapper

Status: experimental (added during v0.2.0 work).

`mrn_netem` ships an in-process loss/delay/jitter model that is fast to iterate
on but cannot exercise real DDS behavior under packet loss. The
`mrn_netem_netns` CLI brings the same `NetworkFaultProfile` YAMLs onto a real
Linux topology: one bridge in the root namespace, one network namespace per
agent, one veth pair per agent, and `tc qdisc add ... netem` applied to **both**
veth ends so traffic in either direction takes the configured hit.

The wrapper requires root for execution. The command translation itself is a
pure function and is unit-tested without root (see
`mrn_netem/test_mrn_netem/test_netns.py`).

## Frame Contract

```
root netns                       agent netns mrn_ns_robot_1
  +--------+        +-----+       +----------+
  | mrn_br0|<------>|veth_|       |veth_     |
  | 10.42  |        |robot|<----->|robot_1_n |
  | .0.1/24|        |_1_h |       |10.42.0.10|
  +--------+        +-----+       +----------+
                 tc netem here     tc netem here
```

- Bridge: `mrn_br0` (default), `10.42.0.1/24` gateway.
- One netns per agent: `mrn_ns_<agent_id>`.
- Host veth: `veth_<agent_id>_h`, attached to the bridge.
- Namespace veth: `veth_<agent_id>_n`, addressed `10.42.0.<10+index>/24`,
  default route via `10.42.0.1`.
- `tc qdisc add ... root netem` on the host veth AND the namespace veth (so an
  outbound packet from the namespace is delayed/dropped before it hits the
  bridge, and an inbound packet from the bridge is delayed/dropped before it
  reaches the application).

Constraints:

- `agent_id` must be alphanumerics or `_` and short enough that
  `veth_<agent_id>_h` fits in 15 characters (Linux `IFNAMSIZ - 1`).
- IPv6 and multicast routing are not configured. DDS must be set to use
  unicast discovery — typically `CYCLONEDDS_URI` with `<Peers>` pointing at the
  other namespaces' IPs, or `ROS_DISCOVERY_SERVER` pointing at the bridge IP.

## Commands

### plan

Prints the JSON plan (agent table, profile, setup/teardown argv lists) without
touching the system. Always run this first to confirm what would happen.

```bash
ros2 run mrn_netem mrn_netem_netns plan \
  --profile mrn_netem/config/loss20_delay80.yaml \
  --agents robot_1,robot_2
```

The output is suitable for piping into `jq` or attaching to an experiment's
provenance:

```bash
ros2 run mrn_netem mrn_netem_netns plan --agents robot_1,robot_2 \
  | jq '.agents[].ip_cidr'
```

### up

Brings the topology up. Requires root.

```bash
sudo --preserve-env=ROS_DISTRO scripts/mrn_netns.sh up \
  --profile mrn_netem/config/loss20_delay80.yaml \
  --agents robot_1,robot_2
```

`--dry-run` prints the same shell commands the script would execute, which is
useful when adapting to a different environment (e.g. a container with a
preexisting bridge).

### down

Tears everything down. Use `--ignore-errors` because `tc qdisc del` will return
non-zero if the qdisc never got created on a partial setup.

```bash
sudo scripts/mrn_netns.sh down --agents robot_1,robot_2 --ignore-errors
```

## Running ROS Nodes In A Namespace

Once `up` succeeds, each robot's stack runs inside its namespace via
`ip netns exec`:

```bash
sudo ip netns exec mrn_ns_robot_1 sudo -u "$USER" \
  env ROS_DOMAIN_ID=51 \
  bash -c 'source /opt/ros/jazzy/setup.bash && source install/setup.bash && \
           ros2 run mrn_demos synthetic_world_node.py --ros-args -p agent_id:=robot_1'
```

Important: the inner `sudo -u "$USER"` drops privileges before exec'ing the
ROS node, so the node runs as the regular user (avoiding root-owned build
caches). Pre-source the ROS overlays so the inner shell finds them.

When using Cyclone DDS, point each namespace at the same bridge for discovery:

```xml
<!-- ~/.ros/cyclonedds.xml inside each namespace -->
<CycloneDDS>
  <Domain>
    <General>
      <Interfaces>
        <NetworkInterface name="veth_robot_1_n"/>
      </Interfaces>
    </General>
    <Discovery>
      <Peers>
        <Peer Address="10.42.0.11"/>
      </Peers>
      <ParticipantIndex>auto</ParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
```

## Smoke Procedure

This is the minimal end-to-end check that proves the wrapper works on a given
host. It is **not** a CI gate — CI workers do not have CAP_NET_ADMIN.

```bash
# 1. Confirm netem is available in the running kernel.
modprobe sch_netem || sudo modprobe sch_netem

# 2. Inspect the plan.
ros2 run mrn_netem mrn_netem_netns plan \
  --profile mrn_netem/config/loss20_delay80.yaml \
  --agents robot_1,robot_2 | tee /tmp/mrn_netns_plan.json

# 3. Bring the topology up.
sudo scripts/mrn_netns.sh up \
  --profile mrn_netem/config/loss20_delay80.yaml \
  --agents robot_1,robot_2

# 4. Generate traffic. ping with the configured delay/loss should match the
#    profile's latency_ms_mean and packet_loss_percent within sampling noise.
sudo ip netns exec mrn_ns_robot_1 ping -c 50 -i 0.1 10.42.0.11 \
  | tail -2

# 5. Tear it down.
sudo scripts/mrn_netns.sh down --agents robot_1,robot_2 --ignore-errors
```

## Troubleshooting

- `RTNETLINK answers: File exists` on `ip link add mrn_br0` — a previous run
  left the bridge behind. Run `down --ignore-errors` first.
- `Error: argument "veth_robot_1_h" is wrong: "name" too long` —
  agent_id is too long for the IFNAMSIZ budget. Shorten the agent id.
- `Operation not supported` on `tc qdisc add ... netem` — the kernel does not
  have the `sch_netem` module loaded. `sudo modprobe sch_netem` should fix it.
- DDS participants in the same namespace see each other but cross-namespace
  discovery fails — discovery is multicast by default and this wrapper does
  not configure multicast routing. Switch DDS to unicast discovery (Cyclone
  `Peers` block or `ROS_DISCOVERY_SERVER`).

## Related Documents

- [`docs/experiments.md`](experiments.md) — how `mrn_netem` profiles feed
  experiment YAMLs in the synthetic path.
- `mrn_netem/config/loss20_delay80.yaml` — the canonical profile used by the
  smoke procedure above.
- [`PLAN.md`](../PLAN.md) §11 — v0.2.0 goal and acceptance.
