# QoS Profiles

The project does not make every topic reliable. Stale reliable messages can be worse than dropped messages.

Initial profiles live in `mrn_comm/config/qos_profiles.yaml`.

## Policy

- high-rate agent state: best effort, short lifespan
- heartbeat: best effort, shallow history
- relative constraints: reliable candidate, explicit TTL
- static agent info: reliable and transient local

All profiles must be evaluated under packet loss, latency, and jitter.

## Replay Benchmark

`experiments/qos_best_effort_vs_reliable.yaml` is the first synthetic QoS
comparison. It runs the same cooperative graph method twice with generated
scenario overrides:

- `qos_profile_best_effort_fast`: `agent_state_fast`, 30% packet loss, 40 ms
  mean latency
- `qos_profile_reliable_constraints`: `relative_constraint`, 5% packet loss,
  100 ms mean latency

The synthetic transport is not a DDS implementation. It makes the network
tradeoff visible in replay artifacts: best-effort-like links should show higher
loss and lower latency; reliable-constraint-like links should show lower loss
and higher latency. The acceptance rules check the observed communication rows
by sweep case and QoS profile name.
