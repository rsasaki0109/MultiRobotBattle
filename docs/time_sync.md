# Time Sync

Time handling must support live robots, simulation, and rosbag replay.

## Required Times

Every V2V measurement must preserve:

- measurement time
- source publish time
- receive time
- graph insert time

`V2VPacketHeader` carries measurement and publish time. Receive and graph insert time are runtime diagnostics.

## Time Authority

Supported authority modes:

- `sim_time`
- `system_time`
- `chrony`
- `ptp`
- `gnss_pps`

Unknown clock offset must be treated as a risk, not as zero offset.

## Rejection Gates

Messages may be rejected when:

- message age exceeds TTL
- message receive age exceeds TTL or the backend's fixed-lag buffer
- clock offset exceeds the configured threshold
- offset uncertainty exceeds the configured threshold
- source publish time is implausibly before measurement time
- source publish time appears to be in the future after clock-offset correction
- target state buffer has no state near measurement time
- ROS time jumps invalidate buffered data

## Clock Offset Sign

`ClockOffsetEstimate.estimated_offset` follows this convention:

```text
remote_time ~= local_time + estimated_offset
```

When validating a packet timestamp in the local receiver clock domain, use
`local_time ~= remote_time - estimated_offset`.

The default sync gate rejects unknown clock offset for live V2V validation.
Replay-only consumers may explicitly set `reject_if_unknown_offset=false` when
all packet timestamps are already in one simulated clock domain.

## Current Implementation

`mrn_sync.time_gate` provides shared Python gates for:

- `validate_clock_offset`: validates pairwise clock offset and offset uncertainty
- `validate_v2v_packet_time`: validates packet time, TTL, clock offset, offset uncertainty, and future skew
- `validate_receive_age`: validates how long a message has been held after receipt

Graph backends use `validate_receive_age` for stored agent states and relative
constraints so stale data cannot silently remain active.

Graph backends also subscribe to `/robot_i/mrn/clock_status` and apply
pairwise clock-offset gates to relative constraints. The cooperative demo keeps
`reject_unknown_clock_offset=false` so startup and replay do not drop all
constraints before the first clock-status sample arrives, but known offsets
above `max_clock_offset_sec` are rejected.
