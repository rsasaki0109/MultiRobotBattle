# Contributing

This repository is infrastructure-first. Keep changes scoped, replayable, and documented.

## Priorities

1. Preserve message, frame, covariance, and time semantics.
2. Keep demos reproducible from launch files, bags, and YAML configs.
3. Prefer small packages with one responsibility.
4. Add diagnostics for failure modes before adding complex algorithms.
5. Keep docs English-first.

## Development

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
colcon test
colcon test-result --verbose
```

## Pull Requests

Every behavior change should include:

- a focused summary
- test or launch instructions
- docs updates when interfaces, frames, time, QoS, or covariance semantics change
- a note about compatibility impact
