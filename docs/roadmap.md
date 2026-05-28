# Roadmap

## Q1: MVP / Alpha

- ROS 2 Jazzy baseline
- message contracts
- synthetic three-robot demo
- centralized cooperative localization skeleton
- packet loss, latency, and clock drift profiles
- RViz and Foxglove visualization assets
- MCAP bag manifest format
- benchmark report scaffold
- CI smoke demo for `/mrn/eval/summary`

Release gate: see `docs/release_checklist.md`.

## Q2: Real Robots, Nav2, Zenoh

- two-robot real bag
- Nav2 adapter
- `map -> robot_i/odom` correction integration
- Zenoh backend experiment
- real packet loss tests

## Q3: Autoware, GNSS, Datasets

- Autoware adapter
- GNSS/ENU utilities
- RTK quality handling
- dataset adapter prototypes
- robust graph backend

## Q4: Distributed and Shared World Hooks

- federated graph summary exchange
- landmark and submap constraints
- cooperative perception message hooks
- RSU agent support
- edge gateway experiments
