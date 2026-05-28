# Architecture Decisions

## ADR-0001: Infrastructure First

Status: Accepted

Context: Navigation stacks, autonomy stacks, and fleet systems already exist. Multi-robot systems still fail around time, frames, covariance, communication, and replay.

Decision: This repository focuses on cooperative robotics infrastructure before planners, controllers, perception models, or dashboards.

Consequences:

- Nav2, Autoware, and Open-RMF integration happens through adapters.
- Message and replay contracts are prioritized over heavy algorithms.

## ADR-0002: Centralized Graph for MVP

Status: Accepted

Context: Distributed graph optimization is difficult to debug without stable interfaces, bags, and metrics.

Decision: The MVP starts with a centralized graph server.

Consequences:

- deterministic baseline
- easier visualization
- not fully distributed

## ADR-0003: Jazzy Baseline

Status: Accepted

Context: ROS 2 Jazzy is a stable LTS baseline. Newer distributions may need package ecosystem catch-up.

Decision: Jazzy is the first supported baseline. Newer LTS CI can be added experimentally.
