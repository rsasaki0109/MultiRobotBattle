# Overview

`multirobot-navigation` is a ROS 2-native infrastructure layer for cooperative localization and multi-robot navigation experiments under real network, time, frame, and covariance uncertainty.

The project deliberately avoids competing with Nav2 planners/controllers, Autoware autonomy stacks, and Open-RMF fleet management. It focuses on the interfaces and tooling that let those systems exchange cooperative localization information safely.

## Core Claim

The missing layer in ROS 2 multi-robot autonomy is not another planner. It is a practical layer for:

- time semantics
- frame semantics
- covariance semantics
- communication observability
- replay and evaluation
- cooperative constraints

## MVP Scope

The MVP covers:

- agent state exchange
- V2V relative pose constraints
- clock and communication status
- centralized cooperative localization graph skeleton
- synthetic replay demos
- network fault profiles
- evaluation metrics

Distributed SLAM, raw cooperative perception, fleet dashboards, and full simulator integrations are intentionally out of MVP scope.
