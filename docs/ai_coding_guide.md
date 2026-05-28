# AI Coding Guide

This repository should remain easy for coding agents to modify safely.

## Principles

- one package, one responsibility
- small public interfaces
- YAML experiments
- explicit acceptance criteria
- docs near every interface
- small bags for CI
- deterministic synthetic scenarios

## Good Tasks

- add one message and document its semantics
- add one covariance validator
- add one loss model
- add one metric with a deterministic test
- add one launch smoke test

## Bad Tasks

- implement cooperative localization
- integrate every simulator
- build fleet management
- add distributed SLAM
- rewrite the core in a new language
