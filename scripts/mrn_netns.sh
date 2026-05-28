#!/usr/bin/env bash
# Thin sudo wrapper around `mrn_netem_netns` so callers do not have to remember
# the entry-point name. Mirrors what `ros2 run mrn_netem mrn_netem_netns ...`
# would invoke, but bypasses sourcing install/ when run from an already-sourced
# shell.
#
# Usage:
#   scripts/mrn_netns.sh plan --agents robot_1,robot_2
#   sudo scripts/mrn_netns.sh up   --profile mrn_netem/config/loss20_delay80.yaml --agents robot_1,robot_2
#   sudo scripts/mrn_netns.sh down --agents robot_1,robot_2 --ignore-errors
#
# Always inspect the plan first (`plan` writes JSON to stdout) before running
# `up` against a live system. See docs/netem_netns.md for the frame contract,
# expected ROS_DOMAIN_ID handling, and troubleshooting.

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec ros2 run mrn_netem mrn_netem_netns --help
fi

exec ros2 run mrn_netem mrn_netem_netns "$@"
