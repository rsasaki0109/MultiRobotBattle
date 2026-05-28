#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-45}"
OUTPUT="${2:-out/report.md}"
JSON_OUTPUT="${3:-out/metrics.json}"
DOMAIN_ID="${ROS_DOMAIN_ID:-51}"
LAUNCH_FILE="${MRN_LAUNCH_FILE:-cooperative_localization.launch.py}"
LAUNCH_ARGS_STRING="${MRN_LAUNCH_ARGS:-}"
LAUNCH_ARGS=()
if [[ -n "${LAUNCH_ARGS_STRING}" ]]; then
  read -r -a LAUNCH_ARGS <<< "${LAUNCH_ARGS_STRING}"
fi

cleanup() {
  if [[ -n "${LAUNCH_PID:-}" ]]; then
    kill -INT -"${LAUNCH_PID}" >/dev/null 2>&1 || true
    sleep 0.5
    kill -TERM -"${LAUNCH_PID}" >/dev/null 2>&1 || true
    wait "${LAUNCH_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

export ROS_DOMAIN_ID="${DOMAIN_ID}"
setsid ros2 launch mrn_demos "${LAUNCH_FILE}" "${LAUNCH_ARGS[@]}" &
LAUNCH_PID=$!

sleep 3
ros2 run mrn_eval mrn_report \
  --duration "${DURATION}" \
  --output "${OUTPUT}" \
  --json-output "${JSON_OUTPUT}"
