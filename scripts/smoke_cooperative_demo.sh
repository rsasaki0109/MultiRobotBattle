#!/usr/bin/env bash
set -euo pipefail

DURATION="${1:-20}"
OUTPUT="${2:-out/smoke_report.md}"
JSON_OUTPUT="${MRN_SMOKE_JSON_OUTPUT:-out/smoke_metrics.json}"
DOMAIN_ID="${ROS_DOMAIN_ID:-62}"
LAUNCH_LOG="${MRN_SMOKE_LAUNCH_LOG:-out/smoke_launch.log}"
SUMMARY_SAMPLE="${MRN_SMOKE_SUMMARY_SAMPLE:-out/smoke_summary.yaml}"
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

mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${LAUNCH_LOG}")" "$(dirname "${SUMMARY_SAMPLE}")"

export ROS_DOMAIN_ID="${DOMAIN_ID}"
setsid ros2 launch mrn_demos "${LAUNCH_FILE}" "${LAUNCH_ARGS[@]}" >"${LAUNCH_LOG}" 2>&1 &
LAUNCH_PID=$!

sleep 3

if ! timeout 20s ros2 topic echo /mrn/eval/summary --once --no-arr >"${SUMMARY_SAMPLE}"; then
  echo "Timed out waiting for /mrn/eval/summary" >&2
  echo "--- launch log ---" >&2
  sed -n '1,160p' "${LAUNCH_LOG}" >&2 || true
  exit 1
fi

ros2 run mrn_eval mrn_report \
  --duration "${DURATION}" \
  --output "${OUTPUT}" \
  --json-output "${JSON_OUTPUT}"

grep -q "robot_2" "${OUTPUT}"
grep -q "cooperative" "${OUTPUT}"
grep -q "local_only" "${OUTPUT}"
grep -q "Network Diagnostics" "${OUTPUT}"
test -s "${JSON_OUTPUT}"
python3 - "${JSON_OUTPUT}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    data = json.load(stream)

rows = data.get("rows", [])
robot_2_coop = [
    row for row in rows
    if row.get("agent_id") == "robot_2" and row.get("method") == "cooperative"
]
if not robot_2_coop:
    raise SystemExit("missing robot_2 cooperative row")
improvement = robot_2_coop[0].get("improvement_vs_local")
if improvement is None or improvement <= 0.2:
    raise SystemExit(f"robot_2 cooperative improvement too small: {improvement}")

network_rows = data.get("network_rows", [])
if not network_rows:
    raise SystemExit("missing network diagnostics rows")
robot_2_links = [
    row for row in network_rows
    if row.get("local_agent_id") == "robot_2" or row.get("remote_agent_id") == "robot_2"
]
if not robot_2_links:
    raise SystemExit("missing robot_2 network diagnostics")
if not any((row.get("loss_rate") or 0.0) > 0.0 for row in network_rows):
    raise SystemExit("network diagnostics did not observe packet loss")
PY

echo "smoke report: ${OUTPUT}"
echo "smoke metrics: ${JSON_OUTPUT}"
