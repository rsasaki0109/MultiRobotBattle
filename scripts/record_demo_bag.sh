#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-bags/mrn_demo_3robots}"
MANIFEST="${MRN_BAG_MANIFEST:-mrn_demos/bags/mrn_demo_3robots_manifest.yaml}"
STORAGE_ID="${MRN_BAG_STORAGE_ID:-mcap}"
PRINT_TOPICS=0

if [[ "${OUTPUT_DIR}" == "--print-topics" ]]; then
  PRINT_TOPICS=1
  OUTPUT_DIR="bags/mrn_demo_3robots"
fi

mapfile -t TOPICS < <(python3 tools/validate_bag_manifest.py "${MANIFEST}" --print-topics)

if [[ "${PRINT_TOPICS}" == "1" ]]; then
  printf '%s\n' "${TOPICS[@]}"
  exit 0
fi

exec ros2 bag record -s "${STORAGE_ID}" -o "${OUTPUT_DIR}" "${TOPICS[@]}"
