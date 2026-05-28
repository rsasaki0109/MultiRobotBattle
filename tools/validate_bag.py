#!/usr/bin/env python3
"""Validate a recorded rosbag2 directory against an MRN manifest.

Reads ``<bag_dir>/metadata.yaml`` written by rosbag2 and cross-checks the
recorded topics, message types, and storage backend against the manifest YAML
documented in ``docs/bag_capture.md``. ``tools/validate_bag_manifest.py``
covers the manifest schema; this tool answers the complementary question:
"did the recorded bag actually contain what the manifest promised?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


class BagValidationError(ValueError):
    pass


def _load_metadata(bag_dir: Path) -> dict:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.is_file():
        raise BagValidationError(
            f"bag metadata.yaml not found at {metadata_path}"
        )
    with open(metadata_path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise BagValidationError("metadata.yaml must be a YAML mapping")
    info = data.get("rosbag2_bagfile_information")
    if not isinstance(info, dict):
        raise BagValidationError(
            "metadata.yaml missing 'rosbag2_bagfile_information' mapping"
        )
    return info


def _extract_topic_types(info: dict) -> dict[str, str]:
    raw = info.get("topics_with_message_count")
    if not isinstance(raw, list):
        raise BagValidationError(
            "metadata.yaml missing 'topics_with_message_count' list"
        )
    topic_types: dict[str, str] = {}
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise BagValidationError(
                f"topics_with_message_count[{index}] must be a mapping"
            )
        metadata = entry.get("topic_metadata")
        if not isinstance(metadata, dict):
            raise BagValidationError(
                f"topics_with_message_count[{index}] missing topic_metadata"
            )
        name = str(metadata.get("name", "")).strip()
        if not name:
            raise BagValidationError(
                f"topics_with_message_count[{index}] missing name"
            )
        message_type = str(metadata.get("type", "")).strip()
        if not message_type:
            raise BagValidationError(f"topic {name} missing message type")
        topic_types[name] = message_type
    return topic_types


def _required_topic_types(manifest: dict) -> dict[str, str]:
    topics = manifest.get("topics")
    if not isinstance(topics, list):
        raise BagValidationError("manifest missing 'topics' list")
    required: dict[str, str] = {}
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        if not bool(topic.get("required", False)):
            continue
        required[str(topic["name"])] = str(topic["type"])
    return required


def validate_bag(bag_dir: Path, manifest: dict) -> dict:
    """Return summary; raise BagValidationError on contract failure."""
    info = _load_metadata(bag_dir)

    storage_identifier = str(info.get("storage_identifier", "")).strip()
    expected_storage = str(
        manifest.get("dataset", {}).get("storage", "mcap")
    ).strip()
    if storage_identifier != expected_storage:
        raise BagValidationError(
            "bag storage_identifier "
            f"{storage_identifier!r} does not match manifest "
            f"dataset.storage {expected_storage!r}"
        )

    bag_topic_types = _extract_topic_types(info)
    required_topic_types = _required_topic_types(manifest)

    missing_topics = sorted(
        topic for topic in required_topic_types if topic not in bag_topic_types
    )
    if missing_topics:
        raise BagValidationError(
            f"bag is missing required topics: {missing_topics}"
        )

    mismatched: list[str] = []
    for topic, expected_type in required_topic_types.items():
        actual_type = bag_topic_types[topic]
        if actual_type != expected_type:
            mismatched.append(
                f"{topic}: manifest={expected_type} bag={actual_type}"
            )
    if mismatched:
        raise BagValidationError(
            "topic message types do not match manifest: " + "; ".join(mismatched)
        )

    return {
        "storage": storage_identifier,
        "topic_count": len(bag_topic_types),
        "required_topic_count": len(required_topic_types),
        "extra_topics": sorted(
            topic for topic in bag_topic_types if topic not in required_topic_types
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if not args.bag_dir.is_dir():
        print(f"bag directory not found: {args.bag_dir}", file=sys.stderr)
        return 1

    with open(args.manifest, "r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)

    try:
        summary = validate_bag(args.bag_dir, manifest)
    except BagValidationError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(
        f"valid bag: {args.bag_dir} storage={summary['storage']} "
        f"topics={summary['topic_count']} "
        f"required={summary['required_topic_count']} "
        f"extra={len(summary['extra_topics'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
