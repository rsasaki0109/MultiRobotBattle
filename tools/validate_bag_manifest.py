#!/usr/bin/env python3
"""Validate an MRN bag manifest."""

from __future__ import annotations

import argparse
import sys
import yaml


REQUIRED_KEYS = {"dataset", "agents", "topics"}
REQUIRED_DATASET_KEYS = {"name", "version", "ros_distro", "storage"}
REQUIRED_AGENT_KEYS = {"id", "base_frame", "odom_frame"}
REQUIRED_TOPIC_KEYS = {"name", "type", "required"}
REQUIRED_TOPICS = {
    "/clock",
    "/tf",
    "/tf_static",
    "/mrn/eval/summary",
    "/mrn/graph/status",
    "/mrn/viz/markers",
}


class ManifestError(ValueError):
    pass


def validate_manifest(data: object) -> list[str]:
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a YAML mapping")

    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ManifestError(f"missing required keys: {sorted(missing)}")

    _validate_dataset(data["dataset"])
    agent_ids = _validate_agents(data["agents"])
    topic_names = _validate_topics(data["topics"])
    _validate_agent_topics(agent_ids, topic_names)
    _validate_faults(data.get("faults", {}), agent_ids)
    _validate_metrics(data.get("metrics", []))
    return topic_names_in_manifest(data)


def topic_names_in_manifest(data: dict, required_only: bool = False) -> list[str]:
    names: list[str] = []
    for topic in data["topics"]:
        if required_only and not bool(topic["required"]):
            continue
        names.append(str(topic["name"]))
    return names


def _validate_dataset(dataset: object) -> None:
    if not isinstance(dataset, dict):
        raise ManifestError("dataset must be a mapping")
    missing = REQUIRED_DATASET_KEYS - set(dataset.keys())
    if missing:
        raise ManifestError(f"dataset missing keys: {sorted(missing)}")
    if dataset["storage"] != "mcap":
        raise ManifestError("dataset.storage must be 'mcap'")


def _validate_agents(agents: object) -> list[str]:
    if not isinstance(agents, list) or not agents:
        raise ManifestError("agents must be a non-empty list")
    agent_ids: list[str] = []
    for index, agent in enumerate(agents):
        if not isinstance(agent, dict):
            raise ManifestError(f"agents[{index}] must be a mapping")
        missing = REQUIRED_AGENT_KEYS - set(agent.keys())
        if missing:
            raise ManifestError(f"agents[{index}] missing keys: {sorted(missing)}")
        agent_id = str(agent["id"])
        if agent_id in agent_ids:
            raise ManifestError(f"duplicate agent id: {agent_id}")
        if not str(agent["base_frame"]).startswith(f"{agent_id}/"):
            raise ManifestError(f"{agent_id} base_frame must be namespaced")
        if not str(agent["odom_frame"]).startswith(f"{agent_id}/"):
            raise ManifestError(f"{agent_id} odom_frame must be namespaced")
        agent_ids.append(agent_id)
    return agent_ids


def _validate_topics(topics: object) -> set[str]:
    if not isinstance(topics, list) or not topics:
        raise ManifestError("topics must be a non-empty list")
    names: set[str] = set()
    for index, topic in enumerate(topics):
        if not isinstance(topic, dict):
            raise ManifestError(f"topics[{index}] must be a mapping")
        missing = REQUIRED_TOPIC_KEYS - set(topic.keys())
        if missing:
            raise ManifestError(f"topics[{index}] missing keys: {sorted(missing)}")
        name = str(topic["name"])
        if not name.startswith("/"):
            raise ManifestError(f"topic must be absolute: {name}")
        if name in names:
            raise ManifestError(f"duplicate topic: {name}")
        if not isinstance(topic["required"], bool):
            raise ManifestError(f"topic required must be bool: {name}")
        names.add(name)

    missing_topics = REQUIRED_TOPICS - names
    if missing_topics:
        raise ManifestError(f"missing required topics: {sorted(missing_topics)}")
    return names


def _validate_agent_topics(agent_ids: list[str], topic_names: set[str]) -> None:
    per_agent_suffixes = [
        "/ground_truth/pose",
        "/local/odometry",
        "/local/gnss_pose",
        "/mrn/agent_state",
        "/mrn/cooperative_odom",
        "/mrn/cooperative_pose",
        "/mrn/relative_constraints",
        "/mrn/comm_status",
        "/mrn/clock_status",
    ]
    missing: list[str] = []
    for agent_id in agent_ids:
        for suffix in per_agent_suffixes:
            topic = f"/{agent_id}{suffix}"
            if topic not in topic_names:
                missing.append(topic)
    if missing:
        raise ManifestError(f"missing per-agent topics: {missing}")


def _validate_faults(faults: object, agent_ids: list[str]) -> None:
    if faults in ({}, None):
        return
    if not isinstance(faults, dict):
        raise ManifestError("faults must be a mapping")
    packet_loss = float(faults.get("packet_loss_percent", 0.0))
    if not 0.0 <= packet_loss <= 100.0:
        raise ManifestError("faults.packet_loss_percent must be in [0, 100]")
    outage = faults.get("gnss_outage", {})
    if outage and not isinstance(outage, dict):
        raise ManifestError("faults.gnss_outage must be a mapping")
    for agent_id in outage:
        if agent_id not in agent_ids:
            raise ManifestError(f"gnss_outage references unknown agent: {agent_id}")


def _validate_metrics(metrics: object) -> None:
    if metrics in ([], None):
        return
    if not isinstance(metrics, list):
        raise ManifestError("metrics must be a list")
    valid = {"ate", "rpe", "nees", "latency", "bandwidth", "packet_loss"}
    unknown = sorted(set(str(metric) for metric in metrics) - valid)
    if unknown:
        raise ManifestError(f"unknown metrics: {unknown}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--print-topics", action="store_true")
    parser.add_argument("--required-only", action="store_true")
    args = parser.parse_args()

    with open(args.manifest, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    try:
        topic_names = validate_manifest(data)
    except ManifestError as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.print_topics:
        topics = topic_names_in_manifest(data, required_only=args.required_only)
        for topic in topics:
            print(topic)
    else:
        print(f"valid manifest: {args.manifest} topics={len(topic_names)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
