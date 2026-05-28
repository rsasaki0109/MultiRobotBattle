"""Tests for tools/validate_bag.py against synthetic bag metadata."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "tools" / "validate_bag.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_bag", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("validate_bag", module)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _write_metadata(bag_dir: Path, *, storage="mcap", topics=None):
    topics = topics if topics is not None else [
        ("/clock", "rosgraph_msgs/msg/Clock"),
        ("/robot_1/mrn/agent_state", "mrn_msgs/msg/AgentState"),
    ]
    info = {
        "rosbag2_bagfile_information": {
            "version": 6,
            "storage_identifier": storage,
            "duration": {"nanoseconds": 1_000_000_000},
            "starting_time": {"nanoseconds_since_epoch": 0},
            "message_count": sum(1 for _ in topics) * 10,
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": name,
                        "type": message_type,
                        "serialization_format": "cdr",
                        "offered_qos_profiles": "",
                    },
                    "message_count": 10,
                }
                for name, message_type in topics
            ],
        }
    }
    metadata_path = bag_dir / "metadata.yaml"
    with open(metadata_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(info, fh)


def _manifest(topics):
    return {
        "dataset": {
            "name": "demo",
            "version": "0.0.1",
            "ros_distro": "jazzy",
            "storage": "mcap",
        },
        "agents": [{"id": "robot_1", "base_frame": "robot_1/base", "odom_frame": "robot_1/odom"}],
        "topics": [
            {"name": name, "type": message_type, "required": required}
            for name, message_type, required in topics
        ],
    }


def test_valid_bag_returns_summary(tmp_path):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    _write_metadata(bag_dir)
    manifest = _manifest([
        ("/clock", "rosgraph_msgs/msg/Clock", True),
        ("/robot_1/mrn/agent_state", "mrn_msgs/msg/AgentState", True),
    ])
    summary = validator.validate_bag(bag_dir, manifest)
    assert summary["storage"] == "mcap"
    assert summary["topic_count"] == 2
    assert summary["required_topic_count"] == 2
    assert summary["extra_topics"] == []


def test_missing_metadata_yaml_raises(tmp_path):
    bag_dir = tmp_path / "empty_bag"
    bag_dir.mkdir()
    with pytest.raises(validator.BagValidationError):
        validator.validate_bag(bag_dir, _manifest([]))


def test_storage_mismatch_rejected(tmp_path):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    _write_metadata(bag_dir, storage="sqlite3")
    with pytest.raises(validator.BagValidationError) as excinfo:
        validator.validate_bag(bag_dir, _manifest([]))
    assert "storage_identifier" in str(excinfo.value)


def test_missing_required_topic_rejected(tmp_path):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    _write_metadata(
        bag_dir,
        topics=[("/clock", "rosgraph_msgs/msg/Clock")],
    )
    manifest = _manifest([
        ("/clock", "rosgraph_msgs/msg/Clock", True),
        ("/robot_1/mrn/agent_state", "mrn_msgs/msg/AgentState", True),
    ])
    with pytest.raises(validator.BagValidationError) as excinfo:
        validator.validate_bag(bag_dir, manifest)
    assert "/robot_1/mrn/agent_state" in str(excinfo.value)


def test_type_mismatch_rejected(tmp_path):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    _write_metadata(
        bag_dir,
        topics=[("/robot_1/mrn/agent_state", "geometry_msgs/msg/PoseStamped")],
    )
    manifest = _manifest([
        ("/robot_1/mrn/agent_state", "mrn_msgs/msg/AgentState", True),
    ])
    with pytest.raises(validator.BagValidationError) as excinfo:
        validator.validate_bag(bag_dir, manifest)
    assert "/robot_1/mrn/agent_state" in str(excinfo.value)
    assert "geometry_msgs/msg/PoseStamped" in str(excinfo.value)


def test_optional_topic_missing_is_ok(tmp_path):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    _write_metadata(
        bag_dir,
        topics=[("/clock", "rosgraph_msgs/msg/Clock")],
    )
    manifest = _manifest([
        ("/clock", "rosgraph_msgs/msg/Clock", True),
        ("/diagnostics", "diagnostic_msgs/msg/DiagnosticArray", False),
    ])
    summary = validator.validate_bag(bag_dir, manifest)
    assert summary["required_topic_count"] == 1


def test_extra_topic_reported(tmp_path):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    _write_metadata(
        bag_dir,
        topics=[
            ("/clock", "rosgraph_msgs/msg/Clock"),
            ("/extra/topic", "std_msgs/msg/String"),
        ],
    )
    manifest = _manifest([("/clock", "rosgraph_msgs/msg/Clock", True)])
    summary = validator.validate_bag(bag_dir, manifest)
    assert summary["extra_topics"] == ["/extra/topic"]


def test_malformed_metadata_root_raises(tmp_path):
    bag_dir = tmp_path / "bag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text(textwrap.dedent("not_a_dict:\n  - 1\n"))
    with pytest.raises(validator.BagValidationError):
        validator.validate_bag(bag_dir, _manifest([]))
