"""Tests for tools/validate_qos_profiles.py against mrn_comm/config/qos_profiles.yaml."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "tools" / "validate_qos_profiles.py"
QOS_YAML_PATH = REPO_ROOT / "mrn_comm" / "config" / "qos_profiles.yaml"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_qos_profiles", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("validate_qos_profiles", module)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def _shipped_yaml():
    with open(QOS_YAML_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _base_valid():
    return {
        "mrn_qos_profiles": {
            "agent_state_fast": {
                "reliability": "best_effort",
                "history": "keep_last",
                "depth": 5,
                "durability": "volatile",
                "lifespan_ms": 300,
            },
            "relative_constraint": {
                "reliability": "reliable",
                "history": "keep_last",
                "depth": 20,
                "durability": "volatile",
                "lifespan_ms": 2000,
            },
            "heartbeat": {
                "reliability": "best_effort",
                "history": "keep_last",
                "depth": 3,
                "deadline_ms": 500,
                "liveliness": "automatic",
            },
            "static_agent_info": {
                "reliability": "reliable",
                "durability": "transient_local",
                "history": "keep_last",
                "depth": 1,
            },
        }
    }


def test_shipped_yaml_passes():
    names = validator.validate_qos_profiles(_shipped_yaml())
    assert set(names) >= {
        "agent_state_fast",
        "relative_constraint",
        "heartbeat",
        "static_agent_info",
    }


def test_root_must_be_mapping():
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles([])


def test_missing_top_level_key():
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles({"something_else": {}})


def test_missing_required_profile():
    data = _base_valid()
    del data["mrn_qos_profiles"]["heartbeat"]
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles(data)


def test_unknown_field_rejected():
    data = _base_valid()
    data["mrn_qos_profiles"]["heartbeat"]["bogus"] = 1
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles(data)


def test_invalid_reliability_rejected():
    data = _base_valid()
    data["mrn_qos_profiles"]["heartbeat"]["reliability"] = "kinda_reliable"
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles(data)


def test_invalid_history_rejected():
    data = _base_valid()
    data["mrn_qos_profiles"]["heartbeat"]["history"] = "keep_some"
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles(data)


def test_invalid_durability_rejected():
    data = _base_valid()
    data["mrn_qos_profiles"]["static_agent_info"]["durability"] = "permanent"
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles(data)


def test_keep_last_requires_depth():
    data = _base_valid()
    del data["mrn_qos_profiles"]["agent_state_fast"]["depth"]
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles(data)


def test_negative_depth_rejected():
    data = _base_valid()
    data["mrn_qos_profiles"]["agent_state_fast"]["depth"] = -1
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles(data)


def test_non_int_lifespan_rejected():
    data = _base_valid()
    data["mrn_qos_profiles"]["agent_state_fast"]["lifespan_ms"] = "300"
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles(data)


def test_missing_reliability_rejected():
    data = _base_valid()
    del data["mrn_qos_profiles"]["heartbeat"]["reliability"]
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles(data)


def test_invalid_liveliness_rejected():
    data = _base_valid()
    data["mrn_qos_profiles"]["heartbeat"]["liveliness"] = "always_on"
    with pytest.raises(validator.QosProfileError):
        validator.validate_qos_profiles(data)
