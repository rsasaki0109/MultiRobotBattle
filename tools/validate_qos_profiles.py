#!/usr/bin/env python3
"""Validate an mrn_comm QoS profile YAML."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

import yaml


ROOT_KEY = "mrn_qos_profiles"

REQUIRED_PROFILES = {
    "agent_state_fast",
    "relative_constraint",
    "heartbeat",
    "static_agent_info",
}

REQUIRED_FIELDS = {"reliability", "history"}

ALLOWED_RELIABILITY = {"reliable", "best_effort"}
ALLOWED_HISTORY = {"keep_last", "keep_all"}
ALLOWED_DURABILITY = {"volatile", "transient_local"}
ALLOWED_LIVELINESS = {"automatic", "manual_by_topic"}

NON_NEGATIVE_INT_FIELDS = {"depth", "lifespan_ms", "deadline_ms"}

ALLOWED_FIELDS = (
    REQUIRED_FIELDS
    | NON_NEGATIVE_INT_FIELDS
    | {"durability", "liveliness"}
)


class QosProfileError(ValueError):
    pass


def validate_qos_profiles(data: object) -> list[str]:
    if not isinstance(data, dict):
        raise QosProfileError("YAML root must be a mapping")
    if ROOT_KEY not in data:
        raise QosProfileError(f"missing top-level key: {ROOT_KEY!r}")
    profiles = data[ROOT_KEY]
    if not isinstance(profiles, dict):
        raise QosProfileError(f"{ROOT_KEY!r} must be a mapping of profile name -> fields")

    missing_profiles = REQUIRED_PROFILES - set(profiles.keys())
    if missing_profiles:
        raise QosProfileError(
            f"missing required profiles: {sorted(missing_profiles)}"
        )

    names: list[str] = []
    for name, fields in profiles.items():
        if not isinstance(name, str) or not name:
            raise QosProfileError(f"profile name must be a non-empty string: {name!r}")
        if not isinstance(fields, dict):
            raise QosProfileError(f"profile {name!r} must be a mapping")
        _validate_profile(name, fields)
        names.append(name)
    return names


def _validate_profile(name: str, fields: dict) -> None:
    missing = REQUIRED_FIELDS - set(fields.keys())
    if missing:
        raise QosProfileError(
            f"profile {name!r} missing required fields: {sorted(missing)}"
        )

    unknown = set(fields.keys()) - ALLOWED_FIELDS
    if unknown:
        raise QosProfileError(
            f"profile {name!r} has unknown fields: {sorted(unknown)}"
        )

    _check_enum(name, fields, "reliability", ALLOWED_RELIABILITY)
    _check_enum(name, fields, "history", ALLOWED_HISTORY)
    if "durability" in fields:
        _check_enum(name, fields, "durability", ALLOWED_DURABILITY)
    if "liveliness" in fields:
        _check_enum(name, fields, "liveliness", ALLOWED_LIVELINESS)

    if fields["history"] == "keep_last" and "depth" not in fields:
        raise QosProfileError(
            f"profile {name!r} uses history=keep_last but is missing depth"
        )

    for key in NON_NEGATIVE_INT_FIELDS:
        if key not in fields:
            continue
        value = fields[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise QosProfileError(
                f"profile {name!r} field {key!r} must be a non-negative integer"
            )
        if value < 0:
            raise QosProfileError(
                f"profile {name!r} field {key!r} must be >= 0 (got {value})"
            )


def _check_enum(name: str, fields: dict, key: str, allowed: Iterable[str]) -> None:
    value = fields[key]
    allowed_set = set(allowed)
    if value not in allowed_set:
        raise QosProfileError(
            f"profile {name!r} field {key!r} must be one of {sorted(allowed_set)}, "
            f"got {value!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to qos_profiles.yaml")
    parser.add_argument(
        "--print-profiles",
        action="store_true",
        help="Print validated profile names, one per line",
    )
    args = parser.parse_args(argv)
    with open(args.path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    try:
        names = validate_qos_profiles(data)
    except QosProfileError as exc:
        print(f"qos profile validation failed: {exc}", file=sys.stderr)
        return 1
    if args.print_profiles:
        for name in names:
            print(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
