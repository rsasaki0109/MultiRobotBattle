#!/usr/bin/env python3
"""Export one topic from a rosbag2 directory to CSV.

For a *pose* topic the output is ``stamp_sec,x,y,z``, which
``mrn_eval_offline_ate`` consumes directly (``--estimated`` / ``--truth``).

For a *geodetic* topic (``sensor_msgs/msg/NavSatFix``) the output is the RTK
input schema ``stamp_sec,lat_deg,lon_deg,alt_m,fix_quality``; feed that to
``mrn_eval_rtk_to_csv`` to linearize it into an ENU truth CSV (the position
cannot be metric without an ENU origin).

Heavy ROS dependencies (``rosbag2_py`` and ``rclpy.serialization``) are
imported lazily inside :func:`main` so that the pure-function extractors in
:mod:`mrn_eval.bag_to_csv` can be unit-tested in a CI environment that does
not have rosbag2_py available.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from mrn_eval.bag_to_csv import (
    extract_geodetic_sample,
    extract_sample,
    is_geodetic_message_type,
    supported_geodetic_message_types,
    supported_message_types,
    write_csv,
)
from mrn_eval.rtk_to_csv import write_rtk_csv


def _iter_bag_messages(bag_dir: Path, storage_id: str, topic: str):
    """Yield (message_type, deserialized_msg) pairs for ``topic`` in ``bag_dir``.

    Imported lazily so ``mrn_eval.bag_to_csv`` stays import-safe without
    rosbag2_py.
    """
    rosbag2_py = importlib.import_module("rosbag2_py")
    serialization = importlib.import_module("rclpy.serialization")
    get_message = importlib.import_module("rosidl_runtime_py.utilities").get_message

    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir), storage_id=storage_id
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_by_topic = {meta.name: meta.type for meta in reader.get_all_topics_and_types()}
    if topic not in type_by_topic:
        raise ValueError(
            f"topic {topic!r} not found in bag; available: {sorted(type_by_topic)}"
        )
    message_type = type_by_topic[topic]
    message_class = get_message(message_type)

    storage_filter = rosbag2_py.StorageFilter(topics=[topic])
    reader.set_filter(storage_filter)

    while reader.has_next():
        recorded_topic, serialized, _bag_timestamp_ns = reader.read_next()
        if recorded_topic != topic:
            continue
        msg = serialization.deserialize_message(serialized, message_class)
        yield message_type, msg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mrn_eval_bag_to_csv",
        description=(
            "Export pose stamps from one topic in a rosbag2 directory to "
            "stamp_sec,x,y,z CSV (input for mrn_eval_offline_ate)."
        ),
    )
    parser.add_argument("bag_dir", type=Path)
    parser.add_argument(
        "--topic",
        required=True,
        help="Topic name to export (must be one of the supported message types).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--storage-id",
        default="mcap",
        help="rosbag2 storage backend (default: mcap).",
    )
    parser.add_argument(
        "--list-types",
        action="store_true",
        help="Print supported message types and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_types:
        for message_type in supported_message_types():
            print(message_type)
        for message_type in supported_geodetic_message_types():
            print(f"{message_type}  (geodetic -> feed mrn_eval_rtk_to_csv)")
        return 0

    if not args.bag_dir.is_dir():
        print(f"error: bag directory not found: {args.bag_dir}", file=sys.stderr)
        return 2

    try:
        iterator = _iter_bag_messages(args.bag_dir, args.storage_id, args.topic)
        pose_samples = []
        geodetic_samples = []
        message_type_seen: str | None = None
        geodetic = False
        for message_type, msg in iterator:
            if message_type_seen is None:
                message_type_seen = message_type
                geodetic = is_geodetic_message_type(message_type)
            try:
                if geodetic:
                    geodetic_samples.append(
                        extract_geodetic_sample(message_type, msg)
                    )
                else:
                    pose_samples.append(extract_sample(message_type, msg))
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
    except ImportError as exc:
        print(
            f"error: rosbag2_py not available ({exc}); install ros-<distro>-rosbag2-py",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    total = len(geodetic_samples) if geodetic else len(pose_samples)
    if total == 0:
        print(
            f"error: no messages on topic {args.topic!r} in {args.bag_dir}",
            file=sys.stderr,
        )
        return 3

    if geodetic:
        written = write_rtk_csv(args.output, geodetic_samples)
        print(
            f"wrote {written} geodetic rows to {args.output} "
            f"(topic={args.topic} type={message_type_seen}); "
            f"feed into mrn_eval_rtk_to_csv to get an ENU truth CSV"
        )
    else:
        written = write_csv(args.output, pose_samples)
        print(
            f"wrote {written} rows to {args.output} "
            f"(topic={args.topic} type={message_type_seen})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
