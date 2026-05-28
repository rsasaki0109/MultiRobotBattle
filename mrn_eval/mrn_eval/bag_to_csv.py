"""Pure-function helpers for exporting bag messages to offline-ATE CSV.

The CLI in ``bag_to_csv_cli`` is what consumes these; tests exercise the
extractors directly with hand-built message-shaped objects so that the
extraction logic is verified without a real bag and without rosbag2_py.

There are two extractor families:

- *pose* extractors take a duck-typed deserialized message and return a
  :class:`~mrn_eval.offline_ate.TrajectorySample` (``stamp_sec,x,y,z``).
- *geodetic* extractors (currently ``sensor_msgs/msg/NavSatFix``) return a
  :class:`~mrn_eval.rtk_to_csv.RtkSample`
  (``stamp_sec,lat_deg,lon_deg,alt_m,fix_quality``). These cannot become a
  metric trajectory directly — they need an ENU origin — so the CLI writes
  them in the RTK input schema for ``mrn_eval_rtk_to_csv`` to linearize.

The CLI is responsible for iterating the bag with rosbag2_py and calling
the right family based on the recorded message type.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable

from mrn_gnss import FixQuality

from .offline_ate import TrajectorySample
from .rtk_to_csv import RtkSample


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _from_header_and_pose(header, position) -> TrajectorySample:
    return TrajectorySample(
        stamp_sec=_stamp_to_sec(header.stamp),
        x=float(position.x),
        y=float(position.y),
        z=float(position.z),
    )


def _extract_agent_state(msg) -> TrajectorySample:
    # AgentState carries the V2V packet header (see mrn_msgs/V2VPacketHeader.msg)
    # whose std_msgs/Header carries the measurement timestamp.
    return _from_header_and_pose(msg.packet.header, msg.pose.pose.position)


def _extract_cooperative_pose(msg) -> TrajectorySample:
    return _from_header_and_pose(msg.header, msg.pose.pose.position)


def _extract_odometry(msg) -> TrajectorySample:
    return _from_header_and_pose(msg.header, msg.pose.pose.position)


def _extract_pose_stamped(msg) -> TrajectorySample:
    return _from_header_and_pose(msg.header, msg.pose.position)


def _extract_pose_with_covariance_stamped(msg) -> TrajectorySample:
    return _from_header_and_pose(msg.header, msg.pose.pose.position)


EXTRACTORS: dict[str, Callable[[object], TrajectorySample]] = {
    "mrn_msgs/msg/AgentState": _extract_agent_state,
    "mrn_msgs/msg/CooperativePose": _extract_cooperative_pose,
    "nav_msgs/msg/Odometry": _extract_odometry,
    "geometry_msgs/msg/PoseStamped": _extract_pose_stamped,
    "geometry_msgs/msg/PoseWithCovarianceStamped": _extract_pose_with_covariance_stamped,
}


def _extract_navsatfix(msg) -> RtkSample:
    # sensor_msgs/NavSatFix carries geodetic lat/lon/alt plus a
    # NavSatStatus; map that status to the closest GGA-style FixQuality.
    return RtkSample(
        stamp_sec=_stamp_to_sec(msg.header.stamp),
        lat_deg=float(msg.latitude),
        lon_deg=float(msg.longitude),
        alt_m=float(msg.altitude),
        fix_quality=FixQuality.from_navsatstatus(int(msg.status.status)),
    )


GEODETIC_EXTRACTORS: dict[str, Callable[[object], RtkSample]] = {
    "sensor_msgs/msg/NavSatFix": _extract_navsatfix,
}


def supported_message_types() -> tuple[str, ...]:
    """Pose message types that map directly to ``stamp_sec,x,y,z``."""
    return tuple(sorted(EXTRACTORS))


def supported_geodetic_message_types() -> tuple[str, ...]:
    """Geodetic message types that map to the RTK input CSV schema."""
    return tuple(sorted(GEODETIC_EXTRACTORS))


def is_geodetic_message_type(message_type: str) -> bool:
    return message_type in GEODETIC_EXTRACTORS


def extract_sample(message_type: str, msg) -> TrajectorySample:
    """Return a TrajectorySample for a known pose message type.

    Raises ``ValueError`` for an unsupported message type so the CLI can
    map it to exit code 2.
    """
    try:
        extractor = EXTRACTORS[message_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported message type {message_type!r}; "
            f"supported: {list(supported_message_types())}"
        ) from exc
    return extractor(msg)


def extract_geodetic_sample(message_type: str, msg) -> RtkSample:
    """Return an RtkSample for a known geodetic message type.

    Raises ``ValueError`` for an unsupported message type so the CLI can
    map it to exit code 2.
    """
    try:
        extractor = GEODETIC_EXTRACTORS[message_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported geodetic message type {message_type!r}; "
            f"supported: {list(supported_geodetic_message_types())}"
        ) from exc
    return extractor(msg)


CSV_HEADER = ("stamp_sec", "x", "y", "z")


def write_csv(path: Path, samples: Iterable[TrajectorySample]) -> int:
    """Write samples to ``path`` as ``stamp_sec,x,y,z`` rows.

    Returns the number of rows written. The caller is responsible for
    ordering (the reader in ``offline_ate.load_trajectory_csv`` sorts on
    load, so writing in bag order is fine).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for sample in samples:
            writer.writerow(
                (
                    f"{sample.stamp_sec:.9f}",
                    f"{sample.x:.6f}",
                    f"{sample.y:.6f}",
                    f"{sample.z:.6f}",
                )
            )
            count += 1
    return count
