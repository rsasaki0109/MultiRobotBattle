"""Network fault profile loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NetworkFaultProfile:
    model: str = "random"
    packet_loss_percent: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_stddev: float = 0.0
    jitter_ms: float = 0.0
    duplicate_percent: float = 0.0
    corrupt_percent: float = 0.0
    good_to_bad_probability: float = 0.0
    bad_to_good_probability: float = 0.0
    loss_percent_when_good: float = 0.0
    loss_percent_when_bad: float = 0.0

    @property
    def loss_rate(self) -> float:
        return self.packet_loss_percent / 100.0

    def to_dict(self) -> dict[str, float | str]:
        return {
            "model": self.model,
            "packet_loss_percent": self.packet_loss_percent,
            "latency_ms_mean": self.latency_ms_mean,
            "latency_ms_stddev": self.latency_ms_stddev,
            "jitter_ms": self.jitter_ms,
            "duplicate_percent": self.duplicate_percent,
            "corrupt_percent": self.corrupt_percent,
            "good_to_bad_probability": self.good_to_bad_probability,
            "bad_to_good_probability": self.bad_to_good_probability,
            "loss_percent_when_good": self.loss_percent_when_good,
            "loss_percent_when_bad": self.loss_percent_when_bad,
        }


def load_network_profile(path: str | Path) -> NetworkFaultProfile:
    profile_path = Path(path)
    with profile_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"invalid network profile: {profile_path}")
    network = data.get("network", data)
    if not isinstance(network, dict):
        raise ValueError(f"network profile must contain a mapping: {profile_path}")
    return NetworkFaultProfile(
        model=str(network.get("model", "random")),
        packet_loss_percent=_percent(network, "packet_loss_percent", 0.0),
        latency_ms_mean=_nonnegative_float(network, "latency_ms_mean", 0.0),
        latency_ms_stddev=_nonnegative_float(network, "latency_ms_stddev", 0.0),
        jitter_ms=_nonnegative_float(network, "jitter_ms", 0.0),
        duplicate_percent=_percent(network, "duplicate_percent", 0.0),
        corrupt_percent=_percent(network, "corrupt_percent", 0.0),
        good_to_bad_probability=_probability(network, "good_to_bad_probability", 0.0),
        bad_to_good_probability=_probability(network, "bad_to_good_probability", 0.0),
        loss_percent_when_good=_percent(network, "loss_percent_when_good", 0.0),
        loss_percent_when_bad=_percent(network, "loss_percent_when_bad", 0.0),
    )


def _nonnegative_float(data: dict[str, Any], key: str, default: float) -> float:
    value = float(data.get(key, default))
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _percent(data: dict[str, Any], key: str, default: float) -> float:
    value = _nonnegative_float(data, key, default)
    if value > 100.0:
        raise ValueError(f"{key} must be <= 100")
    return value


def _probability(data: dict[str, Any], key: str, default: float) -> float:
    value = _nonnegative_float(data, key, default)
    if value > 1.0:
        raise ValueError(f"{key} must be <= 1")
    return value
