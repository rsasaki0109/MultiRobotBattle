"""Deterministic packet loss models for replay tests."""

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class RandomLossModel:
    """Drop packets independently with a fixed probability."""

    loss_rate: float
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.loss_rate <= 1.0:
            raise ValueError("loss_rate must be in [0.0, 1.0]")

    def mask(self, count: int) -> list[bool]:
        rng = random.Random(self.seed)
        return [rng.random() < self.loss_rate for _ in range(count)]
