"""mrn_sim: a deterministic 2D multi-robot world simulator.

The foundation for simulation-based multirobot work: a true world model with
real robot states, kinematics, obstacles, and sensor models, all pure and
ROS-free so they are unit-tested in CI. Both halves of the project plug into the
same world — the localization stack consumes the (noisy) sensor messages it
emits, and the coordination layer's velocity commands drive the robots — so a
single deterministic world closes the loop.

Modules:

- :mod:`kinematics` — unicycle (differential-drive) motion and angle helpers.
- :mod:`world` — robots, obstacles, bounds, and the collision-aware step.
- :mod:`sensors` — geometric sensor models (range/bearing, body-frame relative
  pose, GNSS) with a deterministic Gaussian-noise helper.
"""

from .kinematics import normalize_angle, unicycle_step
from .proximity import in_range_pairs, undirected_in_range
from .sensors import (
    add_gaussian_noise,
    gnss_observation,
    range_bearing,
    relative_pose_body,
)
from .world import Obstacle, Robot, World, step

__all__ = [
    "normalize_angle",
    "unicycle_step",
    "Robot",
    "Obstacle",
    "World",
    "step",
    "range_bearing",
    "relative_pose_body",
    "gnss_observation",
    "add_gaussian_noise",
    "in_range_pairs",
    "undirected_in_range",
]
