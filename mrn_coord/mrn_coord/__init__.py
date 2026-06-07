"""mrn_coord: the multi-robot coordination layer.

This package is the "navigation/coordination" half of multirobot-battle —
the counterpart to the cooperative-localization stack. Where the localization
side answers *where are we*, this side answers *how do we move and what do we
do together*. It follows the same project pattern: pure, ROS-free algorithm
cores that are unit-tested in CI, with thin ROS/CLI wiring layered on top.

Submodules:

- :mod:`mrn_coord.mapf` — multi-agent path finding (collision-free planning on
  a shared grid): space-time A*, Conflict-Based Search, prioritized planning.

Planned: ``formation`` (decentralized formation control reusing the V2V
relative-pose constraints) and ``coverage`` (cooperative exploration / task
allocation).
"""
