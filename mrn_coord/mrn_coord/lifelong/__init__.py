r"""Lifelong / online MAPF: agents that never stop, on a rolling horizon.

One-shot MAPF (``mrn_coord.mapf.cbs`` / ``prioritized_planning``) solves a fixed
set of start→goal pairs once and ends when everyone has arrived. Lifelong MAPF
models the warehouse-robot regime instead: tasks keep arriving, an agent that
reaches its goal is immediately assigned the next one, and the team plans *while
moving*, forever. The figure of merit is **throughput** — tasks completed per
timestep — not makespan.

This package adds that loop on top of the existing single-agent space-time A\*:
:func:`run_lifelong` runs a rolling-horizon, reservation-based prioritized
planner (replan every tick, commit one step, rotate priority for fairness) that
is collision-free by construction and reassigns tasks from a deterministic
stream as they complete.
"""

from .lifelong import LifelongResult, TaskStream, make_warehouse, run_lifelong

__all__ = [
    "LifelongResult",
    "TaskStream",
    "make_warehouse",
    "run_lifelong",
]
