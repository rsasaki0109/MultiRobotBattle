r"""Lifelong / online MAPF: agents that never stop, on a rolling horizon.

One-shot MAPF (``mrn_coord.mapf.cbs`` / ``prioritized_planning``) solves a fixed
set of start→goal pairs once and ends when everyone has arrived. Lifelong MAPF
models the warehouse-robot regime instead: tasks keep arriving, an agent that
reaches its goal is immediately assigned the next one, and the team plans *while
moving*, forever. The figure of merit is **throughput** — tasks completed per
timestep — not makespan.

This package adds that loop on top of PIBT: :func:`run_lifelong` steps a
collision-free configuration each tick and reassigns tasks as they complete.
How free robots are matched to tasks is pluggable (``allocator=``): round-robin
by default, or cost-aware :func:`~mrn_coord.lifelong.allocation.auction` /
:func:`~mrn_coord.lifelong.allocation.hungarian` matching by travel distance,
which lifts throughput by sending the *nearest* free robot to each task.
"""

from .allocation import auction, hungarian
from .lifelong import (
    LifelongResult,
    TaskStream,
    make_warehouse,
    pibt_solve,
    run_lifelong,
)
from .rhcr import run_rhcr
from .token_passing import run_token_passing
from .token_passing_swaps import PickupDelivery, run_tpts

__all__ = [
    "LifelongResult",
    "TaskStream",
    "make_warehouse",
    "pibt_solve",
    "run_lifelong",
    "run_rhcr",
    "run_token_passing",
    "PickupDelivery",
    "run_tpts",
    "auction",
    "hungarian",
]
