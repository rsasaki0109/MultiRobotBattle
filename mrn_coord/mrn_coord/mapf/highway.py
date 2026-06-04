"""Highway heuristics for bounded-suboptimal MAPF (Cohen, Uras & Koenig, 2015).

A Python reproduction of *"Feasibility Study: Using Highways for
Bounded-Suboptimal Multi-Agent Path Finding"* (Cohen, Uras & Koenig, SoCS 2015).
The idea is a piece of *human (or auto-generated) advice* layered on top of
:func:`ecbs`: a **highway** is a set of directed edges marking a preferred flow
direction across the map. ECBS already searches within a suboptimality factor
``w``; highways steer it — among the many ``w``-bounded paths an agent could
take — toward the ones that move *with* the flow. When every agent follows a
consistent circulation, head-on and crossing conflicts largely disappear before
the high level ever has to branch on them, so ECBS expands far fewer nodes for
the *same* cost guarantee.

The mechanism is a tiny, bound-preserving change to ECBS's low level. ECBS runs a
focal search whose OPEN list is ordered by the admissible ``f = g + h`` (the
lower bound that certifies the ``w`` factor) and whose FOCAL sublist — the
``w``-bounded nodes — is ranked by a secondary, inadmissible heuristic. Plain
ECBS ranks FOCAL by "fewest conflicts with the other agents". The **highway
heuristic** adds one more key: among equal-conflict paths, prefer the one with
the fewest *off-highway* moves. Because only the FOCAL *ordering* changes and
OPEN is untouched, the lower bound — and therefore ``cost <= w * optimal`` —
holds exactly; with no highway the secondary key is constant and the search is
byte-for-byte plain ECBS. (This is the "highways as a focal heuristic" variant;
``ecbs(grid, agents, w=w, highways=H)`` is the entry point.)

This module provides the entry wrapper :func:`ecbs_highway` and two standard
highway constructors — a **keep-to-one-side** rule for a two-lane corridor and a
**directed-ring** circulation — that turn a grid into the directed-edge set ECBS
consumes. A highway is *advice*, never a constraint: an agent may still leave it
(at the cost of the secondary key) whenever staying on it would break the ``w``
bound or the conflict count, so correctness never depends on the highway being
"right".
"""

from __future__ import annotations

from .ecbs import ecbs
from .grid import GridWorld


def ecbs_highway(grid: GridWorld, agents: dict, *, w: float = 1.5,
                 highways=frozenset(), max_expansions: int = 100_000,
                 stats: dict | None = None):
    """Bounded-suboptimal ECBS biased by a directed-edge ``highways`` set.

    A thin alias for :func:`ecbs` with ``highways`` supplied — the highway
    heuristic *is* ECBS with the flow-following secondary key. Same signature and
    return value (a :class:`Solution` or ``None``); ``cost <= w * optimal`` holds
    regardless of the highway, which only reorders the FOCAL list."""
    return ecbs(grid, agents, w=w, highways=frozenset(highways),
                max_expansions=max_expansions, stats=stats)


def keep_side_highway(grid: GridWorld, *, axis: str = "x") -> frozenset:
    """A *keep-to-one-side* highway for a corridor a few cells wide.

    Along ``axis`` (``"x"`` for a horizontal corridor, ``"y"`` for vertical), the
    rows (or columns) alternate flow direction — like lanes of opposing traffic —
    so agents going one way and agents going the other naturally pick different
    lanes instead of meeting head-on. Returns the set of directed edges
    ``(from_cell, to_cell)`` of the implied flow."""
    edges = set()
    cells = [(x, y) for x in range(grid.width) for y in range(grid.height)
             if grid.is_free((x, y))]
    for (x, y) in cells:
        if axis == "x":
            forward = 1 if (y % 2 == 0) else -1     # even rows ->, odd rows <-
            nb = (x + forward, y)
        else:
            forward = 1 if (x % 2 == 0) else -1     # even cols v, odd cols ^
            nb = (x, y + forward)
        if grid.is_free(nb):
            edges.add(((x, y), nb))
    return frozenset(edges)


def ring_highway(grid: GridWorld, cells) -> frozenset:
    """A directed circulation around an ordered cycle ``cells``.

    ``cells`` is a sequence of free cells forming a simple loop (consecutive
    cells, and the last, adjacent); the highway directs the flow one way around
    it (``cells[i] -> cells[i+1]``), so agents circulating the ring never run
    into one another head-on. Returns the directed-edge set."""
    seq = list(cells)
    edges = set()
    for i in range(len(seq)):
        a, b = seq[i], seq[(i + 1) % len(seq)]
        if grid.is_free(a) and grid.is_free(b):
            edges.add((a, b))
    return frozenset(edges)
