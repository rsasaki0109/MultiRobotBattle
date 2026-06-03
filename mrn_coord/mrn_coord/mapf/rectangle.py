"""Rectangle symmetry reasoning for Conflict-Based Search.

Li, Harabor, Stuckey, Felner & Koenig, *"Symmetry-Breaking Constraints for
Grid-Based Multi-Agent Path Finding"* (AAAI 2019) and its journal extension
*"Pairwise symmetry reasoning for multi-agent path finding search"* (AIJ 2021).

A **rectangle symmetry** arises when two agents cross the same open rectangular
region moving in the same direction: *every* pair of their Manhattan-optimal
paths collides somewhere inside the rectangle. Plain CBS resolves this one
colliding cell at a time, and because shifting either agent by one cell just
moves the collision, it must branch through an exponential number of
symmetric permutations before the rectangle is escaped.

A **barrier constraint** breaks the whole symmetry in a single split. For a
rectangle with start corner ``Rs`` and goal corner ``Rg``, agent ``a1``'s exit
border ``R1·Rg`` and agent ``a2``'s exit border ``R2·Rg`` (where
``R1 = (Rs.x, Rg.y)`` and ``R2 = (Rg.x, Rs.y)``), the two children are:

- block ``a1`` from every cell on ``R1·Rg`` at the (Manhattan) time it would
  reach it, or
- block ``a2`` from every cell on ``R2·Rg`` likewise.

These two barriers are *mutually disjunctive*: if both agents crossed their full
exit borders on time they would collide, so every conflict-free solution
satisfies at least one child — the split preserves CBS's optimality and
completeness while collapsing all the permutations at once.

This module finds the rectangle (generalized to MDD singletons that bracket a
vertex conflict, so it fires on path *segments*, not just whole start-to-goal
paths) and returns the two barrier constraint sets. It is wired into
:func:`mrn_coord.mapf.cbsh.cbsh` behind ``rectangle=True`` (off by default).
"""

from __future__ import annotations

from .mdd import Mdd


def _sign(d: int) -> int:
    return (d > 0) - (d < 0)


def _singletons(mdd: Mdd) -> list:
    """The ``(cell, time)`` levels of the MDD that have width 1 — the cells the
    agent is *pinned* to on every optimal path."""
    out = []
    for t in range(mdd.cost + 1):
        lvl = mdd.levels[t]
        if len(lvl) == 1:
            (cell,) = tuple(lvl)
            out.append((cell, t))
    return out


def find_rectangle_barriers(mdd1: Mdd, mdd2: Mdd, conflict_time: int):
    """Find a rectangle symmetry bracketing a vertex conflict at ``conflict_time``.

    ``mdd1`` / ``mdd2`` are the two agents' MDDs at the current node. Returns
    ``(barrier1, barrier2, klass)`` where ``barrier1`` is the set of
    ``(cell, time)`` vertex constraints for agent 1 (its exit border ∩ its MDD)
    and ``barrier2`` likewise for agent 2, and ``klass`` is the rectangle's type
    (2 cardinal, 1 semi, 0 non) for prioritization. Returns ``None`` if no valid
    rectangle is found.

    Generalized version: ``S_i`` and ``G_i`` range over the MDD singletons that
    bracket the conflict (entry singletons at time ``<= conflict_time``, exit
    singletons at time ``>= conflict_time``); all combinations are tried and the
    best (highest type, then largest area) is returned.
    """
    ent1 = [s for s in _singletons(mdd1) if s[1] <= conflict_time]
    ext1 = [s for s in _singletons(mdd1) if s[1] >= conflict_time]
    ent2 = [s for s in _singletons(mdd2) if s[1] <= conflict_time]
    ext2 = [s for s in _singletons(mdd2) if s[1] >= conflict_time]

    best = None
    best_key = None
    for S1, st1 in ent1:
        for G1, gt1 in ext1:
            if gt1 <= st1:
                continue
            for S2, st2 in ent2:
                for G2, gt2 in ext2:
                    if gt2 <= st2:
                        continue
                    cand = _try_rectangle(
                        mdd1, mdd2, S1, st1, G1, gt1, S2, st2, G2, gt2)
                    if cand is None:
                        continue
                    klass, area, b1, b2 = cand
                    key = (klass, area)
                    if best_key is None or key > best_key:
                        best_key = key
                        best = (b1, b2, klass)
    return best


def _try_rectangle(mdd1, mdd2, S1, st1, G1, gt1, S2, st2, G2, gt2):
    """Validate one candidate rectangle and, if valid, build both barriers.

    Returns ``(klass, area, barrier1, barrier2)`` or ``None``.
    """
    # Both segments must be 2-D and Manhattan-optimal (no waiting inside).
    dx1, dy1 = _sign(G1[0] - S1[0]), _sign(G1[1] - S1[1])
    dx2, dy2 = _sign(G2[0] - S2[0]), _sign(G2[1] - S2[1])
    if dx1 == 0 or dy1 == 0 or dx2 == 0 or dy2 == 0:
        return None
    # Same movement direction on each axis (the rectangle-symmetry precondition).
    if dx1 != dx2 or dy1 != dy2:
        return None
    dx, dy = dx1, dy1
    if abs(S1[0] - G1[0]) + abs(S1[1] - G1[1]) != gt1 - st1:
        return None
    if abs(S2[0] - G2[0]) + abs(S2[1] - G2[1]) != gt2 - st2:
        return None
    # Phase lock: time_i(x, y) = st_i + dx*(x - S_i.x) + dy*(y - S_i.y). The two
    # agents collide throughout the rectangle iff these parameterizations agree.
    if st1 - dx * S1[0] - dy * S1[1] != st2 - dx * S2[0] - dy * S2[1]:
        return None

    rsx = max(min(S1[0], G1[0]), min(S2[0], G2[0]))
    rsy = max(min(S1[1], G1[1]), min(S2[1], G2[1]))
    rgx = min(max(S1[0], G1[0]), max(S2[0], G2[0]))
    rgy = min(max(S1[1], G1[1]), max(S2[1], G2[1]))
    # Non-degenerate area, oriented along the motion.
    if (rgx - rsx) * dx < 1 or (rgy - rsy) * dy < 1:
        return None

    # Barrier assignment. The agent leading on the x-axis (entering the rectangle
    # already advanced in x, so it must climb out along the y = Rg.y border) is
    # blocked on R1·Rg; the other, leading on y, on R2·Rg. mx_i = min(S_i.x,
    # G_i.x) is each agent's near-x edge; the larger one is the x-leader.
    mx1, mx2 = min(S1[0], G1[0]), min(S2[0], G2[0])
    my1, my2 = min(S1[1], G1[1]), min(S2[1], G2[1])
    if mx1 == mx2 or my1 == my2:
        return None  # cannot tell the two apart -> not a clean rectangle
    a1_leads_x = (mx1 > mx2) if dx > 0 else (mx1 < mx2)
    a1_leads_y = (my1 > my2) if dy > 0 else (my1 < my2)
    if a1_leads_x == a1_leads_y:
        return None  # one agent must lead x and the other y

    R1 = (rsx, rgy)  # corner on agent's y-exit border
    R2 = (rgx, rsy)  # corner on agent's x-exit border
    if a1_leads_x:
        b1 = _barrier_y(mdd1, R1, rgx, dx, st1, S1, dy)
        b2 = _barrier_x(mdd2, R2, rgy, dy, st2, S2, dx)
    else:
        b1 = _barrier_x(mdd1, R2, rgy, dy, st1, S1, dx)
        b2 = _barrier_y(mdd2, R1, rgx, dx, st2, S2, dy)
    if not b1 or not b2:
        return None
    # A barrier is *cardinal* for its agent when it cuts the MDD — every optimal
    # path crosses it, so the agent must lengthen its path to avoid it.
    klass = int(_cuts_mdd(mdd1, b1)) + int(_cuts_mdd(mdd2, b2))
    area = (abs(rgx - rsx) + 1) * (abs(rgy - rsy) + 1)
    return klass, area, frozenset(b1), frozenset(b2)


def _barrier_y(mdd, corner, rgx, dx, st, S, dy):
    """Barrier along the ``y = corner.y`` (Rg-side horizontal) border, from
    ``corner`` to ``Rg``, with each cell forbidden at the agent's Manhattan time;
    intersected with the agent's MDD."""
    y = corner[1]
    out = []
    x = corner[0]
    while True:
        t = st + dx * (x - S[0]) + dy * (y - S[1])
        if 0 <= t <= mdd.cost and (x, y) in mdd.levels[t]:
            out.append(((x, y), t))
        if x == rgx:
            break
        x += dx
    return out


def _barrier_x(mdd, corner, rgy, dy, st, S, dx):
    """Barrier along the ``x = corner.x`` (Rg-side vertical) border, from
    ``corner`` to ``Rg``, intersected with the agent's MDD."""
    x = corner[0]
    out = []
    y = corner[1]
    while True:
        t = st + dx * (x - S[0]) + dy * (y - S[1])
        if 0 <= t <= mdd.cost and (x, y) in mdd.levels[t]:
            out.append(((x, y), t))
        if y == rgy:
            break
        y += dy
    return out


def _cuts_mdd(mdd: Mdd, barrier) -> bool:
    """Whether removing the barrier ``(cell, time)`` pairs disconnects the MDD —
    i.e. every optimal path crosses the barrier, so it is a *cardinal* cut and
    avoiding it must cost the agent extra. Forward reachability through the MDD
    levels, skipping barred cells; cut iff the goal becomes unreachable."""
    blocked = set(barrier)
    reach = {cell for cell in mdd.levels[0] if (cell, 0) not in blocked}
    for t in range(mdd.cost):
        nxt = set()
        upper = mdd.levels[t + 1]
        for cell in reach:
            x, y = cell
            for nb in ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in upper and (nb, t + 1) not in blocked:
                    nxt.add(nb)
        reach = nxt
        if not reach:
            return True
    return mdd.goal not in reach
