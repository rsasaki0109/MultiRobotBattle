"""Continuous-time Conflict-Based Search (CCBS).

A Python reproduction of Andreychuk, Yakovlev, Atzmon & Stern's *"Multi-Agent
Pathfinding with Continuous Time"* (IJCAI 2019; AIJ 2022). Classical CBS lives on
a discrete clock: every move takes exactly one timestep, conflicts are
"same cell, same integer tick" or "swap", and an agent that must yield waits a
*whole* timestep. CCBS drops that clock. It plans over **continuous time** on a
geometric roadmap:

- The roadmap here is the 8-connected grid: cardinal moves take time ``1``,
  diagonal moves take time ``sqrt(2)`` (so durations are irrational — the
  discrete model cannot even represent an optimal diagonal plan). Speed is 1.
- Each agent is a **disk** of radius ``r``. Two agents *collide* whenever the
  distance between their centres drops below ``2r`` at *any* real instant — not
  just when they share a vertex. Two agents whose paths cross mid-edge (e.g. the
  two diagonals of a unit square) collide geometrically while sharing no vertex
  and no edge: a conflict classical CBS is blind to.

The three levels mirror CBS, lifted to continuous time:

- :func:`_plan_continuous` — the low level: **continuous-time SIPP**. Each node
  carries real-valued *safe intervals* (the complement of the times a constraint
  forbids it); each move carries forbidden *start* intervals. An agent waits any
  real duration for free, so a yield costs only the minimal real time to clear.
- :func:`first_collision` — exact continuous collision detection between two
  piecewise-linear trajectories, by solving the quadratic distance on each
  shared linear segment. This is both the conflict detector and the independent
  oracle the gate verifies solutions against.
- :func:`ccbs` — the high level: best-first over a constraint tree by continuous
  sum-of-costs. On the first collision it computes, for each agent, the
  **unsafe interval** of *starting its colliding action* (an edge-start interval
  for a move, a vertex interval for a wait), and branches one agent each way.
  The first collision-free node popped is optimal in continuous time.

Honest scope (see ``docs/coordination.md``): the unsafe interval is derived from
the two conflicting *actions* (the sound, local computation the paper uses), via
an exact collision predicate located by bisection rather than closed-form
case-work — same interval to a tight tolerance, rounded outward so the resolved
plan clears with real separation. Equal radii; 8-connected roadmap.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

from .grid import GridWorld

INF = float("inf")
_EPS = 1e-9
_TOL = 1e-7

_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1),
         (1, 1), (1, -1), (-1, 1), (-1, -1)]


@dataclass
class CCBSSolution:
    """Continuous-time plans: ``trajectories[agent]`` is a list of
    ``(cell, time)`` waypoints (the agent moves linearly between consecutive
    waypoints and parks at the last one), ``cost`` is the continuous
    sum-of-costs (each agent's real arrival time at its goal)."""

    trajectories: dict
    cost: float


# --------------------------------------------------------------------------- #
# Roadmap geometry                                                            #
# --------------------------------------------------------------------------- #
def neighbors8(grid: GridWorld, cell):
    """Free 8-connected neighbours, refusing to cut a diagonal through a blocked
    corner (both orthogonal cells must be free)."""
    x, y = cell
    out = []
    for dx, dy in _DIRS:
        n = (x + dx, y + dy)
        if not grid.is_free(n):
            continue
        if dx != 0 and dy != 0:
            if not (grid.is_free((x + dx, y)) and grid.is_free((x, y + dy))):
                continue
        out.append(n)
    return out


def _dur(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _euclid(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pos_at(traj, t):
    """Linearly-interpolated position of a trajectory at time ``t`` (clamped to
    the endpoints outside its span)."""
    if t <= traj[0][1]:
        return (float(traj[0][0][0]), float(traj[0][0][1]))
    for k in range(len(traj) - 1):
        (c0, t0), (c1, t1) = traj[k], traj[k + 1]
        if t0 <= t <= t1:
            if t1 - t0 < _EPS:
                return (float(c1[0]), float(c1[1]))
            a = (t - t0) / (t1 - t0)
            return (c0[0] + (c1[0] - c0[0]) * a, c0[1] + (c1[1] - c0[1]) * a)
    return (float(traj[-1][0][0]), float(traj[-1][0][1]))


# --------------------------------------------------------------------------- #
# Continuous collision detection                                              #
# --------------------------------------------------------------------------- #
def _seg_below(r0, r1, t0, t1, thresh):
    """Earliest ``t`` in ``[t0, t1]`` at which the relative position, linear from
    ``r0`` (at t0) to ``r1`` (at t1), has norm ``< thresh`` — or ``None``.

    ``|r0 + u (r1 - r0)|^2 < thresh^2`` is a quadratic in ``u in [0, 1]``."""
    th2 = thresh * thresh - 1e-12
    dx, dy = r1[0] - r0[0], r1[1] - r0[1]
    A = dx * dx + dy * dy
    B = 2.0 * (r0[0] * dx + r0[1] * dy)
    C = r0[0] * r0[0] + r0[1] * r0[1] - th2
    if A < _EPS:  # constant separation over the segment
        return t0 if C < 0 else None
    disc = B * B - 4 * A * C
    if disc <= 0:
        return None
    sq = math.sqrt(disc)
    u_lo = (-B - sq) / (2 * A)
    u_hi = (-B + sq) / (2 * A)
    lo = max(u_lo, 0.0)
    hi = min(u_hi, 1.0)
    if lo > hi:
        return None
    return t0 + (t1 - t0) * lo


def first_collision(traj_i, traj_j, radius):
    """Earliest real time the two disks (radius ``radius`` each) are closer than
    ``2 * radius``, or ``None``. Exact: the position pair is piecewise-linear, so
    the squared distance is a quadratic on each shared breakpoint interval."""
    thresh = 2.0 * radius
    bps = sorted({t for _, t in traj_i} | {t for _, t in traj_j})
    bps.append(bps[-1] + 1.0)  # one tail interval with both agents parked
    for k in range(len(bps) - 1):
        t0, t1 = bps[k], bps[k + 1]
        ai, aj = pos_at(traj_i, t0), pos_at(traj_j, t0)
        bi, bj = pos_at(traj_i, t1), pos_at(traj_j, t1)
        r0 = (ai[0] - aj[0], ai[1] - aj[1])
        r1 = (bi[0] - bj[0], bi[1] - bj[1])
        tc = _seg_below(r0, r1, t0, t1, thresh)
        if tc is not None:
            return tc
    return None


def min_separation(traj_i, traj_j):
    """The closest the two centres ever come — the gate's geometric oracle."""
    best = INF
    bps = sorted({t for _, t in traj_i} | {t for _, t in traj_j})
    bps.append(bps[-1] + 1.0)
    for k in range(len(bps) - 1):
        t0, t1 = bps[k], bps[k + 1]
        ai, aj = pos_at(traj_i, t0), pos_at(traj_j, t0)
        bi, bj = pos_at(traj_i, t1), pos_at(traj_j, t1)
        rx0, ry0 = ai[0] - aj[0], ai[1] - aj[1]
        rx1, ry1 = bi[0] - bj[0], bi[1] - bj[1]
        # minimise |r0 + u (r1-r0)| over u in [0,1]
        dx, dy = rx1 - rx0, ry1 - ry0
        A = dx * dx + dy * dy
        if A < _EPS:
            u = 0.0
        else:
            u = max(0.0, min(1.0, -(rx0 * dx + ry0 * dy) / A))
        mx, my = rx0 + u * dx, ry0 + u * dy
        best = min(best, math.hypot(mx, my))
    return best


# --------------------------------------------------------------------------- #
# Action identification + unsafe-interval geometry                            #
# --------------------------------------------------------------------------- #
def _action_at(traj, tc):
    """What the agent is doing at time ``tc``: ``("move", frm, to, t0, dur)`` or
    ``("wait", node, lo, hi)``. Moves take precedence at a waypoint boundary."""
    for k in range(len(traj) - 1):
        (c0, t0), (c1, t1) = traj[k], traj[k + 1]
        if t0 - _EPS <= tc <= t1 + _EPS:
            if c0 != c1:
                return ("move", c0, c1, t0, t1 - t0)
            return ("wait", c0, t0, t1)
    return ("wait", traj[-1][0], traj[-1][1], INF)


def _act_window_pos(act):
    """A fixed action as (window_lo, window_hi, pos(t))."""
    if act[0] == "move":
        _, frm, to, t0, dur = act
        t1 = t0 + dur

        def p(t):
            if dur < _EPS:
                return (float(to[0]), float(to[1]))
            a = max(0.0, min(1.0, (t - t0) / dur))
            return (frm[0] + (to[0] - frm[0]) * a, frm[1] + (to[1] - frm[1]) * a)

        return t0, t1, p
    _, node, lo, hi = act
    return lo, hi, (lambda t: (float(node[0]), float(node[1])))


def _move_hits(i_from, i_to, i_dur, s, jact, radius):
    """Does agent i, performing move ``i_from -> i_to`` starting at time ``s``,
    collide with the fixed other action ``jact`` (only the temporally
    overlapping, moving phases matter)?"""
    thresh = 2.0 * radius
    jlo, jhi, jpos = _act_window_pos(jact)
    lo = max(s, jlo)
    hi = min(s + i_dur, jhi)
    if lo > hi - _EPS:
        return False

    def ipos(t):
        if i_dur < _EPS:
            return (float(i_to[0]), float(i_to[1]))
        a = max(0.0, min(1.0, (t - s) / i_dur))
        return (i_from[0] + (i_to[0] - i_from[0]) * a,
                i_from[1] + (i_to[1] - i_from[1]) * a)

    pi0, pj0 = ipos(lo), jpos(lo)
    pi1, pj1 = ipos(hi), jpos(hi)
    r0 = (pi0[0] - pj0[0], pi0[1] - pj0[1])
    r1 = (pi1[0] - pj1[0], pi1[1] - pj1[1])
    return _seg_below(r0, r1, lo, hi, thresh) is not None


def _unsafe_around(predicate, s0, span):
    """Given ``predicate(s)`` true at ``s0`` and false far enough away, return the
    maximal contiguous ``[lo, hi]`` of true values containing ``s0`` (bisection)."""
    left = s0 - span
    right = s0 + span
    # widen until both ends are clear (bounded — the overlap window is finite)
    tries = 0
    while predicate(left) and tries < 40:
        left -= span
        tries += 1
    while predicate(right) and tries < 80:
        right += span
        tries += 1
    # bisect false(left) -> true(s0)
    a, b = left, s0
    while b - a > _TOL:
        m = 0.5 * (a + b)
        if predicate(m):
            b = m
        else:
            a = m
    lo = b
    # bisect true(s0) -> false(right)
    a, b = s0, right
    while b - a > _TOL:
        m = 0.5 * (a + b)
        if predicate(m):
            a = m
        else:
            b = m
    hi = a
    return lo, hi


def _vertex_unsafe(node, jact, tc, radius):
    """Interval of times agent i sitting at ``node`` collides with fixed ``jact``,
    the one containing ``tc``. Returns ``(lo, hi)`` (``hi`` may be ``INF``)."""
    thresh = 2.0 * radius
    jlo, jhi, jpos = _act_window_pos(jact)
    if jact[0] == "wait":
        # constant separation; either the whole overlap collides or none does
        if _euclid(node, (jpos(jlo)[0], jpos(jlo)[1])) < thresh:
            return jlo, jhi
        return tc, tc  # degenerate (shouldn't happen)
    # j moving: dist(node, line(t)) < thresh on [jlo, jhi] -> quadratic in t
    p0 = jpos(jlo)
    p1 = jpos(jhi)
    r0 = (node[0] - p0[0], node[1] - p0[1])
    r1 = (node[0] - p1[0], node[1] - p1[1])
    th2 = thresh * thresh
    dx, dy = r1[0] - r0[0], r1[1] - r0[1]
    A = dx * dx + dy * dy
    B = 2.0 * (r0[0] * dx + r0[1] * dy)
    C = r0[0] * r0[0] + r0[1] * r0[1] - th2
    if A < _EPS:
        return (jlo, jhi) if C < 0 else (tc, tc)
    disc = B * B - 4 * A * C
    if disc <= 0:
        return (tc, tc)
    sq = math.sqrt(disc)
    u_lo = max(0.0, (-B - sq) / (2 * A))
    u_hi = min(1.0, (-B + sq) / (2 * A))
    return jlo + (jhi - jlo) * u_lo, jlo + (jhi - jlo) * u_hi


def _constraint_for(act, other_act, tc, radius):
    """The CCBS constraint that resolves this agent's side of the collision:
    ``("edge", frm, to, lo, hi)`` (forbid starting the move in [lo,hi]) for a
    move, ``("vertex", node, lo, hi)`` (forbid occupying the node in [lo,hi]) for
    a wait. Intervals are rounded outward by the tolerance so the replan clears."""
    if act[0] == "move":
        _, frm, to, t0, dur = act
        span = dur + 2.0
        jlo, jhi, _ = _act_window_pos(other_act)
        if jhi != INF:
            span += (jhi - jlo)

        def pred(s):
            return _move_hits(frm, to, dur, s, other_act, radius)

        lo, hi = _unsafe_around(pred, t0, span)
        return ("edge", frm, to, lo - _TOL, hi + _TOL)
    _, node, _, _ = act
    lo, hi = _vertex_unsafe(node, other_act, tc, radius)
    hi_out = hi + _TOL if hi != INF else INF
    return ("vertex", node, lo - _TOL, hi_out)


# --------------------------------------------------------------------------- #
# Low level: continuous-time SIPP                                             #
# --------------------------------------------------------------------------- #
def _complement(intervals):
    """Complement of a union of ``[lo, hi)`` busy intervals within ``[0, INF)``,
    as a sorted list of safe ``[lo, hi)`` intervals."""
    if not intervals:
        return [(0.0, INF)]
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + _EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    safe = []
    cur = 0.0
    for lo, hi in merged:
        if lo > cur + _EPS:
            safe.append((cur, lo))
        cur = max(cur, hi)
    safe.append((cur, INF))
    return safe


def _earliest_allowed(earliest, latest, forbidden):
    """Smallest ``t`` in ``[earliest, latest]`` outside every forbidden ``[a, b)``
    start interval, or ``None``."""
    t = earliest
    moved = True
    while moved:
        moved = False
        for a, b in forbidden:
            if a - _EPS <= t < b - _EPS:
                t = b
                moved = True
        if t > latest + _EPS:
            return None
    return t if t <= latest + _EPS else None


def _plan_continuous(grid, start, goal, vertex_unsafe, edge_unsafe, radius,
                     max_cost):
    """Minimal continuous-time ``start -> goal`` trajectory honoring the unsafe
    intervals, as a list of ``(cell, time)`` waypoints, or ``None``."""
    safe_cache = {}

    def safe(node):
        iv = safe_cache.get(node)
        if iv is None:
            iv = _complement(vertex_unsafe.get(node, []))
            safe_cache[node] = iv
        return iv

    s_iv = [iv for iv in safe(start) if iv[0] <= 0.0 <= iv[1]]
    if not s_iv:
        return None
    s_lo, s_hi = s_iv[0]
    goal_busy = vertex_unsafe.get(goal, [])
    goal_clear = max((hi for _, hi in goal_busy if hi != INF), default=0.0)

    counter = itertools.count()
    start_state = (start, s_lo)
    open_heap = [(_euclid(start, goal), 0.0, next(counter), start, s_lo, s_hi)]
    best = {start_state: 0.0}
    came = {}

    while open_heap:
        _, g, _, node, lo, hi = heapq.heappop(open_heap)
        if g > best.get((node, lo), g) + _EPS:
            continue
        if node == goal and hi == INF and g >= goal_clear - _EPS:
            return _reconstruct(came, best, (node, lo), start)
        if g > max_cost + _EPS:
            continue
        for v in neighbors8(grid, node):
            d = _dur(node, v)
            forbidden = edge_unsafe.get((node, v), ())
            for (vlo, vhi) in safe(v):
                earliest_dep = max(g, vlo - d)
                latest_dep = min(hi, vhi - d)
                if earliest_dep > latest_dep + _EPS:
                    continue
                dep = _earliest_allowed(earliest_dep, latest_dep, forbidden)
                if dep is None:
                    continue
                arr = dep + d
                if arr > max_cost + _EPS:
                    continue
                ns = (v, vlo)
                if arr < best.get(ns, INF) - _EPS:
                    best[ns] = arr
                    came[ns] = (node, lo, dep, arr)
                    heapq.heappush(
                        open_heap,
                        (arr + _euclid(v, goal), arr, next(counter), v, vlo, vhi))
    return None


def _reconstruct(came, best, end_state, start):
    chain = [end_state]
    while chain[-1] in came:
        pnode, plo, dep, arr = came[chain[-1]]
        chain.append((pnode, plo))
    chain.reverse()
    # Build waypoints: at each state we know arrival; the stored `dep` is when we
    # left the previous node, so a wait shows as two waypoints at the same cell.
    traj = [(start, 0.0)]
    for st in chain[1:]:
        pnode, plo, dep, arr = came[st]
        node = st[0]
        last_cell, last_t = traj[-1]
        if dep > last_t + _EPS:  # waited at the previous node until `dep`
            traj.append((last_cell, dep))
        traj.append((node, arr))
    return traj


# --------------------------------------------------------------------------- #
# High level: CCBS                                                            #
# --------------------------------------------------------------------------- #
def _cost(trajs):
    return sum(t[-1][1] for t in trajs.values())


def _first_pair_collision(trajs, radius):
    """The earliest collision across all agent pairs: ``(tc, a, b)`` or None."""
    best = None
    ids = list(trajs)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            tc = first_collision(trajs[a], trajs[b], radius)
            if tc is not None and (best is None or tc < best[0] - _EPS):
                best = (tc, a, b)
    return best


def ccbs(grid: GridWorld, agents: dict, *, radius: float = 0.4,
         max_expansions: int = 20_000, max_cost: float = 1e6,
         stats: dict | None = None):
    """Solve a MAPF instance in continuous time on the 8-connected roadmap.

    ``agents`` maps an id to ``(start, goal)``. Each agent is a disk of radius
    ``radius`` moving at unit speed. Returns a :class:`CCBSSolution` whose
    trajectories never bring two centres within ``2 * radius``, or ``None`` if
    infeasible / the expansion budget is exhausted. ``stats["expansions"]`` is the
    number of high-level nodes expanded."""

    def plan(agent, v_un, e_un):
        start, goal = agents[agent]
        return _plan_continuous(grid, start, goal, v_un, e_un, radius, max_cost)

    v_con = {a: {} for a in agents}
    e_con = {a: {} for a in agents}
    trajs = {}
    for a in agents:
        tr = plan(a, v_con[a], e_con[a])
        if tr is None:
            return None
        trajs[a] = tr

    counter = itertools.count()
    open_heap = [(_cost(trajs), next(counter), v_con, e_con, trajs)]
    expansions = 0

    while open_heap:
        cost, _, v_con, e_con, trajs = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            if stats is not None:
                stats["expansions"] = expansions
            return None

        hit = _first_pair_collision(trajs, radius)
        if hit is None:
            if stats is not None:
                stats["expansions"] = expansions
            return CCBSSolution(trajectories=dict(trajs), cost=cost)

        tc, a, b = hit
        act_a = _action_at(trajs[a], tc)
        act_b = _action_at(trajs[b], tc)
        con_a = _constraint_for(act_a, act_b, tc, radius)
        con_b = _constraint_for(act_b, act_a, tc, radius)

        for agent, con in ((a, con_a), (b, con_b)):
            child_v = {k: dict(v) for k, v in v_con.items()}
            child_e = {k: dict(v) for k, v in e_con.items()}
            if con[0] == "vertex":
                _, node, lo, hi = con
                child_v[agent].setdefault(node, [])
                child_v[agent][node] = child_v[agent][node] + [(lo, hi)]
            else:
                _, frm, to, lo, hi = con
                child_e[agent].setdefault((frm, to), [])
                child_e[agent][(frm, to)] = child_e[agent][(frm, to)] + [(lo, hi)]
            tr = plan(agent, child_v[agent], child_e[agent])
            if tr is None:
                continue
            child_trajs = dict(trajs)
            child_trajs[agent] = tr
            heapq.heappush(
                open_heap,
                (_cost(child_trajs), next(counter), child_v, child_e,
                 child_trajs))

    if stats is not None:
        stats["expansions"] = expansions
    return None


def shortest_trajectory(grid: GridWorld, start, goal):
    """The agent's unconstrained continuous-time shortest path (8-connected),
    as ``(cell, time)`` waypoints — the uncoordinated baseline."""
    return _plan_continuous(grid, start, goal, {}, {}, 0.0, 1e6)
