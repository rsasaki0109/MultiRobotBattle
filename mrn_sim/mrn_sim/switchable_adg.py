"""Switchable Action Dependency Graph: robust MAPF execution under delays.

A MAPF plan is collision-free only if every robot keeps to the schedule, which
real robots cannot — sensing, kinematics and traffic make them late. The
**Action Dependency Graph** (ADG; Hönig et al. 2016/2019) decouples *ordering*
from *timing*: from a plan it records, at every shared cell, the order in which
agents pass through it, as precedence edges "agent ``a`` may enter cell ``c``
only after agent ``p`` has left it". Executing while merely respecting those
edges (the Temporal Plan Graph gate in :mod:`mrn_sim.mapf_exec`) is collision-
free *whatever the timing* — a delayed robot just makes its followers wait. The
ADG is acyclic, so this never deadlocks.

But a *fixed* order is wasteful: if the robot scheduled to cross a junction first
is the one that got delayed, everyone behind it stalls for the whole delay, even
when the junction is free. The **Switchable ADG** (Berndt, Palmieri, et al.,
*Receding-Horizon Re-ordering of Multi-Agent Execution Schedules*, IROS 2020 /
T-RO 2024) makes each passing-order edge *reversible*: when a ready robot is
blocked behind a delayed one at a shared cell, flip the order so the ready robot
goes first — **provided the flip keeps the graph acyclic**. An acyclic graph is
deadlock-free, so the reversal is safe exactly when it introduces no cycle; the
check is a single reachability query. Re-ordering recovers the throughput a fixed
order throws away, with the same hard collision-free *and* deadlock-free
guarantees.

This module builds the ADG from a plan (reusing :mod:`mrn_sim.mapf_exec`'s
milestone extraction), and runs a discrete delay-execution simulator in two
modes — ``fixed`` (the plain ADG) and ``switchable`` — so one run pair isolates
exactly what re-ordering buys. Pure and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mapf_exec import _milestones


@dataclass
class AdgEdge:
    """A passing-order precedence at a shared cell ``cell``.

    Planned order: ``first`` occupies ``cell`` at its milestone ``kf`` *before*
    ``second`` does at milestone ``ks``. The constraint (not ``reversed``) is
    that ``second`` may complete milestone ``ks`` only once ``first`` has
    departed ``cell`` (reached milestone ``kf + 1``). ``switchable`` is True only
    when *either* order is feasible — both occupants leave the cell, so neither
    is parked on it as a goal. Reversing swaps which agent waits for the other.
    """

    cell: tuple
    first: object
    kf: int
    second: object
    ks: int
    switchable: bool
    reversed: bool = False

    def waiter(self):
        """(agent, milestone) that must wait, given the current orientation."""
        return (self.second, self.ks) if not self.reversed else (self.first, self.kf)

    def blocker(self):
        """(agent, milestone) the waiter is waiting on (the depart milestone)."""
        return (self.first, self.kf + 1) if not self.reversed else (self.second, self.ks + 1)


def build_adg(paths):
    """Build the ADG of a MAPF solution: ``(cells, edges)``.

    ``cells[a]`` is agent ``a``'s ordered list of distinct cells (waits
    collapsed). ``edges`` is the list of :class:`AdgEdge` passing-order
    constraints, one per consecutive pair of occupants at each shared cell, in
    planned (arrival-time) order.
    """
    cells, steps = _milestones(paths)
    last = {a: len(cells[a]) - 1 for a in cells}
    occ: dict = {}
    for a in cells:
        for k, c in enumerate(cells[a]):
            occ.setdefault(c, []).append((steps[a][k], a, k))
    edges = []
    for c, lst in occ.items():
        lst.sort()                                   # planned order by arrival
        for idx in range(1, len(lst)):
            _, second, ks = lst[idx]
            _, first, kf = lst[idx - 1]
            if first == second:
                continue
            # Switchable only if *both* occupants leave c (neither parks here as
            # its goal); otherwise the goal-occupant must be last and the order
            # is fixed.
            switchable = kf < last[first] and ks < last[second]
            edges.append(AdgEdge(c, first, kf, second, ks, switchable))
    return cells, edges


def build_sadg(paths):
    """Build the Switchable ADG for re-ordering: ``(cells, edges)`` with a
    precedence edge for **every pair** of agents sharing a cell — not just
    consecutive ones as in :func:`build_adg`.

    This distinction matters the moment a cell is shared by *three or more*
    agents. :func:`build_adg`'s consecutive-pair chain ``a1 -> a2 -> a3`` encodes
    the total order only *transitively*; reversing the middle edge to ``a3 -> a2``
    leaves ``a1`` and ``a3`` mutually unconstrained, so a re-ordering can let both
    onto the cell at once. Materialising **all** pairs keeps mutual exclusion
    enforced for every pair independently, so any acyclic orientation is a genuine
    total order at each cell — collision-free under arbitrary re-ordering. (The
    plain ADG never hit this because reversals there were only ever validated on
    2-agent cells.) Initial orientations match the planned arrival order, so on a
    cell shared by two this is identical to :func:`build_adg`.
    """
    cells, steps = _milestones(paths)
    last = {a: len(cells[a]) - 1 for a in cells}
    occ: dict = {}
    for a in cells:
        for k, c in enumerate(cells[a]):
            occ.setdefault(c, []).append((steps[a][k], a, k))
    edges = []
    for c, lst in occ.items():
        lst.sort()                                   # planned order by arrival
        for j in range(len(lst)):
            for i in range(j):
                _, first, kf = lst[i]
                _, second, ks = lst[j]
                if first == second:
                    continue
                switchable = kf < last[first] and ks < last[second]
                edges.append(AdgEdge(c, first, kf, second, ks, switchable))
    return cells, edges


def _precedences(cells, edges):
    """Directed precedence edges over ``(agent, milestone)`` nodes.

    Type-1 (intra-agent) chain ``(a, k) -> (a, k+1)`` plus the oriented type-2
    passing-order edges. A cycle here is a deadlock.
    """
    adj: dict = {}

    def link(u, v):
        adj.setdefault(u, set()).add(v)

    for a, cs in cells.items():
        for k in range(len(cs) - 1):
            link((a, k), (a, k + 1))
    for e in edges:
        (wa, wk), (ba, bk) = e.waiter(), e.blocker()
        link((ba, bk), (wa, wk))                     # blocker departs before waiter enters
    return adj


def _reaches(adj, src, dst):
    """Is ``dst`` reachable from ``src`` in the precedence graph?"""
    if src == dst:
        return True
    stack = [src]
    seen = {src}
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):  # noqa: SIM118
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _reversal_is_safe(cells, edges, e):
    """Would reversing edge ``e`` keep the precedence graph acyclic?

    Reversing makes ``first`` wait for ``second``: it adds the precedence
    ``(second, ks+1) -> (first, kf)``. That closes a cycle iff ``(second, ks+1)``
    is already reachable *from* ``(first, kf)`` in the graph without this edge.
    """
    others = [x for x in edges if x is not e]
    adj = _precedences(cells, others)
    new_from, new_to = (e.second, e.ks + 1), (e.first, e.kf)
    return not _reaches(adj, new_to, new_from)


@dataclass
class AdgRunResult:
    """Outcome of a delay-execution run over the ADG."""

    makespan: int                 # ticks until every robot reached its goal
    finished: bool                # all robots arrived (False == deadlock/timeout)
    switches: int                 # passing-order reversals applied (0 when fixed)
    deadlock: bool                # the precedence graph held a cycle at any point
    history: list = None          # [ {agent: cell} per tick ] if kept

    def as_dict(self) -> dict:
        return {
            "makespan": self.makespan,
            "finished": self.finished,
            "switches": self.switches,
            "deadlock": self.deadlock,
        }


def schedule_is_collision_free(history) -> bool:
    """No two robots share a cell, and none swap cells, on the executed schedule."""
    for prev, cur in zip(history, history[1:]):
        if len(set(cur.values())) != len(cur):
            return False                              # vertex: shared cell
        for a in cur:
            for b in cur:
                if a < b and cur[a] == prev[b] and cur[b] == prev[a]:
                    return False                      # edge: head-on swap
    return True


def simulate(cells, edges, start_delay, *, switchable, max_ticks=None,
             keep_history=False):
    """Run the ADG under a start-delay model; return an :class:`AdgRunResult`.

    Each robot advances one milestone per tick once its precedences are met, but
    robot ``a`` is frozen at its start until tick ``start_delay[a]``. With
    ``switchable`` on, a ready robot blocked behind a frozen one at a *switchable*
    passing-order edge flips that edge (so it goes first) whenever the flip keeps
    the graph acyclic. Motion is collision-free by construction (the precedences
    preserve the plan's cell orderings) and deadlock-free (only acyclic-preserving
    flips are applied).
    """
    ids = list(cells)
    last = {a: len(cells[a]) - 1 for a in ids}
    done = {a: 0 for a in ids}
    if max_ticks is None:
        max_ticks = sum(last.values()) + sum(start_delay.values()) + len(ids) + 5
    switches = 0
    deadlock = False
    history = [] if keep_history else None

    def ready_to_advance(a, tick):
        """Can ``a`` complete its next milestone this tick?"""
        if done[a] >= last[a] or tick < start_delay.get(a, 0):
            return False
        nk = done[a] + 1
        for e in edges:
            wa, wk = e.waiter()
            if wa == a and wk == nk:
                ba, bk = e.blocker()
                if done[ba] < bk:
                    return False
        return True

    for tick in range(max_ticks):
        if keep_history:
            history.append({a: cells[a][done[a]] for a in ids})
        if all(done[a] == last[a] for a in ids):
            return AdgRunResult(tick, True, switches, deadlock, history)

        if switchable:
            # Flip a switchable edge whose waiter is ready and stuck behind a
            # frozen/not-yet-departed blocker, when the flip stays acyclic.
            for e in edges:
                if not e.switchable:
                    continue
                wa, wk = e.waiter()
                ba, bk = e.blocker()
                if done[wa] + 1 != wk or done[ba] >= bk:
                    continue                          # waiter not poised, or blocker cleared
                if tick < start_delay.get(wa, 0):
                    continue                          # the waiter is itself frozen
                if done[ba] + 1 <= last[ba] and ready_to_advance(ba, tick):
                    continue                          # blocker is moving anyway; no gain
                if _reversal_is_safe(cells, edges, e):
                    e.reversed = not e.reversed
                    switches += 1

        advanced = {a: ready_to_advance(a, tick) for a in ids}
        for a in ids:
            if advanced[a]:
                done[a] += 1

    finished = all(done[a] == last[a] for a in ids)
    return AdgRunResult(max_ticks, finished, switches, not finished, history)


# --------------------------------------------------------------------------- #
# Receding-Horizon re-ordering (Berndt et al., T-RO 2024)                      #
#                                                                              #
# The reactive ``simulate(switchable=True)`` above flips one edge at a time,   #
# myopically: it reverses any switchable edge whose waiter is stuck behind a   #
# not-moving blocker. That is locally sensible but globally short-sighted —    #
# letting one ready robot jump ahead can delay another that many more robots   #
# wait on, raising the *cumulative* completion time. The T-RO 2024 method      #
# instead solves, at every step, a small integer program over the switchable   #
# edges in a horizon: choose the acyclic (deadlock-free) orientation that       #
# MINIMIZES the cumulative route-completion time, predicted by rolling the      #
# schedule out under the currently-observed delays — then re-solve as          #
# execution proceeds (receding horizon, recursively feasible).                 #
# --------------------------------------------------------------------------- #
def _is_acyclic(cells, edges) -> bool:
    """Does the precedence graph (under the edges' current orientations) hold no
    cycle? A cycle is a deadlock, so an orientation is admissible iff acyclic."""
    adj = _precedences(cells, edges)
    color: dict = {}                                  # 0 = visiting, 1 = done

    def visit(u):
        color[u] = 0
        for v in adj.get(u, ()):  # noqa: SIM118
            c = color.get(v)
            if c == 0:
                return False                          # back-edge -> cycle
            if c is None and not visit(v):
                return False
        color[u] = 1
        return True

    # Any cycle lies entirely among nodes with out-edges, so iterating the
    # adjacency keys suffices to detect one.
    return all(visit(u) for u in list(adj) if color.get(u) is None)


def _rollout(cells, edges, done0, rem0, max_ticks):
    """Predict completion from state ``done0`` with FIXED edge orientations and
    per-robot remaining freeze ``rem0`` (ticks a robot still cannot move). Returns
    ``(finish_tick_per_agent, finished)``; unfinished robots are charged
    ``max_ticks`` (a penalty that keeps deadlocked orientations uncompetitive)."""
    done = dict(done0)
    rem = dict(rem0)
    last = {a: len(cells[a]) - 1 for a in cells}
    finish = {a: (0 if done[a] == last[a] else None) for a in cells}

    def ready(a):
        if done[a] >= last[a]:
            return False
        if done[a] == 0 and rem.get(a, 0) > 0:
            return False
        nk = done[a] + 1
        for e in edges:
            wa, wk = e.waiter()
            if wa == a and wk == nk:
                ba, bk = e.blocker()
                if done[ba] < bk:
                    return False
        return True

    t = 0
    while t < max_ticks and not all(done[a] == last[a] for a in cells):
        adv = {a: ready(a) for a in cells}
        for a in cells:
            if done[a] == 0 and rem.get(a, 0) > 0:
                rem[a] -= 1
        for a in cells:
            if adv[a]:
                done[a] += 1
                if done[a] == last[a] and finish[a] is None:
                    finish[a] = t + 1
        t += 1
    finished = all(done[a] == last[a] for a in cells)
    for a in finish:
        if finish[a] is None:
            finish[a] = max_ticks
    return finish, finished


def _live_switchables(cells, edges, done):
    """Switchable edges whose ordering decision is still open — **neither occupant
    has yet entered** the shared cell (``done < milestone`` for both), so either
    order is still physically realizable. Once a robot is *on* the cell the order
    is committed; reversing then would push the other robot onto an occupied cell
    (a collision), so such edges are excluded — this is what keeps re-ordering
    recursively feasible."""
    out = []
    for e in edges:
        if not e.switchable:
            continue
        if done[e.first] < e.kf and done[e.second] < e.ks:
            out.append(e)
    return out


def reorder_milp(cells, edges, done, rem, horizon, max_ticks):
    """Re-order: pick, for the next ``horizon`` open switchable edges, the acyclic
    orientation that minimizes predicted cumulative completion time, and apply it.

    This is the (small) integer program of Berndt et al. solved exactly by
    enumeration — bounded to ``horizon`` edges so it stays low-dimensional. Other
    edges keep their current orientation. Returns the number of edges whose
    orientation changed. Ties break toward fewer reversals then lexicographically,
    so the result is deterministic."""
    cand = _live_switchables(cells, edges, done)
    cand.sort(key=lambda e: (min(e.kf, e.ks), e.first, e.second))
    cand = cand[:horizon]
    if not cand:
        return 0
    base = [e.reversed for e in cand]

    best = None                                       # (cost, nflips, bits)
    for mask in range(1 << len(cand)):
        bits = [(mask >> i) & 1 for i in range(len(cand))]
        for e, b in zip(cand, bits):
            e.reversed = bool(b)
        if not _is_acyclic(cells, edges):
            continue
        finish, finished = _rollout(cells, edges, done, rem, max_ticks)
        cost = sum(finish.values())
        nflips = sum(1 for b, b0 in zip(bits, base) if b != b0)
        key = (0 if finished else 1, cost, nflips, tuple(bits))
        if best is None or key < best[0]:
            best = (key, bits)
    if best is None:                                  # no acyclic option (shouldn't happen)
        for e, b in zip(cand, base):
            e.reversed = b
        return 0
    changed = 0
    for e, b, b0 in zip(cand, best[1], base):
        e.reversed = bool(b)
        changed += int(bool(b) != b0)
    return changed


def simulate_rhc(cells, edges, start_delay, *, horizon=4, max_ticks=None,
                 keep_history=False):
    """Run the SADG under start-delays with **receding-horizon re-ordering**.

    At every tick, before advancing, re-solve :func:`reorder_milp` on the current
    state (the receding horizon) so the passing order is continually re-optimized
    for minimum cumulative completion time, always keeping the graph acyclic.
    Collision-free *and* deadlock-free by construction, like :func:`simulate`, but
    globally cheaper than the reactive single-flip. Returns an
    :class:`AdgRunResult` (``switches`` counts applied reversals)."""
    ids = list(cells)
    last = {a: len(cells[a]) - 1 for a in ids}
    done = {a: 0 for a in ids}
    if max_ticks is None:
        max_ticks = sum(last.values()) + sum(start_delay.values()) + len(ids) + 5
    switches = 0
    history = [] if keep_history else None

    def ready_to_advance(a, tick):
        if done[a] >= last[a] or tick < start_delay.get(a, 0):
            return False
        nk = done[a] + 1
        for e in edges:
            wa, wk = e.waiter()
            if wa == a and wk == nk:
                ba, bk = e.blocker()
                if done[ba] < bk:
                    return False
        return True

    for tick in range(max_ticks):
        if keep_history:
            history.append({a: cells[a][done[a]] for a in ids})
        if all(done[a] == last[a] for a in ids):
            return AdgRunResult(tick, True, switches, False, history)
        rem = {a: max(0, start_delay.get(a, 0) - tick) if done[a] == 0 else 0
               for a in ids}
        switches += reorder_milp(cells, edges, done, rem, horizon, max_ticks)
        advanced = {a: ready_to_advance(a, tick) for a in ids}
        for a in ids:
            if advanced[a]:
                done[a] += 1

    finished = all(done[a] == last[a] for a in ids)
    return AdgRunResult(max_ticks, finished, switches, not finished, history)


def cumulative_completion(cells, history) -> int:
    """Cumulative route-completion time over an executed ``history``: the sum, per
    robot, of the first tick it stands on its goal cell (the T-RO objective).
    Robots that never arrive are charged the horizon length."""
    goal = {a: cells[a][-1] for a in cells}
    finish: dict = {}
    for t, snap in enumerate(history):
        for a, c in snap.items():
            if a not in finish and c == goal[a]:
                finish[a] = t
    horizon = len(history)
    return sum(finish.get(a, horizon) for a in cells)
