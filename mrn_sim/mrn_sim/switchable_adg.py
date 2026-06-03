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
