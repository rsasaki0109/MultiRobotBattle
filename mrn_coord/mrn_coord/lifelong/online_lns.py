"""Online LNS for lifelong MAPF: repair the plan instead of rebuilding it.

The other planning engine here, RHCR (:mod:`mrn_coord.lifelong.rhcr`), solves a
fresh Windowed MAPF instance from scratch every replan period. That is the
**CENTRAL** strategy -- replan *every* agent each boundary -- and it is the
expensive baseline the lifelong-MAPF literature contrasts against. **Online
LNS** keeps the team's committed paths between boundaries and only *repairs*
what must change: the agents that just finished a task (their old path is spent)
plus a small **Large-Neighborhood-Search** destroy set chosen to relieve the
worst detour. Each repaired agent replans around everyone else's frozen path
(the package's space-time A\\*), so motion stays **collision-free by
construction**, exactly as in one-shot :func:`mrn_coord.mapf.mapf_lns`.

Because a boundary replans a bounded handful of agents rather than the whole
team, online LNS does a small fraction of CENTRAL's per-boundary planning work
while keeping the team flowing at a comparable throughput -- the anytime,
incremental trade the approach is built for. A single ``mode`` flag selects
between the two, so one run pair isolates exactly what reusing the previous plan
buys. Pure and deterministic given the seed.
"""

from __future__ import annotations

import random

from mrn_coord.mapf.lns import _worst_neighborhood
from mrn_coord.mapf.space_time_astar import plan_path

from .lifelong import LifelongResult, _bfs_dist, make_assigner


def _frozen_reservations(plan, ids, subset, horizon):
    """Vertex/edge reservations from every agent *not* in ``subset``.

    Each frozen agent reserves its cell at every time and holds its final
    (goal) cell to ``horizon``, so a repaired path can never collide with a
    committed one or slip through a held goal. Edges block swaps across moves.
    """
    vertex: set = set()
    edge: set = set()
    for a in ids:
        if a in subset:
            continue
        p = plan[a]
        for t in range(horizon + 1):
            vertex.add((p[t] if t < len(p) else p[-1], t))
        for t in range(min(len(p) - 1, horizon)):
            edge.add((p[t + 1], p[t], t + 1))
    return frozenset(vertex), frozenset(edge)


def run_online_lns(
    grid,
    starts: dict,
    stream,
    *,
    mode: str = "lns",
    max_steps: int = 256,
    replan_period: int = 5,
    neighborhood: int = 4,
    keep_history: bool = False,
    allocator: str = "stream",
    open_tasks: int | None = None,
    horizon: int | None = None,
    seed: int = 0,
    stats: dict | None = None,
) -> LifelongResult:
    """Run lifelong MAPF by online plan repair for ``max_steps`` ticks.

    ``starts`` maps agent id -> current cell. Every ``replan_period`` ticks the
    team repairs its committed paths toward the current goals:

    - ``mode="central"`` -- replan **every** agent from scratch (the CENTRAL
      baseline);
    - ``mode="lns"`` -- replan only the agents that just need a new path (a
      completed task) plus a destroy neighborhood of up to ``neighborhood``
      agents around the worst detour, keeping everyone else's path frozen.

    Each repaired agent plans around the frozen paths, so motion is
    collision-free by construction. ``allocator`` / ``open_tasks`` match free
    robots to tasks as in :func:`mrn_coord.lifelong.run_lifelong`. ``stats``, if
    given, records the per-boundary planning effort (single-agent replans). Pass
    ``keep_history=True`` to capture per-step positions. Returns a
    :class:`~mrn_coord.lifelong.LifelongResult`.
    """
    if mode not in ("lns", "central"):
        raise ValueError(f"unknown mode: {mode!r}")
    ids = sorted(starts)
    pos = {a: starts[a] for a in ids}
    plan = {a: [pos[a]] for a in ids}        # committed path from now (index 0 == pos)
    goal: dict = {}
    has_task = {a: False for a in ids}
    assigned_at: dict = {}
    elapsed = {a: 0 for a in ids}
    dist_cache: dict = {}

    def dist_to(g):
        if g not in dist_cache:
            dist_cache[g] = _bfs_dist(grid, g)
        return dist_cache[g]

    H = horizon if horizon is not None else 2 * (grid.width + grid.height) + 4
    rng = random.Random(seed)

    assign = make_assigner(grid, ids, stream, allocator, open_tasks, dist_to,
                           goal, has_task, assigned_at, elapsed)
    for a in ids:
        goal[a] = pos[a]
    assign(ids, 0, pos)

    replans = 0
    rejected = 0
    boundaries = 0

    def repair(subset):
        """Replan ``subset`` around the frozen paths -- **all or nothing**.

        Each agent is planned in turn (longest-detour first) around the frozen
        paths and the already-replanned subset agents, so a committed result is
        collision-free by construction. If *any* agent finds no path the whole
        repair is rejected and the previous committed plans are kept untouched --
        because holding a single blocked agent in place would break a path an
        earlier-planned agent was routed through, a real collision. The prior
        plans are themselves collision-free, so rejecting is always safe; the
        agent simply retries at the next boundary. Mirrors one-shot
        :func:`mrn_coord.mapf.mapf_lns`'s reject-on-failure repair.
        """
        nonlocal replans, rejected
        vertex, edge = _frozen_reservations(plan, ids, subset, H)
        vertex = set(vertex)
        edge = set(edge)
        # longest-detour first (id breaks ties -> order independent of set order).
        order = sorted(subset, key=lambda a: (-(len(plan[a]) - 1), a))
        new_plans = {}
        for a in order:
            replans += 1
            if pos[a] == goal[a]:
                p = [pos[a]]
            else:
                p = plan_path(grid, pos[a], goal[a], frozenset(vertex),
                              frozenset(edge), max_time=H)
            if p is None:
                rejected += 1
                return                         # reject: leave every plan as-is
            new_plans[a] = p
            for t in range(H + 1):
                vertex.add((p[t] if t < len(p) else p[-1], t))
            for t in range(min(len(p) - 1, H)):
                edge.add((p[t + 1], p[t], t + 1))
        plan.update(new_plans)                 # commit only once all succeeded

    def destroy_set(stale):
        """The agents to repair this boundary: the stale ones plus, in LNS mode,
        a worst-detour neighborhood; in CENTRAL mode, everyone."""
        if mode == "central":
            return set(ids)
        subset = set(stale)
        k = min(neighborhood, len(ids))
        if k > len(subset):
            shortest = {a: dist_to(goal[a]).get(pos[a], 0) for a in ids}
            extra = _worst_neighborhood(plan, shortest, rng, k)
            subset |= extra
        return subset

    completed = 0
    per_agent = {a: 0 for a in ids}
    service_times: list = []
    history: list = []
    goal_history: list = []
    completions: list = []
    stale: set = set(ids)                    # everyone needs a first plan

    for step in range(max_steps):
        # 1. completions + reassignment; a reassigned agent's path is now stale.
        # Crucially we do NOT touch plan[a] here -- between boundaries every agent
        # follows its committed (mutually collision-free) path verbatim, so the
        # team stays collision-free by construction. Completion is bookkeeping
        # only; the new goal takes effect at the next boundary's repair. (Freezing
        # a completer at its goal mid-period would break a path another agent was
        # allowed to route through that cell, which is a real collision.)
        done = [a for a in ids if has_task[a] and pos[a] == goal[a]]
        for a in done:
            completed += 1
            per_agent[a] += 1
            service_times.append(step - assigned_at[a])
            has_task[a] = False
            goal[a] = pos[a]
        if done:
            assign(done, step, pos)
            stale |= set(done)
        completions.append(len(done))

        if keep_history:
            history.append(dict(pos))
            goal_history.append({a: (goal[a] if has_task[a] else pos[a])
                                 for a in ids})

        # 2. replanning boundary: repair the plan (CENTRAL = all, LNS = bounded).
        if step % replan_period == 0:
            repair(destroy_set(stale))
            stale = set()
            boundaries += 1

        # 3. advance one tick along the committed plans.
        newpos = {}
        for a in ids:
            if len(plan[a]) > 1:
                newpos[a] = plan[a][1]
                plan[a] = plan[a][1:]
            else:
                newpos[a] = plan[a][0]
        pos = newpos
        for a in ids:
            elapsed[a] += 1

    final_done = 0
    for a in ids:
        if has_task[a] and pos[a] == goal[a]:
            completed += 1
            per_agent[a] += 1
            final_done += 1
            service_times.append(max_steps - assigned_at[a])
    completions.append(final_done)
    if keep_history:
        history.append(dict(pos))
        goal_history.append({a: (goal[a] if has_task[a] else pos[a]) for a in ids})

    avg_service = (sum(service_times) / len(service_times)) if service_times else 0.0
    result = LifelongResult(
        steps=max_steps,
        agents=len(ids),
        completed=completed,
        throughput=(completed / max_steps if max_steps else 0.0),
        per_agent=per_agent,
        avg_service_time=avg_service,
        max_wait=(max(service_times) if service_times else 0),
        history=history,
        goal_history=goal_history,
        completions=completions,
    )
    result.replans = replans
    result.rejected = rejected
    result.boundaries = boundaries
    if stats is not None:
        stats.update(mode=mode, steps=max_steps, replans=replans,
                     rejected=rejected, boundaries=boundaries, completed=completed)
    return result
