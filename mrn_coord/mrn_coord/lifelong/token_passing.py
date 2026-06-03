"""Token Passing (TP) for lifelong MAPF.

Ma, Li, Kumar & Koenig, *Lifelong Multi-Agent Path Finding for Online Pickup
and Delivery Tasks* (AAAI 2017). The third lifelong engine here, and a third
distinct paradigm: :func:`mrn_coord.lifelong.run_lifelong` steps **PIBT** (a
greedy one-timestep rule), :func:`mrn_coord.lifelong.run_rhcr` solves a
**windowed** batch MAPF every few ticks; Token Passing plans **full
space-time paths** into a shared *token* of reservations.

The token is the set of every agent's committed future path. Agents update it
**one at a time**: a free agent reads the token, plans a minimal-time path to
its goal that avoids every other agent's reserved cells and swaps (a
cooperative / WHCA\\*-style reservation search, here the package's space-time
A\\* with the token rendered as vertex/edge constraints), and writes that path
back. Because each agent commits a full conflict-free path against the others'
commitments, the team is **collision-free by construction** — no per-step
rule, no fallback rollout.

The price of committing whole paths is that a reservation search can fail to
find a path (where PIBT/RHCR always have a safe one-step fallback). Token
Passing avoids deadlock the way the paper does — by **parking**: an agent with
no task it can reach holds an endpoint cell rather than wandering into others'
way, and the warehouse is *well-formed* (enough endpoints that every agent can
rest without blocking any task). On such a map TP stays **live** — every task
is eventually served. On a congested map where the well-formed property fails,
reservation planning can stall (the documented reason RHCR falls back to PIBT
in cramped aisles); TP is scoped to, and gated on, the well-formed regime.

Scored, like every lifelong run, by **throughput** (tasks completed per
timestep). Pure and deterministic; shares the task allocators
(:mod:`mrn_coord.lifelong.allocation`) with the other engines for an
apples-to-apples comparison of the *motion* paradigm.
"""

from __future__ import annotations

from mrn_coord.mapf.space_time_astar import plan_path

from .lifelong import LifelongResult, _bfs_dist, make_assigner


def _cell_at(path: list, t: int):
    """Agent's cell at time ``t``: it holds the last cell once the path ends."""
    return path[t] if t < len(path) else path[-1]


def _reservations(plan: dict, ids, exclude, horizon: int):
    """Render the token (every *other* agent's committed path) as constraints.

    Returns ``(vertex, edge)`` sets in :func:`plan_path`'s format. Each other
    agent reserves its cell at every time, and — crucially for a lifelong run —
    holds its final (resting) cell for all later times up to ``horizon``, so a
    parked agent's endpoint is reserved indefinitely. Edge reservations forbid
    swapping across any move the other agent makes.
    """
    vertex = set()
    edge = set()
    for b in ids:
        if b == exclude:
            continue
        p = plan[b]
        for t in range(horizon + 1):
            vertex.add((_cell_at(p, t), t))
        # forbid swapping across b's move p[t] -> p[t+1]: the planning agent must
        # not make the reverse traversal arriving at t+1 (plan_path keys edge
        # constraints (frm, to, arrival_time), so the time is t+1, not t).
        for t in range(min(len(p) - 1, horizon)):
            edge.add((p[t + 1], p[t], t + 1))
    return frozenset(vertex), frozenset(edge)


def run_token_passing(
    grid,
    starts: dict,
    stream,
    *,
    max_steps: int = 256,
    keep_history: bool = False,
    allocator: str = "stream",
    open_tasks: int | None = None,
    horizon: int | None = None,
    homes: dict | None = None,
) -> LifelongResult:
    """Run lifelong MAPF under Token Passing for ``max_steps`` ticks.

    ``starts`` maps agent id -> current :class:`~mrn_coord.mapf.Cell`. Agents
    are tasked exactly as in :func:`mrn_coord.lifelong.run_lifelong` (the
    ``allocator`` / ``open_tasks`` knobs), then move by committing full
    reservation-respecting paths into a shared token. Movement is collision-free
    by construction.

    ``homes`` gives each agent a *parking* cell to rest on when it has no task
    (default: its start). For Token Passing to stay live the instance must be
    **well-formed**: home cells are disjoint from the task endpoints the
    ``stream`` hands out, so a resting agent — which reserves its home cell
    forever — never blocks a task endpoint another agent must reach. ``horizon``
    caps the per-agent planning depth (default a multiple of the grid diameter).
    Returns a :class:`~mrn_coord.lifelong.LifelongResult`; pass
    ``keep_history=True`` to capture per-step positions for rendering.
    """
    ids = sorted(starts)
    pos = {a: starts[a] for a in ids}
    home = dict(homes) if homes is not None else {a: starts[a] for a in ids}
    # plan[a] is the agent's committed path from *now* (index 0 == current cell).
    plan = {a: [pos[a]] for a in ids}
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

    assign = make_assigner(grid, ids, stream, allocator, open_tasks, dist_to,
                           goal, has_task, assigned_at, elapsed)
    for a in ids:
        goal[a] = pos[a]                  # default: hold position until tasked
    assign(ids, 0, pos)

    completed = 0
    per_agent = {a: 0 for a in ids}
    service_times: list = []
    history: list = []
    goal_history: list = []
    completions: list = []
    replans = 0
    blocked = 0

    for step in range(max_steps):
        # 1. completions + reassignment (mirrors run_lifelong's accounting).
        done = [a for a in ids if has_task[a] and pos[a] == goal[a]]
        for a in done:
            completed += 1
            per_agent[a] += 1
            service_times.append(step - assigned_at[a])
            has_task[a] = False
            goal[a] = pos[a]
            plan[a] = [pos[a]]
        if done:
            assign(done, step, pos)
        completions.append(len(done))

        if keep_history:
            history.append(dict(pos))
            goal_history.append({a: (goal[a] if has_task[a] else pos[a])
                                 for a in ids})

        # 2. token update: every agent whose committed plan is spent re-plans,
        # one at a time, reading the current token (others' committed paths) as
        # reservations. Longest-unfinished-first, matching the other engines.
        order = sorted(ids, key=lambda a: (-elapsed[a], a))
        for a in order:
            if len(plan[a]) > 1:
                continue                  # still executing a committed path
            # head for the task goal, or — when idle — for the parking home.
            target = goal[a] if has_task[a] else home[a]
            if pos[a] == target:
                plan[a] = [pos[a]]        # already there: hold (reserve the cell)
                continue
            vertex, edge = _reservations(plan, ids, a, H)
            p = plan_path(grid, pos[a], target, vertex, edge, max_time=H)
            replans += 1
            if p is not None and len(p) > 1:
                plan[a] = p
            else:
                plan[a] = [pos[a]]        # blocked: hold in place, retry next tick
                blocked += 1

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

    # final-tick completions.
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
    result.blocked = blocked
    return result
