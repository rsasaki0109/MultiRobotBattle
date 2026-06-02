"""Rolling-Horizon Collision Resolution (RHCR) for lifelong MAPF.

Li, Tinka, Kiesel, Durham, Kumar & Koenig, *Lifelong Multi-Agent Path Finding in
Large-Scale Warehouses* (AAAI 2021). The default lifelong engine here
(:func:`mrn_coord.lifelong.run_lifelong`) steps **PIBT**: a greedy,
single-timestep, collision-free move. RHCR is the planning alternative — it
decomposes the endless problem into a sequence of **Windowed MAPF** instances:

- every ``replan_period`` (``h``) timesteps it re-plans full paths toward the
  current goals, but the windowed solver only **resolves collisions within the
  next ``window`` (``w >= h``) timesteps** and ignores everything beyond — those
  far conflicts get replanned away before they ever matter;
- the team then *commits* and executes the next ``h`` steps of that plan and the
  cycle repeats.

Bounding conflict resolution to a window is what lets it scale (the windowed
solver only ever reasons about a short horizon), and the lookahead is what lets
it sidestep the local traps a one-step-greedy stepper can walk into. The
windowed solver is pluggable: **PBS** (:mod:`mrn_coord.mapf.pbs`, the paper's
recommended choice and the default), fixed-order prioritized planning (``"pp"``),
or PIBT rollout (``"pibt"``). PBS / PP fall back to a PIBT rollout for any window
they cannot fully resolve, so a run is *always* collision-free and live.

Scored, like all lifelong runs, by **throughput** (tasks completed per timestep).
Pure and deterministic.
"""

from __future__ import annotations

from mrn_coord.mapf import pbs_paths
from mrn_coord.mapf.conflicts import cell_at
from mrn_coord.mapf.pbs import _plan_under

from .lifelong import LifelongResult, _bfs_dist, _Pibt, make_assigner


def _pibt_rollout(grid, pos, goal, elapsed, window, dist_to) -> dict:
    """A ``window``-step PIBT rollout from ``pos``; always collision-free.

    The guaranteed-safe fallback (and the ``solver="pibt"`` window): run the
    one-step PIBT move ``window`` times and stack the configurations into a path
    per agent. Priority order is longest-unfinished-first, matching
    :func:`run_lifelong`.
    """
    order = sorted(pos, key=lambda a: (-elapsed[a], a))
    cur = dict(pos)
    paths = {a: [cur[a]] for a in pos}
    for _ in range(window):
        pibt = _Pibt(grid, cur, goal, {a: dist_to(goal[a]) for a in pos})
        for a in order:
            if a not in pibt.next_pos:
                pibt.decide(a)
        cur = pibt.next_pos
        for a in pos:
            paths[a].append(cur[a])
    return paths


def _windowed_pp(grid, agents, order_hint, window):
    """Fixed-order prioritized planning within ``window`` (``None`` if it fails)."""
    paths: dict = {}
    for a in order_hint:
        start, goal = agents[a]
        p = _plan_under(grid, start, goal, list(paths.values()), window)
        if p is None:
            return None
        paths[a] = p
    return paths


def _plan_window(grid, pos, goal, elapsed, window, solver, dist_to, max_nodes) -> dict:
    """Plan a window of motion from ``pos`` toward ``goal``; never returns ``None``.

    Dispatches on ``solver`` and falls back to a PIBT rollout for anything the
    planning solver cannot resolve, so the returned ``{agent: path}`` is always
    collision-free within ``window`` and every agent has somewhere to go.
    """
    agents = {a: (pos[a], goal[a]) for a in pos}
    order_hint = sorted(pos, key=lambda a: (-elapsed[a], a))
    if solver == "pbs":
        paths = pbs_paths(grid, agents, window=window, order_hint=order_hint,
                          max_nodes=max_nodes)
    elif solver == "pp":
        paths = _windowed_pp(grid, agents, order_hint, window)
    elif solver == "pibt":
        paths = None
    else:
        raise ValueError(f"unknown windowed solver: {solver!r}")
    if paths is None:
        return _pibt_rollout(grid, pos, goal, elapsed, window, dist_to)
    return paths


def run_rhcr(
    grid,
    starts: dict,
    stream,
    *,
    max_steps: int = 256,
    window: int = 16,
    replan_period: int | None = None,
    solver: str = "pbs",
    keep_history: bool = False,
    allocator: str = "stream",
    open_tasks: int | None = None,
    max_nodes: int = 10_000,
) -> LifelongResult:
    """Run lifelong MAPF under RHCR for ``max_steps`` ticks; return throughput.

    ``starts`` maps agent id -> current :class:`~mrn_coord.mapf.Cell`. Every
    ``replan_period`` ticks the team re-plans toward its current goals with the
    windowed ``solver`` (resolving conflicts only ``window`` steps ahead) and
    commits the next ``replan_period`` steps. ``replan_period`` defaults to
    ``window`` (commit the whole resolved window) and must satisfy ``1 <= h <=
    w``. ``allocator`` / ``open_tasks`` match free robots to tasks exactly as in
    :func:`run_lifelong`. Movement is collision-free by construction; pass
    ``keep_history=True`` to capture per-step positions for rendering.
    """
    h = replan_period if replan_period is not None else window
    if not (1 <= h <= window):
        raise ValueError(f"replan_period must satisfy 1 <= h <= window; got {h}, {window}")

    ids = sorted(starts)
    pos = {a: starts[a] for a in ids}
    goal: dict = {}
    has_task = {a: False for a in ids}
    assigned_at: dict = {}
    elapsed = {a: 0 for a in ids}
    dist_cache: dict = {}

    def dist_to(g):
        if g not in dist_cache:
            dist_cache[g] = _bfs_dist(grid, g)
        return dist_cache[g]

    assign = make_assigner(grid, ids, stream, allocator, open_tasks, dist_to,
                           goal, has_task, assigned_at, elapsed)
    for a in ids:
        goal[a] = pos[a]                  # default: hold position until tasked

    completed = 0
    per_agent = {a: 0 for a in ids}
    service_times: list = []
    history: list = []
    goal_history: list = []

    step = 0
    while step < max_steps:
        # 1. replanning boundary: task any idle robot, then plan a fresh window.
        idle = [a for a in ids if not has_task[a]]
        if idle:
            assign(idle, step, pos)
        plan = _plan_window(grid, pos, goal, elapsed, window, solver, dist_to,
                            max_nodes)

        # 2. commit the next h steps (never past max_steps), catching completions.
        end = min(step + h, max_steps)
        for k in range(1, end - step + 1):
            if keep_history:
                history.append(dict(pos))
                goal_history.append({a: (goal[a] if has_task[a] else pos[a])
                                     for a in ids})
            pos = {a: cell_at(plan[a], k) for a in ids}
            for a in ids:
                elapsed[a] += 1
            now = step + k
            for a in ids:
                if has_task[a] and pos[a] == goal[a]:
                    completed += 1
                    per_agent[a] += 1
                    service_times.append(now - assigned_at[a])
                    has_task[a] = False
                    goal[a] = pos[a]       # idle at goal until the next boundary
        step = end

    if keep_history:
        history.append(dict(pos))
        goal_history.append({a: (goal[a] if has_task[a] else pos[a]) for a in ids})

    avg_service = (sum(service_times) / len(service_times)) if service_times else 0.0
    return LifelongResult(
        steps=max_steps,
        agents=len(ids),
        completed=completed,
        throughput=(completed / max_steps if max_steps else 0.0),
        per_agent=per_agent,
        avg_service_time=avg_service,
        max_wait=(max(service_times) if service_times else 0),
        history=history,
        goal_history=goal_history,
    )
