"""Lifelong MAPF stepped by PIBT (Priority Inheritance with Backtracking).

One-shot solvers (CBS / prioritized planning) route a fixed set of start→goal
pairs once. Lifelong MAPF keeps going: an agent that reaches its goal is handed
the next task immediately, so the team must plan *while moving*, indefinitely —
the warehouse-robot regime, scored by **throughput** (tasks per timestep).

The per-timestep collision-free move is computed by **PIBT** (Okumura et al.):
agents are ordered by priority, each greedily steps toward its goal along the
obstacle-aware distance gradient, and when a high-priority agent wants the cell
of a lower one, it *pushes* it (priority inheritance) — recursively — and
backtracks to its next-best cell if the push fails. PIBT guarantees a
collision-free configuration each step (no vertex sharing, no swaps) and avoids
the deadlock that naive reservation planning hits when a forced-to-wait agent
sits in a cell another agent already claimed. Priorities rise the longer an
agent's current task goes unfinished, so no task starves. Pure and
deterministic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from mrn_coord.mapf import Cell, GridWorld


@dataclass
class TaskStream:
    """A deterministic, endless supply of goal cells (round-robin over ``pool``)."""

    pool: list
    _i: int = 0

    def next_goal(self, avoid: Cell | None = None) -> Cell:
        for _ in range(len(self.pool)):
            goal = self.pool[self._i % len(self.pool)]
            self._i += 1
            if goal != avoid:
                return goal
        return self.pool[self._i % len(self.pool)]


@dataclass
class LifelongResult:
    """Metrics from a lifelong run (all deterministic)."""

    steps: int
    agents: int
    completed: int                       # total tasks finished
    throughput: float                    # completed / steps
    per_agent: dict                      # id -> tasks completed
    avg_service_time: float              # mean steps from assignment to completion
    max_wait: int                        # longest a single task took
    history: list = field(default_factory=list)   # [ {id: cell} per step ] if kept

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "agents": self.agents,
            "completed": self.completed,
            "throughput": round(self.throughput, 4),
            "avg_service_time": round(self.avg_service_time, 3),
            "max_wait": self.max_wait,
        }


def _bfs_dist(grid: GridWorld, goal: Cell) -> dict:
    """4-connected BFS distance from every free cell to ``goal`` (obstacle-aware)."""
    dist = {goal: 0}
    q = deque([goal])
    while q:
        cell = q.popleft()
        d = dist[cell]
        for nb in grid.neighbors(cell):
            if nb not in dist:
                dist[nb] = d + 1
                q.append(nb)
    return dist


class _Pibt:
    """One PIBT timestep over the current configuration."""

    def __init__(self, grid, pos, goal, dist_to):
        self.grid = grid
        self.pos = pos                  # agent -> current cell
        self.goal = goal                # agent -> goal cell
        self.dist_to = dist_to          # agent -> {cell: dist-to-goal}
        self.occupant = {c: a for a, c in pos.items()}   # current cell -> agent
        self.next_pos = {}              # agent -> chosen next cell
        self.next_occ = {}              # chosen next cell -> agent

    def _candidates(self, a):
        # neighbours (incl. wait), nearest-to-goal first; prefer moving over
        # waiting on ties so the team keeps flowing. Unknown dist = +inf.
        d = self.dist_to[a]
        big = len(d) + 1
        here = self.pos[a]
        cells = self.grid.neighbors(here)
        return sorted(cells, key=lambda c: (d.get(c, big), c == here, c))

    def decide(self, a, pusher=None) -> bool:
        """Choose ``a``'s next cell; push lower-priority occupants. Returns success.

        ``wait`` is just the candidate ``c == pos[a]``; if every candidate is
        already claimed (a pushed agent that is boxed in), this returns ``False``
        without claiming anything, so the pusher backtracks. A top-level agent
        always has its own cell free to wait in, so it never fails.
        """
        for c in self._candidates(a):
            if c in self.next_occ:
                continue                          # cell already taken this step
            if pusher is not None and c == self.pos[pusher]:
                continue                          # would swap with the pusher
            self.next_pos[a] = c
            self.next_occ[c] = a
            other = self.occupant.get(c)
            if other is not None and other != a and other not in self.next_pos:
                if self.decide(other, pusher=a):
                    return True
                # push failed: release c and try a's next candidate
                del self.next_occ[c]
                del self.next_pos[a]
                continue
            return True
        return False


def run_lifelong(
    grid: GridWorld,
    starts: dict,
    stream: TaskStream,
    *,
    max_steps: int = 256,
    keep_history: bool = False,
    horizon: int | None = None,   # accepted for API symmetry; unused by PIBT
) -> LifelongResult:
    """Run lifelong MAPF for ``max_steps`` ticks and return throughput metrics.

    ``starts`` maps agent id -> current :class:`Cell`. Each agent gets an initial
    goal from ``stream`` and a fresh goal whenever it arrives. Movement is
    collision-free by construction (PIBT). Returns a :class:`LifelongResult`;
    pass ``keep_history=True`` to also capture per-step positions for rendering.
    """
    del horizon
    ids = sorted(starts)
    pos = {a: starts[a] for a in ids}
    goal = {}
    assigned_at = {}
    elapsed = {a: 0 for a in ids}        # ticks since current task assigned (priority)
    dist_cache: dict = {}

    def dist_to(g):
        if g not in dist_cache:
            dist_cache[g] = _bfs_dist(grid, g)
        return dist_cache[g]

    completed = 0
    per_agent = {a: 0 for a in ids}
    service_times: list = []
    history: list = []

    for a in ids:
        goal[a] = stream.next_goal(avoid=pos[a])
        assigned_at[a] = 0

    for step in range(max_steps):
        # 1. completions + reassignment.
        for a in ids:
            if pos[a] == goal[a]:
                completed += 1
                per_agent[a] += 1
                service_times.append(step - assigned_at[a])
                goal[a] = stream.next_goal(avoid=pos[a])
                assigned_at[a] = step
                elapsed[a] = 0

        if keep_history:
            history.append(dict(pos))

        # 2. PIBT step. Priority: longest-unfinished task first, tie-break by id.
        pibt = _Pibt(grid, pos, goal, {a: dist_to(goal[a]) for a in ids})
        order = sorted(ids, key=lambda a: (-elapsed[a], a))
        for a in order:
            if a not in pibt.next_pos:
                pibt.decide(a)

        pos = pibt.next_pos
        for a in ids:
            elapsed[a] += 1

    # final-tick completions.
    for a in ids:
        if pos[a] == goal[a]:
            completed += 1
            per_agent[a] += 1
            service_times.append(max_steps - assigned_at[a])
    if keep_history:
        history.append(dict(pos))

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
    )


def make_warehouse(rows: int = 3, cols: int = 4, *, aisle: int = 1):
    """Build a small warehouse grid: ``2x2`` shelf blocks on a lattice with aisles.

    Returns ``(grid, endpoints)`` where ``endpoints`` is a deterministic list of
    free cells flanking the shelves — the pickup/dropoff stations a
    :class:`TaskStream` cycles through.
    """
    block = 2
    pitch = block + aisle
    width = aisle + cols * pitch
    height = aisle + rows * pitch

    blocked = set()
    for r in range(rows):
        for c in range(cols):
            x0 = aisle + c * pitch
            y0 = aisle + r * pitch
            for dx in range(block):
                for dy in range(block):
                    blocked.add((x0 + dx, y0 + dy))
    grid = GridWorld(width, height, blocked=frozenset(blocked))

    endpoints = []
    for r in range(rows):
        for c in range(cols):
            x0 = aisle + c * pitch
            y0 = aisle + r * pitch
            for cell in ((x0 - 1, y0), (x0 + block, y0 + block - 1)):
                if grid.is_free(cell):
                    endpoints.append(cell)
    return grid, endpoints
