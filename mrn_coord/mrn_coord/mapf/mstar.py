"""M*: subdimensional expansion (Wagner & Choset, 2011/2015).

An optimal (sum-of-costs) MAPF solver built on a different idea from CBS. CBS
plans each agent alone and branches *constraints* when paths conflict; M* plans
in the *joint* configuration space but keeps the search low-dimensional almost
everywhere, raising the dimension only where agents actually interact.

The machinery has three parts:

- **Individual optimal policy** ``phi_i``. For each agent we run a backward BFS
  from its goal to get the true cost-to-go ``dist_i`` over free cells. From a
  cell, ``phi_i`` is the *set* of neighbors that strictly decrease ``dist_i``
  (every step that lies on some shortest path), or — at the goal — just waiting
  there. The sum of ``dist_i`` over agents is an admissible, consistent
  heuristic on the joint space.

- **The collision set** ``C(v)`` attached to each joint configuration ``v``
  (a tuple of one cell per agent). When the search expands ``v`` it builds the
  *limited* neighbor set: an agent **in** ``C(v)`` branches over **all** its grid
  moves (the full local dimension), while an agent **not** in ``C(v)`` is pinned
  to its individual policy ``phi_i`` (one dimension collapsed to its optimal
  steps). With every collision set empty the search is essentially ``n``
  independent shortest paths threaded together; coupling appears only where
  ``C(v)`` is non-empty.

- **Backpropagation.** When generating a neighbor reveals a collision between
  agents (they share a cell, or swap across the edge), those agents are added to
  the collision set of the *predecessor* and the growth is propagated backward
  along the recorded ``back_set`` of every configuration that can reach it,
  reopening each so it re-expands with the larger — now higher-dimensional —
  branching. This is *subdimensional expansion*: the joint search inflates to
  full dimension only on the configurations that lie on a path into a real
  interaction, and stays one-dimensional-per-agent everywhere else.

The first time the goal configuration is popped, A* optimality (the heuristic is
consistent) plus M*'s reopening rule guarantee the cost is the optimal
sum-of-costs — the **same** optimum :func:`mrn_coord.mapf.cbs.cbs` returns. What
M* buys for that optimum is locality: on instances where agents mostly pass each
other freely it expands a tiny fraction of the joint configurations a fully
coupled joint A* would (:func:`joint_astar` here is that baseline), because the
collision set — and with it the search dimension — stays small.

This module reproduces *basic* M*. The recursive variant rM*, which further
splits a collision set into independent sub-problems, is a refinement of the
same backpropagation and is left out; basic M* already exhibits the defining
subdimensional behavior this gate pins.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque

from .grid import Cell, GridWorld
from .solution import Solution


def _dist_to_goal(grid: GridWorld, goal: Cell) -> dict[Cell, int]:
    """Backward BFS cost-to-go from ``goal`` over free cells (4-connected)."""
    if not grid.is_free(goal):
        return {}
    dist = {goal: 0}
    q = deque([goal])
    while q:
        cell = q.popleft()
        x, y = cell
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if grid.is_free(nb) and nb not in dist:
                dist[nb] = dist[cell] + 1
                q.append(nb)
    return dist


class _MStar:
    """One M* run. Holds the per-agent policies and the joint search state."""

    def __init__(self, grid: GridWorld, agents: dict, max_expansions: int):
        self.grid = grid
        self.ids = list(agents)
        self.n = len(self.ids)
        self.starts = tuple(agents[a][0] for a in self.ids)
        self.goals = tuple(agents[a][1] for a in self.ids)
        self.max_expansions = max_expansions
        # Per-agent cost-to-go; an empty map means the goal is unreachable.
        self.dist = [_dist_to_goal(grid, g) for g in self.goals]
        # Cache of individual optimal steps from a (agent index, cell).
        self._policy: dict[tuple[int, Cell], tuple[Cell, ...]] = {}

    # -- individual policy ------------------------------------------------
    def _optimal_steps(self, i: int, cell: Cell) -> tuple[Cell, ...]:
        """Neighbors of ``cell`` on a shortest path to agent ``i``'s goal.

        At the goal the only optimal action is to wait there (cost 0). Off the
        goal it is every neighbor whose cost-to-go is one less — all moves that
        keep the agent on *some* individually optimal path.
        """
        key = (i, cell)
        cached = self._policy.get(key)
        if cached is not None:
            return cached
        di = self.dist[i]
        if cell == self.goals[i]:
            steps: tuple[Cell, ...] = (cell,)
        else:
            here = di[cell]
            steps = tuple(nb for nb in self.grid.neighbors(cell)
                          if di.get(nb, here) == here - 1)
        self._policy[key] = steps
        return steps

    def _h(self, config: tuple) -> int:
        return sum(self.dist[i][config[i]] for i in range(self.n))

    def _collisions(self, u: tuple, v: tuple) -> frozenset:
        """Agent indices in a vertex (shared cell) or swap collision on u->v."""
        bad: set[int] = set()
        # vertex: two agents on the same cell in v
        seen: dict[Cell, int] = {}
        for i in range(self.n):
            j = seen.get(v[i])
            if j is not None:
                bad.add(i)
                bad.add(j)
            else:
                seen[v[i]] = i
        # edge (swap): i and j exchange cells across the step
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if u[i] == v[j] and u[j] == v[i] and u[i] != u[j]:
                    bad.add(i)
                    bad.add(j)
        return frozenset(bad)

    def _neighbors(self, config: tuple, settled: frozenset, cset: frozenset):
        """Generate ``(next_config, next_settled, edge_cost)`` successors.

        For each agent: a *settled* one is frozen at its goal at zero cost; an
        agent in the collision set branches over all its grid moves; everyone
        else follows its individual optimal policy. An unsettled agent sitting on
        its goal may *settle* (cost 0, the final rest CBS gets for free) or — only
        when it is coupled and might still have to move aside — keep waiting
        unsettled at cost 1. Every other step costs 1. So the accumulated cost is
        exactly the sum-of-costs (each agent pays until its final arrival), even
        when an agent must vacate its goal and return.
        """
        per_agent = []
        for i in range(self.n):
            gi = self.goals[i]
            if i in settled:
                per_agent.append([(gi, True, 0)])
                continue
            if i in cset:
                moves = self.grid.neighbors(config[i])
            else:
                moves = self._optimal_steps(i, config[i])
            on_goal = config[i] == gi
            opts = []
            for m in moves:
                if on_goal and m == gi:
                    opts.append((m, True, 0))      # settle for good (free)
                    if i in cset:
                        opts.append((m, False, 1))  # wait on, may yet vacate
                else:
                    opts.append((m, False, 1))
            per_agent.append(opts)
        for combo in itertools.product(*per_agent):
            v = tuple(c[0] for c in combo)
            newly = frozenset(i for i, c in enumerate(combo) if c[1])
            nsettled = settled | newly if newly - settled else settled
            cost = sum(c[2] for c in combo)
            yield v, nsettled, cost

    # -- search -----------------------------------------------------------
    def solve(self, stats: dict | None = None):
        for i in range(self.n):
            if self.dist[i].get(self.starts[i]) is None:
                return None  # an agent cannot reach its goal at all

        # A search node is ``(config, settled)``: the joint position tuple plus
        # the set of agents that have settled for good at their goals.
        start = (self.starts, frozenset())
        goal_cfg = self.goals
        INF = float("inf")
        g: dict[tuple, float] = {start: 0}
        parent: dict[tuple, tuple] = {}
        cset: dict[tuple, frozenset] = {start: frozenset()}
        back: dict[tuple, set] = {start: set()}
        in_open: set[tuple] = set()

        counter = itertools.count()
        open_heap: list = []

        def push(node: tuple):
            if node in in_open:
                return
            in_open.add(node)
            heapq.heappush(open_heap,
                           (g[node] + self._h(node[0]), next(counter), node))

        def backprop(vk: tuple, grown: frozenset):
            # Iterative: grow vk's collision set by `grown` and propagate the
            # (now larger) set back along every predecessor, reopening as we go.
            stack = [(vk, grown)]
            while stack:
                v, add = stack.pop()
                cur = cset.get(v, frozenset())
                if add <= cur:
                    continue
                merged = cur | add
                cset[v] = merged
                if g.get(v, INF) < INF:
                    push(v)  # reopen so v re-expands at full local dimension
                for vm in back.get(v, ()):
                    stack.append((vm, merged))

        push(start)
        expansions = 0
        while open_heap:
            _, _, vk = heapq.heappop(open_heap)
            in_open.discard(vk)
            expansions += 1
            if expansions > self.max_expansions:
                if stats is not None:
                    stats["expansions"] = expansions
                    stats["max_collision_set"] = self._max_cset(cset)
                return None

            if vk[0] == goal_cfg:
                if stats is not None:
                    stats["expansions"] = expansions
                    stats["max_collision_set"] = self._max_cset(cset)
                return self._reconstruct(parent, vk, g[vk])

            uc = vk[0]
            ck = cset.get(vk, frozenset())
            for vc, vsettled, ec in self._neighbors(uc, vk[1], ck):
                vl = (vc, vsettled)
                back.setdefault(vl, set()).add(vk)
                col = self._collisions(uc, vc)
                backprop(vk, cset.get(vl, frozenset()) | col)
                if col:
                    continue  # illegal step (overlap/swap) — never relax through
                ng = g[vk] + ec
                if ng < g.get(vl, INF):
                    g[vl] = ng
                    parent[vl] = vk
                    cset.setdefault(vl, frozenset())
                    push(vl)

        if stats is not None:
            stats["expansions"] = expansions
            stats["max_collision_set"] = self._max_cset(cset)
        return None

    @staticmethod
    def _max_cset(cset: dict) -> int:
        return max((len(s) for s in cset.values()), default=0)

    def _reconstruct(self, parent: dict, goal: tuple, cost: int) -> Solution:
        nodes = [goal]
        cur = goal
        while cur in parent:
            cur = parent[cur]
            nodes.append(cur)
        nodes.reverse()
        configs = [node[0] for node in nodes]
        paths: dict = {}
        for idx, a in enumerate(self.ids):
            seq = [cfg[idx] for cfg in configs]
            # Trim trailing waits at the goal so sum-of-costs counts true arrival
            # time (the joint trace pads every agent to the makespan).
            goal_cell = self.goals[idx]
            while len(seq) > 1 and seq[-1] == goal_cell and seq[-2] == goal_cell:
                seq.pop()
            paths[a] = seq
        return Solution(paths=paths, cost=cost)


def mstar(grid: GridWorld, agents: dict, *, max_expansions: int = 200_000,
          stats: dict | None = None):
    """Solve a MAPF instance optimally (sum-of-costs) via subdimensional expansion.

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns a
    :class:`mrn_coord.mapf.solution.Solution` whose paths are collision-free and
    minimal in sum-of-costs — the same optimum :func:`mrn_coord.mapf.cbs.cbs`
    finds — or ``None`` if the instance is infeasible or the expansion budget is
    exhausted. If ``stats`` is given, ``stats["expansions"]`` is the number of
    joint configurations popped and ``stats["max_collision_set"]`` the largest
    collision set the search ever formed (its peak coupling dimension); compare
    the expansions against :func:`joint_astar` to see subdimensional expansion
    pay off.
    """
    return _MStar(grid, agents, max_expansions).solve(stats)


def joint_astar(grid: GridWorld, agents: dict, *, max_expansions: int = 200_000,
                stats: dict | None = None):
    """Fully coupled optimal baseline: A* over the joint configuration space.

    Every agent always branches over *all* its grid moves (no individual-policy
    pinning, no collision set), so the search explores the full product space
    and expands far more configurations than :func:`mstar` for the identical
    optimum. This is the straw man M* is built to beat; the gate compares their
    expansion counts on the same instances. Tractable only for a few agents.
    """
    ids = list(agents)
    n = len(ids)
    starts = tuple(agents[a][0] for a in ids)
    goals = tuple(agents[a][1] for a in ids)
    dist = [_dist_to_goal(grid, g) for g in goals]
    for i in range(n):
        if dist[i].get(starts[i]) is None:
            return None

    def h(cfg):
        return sum(dist[i][cfg[i]] for i in range(n))

    def edge_cost(u, v):
        return sum(1 for i in range(n) if not (u[i] == v[i] == goals[i]))

    def legal(u, v):
        if len(set(v)) != n:  # vertex collision
            return False
        for i in range(n):
            for j in range(i + 1, n):
                if u[i] == v[j] and u[j] == v[i] and u[i] != u[j]:
                    return False
        return True

    INF = float("inf")
    g = {starts: 0}
    parent: dict = {}
    counter = itertools.count()
    open_heap = [(h(starts), next(counter), starts)]
    closed: set = set()
    expansions = 0
    while open_heap:
        _, _, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        closed.add(u)
        expansions += 1
        if expansions > max_expansions:
            if stats is not None:
                stats["expansions"] = expansions
            return None
        if u == goals:
            if stats is not None:
                stats["expansions"] = expansions
            configs = [u]
            cur = u
            while cur in parent:
                cur = parent[cur]
                configs.append(cur)
            configs.reverse()
            paths = {}
            for idx, a in enumerate(ids):
                seq = [cfg[idx] for cfg in configs]
                gc = goals[idx]
                while len(seq) > 1 and seq[-1] == gc and seq[-2] == gc:
                    seq.pop()
                paths[a] = seq
            return Solution(paths=paths, cost=g[u])
        per_agent = [grid.neighbors(u[i]) for i in range(n)]
        for v in itertools.product(*per_agent):
            if not legal(u, v):
                continue
            ng = g[u] + edge_cost(u, v)
            if ng < g.get(v, INF):
                g[v] = ng
                parent[v] = u
                heapq.heappush(open_heap, (ng + h(v), next(counter), v))
    if stats is not None:
        stats["expansions"] = expansions
    return None
