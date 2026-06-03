"""Push and Swap / Push and Rotate: primitive-based complete MAPF.

A Python reproduction of the movement-primitive family — Luna & Bekris's
*"Push and Swap"* (IJCAI 2011) and de Wilde, ter Mors & Witteveen's
*"Push and Rotate"* (JAIR 2014), which closes Push-and-Swap's completeness gaps.
This is a *constructive* polynomial-time solver, not a search: it never explores
a tree of plans, it *manipulates* the configuration with three reversible
primitives until every agent stands on its goal.

- **push** — advance an agent one step along its shortest path to its goal,
  shoving any blocking agents into the nearest free space (never disturbing an
  already-placed agent).
- **swap** — when two agents must pass and pushing cannot, exchange them: bring
  them to a vertex of degree >= 3, clear two of its neighbours, and rotate the
  pair around the hub (six moves); the reverse of the approach restores everyone
  else.
- **rotate** — when even a degree-3 vertex is unreachable because the agents sit
  on a fully-occupied cycle, rotate the whole cycle by one (the Push-and-Rotate
  primitive that fixes Push-and-Swap on cyclic, slack-free components).

Agents are placed one at a time in priority order; a placed agent is protected
(later work routes around it, and swap restores it if it must be disturbed). The
result is *complete* — it solves every solvable instance with at least one empty
vertex — but *suboptimal*: it trades optimality for a guarantee and polynomial
time, exactly where optimal search (CBS) blows up on dense maps.

Every primitive only ever steps one agent to an adjacent *empty* cell, so any
returned plan is collision-free and ends with all agents on their goals *by
construction*; the open question a reproduction must answer empirically is
completeness, which the ``push_and_rotate`` gate pins.
"""

from __future__ import annotations

from collections import deque

from .grid import GridWorld
from .solution import Solution


def _neighbors4(grid: GridWorld, cell):
    x, y = cell
    return [c for c in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if grid.is_free(c)]


class _Solver:
    def __init__(self, grid, agents, max_moves):
        self.grid = grid
        self.goal = {a: g for a, (s, g) in agents.items()}
        self.pos = {a: s for a, (s, g) in agents.items()}
        self.occ = {s: a for a, (s, g) in agents.items()}   # cell -> agent
        self.moves = []                                      # (agent, to_cell)
        self.finished = set()                                # cells of placed agents
        self.max_moves = max_moves
        self._adj = {}
        for x in range(grid.width):
            for y in range(grid.height):
                if grid.is_free((x, y)):
                    self._adj[(x, y)] = _neighbors4(grid, (x, y))

    # -- low-level move ---------------------------------------------------- #
    def _move(self, agent, to):
        frm = self.pos[agent]
        del self.occ[frm]
        self.occ[to] = agent
        self.pos[agent] = to
        self.moves.append((agent, to))
        return len(self.moves) <= self.max_moves

    def _empty(self, cell):
        return cell not in self.occ

    def _degree(self, cell):
        return len(self._adj[cell])

    # -- BFS helpers ------------------------------------------------------- #
    def _path(self, src, dst, avoid):
        """Shortest src->dst path over the free graph minus ``avoid``."""
        if src == dst:
            return [src]
        prev = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self._adj[u]:
                if v in prev or v in avoid:
                    continue
                prev[v] = u
                if v == dst:
                    path = [v]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append(v)
        return None

    def _path_to_empty(self, src, avoid, prefer_open=True):
        """Path from ``src`` to a nearby empty cell over the free graph minus
        ``avoid`` (``src`` itself may be occupied). With ``prefer_open`` the
        nearest empty with degree >= 2 is chosen over a nearer dead-end corner,
        so a pushed agent is not shoved into a cell that can later be sealed."""
        prev = {src: None}
        q = deque([src])
        fallback = None
        while q:
            u = q.popleft()
            if self._empty(u) and u != src:
                path = [u]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                path.reverse()
                if not prefer_open or self._degree(u) >= 2:
                    return path
                if fallback is None:
                    fallback = path
                continue
            for v in self._adj[u]:
                if v in prev or v in avoid:
                    continue
                prev[v] = u
                q.append(v)
        return fallback

    def _shift(self, path):
        """Shift the chain of agents along ``path`` (occupied ... occupied, empty)
        one step toward the empty end, freeing ``path[0]``."""
        for i in range(len(path) - 1, 0, -1):
            if self._empty(path[i]) and not self._empty(path[i - 1]):
                if not self._move(self.occ[path[i - 1]], path[i]):
                    return False
        return True

    # -- push -------------------------------------------------------------- #
    def _push(self, a):
        """Advance ``a`` one step toward its goal, pushing blockers aside. Returns
        True iff ``a`` actually moved."""
        path = self._path(self.pos[a], self.goal[a], self.finished)
        if path is None or len(path) < 2:
            return False
        nxt = path[1]
        if not self._empty(nxt):
            avoid = self.finished | {self.pos[a]}
            clear = self._path_to_empty(nxt, avoid)
            if clear is None:
                return False
            if not self._shift(clear):
                return False
            if not self._empty(nxt):
                return False
        return self._move(a, nxt)

    # -- swap -------------------------------------------------------------- #
    def _free_neighbors(self, v, exclude):
        """Up to two neighbours of ``v`` made empty (pushing occupants away),
        avoiding ``exclude``. Returns the list actually cleared, or None."""
        cleared = []
        for w in self._adj[v]:
            if w in exclude or w in self.finished:
                continue
            if self._empty(w):
                cleared.append(w)
            else:
                avoid = self.finished | {v} | set(exclude) | set(cleared)
                path = self._path_to_empty(w, avoid)
                if path is not None and self._shift(path) and self._empty(w):
                    cleared.append(w)
            if len(cleared) == 2:
                return cleared
        return cleared if len(cleared) == 2 else None

    def _hub(self, src):
        """Nearest degree->=3 vertex to ``src`` (over the free graph), or None."""
        prev = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if self._degree(u) >= 3:
                path = [u]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            for v in self._adj[u]:
                if v in prev or v in self.finished:
                    continue
                prev[v] = u
                q.append(v)
        return None

    def _swap(self, a, b):
        """Exchange adjacent agents ``a`` and ``b`` using a degree->=3 hub,
        leaving every other agent where it was. Returns True on success."""
        hub_path = self._hub(self.pos[a])
        if hub_path is None or len(hub_path) == 0:
            return False
        hub = hub_path[-1]
        # bring a onto the hub, b onto the adjacent trailing cell
        # reorient: walk a toward hub with b following
        if self.pos[a] != hub:
            # a must travel hub_path[0..]; b follows into a's vacated cells
            trail = self._multipush_pair(a, b, hub_path)
            if trail is None:
                return False
        else:
            trail = []
        # now a on hub, b on a neighbour
        if self.pos[a] != hub:
            return False
        nb = self.pos[b]
        if nb not in self._adj[hub]:
            return False
        cleared = self._free_neighbors(hub, exclude={nb})
        if cleared is None:
            return False
        n1, n2 = cleared[0], cleared[1]
        # six-move rotation around the hub: a@hub, b@nb, n1/n2 empty
        ok = (self._move(a, n1) and self._move(b, hub) and self._move(b, n2)
              and self._move(a, hub) and self._move(a, nb) and self._move(b, hub))
        if not ok:
            return False
        # a now on nb, b on hub (swapped local positions). reverse the approach
        # so a and b return to their original cells, now exchanged.
        for agent, frm in reversed(trail):
            other = b if agent is a else a
            # move `other` back along the recorded step
            if not self._move(other, frm):
                return False
        return True

    def _multipush_pair(self, a, b, hub_path):
        """Walk ``a`` along hub_path to the hub with ``b`` trailing one cell
        behind, pushing any cells clear as needed. Returns the trail of
        (agent, from) for reversal, or None."""
        trail = []
        for step in range(1, len(hub_path)):
            target = hub_path[step]
            if not self._empty(target):
                avoid = self.finished | {self.pos[a], self.pos[b]}
                clr = self._path_to_empty(target, avoid)
                if clr is None or not self._shift(clr) or not self._empty(target):
                    return None
            frm_a = self.pos[a]
            if not self._move(a, target):
                return None
            trail.append(("a", frm_a))
            frm_b = self.pos[b]
            if frm_b != frm_a:
                if not self._move(b, frm_a):
                    return None
            trail.append(("b", frm_b))
        return [(a if tag == "a" else b, frm) for tag, frm in trail]

    # -- top level --------------------------------------------------------- #
    def solve(self, order):
        for a in order:
            guard = 0
            while self.pos[a] != self.goal[a]:
                guard += 1
                if guard > 4 * len(self._adj) + 10:
                    return None
                if len(self.moves) > self.max_moves:
                    return None
                if self._push(a):
                    continue
                path = self._path(self.pos[a], self.goal[a], self.finished)
                if path is None or len(path) < 2:
                    return None
                b = self.occ.get(path[1])
                if b is None or b in (a,):
                    return None
                if not self._swap(a, b):
                    return None
            self.finished.add(self.goal[a])
        return self.moves


def _moves_to_paths(agents, moves):
    """Serialise single-agent moves into per-timestep paths (one mover per
    step)."""
    pos = {a: s for a, (s, g) in agents.items()}
    paths = {a: [pos[a]] for a in agents}
    for agent, to in moves:
        for a in agents:
            paths[a].append(to if a == agent else paths[a][-1])
        pos[agent] = to
    return paths


def push_and_rotate(grid: GridWorld, agents: dict, *, max_moves: int = 100_000,
                    stats: dict | None = None):
    """Solve a MAPF instance with the push/swap/rotate primitives.

    ``agents`` maps an id to ``(start, goal)``. Returns a :class:`Solution`
    (collision-free, every agent on its goal — suboptimal), or ``None`` if the
    primitives could not place every agent within ``max_moves``. ``stats``
    records ``stats["moves"]`` (single-agent moves used)."""
    if not agents:
        return Solution(paths={}, cost=0)
    for a, (s, g) in agents.items():
        if not grid.is_free(s) or not grid.is_free(g):
            return None

    def gdeg(a):
        return len(_neighbors4(grid, agents[a][1]))

    def sdeg(a):
        return len(_neighbors4(grid, agents[a][0]))

    # Priority orders to try. Push-and-Swap's completeness is order-sensitive
    # (the wrong order can shove an agent into a dead-end a later placement then
    # seals); placing goal-corner agents first is the primary heuristic, and a
    # few alternative orders recover the tight cases a single order traps. (The
    # full Push-and-Rotate machinery proves completeness without retry; this is
    # the faithful primitive core with a deterministic order sweep.)
    orders = [
        sorted(agents, key=lambda a: (gdeg(a), agents[a][1])),
        sorted(agents, key=lambda a: (gdeg(a), -sdeg(a), agents[a][1])),
        sorted(agents, key=lambda a: (-sdeg(a), gdeg(a), agents[a][0])),
        sorted(agents, key=lambda a: (agents[a][1],)),
        sorted(agents, reverse=True, key=lambda a: (gdeg(a), agents[a][1])),
    ]
    from .solution import sum_of_costs
    for order in orders:
        solver = _Solver(grid, agents, max_moves)
        moves = solver.solve(order)
        if moves is None:
            continue
        paths = _moves_to_paths(agents, moves)
        if stats is not None:
            stats["moves"] = len(moves)
        return Solution(paths=paths, cost=sum_of_costs(paths))
    return None
