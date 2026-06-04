"""Bibox: polynomial-time complete MAPF on biconnected graphs (Surynek, 2009).

A Python reproduction of **Bibox** — Pavel Surynek's *"A Novel Approach to Path
Planning for Multiple Robots in Bi-connected Graphs"* (ICRA 2009; journal
version *"A Complete Multi-robot Path-planning Algorithm"*, AAMAS 2014). Bibox is
a *constructive* solver, not a search: on any **biconnected** (2-vertex-connected)
graph with **at least two unoccupied vertices** it places every agent on its goal
in O(n^3) moves, and it is *complete* on that class — exactly the regime where
optimal search (CBS) blows up.

The distinctive idea is an **open ear decomposition** of the graph: a basic cycle
``L0`` plus a sequence of *ears* (chains whose two endpoints lie on the part built
so far and whose interior vertices are new). Bibox solves the derived ears one by
one **in reverse order** (``Lr`` down to ``L1``), locking each ear's interior once
its goal agents are in place, and finally solves the basic cycle. An ear is filled
by **rotating the cycle** formed by the ear plus a return path through the already
built subgraph: agents are staged at the ear's entrance and a single rotation
conveys each one step into the ear's interior, shoving whatever was there out
through the exit. Because every locked ear holds exactly its goal blanks, the
working subgraph always keeps the two free vertices the rotations need (Surynek's
Prop. 5); a *BorrowBlanks* goal transform first guarantees the two blanks live in
the basic cycle, undone by *ReturnBlanks* once solving finishes.

**Faithful core / honest scope.** The ear decomposition, the reverse-order ear
solving by cycle rotation, the locking, and the borrow/return transform are
reproduced directly from the algorithm. The *basic cycle* — which Surynek solves
with the six-stage ``BringAgentsTogether`` maneuver that borrows the first ear as a
reordering buffer — is closed here on the small theta region ``L0 ∪ int(L1)`` by a
generic biconnected-region endgame (peel vertices while the remainder stays
biconnected, then finish the O(1) core by an exact joint search). This realises the
same guarantee (the chord of ``L1`` provides the bypass a pure cycle lacks) through
the region's biconnectivity rather than the literal maneuver. The implementation
targets **undirected** graphs, so the directed-graph *escape doors* are unneeded.

Every primitive only ever steps one agent into an adjacent empty cell, so any plan
returned is collision-free and ends with all agents on their goals *by
construction*. ``bibox`` returns ``None`` outside its class — graphs that are not
biconnected, or instances with fewer than two blanks.
"""

from __future__ import annotations

from collections import deque

from .grid import GridWorld
from .push_and_rotate import _moves_to_paths
from .solution import Solution, sum_of_costs


# --------------------------------------------------------------------------- #
# Graph helpers (undirected adjacency over the grid's free cells)
# --------------------------------------------------------------------------- #
def _build_adjacency(grid: GridWorld) -> dict:
    free = [(x, y) for x in range(grid.width) for y in range(grid.height)
            if grid.is_free((x, y))]
    fset = set(free)
    adj = {c: set() for c in free}
    for (x, y) in free:
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb in fset:
                adj[(x, y)].add(nb)
    return adj


def _connected(adj: dict, verts=None) -> bool:
    verts = set(adj) if verts is None else set(verts)
    if not verts:
        return True
    start = next(iter(verts))
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in verts and v not in seen:
                seen.add(v)
                q.append(v)
    return seen == verts


def _biconnected(adj: dict, verts=None) -> bool:
    """Connected with no articulation vertex in the induced subgraph."""
    verts = set(adj) if verts is None else set(verts)
    if len(verts) < 3:
        return False
    if not _connected(adj, verts):
        return False
    for v in verts:
        if not _connected(adj, verts - {v}):
            return False
    return True


def _bfs_path(adj, src, dst, allowed, forbid_edge=None):
    """Shortest src->dst path through ``allowed`` vertices, optionally skipping
    a single edge."""
    if src == dst:
        return [src]
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in allowed or v in prev:
                continue
            if forbid_edge and {u, v} == forbid_edge:
                continue
            prev[v] = u
            if v == dst:
                path = [v]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(v)
    return None


# --------------------------------------------------------------------------- #
# Open ear decomposition (greedy, deterministic)
# --------------------------------------------------------------------------- #
def _ear_decomposition(adj: dict):
    """Return ``(basic_cycle, ears)``: the basic cycle as a vertex list in cyclic
    order, and each ear as ``[x0, <interior...>, xk]`` with endpoints in the prefix
    and new interior vertices. Only non-trivial ears (>=1 interior vertex) are
    returned. ``(None, None)`` if a bridge is hit (not biconnected)."""
    verts = set(adj)
    u = min(verts)
    v = min(adj[u])
    back = _bfs_path(adj, v, u, verts, forbid_edge={u, v})
    if back is None:
        return None, None
    basic = back[:]
    built = set(basic)
    ears = []
    while built != verts:
        x = y = None
        for bx in sorted(built):
            outs = sorted(n for n in adj[bx] if n not in built)
            if outs:
                x, y = bx, outs[0]
                break
        if x is None:
            break
        prev = {y: None}
        q = deque([y])
        closure = None
        unbuilt = verts - built
        while q and closure is None:
            w = q.popleft()
            for z in sorted(adj[w]):
                if z in built and z != x:
                    closure = (w, z)
                    break
                if z in unbuilt and z not in prev:
                    prev[z] = w
                    q.append(z)
        if closure is None:
            for w in prev:
                for z in sorted(adj[w]):
                    if z in built:
                        closure = (w, z)
                        break
                if closure:
                    break
        w, z = closure
        interior = [w]
        while prev[interior[-1]] is not None:
            interior.append(prev[interior[-1]])
        interior = interior[::-1]
        ears.append([x] + interior + [z])
        built |= set(interior)
    return basic, ears


def ear_decomposition(grid: GridWorld):
    """Public helper: open ear decomposition of ``grid``'s free graph, returned as
    ``(basic_cycle, ears)`` (see :func:`_ear_decomposition`)."""
    return _ear_decomposition(_build_adjacency(grid))


# --------------------------------------------------------------------------- #
# Bibox solver state + primitives
# --------------------------------------------------------------------------- #
class _Bibox:
    def __init__(self, adj, agents):
        self.adj = adj
        self.goal = {a: g for a, (s, g) in agents.items()}
        self.pos = {a: s for a, (s, g) in agents.items()}
        self.occ = {s: a for a, (s, g) in agents.items()}
        self.moves = []
        self.frozen = set()

    # -- elementary move --------------------------------------------------- #
    def _empty(self, cell):
        return cell not in self.occ

    def _step(self, agent, to):
        frm = self.pos[agent]
        assert frm not in self.frozen, f"moved frozen agent at {frm}"
        assert self._empty(to), f"step into occupied {to}"
        assert to in self.adj[frm], f"non-adjacent step {frm}->{to}"
        del self.occ[frm]
        self.occ[to] = agent
        self.pos[agent] = to
        self.moves.append((agent, to))

    def _goal_agent(self, cell):
        for a, g in self.goal.items():
            if g == cell:
                return a
        return None

    # -- routing helpers --------------------------------------------------- #
    def _path_to_blank(self, src, allowed, avoid):
        prev = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if self._empty(u) and u != src:
                path = [u]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            for v in sorted(self.adj[u]):
                if v in allowed and v not in avoid and v not in prev:
                    prev[v] = u
                    q.append(v)
        return None

    def _shift_chain(self, path):
        """``path = [c0(occupied), ..., ck(empty)]``; cascade so c0 ends empty."""
        for i in range(len(path) - 1, 0, -1):
            self._step(self.occ[path[i - 1]], path[i])

    def _move_agent(self, agent, target, allowed):
        """Move ``agent`` to ``target`` inside ``allowed``; other non-frozen agents
        shuffle freely. Needs ``allowed`` connected with a blank."""
        guard = 0
        while self.pos[agent] != target:
            guard += 1
            if guard > 100000:
                return False
            path = _bfs_path(self.adj, self.pos[agent], target, allowed)
            if path is None or len(path) < 2:
                return False
            nxt = path[1]
            if not self._empty(nxt):
                ev = self._path_to_blank(nxt, allowed, avoid={self.pos[agent]})
                if ev is None:
                    return False
                self._shift_chain(ev)
            self._step(agent, nxt)
        return True

    # -- cycle rotation (uniform one-step forward shift) ------------------- #
    def _rotate_once(self, cyc, forward=True):
        seq = list(cyc) if forward else list(cyc)[::-1]
        L = len(seq)
        orig = [self.occ.get(c) for c in seq]
        blanks = [i for i in range(L) if orig[i] is None]
        assert blanks, "rotate needs a blank on the cycle"
        for b in blanks:
            j = (b - 1) % L
            while orig[j] is not None and j != b:
                self._step(orig[j], seq[(j + 1) % L])
                j = (j - 1) % L

    # -- solve one derived ear by rotation (Algorithm 1, undirected) ------- #
    def _solve_ear(self, ear, Dprev):
        interior = ear[1:-1]
        z = len(interior)
        if z == 0:
            return True
        x0, xk = ear[0], ear[-1]
        pi = _bfs_path(self.adj, xk, x0, Dprev)
        if pi is None:
            return False
        cyc = list(ear) + pi[1:-1]
        if len(set(cyc)) != len(cyc):
            return False
        # prime: convey a blank into D_{i-1} so staging can run
        guard = 0
        while not any(self._empty(c) for c in Dprev):
            self._rotate_once(cyc, forward=True)
            guard += 1
            if guard > 100000:
                return False
        # stack the goal pattern at the entrance, one rotation each. A goal agent
        # is staged at x0; a blank-goal interior cell stages an empty x0 (the
        # rotation then conveys a blank into the interior).
        for l in range(z, 0, -1):
            q = self._goal_agent(interior[l - 1])
            if q is not None:
                if self.pos[q] != x0:
                    if not self._move_agent(q, x0, Dprev):
                        return False
            else:
                if not self._empty(x0):
                    ev = self._path_to_blank(x0, Dprev, avoid=set())
                    if ev is None:
                        return False
                    self._shift_chain(ev)
            # the rotation needs a blank on the cycle; if none, free the exit
            # (route its occupant through D_{i-1}, never touching the staged
            # entrance or the partly stacked interior)
            if not any(self._empty(c) for c in cyc):
                ev = self._path_to_blank(xk, Dprev, avoid={x0} | set(interior))
                if ev is None:
                    return False
                self._shift_chain(ev)
            if not any(self._empty(c) for c in cyc):
                return False
            self._rotate_once(cyc, forward=True)
        for cell in interior:
            a = self._goal_agent(cell)
            if a is not None and self.pos[a] != cell:
                return False
        return True

    # -- solve a biconnected region (peel + brute endgame) ----------------- #
    def _solve_region(self, region):
        remaining = set(region)
        CORE = 7
        while len(remaining) > CORE and _biconnected(self.adj, remaining):
            cand = None
            for v in sorted(remaining):
                if len(remaining) - 1 >= 3 and _biconnected(self.adj, remaining - {v}):
                    cand = v
                    break
            if cand is None:
                break
            a = self._goal_agent(cand)
            if a is not None and self.goal[a] == cand:
                if not self._move_agent(a, cand, remaining):
                    return False
            else:
                if not self._empty(cand):
                    ev = self._path_to_blank(cand, remaining, avoid=set())
                    if ev is None:
                        return False
                    self._shift_chain(ev)
            self.frozen = self.frozen | {cand}
            remaining.discard(cand)
        return self._brute_region(remaining)

    def _brute_region(self, cells, max_states=400000):
        cellset = set(cells)
        agents_here = sorted((a for a in self.pos if self.pos[a] in cellset),
                             key=lambda a: self.pos[a])
        if not agents_here:
            return True
        start = tuple(self.pos[a] for a in agents_here)
        target = tuple(self.goal[a] for a in agents_here)
        if any(self.goal[a] not in cellset for a in agents_here):
            return False
        if start == target:
            return True
        adj_local = {c: [n for n in self.adj[c] if n in cellset] for c in cellset}
        parent = {start: None}
        q = deque([start])
        found = None
        while q and len(parent) < max_states:
            cfg = q.popleft()
            occ_cells = set(cfg)
            for idx, a in enumerate(agents_here):
                cur = cfg[idx]
                for nb in adj_local[cur]:
                    if nb not in occ_cells:
                        nc = cfg[:idx] + (nb,) + cfg[idx + 1:]
                        if nc not in parent:
                            parent[nc] = (cfg, a, nb)
                            if nc == target:
                                found = nc
                                q.clear()
                                break
                            q.append(nc)
                if found:
                    break
        if found is None:
            return False
        seq = []
        cur = found
        while parent[cur] is not None:
            pc, a, to = parent[cur]
            seq.append((a, to))
            cur = pc
        for mv in reversed(seq):
            self._step(*mv)
        return True

    # -- orchestration ----------------------------------------------------- #
    def solve(self, basic, ears):
        frozen = set()
        # solve derived ears L_r .. L_2 by cycle rotation, locking each interior
        for j in range(len(ears) - 1, 0, -1):
            Dprev = set(basic)
            for k in range(j):
                Dprev |= set(ears[k][1:-1])
            self.frozen = frozen
            if not self._solve_ear(ears[j], Dprev):
                return False
            frozen = frozen | set(ears[j][1:-1])
        # close the basic cycle on the theta region L0 ∪ int(L1)
        region = set(basic)
        if ears:
            region |= set(ears[0][1:-1])
        self.frozen = frozen
        return self._solve_region(region)


# --------------------------------------------------------------------------- #
# Goal transform: BorrowBlanks / ReturnBlanks (Section 6.2)
# --------------------------------------------------------------------------- #
def _borrow_blanks(adj, goal, basic_set):
    """Slide goal-blanks into the basic cycle until it carries >=2 of them,
    recording each slide path so :func:`_return_blanks` can undo it. Returns the
    list of paths, or ``None`` if the consolidation is impossible."""
    def goal_blanks():
        return set(adj) - set(goal.values())

    paths = []
    guard = 0
    while len(goal_blanks() & basic_set) < 2:
        guard += 1
        if guard > 50:
            return None
        gb = goal_blanks()
        outside = sorted(gb - basic_set)
        targets = sorted(basic_set - gb)
        if not outside or not targets:
            return None
        done = False
        for g in outside:
            allowed = set(adj) - (gb - {g})
            for n in targets:
                if n not in allowed:
                    continue
                path = _bfs_path(adj, g, n, allowed)
                if path is None or len(path) < 2:
                    continue
                inv = {gg: a for a, gg in goal.items()}
                if not all(path[i] in inv for i in range(1, len(path))):
                    continue
                for i in range(len(path) - 1):
                    goal[inv[path[i + 1]]] = path[i]
                paths.append(path)
                done = True
                break
            if done:
                break
        if not done:
            return None
    return paths


# --------------------------------------------------------------------------- #
# Public solver
# --------------------------------------------------------------------------- #
def bibox(grid: GridWorld, agents: dict, *, max_moves: int = 200000,
          stats: dict | None = None):
    """Solve a MAPF instance on a biconnected grid with Bibox.

    ``agents`` maps an id to ``(start, goal)``. Returns a :class:`Solution`
    (collision-free, every agent on its goal — suboptimal), or ``None`` when the
    instance is outside Bibox's class (graph not biconnected, fewer than two
    blanks) or could not be solved. ``stats`` records ``moves``, ``ears``,
    ``basic_cycle_len`` and ``blanks``."""
    if not agents:
        return Solution(paths={}, cost=0)
    for a, (s, g) in agents.items():
        if not grid.is_free(s) or not grid.is_free(g):
            return None
    adj = _build_adjacency(grid)
    if not _connected(adj):
        return None
    blanks = len(adj) - len(agents)
    if stats is not None:
        stats["blanks"] = blanks
    if blanks < 2:
        return None

    bc, ears = _ear_decomposition(adj)
    if bc is None:
        return None
    if stats is not None:
        stats["basic_cycle_len"] = len(bc)
        stats["ears"] = len(ears)

    orig_goal = {a: g for a, (s, g) in agents.items()}
    solver = _Bibox(adj, agents)
    if not ears:
        ok = solver._solve_region(set(adj))               # pure cycle
    else:
        if not _biconnected(adj):
            return None
        ret_paths = _borrow_blanks(adj, solver.goal, set(bc))
        if ret_paths is None:
            return None
        ok = solver.solve(bc, ears)
        if ok:
            solver.frozen = set()
            for path in reversed(ret_paths):
                if not solver._empty(path[-1]):
                    ok = False
                    break
                solver._shift_chain(path)

    if not ok or any(solver.pos[a] != orig_goal[a] for a in agents):
        return None
    if len(solver.moves) > max_moves:
        return None
    if stats is not None:
        stats["moves"] = len(solver.moves)
    paths = _moves_to_paths({a: (s, g) for a, (s, g) in agents.items()},
                            solver.moves)
    return Solution(paths=paths, cost=sum_of_costs(paths))
