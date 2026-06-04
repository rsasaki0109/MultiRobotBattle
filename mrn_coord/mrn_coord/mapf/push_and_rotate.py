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

For a *fully packed* rectangle (the 15-puzzle regime, where the greedy primitives
stall for want of slack) the solver dispatches a constructive row/column
reduction instead: a direct one for two or more empty cells, and -- because that
reduction can strand a *single* blank in a finished corner -- a tracked-agent BFS
endgame for exactly one empty cell (place each tile, or last-two pair, by an exact
search over the unsolved region that tracks only the agents being placed; every
other tile is an anonymous filler, so the state stays tiny and the blank can never
be stranded). Both stay constructive: still one agent into an adjacent empty cell.

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
        self.placed = set()                                  # ids of placed agents
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

    # -- snapshot/restore -------------------------------------------------- #
    # A primitive (swap, rotate) that fails partway has already appended moves;
    # rolling back lets the caller try the next primitive on a clean state, so
    # only a fully-successful primitive ever commits moves to the plan.
    def _snapshot(self):
        return (dict(self.pos), dict(self.occ), len(self.moves))

    def _restore(self, snap):
        self.pos = dict(snap[0])
        self.occ = dict(snap[1])
        del self.moves[snap[2]:]

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

    # -- rotate ------------------------------------------------------------ #
    # Push-and-Rotate's addition over Push-and-Swap. When ``a`` is blocked by an
    # agent ``b`` it must pass, the region is too packed to clear a degree->=3 hub
    # for a swap, but ``a`` and ``b`` sit on a *cycle*: a single empty cell brought
    # onto the cycle lets us rotate the whole ring by one, advancing ``a`` past
    # ``b`` without ever touching a placed agent. This is exactly the near-packed,
    # cyclic regime where the bare push/swap core gets stuck.
    def _cycle_through_edge(self, u, v, avoid):
        """The shortest simple cycle containing the edge ``u``-``v`` over the free
        graph minus ``avoid``, returned as a ring ``[u, v, ...]`` whose ``+1``
        direction carries ``u`` to ``v``. ``None`` if ``u``-``v`` is a bridge
        (a tree-like region — swap territory, not rotate)."""
        prev = {v: None}
        q = deque([v])
        while q:
            x = q.popleft()
            for w in self._adj[x]:
                if w in prev or w in avoid:
                    continue
                if x == v and w == u:        # forbid the direct edge: need a detour
                    continue
                prev[w] = x
                if w == u:                   # closed the loop u .. v .. u
                    seq = []
                    cur = w
                    while cur is not None:
                        seq.append(cur)
                        cur = prev[cur]
                    seq.reverse()            # [v, c1, ..., ck, u]
                    return [u, v] + seq[1:-1]
                q.append(w)
        return None

    def _bring_blank_onto(self, cycle):
        """Shift a nearby empty cell onto some vertex of ``cycle`` (other than the
        agent's own cell ``cycle[0]``), without crossing the cycle or disturbing a
        placed agent. Returns True if a cycle vertex is now empty."""
        cyset = set(cycle)
        for w in cycle[1:]:
            avoid = self.finished | (cyset - {w})
            path = self._path_to_empty(w, avoid, prefer_open=False)
            if path is not None and len(path) >= 2 and self._shift(path) \
                    and self._empty(w):
                return True
        return False

    def _rotate_advance(self, a):
        """Advance ``a`` one step toward its goal by rotating a cycle, when push
        and swap cannot. Returns True iff ``a`` moved."""
        u = self.pos[a]
        path = self._path(u, self.goal[a], self.finished)
        if path is None or len(path) < 2:
            return False
        v = path[1]
        cycle = self._cycle_through_edge(u, v, self.finished)
        if cycle is None:
            return False                     # bridge, not a cycle
        if not any(self._empty(c) for c in cycle) \
                and not self._bring_blank_onto(cycle):
            return False
        idx = {c: i for i, c in enumerate(cycle)}
        k = len(cycle)
        guard = 0
        # Roll the empty backward around the ring toward ``a``: each step shifts one
        # agent one position in the ``+1`` direction. When the empty reaches ``v``
        # (``b`` has vacated it), ``a`` itself steps in.
        while self.pos[a] != v:
            guard += 1
            if guard > 2 * k + 4:
                return False
            ahead = [i for c, i in idx.items() if i >= 1 and self._empty(c)]
            if not ahead:
                return False
            j = min(ahead)                   # nearest empty ahead of a (at index 0)
            if j == 1:                       # v is empty -> a steps into it
                if not self._move(a, v):
                    return False
            else:                            # cycle[j-1] is occupied (j is nearest)
                occ = self.occ.get(cycle[j - 1])
                if occ is None or not self._move(occ, cycle[j]):
                    return False
        return True

    # -- residual subproblem ---------------------------------------------- #
    # Greedy priority placement can paint itself into a corner: once most agents
    # are frozen, the last few sit in a small pocket walled off by placed agents,
    # where neither push/swap nor a single cycle-rotate can untangle them. This is
    # exactly the cyclic, near-packed residual Push-and-Rotate's subproblem
    # decomposition exists to handle. Rather than reproduce de Wilde's polynomial
    # decomposition + resolve bookkeeping, we solve that residual *pocket* exactly:
    # a breadth-first search over the configurations of the not-yet-placed agents
    # within their connected free component (placed agents are walls). BFS only
    # ever steps one agent into an adjacent empty cell, so the plan it returns is
    # collision-free and on-goal by construction; being exhaustive, it is complete
    # whenever the residual is solvable with the placed agents held fixed. It is
    # bounded (cells <= ``limit``, nodes <= ``node_cap``) so it stays a fast
    # endgame, not an exponential search over the whole instance.
    def _bfs_region(self, region, node_cap=200_000):
        """Exactly solve the not-yet-placed agents inside ``region`` (cells outside
        ``region`` are walls) by BFS over configurations. Only steps an agent into
        an adjacent empty cell, so the plan is valid by construction; exhaustive,
        so complete whenever the region's subproblem is solvable in isolation."""
        cells = sorted(region)
        cidx = {c: i for i, c in enumerate(cells)}
        adj = {c: [d for d in self._adj[c] if d in region] for c in region}
        aset = {self.occ[c] for c in cells
                if c in self.occ and self.occ[c] not in self.placed}
        if not aset:
            return True
        target = {}
        for a in aset:
            if self.goal[a] not in region:
                return False
            target[cidx[self.goal[a]]] = a
        start = tuple(self.occ.get(c) if self.occ.get(c) in aset else None
                      for c in cells)

        def is_goal(st):
            return all(st[i] == ag for i, ag in target.items())

        seen = {start: None}
        bq = deque([start])
        found = None
        while bq:
            st = bq.popleft()
            if is_goal(st):
                found = st
                break
            if len(seen) > node_cap:
                return False
            for i, occ_a in enumerate(st):
                if occ_a is None:
                    continue
                for d in adj[cells[i]]:
                    j = cidx[d]
                    if st[j] is None:
                        nxt = list(st)
                        nxt[j], nxt[i] = occ_a, None
                        nxt = tuple(nxt)
                        if nxt not in seen:
                            seen[nxt] = (st, occ_a, d)
                            bq.append(nxt)
        if found is None:
            return False
        chain = []
        cur = found
        while seen[cur] is not None:
            prev, ag, to = seen[cur]
            chain.append((ag, to))
            cur = prev
        for ag, to in reversed(chain):
            if not self._move(ag, to):
                return False
        for a in aset:
            self.placed.add(a)
            self.finished.add(self.goal[a])
        return True

    def _solve_residual(self, limit=9, node_cap=200_000):
        rem = [a for a in self.pos if a not in self.placed]
        if not rem:
            return True
        # connected free component of the residual, placed cells as walls
        comp = set()
        q = deque([self.pos[rem[0]]])
        comp.add(self.pos[rem[0]])
        while q:
            u = q.popleft()
            for v in self._adj[u]:
                if v in self.finished or v in comp:
                    continue
                comp.add(v)
                q.append(v)
        if len(comp) > limit:
            return False
        for a in rem:
            if self.pos[a] not in comp or self.goal[a] not in comp:
                return False               # an agent or its goal escapes the pocket
        return self._bfs_region(comp, node_cap=node_cap)

    # -- grid reduction (the dense, rectangular endgame) ------------------- #
    # On a fully packed rectangular region the greedy primitives stall almost
    # immediately, and the residual pocket is the whole rectangle -- far too big to
    # brute-force. This is the 15-puzzle regime, and Push-and-Rotate dispatches it
    # with a *constructive* row reduction, NOT a search (search is what CBS does,
    # and exactly what blows up here). We place the rectangle's tiles top row by
    # top row -- each interior tile by walking it to its cell with the blank as a
    # cursor, the two rightmost tiles of a row by the standard corner rotation that
    # avoids trapping the blank -- locking each finished row, until a two-row strip
    # remains that the exact ``_bfs_region`` finishes. Every step still moves one
    # agent into an adjacent empty cell, so validity by construction is preserved.
    # Scope: the rectangle's empty cells must lie in that bottom strip (its upper
    # rows are full at the goal) -- the canonical packed-formation target.
    def _route_blank_to(self, dest, avoid):
        """Bring an empty cell to ``dest`` without crossing ``avoid``."""
        if self._empty(dest):
            return True
        if dest in avoid:
            return False
        path = self._path_to_empty(dest, avoid, prefer_open=False)
        if path is None or len(path) < 2:
            return False
        return self._shift(path)

    def _move_tile(self, tile, target, locked):
        """Walk ``tile`` to ``target`` one cell at a time, never moving a ``locked``
        tile: route the blank to the next cell, then step the tile in."""
        guard = 0
        while self.pos[tile] != target:
            guard += 1
            if guard > 6 * len(self._adj):
                return False
            cur = self.pos[tile]
            path = self._path(cur, target, locked)
            if path is None or len(path) < 2:
                return False
            nb = path[1]
            if not self._route_blank_to(nb, locked | {cur}):
                return False
            if not self._empty(nb) or not self._move(tile, nb):
                return False
        return True

    def _place_pair(self, a_near, a_far, near, far, work, locked):
        """Place a trapped pair: ``a_near`` ends on ``near``, ``a_far`` on the
        corner ``far`` (reachable only through ``near`` or the open ``work`` cell).
        Bring ``a_near`` to the corner and ``a_far`` to ``work``, then rotate them
        into place — a maneuver that never strands the blank behind the corner."""
        if self.pos[a_near] == near and self.pos[a_far] == far:
            return True
        if not self._move_tile(a_near, far, locked):
            return False
        if not self._move_tile(a_far, work, locked | {far}):
            return False
        if not self._route_blank_to(near, locked | {far, work}):
            return False
        return self._move(a_near, near) and self._move(a_far, far)

    def _reduce_row(self, y, x0, x1, want, locked):
        """Place row ``y`` left to right; the two rightmost cells go in as a pair,
        using the row below (``y+1``) as the corner's workspace."""
        for x in range(x0, x1 - 1):
            tgt = (x, y)
            a = want.get(tgt)
            if a is None or not self._move_tile(a, tgt, locked):
                return False
            locked.add(tgt)
        c0, c1 = (x1 - 1, y), (x1, y)
        a0, a1 = want.get(c0), want.get(c1)
        if a0 is None or a1 is None \
                or not self._place_pair(a0, a1, c0, c1, (x1, y + 1), locked):
            return False
        locked.add(c0)
        locked.add(c1)
        return True

    def _reduce_col(self, x, ytop, ybot, want, locked):
        """Place column ``x`` of the final two-row strip as a pair (the two cells
        share the trap), using the cell to the right as the corner's workspace."""
        ct, cb = (x, ytop), (x, ybot)
        at, ab = want.get(ct), want.get(cb)
        if at is None or ab is None \
                or not self._place_pair(ab, at, cb, ct, (x + 1, ytop), locked):
            return False
        locked.add(ct)
        locked.add(cb)
        return True

    def _grid_reduction(self, comp):
        xs = [c[0] for c in comp]
        ys = [c[1] for c in comp]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        W, H = x1 - x0 + 1, y1 - y0 + 1
        if len(comp) != W * H or W < 3 or H < 2:
            return False                   # not a full rectangle the reduction fits
        if not any(self._empty(c) for c in comp):
            return False
        want = {self.goal[a]: a for a in self.pos if a not in self.placed}
        locked = set(self._adj) - set(comp)
        # peel full rows from the top until exactly two rows remain ...
        y = y0
        while (y1 - y) >= 2:
            if not self._reduce_row(y, x0, x1, want, locked):
                return False
            y += 1
        # ... then peel columns of the two-row strip from the left to a 2x3 corner.
        x = x0
        while (x1 - x) >= 3:
            if not self._reduce_col(x, y, y1, want, locked):
                return False
            x += 1
        strip = {(xx, yy) for xx in range(x, x1 + 1) for yy in range(y, y1 + 1)}
        return self._bfs_region(strip)

    def solve_reduction(self):
        """Solve a fully packed rectangle by constructive row reduction. Returns the
        move list, or ``None`` if the region is not a rectangle the reduction
        handles (the caller then falls back to the primitive order-sweep)."""
        if not self._grid_reduction(sorted(self._adj)):
            return None
        if any(self.pos[a] != self.goal[a] for a in self.pos):
            return None
        return self.moves

    # -- single-blank endgame (the 15-puzzle proper) ----------------------- #
    # With exactly one empty cell a packed rectangle is the (W*H - 1)-puzzle, and
    # the row/column reduction above can paint the lone blank into a corner where
    # every neighbour is a frozen tile (with two empties the spare slack escapes;
    # with one it does not). The fix is to stop steering the blank by hand: place
    # each tile -- or each last-two pair -- with an exact BFS over the *whole*
    # unsolved region that tracks ONLY the one or two agents being placed. Every
    # other tile is an interchangeable filler, so the state is just
    # ``(blank cell, tracked-agent cells)`` -- tiny and independent of region size
    # -- and the search, being exhaustive, can never strand the blank: if a
    # legal move sequence places the target, BFS finds it. The reduction order
    # keeps each subproblem solvable; the BFS keeps each step constructive (it
    # only ever steps one agent into an adjacent empty cell, so validity by
    # construction is preserved).
    def _bfs_place(self, region, goals, node_cap=2_000_000):
        """Move the agents named in ``goals`` (cell -> agent) onto their cells by an
        exact BFS over ``region`` (cells outside ``region`` are walls). Untracked
        tiles are anonymous fillers, so the state is ``(blank, tracked cells)``;
        blank transitions are recorded and replayed on the real occupancy, so
        filler anonymity is free. Requires exactly one empty cell in ``region``."""
        radj = {c: [d for d in self._adj[c] if d in region] for c in region}
        agents = sorted(goals.values())
        want = {a: gc for gc, a in goals.items()}
        blanks = [c for c in region if c not in self.occ]
        if len(blanks) != 1:
            return False
        pos0 = {}
        for c in region:
            a = self.occ.get(c)
            if a in want:
                pos0[a] = c
        if len(pos0) != len(agents):
            return False                       # a tracked agent is not in region
        start = (blanks[0], tuple(pos0[a] for a in agents))

        def is_goal(st):
            return all(st[1][i] == want[a] for i, a in enumerate(agents))

        seen = {start: None}
        q = deque([start])
        found = None
        while q:
            st = q.popleft()
            if is_goal(st):
                found = st
                break
            if len(seen) > node_cap:
                return False
            blank, ps = st
            for v in radj[blank]:
                nps = list(ps)
                for i in range(len(agents)):
                    if ps[i] == v:             # a tracked tile slides into blank
                        nps[i] = blank
                        break
                nst = (v, tuple(nps))
                if nst not in seen:
                    seen[nst] = (st, blank, v)  # blank moved blank->v
                    q.append(nst)
        if found is None:
            return False
        chain = []
        cur = found
        while seen[cur] is not None:
            prev, bfrom, bto = seen[cur]
            chain.append((bfrom, bto))
            cur = prev
        for bfrom, bto in reversed(chain):     # replay: tile at bto slides to bfrom
            tile = self.occ.get(bto)
            if tile is None or not self._move(tile, bfrom):
                return False
        return True

    def solve_unit(self):
        """Solve a fully packed rectangle that has exactly one empty cell by the
        tracked-agent reduction. Returns the move list, or ``None`` if the region
        is not such a rectangle (caller falls back to the order-sweep)."""
        cells = sorted(self._adj)
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        W, H = x1 - x0 + 1, y1 - y0 + 1
        if len(cells) != W * H or W < 2 or H < 2:
            return None
        if sum(1 for c in cells if self._empty(c)) != 1:
            return None
        want = {self.goal[a]: a for a in self.pos}
        allcells = set(cells)
        frozen = set()

        def region():
            return allcells - frozen

        # rows top-down, leaving the bottom two rows ...
        y = y0
        while (y1 - y) >= 2:
            for x in range(x0, x1 - 1):                 # cols 0..W-3 one at a time
                t = (x, y)
                if not self._bfs_place(region(), {t: want[t]}):
                    return None
                frozen.add(t)
            cL, cR = (x1 - 1, y), (x1, y)               # the last two as a pair
            if not self._bfs_place(region(), {cL: want[cL], cR: want[cR]}):
                return None
            frozen.add(cL)
            frozen.add(cR)
            y += 1
        # ... then the two-row strip: columns left-to-right, leaving a 2x2 corner.
        ybot = y1
        x = x0
        while (x1 - x) >= 2:
            cT, cB = (x, y), (x, ybot)
            if not self._bfs_place(region(), {cT: want[cT], cB: want[cB]}):
                return None
            frozen.add(cT)
            frozen.add(cB)
            x += 1
        quad = {(xx, yy) for xx in range(x, x1 + 1) for yy in range(y, y1 + 1)}
        goals = {c: want[c] for c in quad if c in want}
        if not self._bfs_place(quad, goals):
            return None
        if any(self.pos[a] != self.goal[a] for a in self.pos):
            return None
        return self.moves

    # -- top level --------------------------------------------------------- #
    def solve(self, order, *, allow_rotate=True, allow_residual=True):
        # The two fallbacks past push/swap — ``rotate`` and the residual-pocket
        # BFS — are Push-and-Rotate's completions, *not* part of the bare
        # Push-and-Swap core. ``allow_rotate``/``allow_residual`` default True so
        # ``push_and_rotate`` is byte-identical; ``push_and_swap`` turns them off
        # to expose the Luna-&-Bekris algorithm's exact completeness gap.
        for a in order:
            guard = 0
            while self.pos[a] != self.goal[a]:
                guard += 1
                if guard > 8 * len(self._adj) + 10:
                    return None
                if len(self.moves) > self.max_moves:
                    return None
                if self._push(a):
                    continue
                path = self._path(self.pos[a], self.goal[a], self.finished)
                if path is None or len(path) < 2:
                    if allow_residual and self._solve_residual():
                        break
                    return None
                b = self.occ.get(path[1])
                if b is None or b in (a,):
                    if allow_residual and self._solve_residual():
                        break
                    return None
                # Push and Swap's primitive; when it cannot find/clear a hub (the
                # packed, cyclic regime) fall back to Push and Rotate's rotate, and
                # to an exact solve of the small residual pocket as a last resort.
                snap = self._snapshot()
                if self._swap(a, b):
                    continue
                self._restore(snap)
                if allow_rotate and self._rotate_advance(a):
                    continue
                if allow_residual and self._solve_residual():
                    break
                return None
            self.finished.add(self.goal[a])
            self.placed.add(a)
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

    def _finish(moves):
        paths = _moves_to_paths(agents, moves)
        if stats is not None:
            stats["moves"] = len(moves)
        return Solution(paths=paths, cost=sum_of_costs(paths))

    # The greedy primitives stall on a fully packed rectangle (the 15-puzzle
    # regime); dispatch the constructive endgames first, falling back to the
    # primitive order-sweep for everything they do not cover. A single empty cell
    # needs the tracked-agent BFS (``solve_unit``) because the row reduction can
    # strand the lone blank; two or more empties use the row reduction directly.
    unit = _Solver(grid, agents, max_moves).solve_unit()
    if unit is not None:
        return _finish(unit)
    reduced = _Solver(grid, agents, max_moves).solve_reduction()
    if reduced is not None:
        return _finish(reduced)
    for order in orders:
        solver = _Solver(grid, agents, max_moves)
        moves = solver.solve(order)
        if moves is None:
            continue
        return _finish(moves)
    return None
