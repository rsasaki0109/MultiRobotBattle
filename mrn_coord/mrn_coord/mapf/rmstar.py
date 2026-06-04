"""rM*: recursive M* (Wagner & Choset, 2011/2015).

Basic :func:`mrn_coord.mapf.mstar.mstar` keeps a single *flat* collision set per
joint configuration: the moment any two agents collide it adds **both** to that
set, and — because backpropagation floods the collision set backward along every
predecessor — independent collisions that share a common ancestor configuration
(the start always is one) get **unioned**. A configuration carrying the union
then branches over the full local dimension of *every* agent in it, so the search
dimension is the size of the *union of all* interactions, not the size of any one
of them.

rM* removes exactly that over-coupling. Instead of a flat set it maintains a
**partition** of the agents into independent collision groups, and it couples
only the agents that *actually* collide with each other (pairwise), never the
incidental others that merely collided at the same timestep elsewhere. Two groups
merge only when an agent of one genuinely collides with an agent of the other.
Maintaining a partition this way is equivalent to recursively decomposing the
collision set into independent sub-problems — the "recursive" in recursive M* —
so the peak coupling dimension is the size of the largest *irreducible*
interacting group, not the union of all collisions across the instance.

Within a coupled group rM* does not branch every member over all its moves the
way basic M* does for collision-set agents; it branches the group over its joint
**optimal policy** — the set of jointly-optimal, collision-free sub-moves of just
that group, read off an exact sum-of-costs cost-to-go computed over the group's
own joint space (with the same ``(config, settled)`` done-bit basic M* uses, so
vacate-and-return is priced correctly). An uncoupled agent follows its individual
optimal policy exactly as in basic M*. A group larger than
:data:`_RMStar.MAX_EXACT_GROUP` falls back to basic M*'s branch-all behaviour
(still correct, just unrefined) so the per-group cost-to-go never blows up.

The result is the **same** optimal sum-of-costs basic M* and CBS return, but on
instances whose collisions decompose — several pairwise interactions in disjoint
regions — the partition keeps each group at its true (small) size while basic M*
inflates to their union. There the peak group stays constant as independent
collisions are added and the joint expansion count grows polynomially where basic
M* (and a fully coupled joint A*) grow exponentially. This module reuses
:func:`mrn_coord.mapf.mstar._dist_to_goal` for the per-agent cost-to-go; basic
:mod:`mrn_coord.mapf.mstar` is left byte-for-byte unchanged.
"""

from __future__ import annotations

import heapq
import itertools

from .grid import Cell, GridWorld
from .mstar import _dist_to_goal
from .solution import Solution


def _legal(u: tuple, v: tuple) -> bool:
    """No two agents share a cell in ``v`` and none swap across ``u``->``v``."""
    n = len(u)
    if len(set(v)) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if u[i] == v[j] and u[j] == v[i] and u[i] != u[j]:
                return False
    return True


def _collision_pairs(u: tuple, v: tuple) -> set:
    """Pairs of agents that *actually* collide on ``u``->``v``.

    Unlike basic M*'s flat collision *set*, rM* couples only genuinely-colliding
    agents, so this returns the unordered pairs that share a cell in ``v`` or
    swap across the edge. Keeping collisions pairwise is what lets independent
    simultaneous interactions stay in separate groups instead of being unioned
    through spurious all-pairs combinations of the colliding set.
    """
    n = len(u)
    pairs: set = set()
    seen: dict[Cell, int] = {}
    for i in range(n):
        j = seen.get(v[i])
        if j is not None:
            pairs.add(frozenset((i, j)))
        else:
            seen[v[i]] = i
    for i in range(n):
        for j in range(i + 1, n):
            if u[i] == v[j] and u[j] == v[i] and u[i] != u[j]:
                pairs.add(frozenset((i, j)))
    return pairs


class _RMStar:
    """One rM* run: per-agent policies, group cost-to-go caches, joint search."""

    #: Groups up to this size get the exact optimal-policy treatment; larger
    #: groups fall back to basic M*'s branch-all (correct but unrefined), which
    #: keeps the per-group cost-to-go computation bounded.
    MAX_EXACT_GROUP = 3
    #: Cap on the states a group's cost-to-go map may visit. A group whose joint
    #: space exceeds this (a large group on an open grid) abandons the exact
    #: policy and branches all moves instead -- still correct, and cheaper than
    #: materialising a huge cost-to-go map for a policy the search barely queries.
    GROUP_CTG_CAP = 60_000

    def __init__(self, grid: GridWorld, agents: dict, max_expansions: int):
        self.grid = grid
        self.ids = list(agents)
        self.n = len(self.ids)
        self.start = tuple(agents[a][0] for a in self.ids)
        self.goal = tuple(agents[a][1] for a in self.ids)
        self.dist = [_dist_to_goal(grid, g) for g in self.goal]
        self.max_expansions = max_expansions
        # frozenset(local idx) -> {(subcfg, sub_settled): optimal cost-to-go}
        self._ctg: dict = {}
        # (frozenset idx, subcfg, sub_settled) -> [(subcfg, sub_settled, cost)]
        self._pol: dict = {}

    def _h(self, cfg: tuple) -> int:
        return sum(self.dist[i][cfg[i]] for i in range(self.n))

    # -- individual policy (uncoupled agent) ------------------------------
    def _steps1(self, i: int, cell: Cell) -> tuple:
        gi = self.goal[i]
        if cell == gi:
            return (gi,)
        here = self.dist[i][cell]
        return tuple(nb for nb in self.grid.neighbors(cell)
                     if self.dist[i].get(nb, here + 1) == here - 1)

    # -- exact settle-aware group cost-to-go ------------------------------
    def _group_ctg(self, group: frozenset):
        """Optimal sum-of-costs cost-to-go over a group's joint ``(config,
        settled)`` space, by backward Dijkstra from the all-settled goal.

        Returns ``None`` if the map would exceed :data:`GROUP_CTG_CAP` states, so
        the caller falls back to branch-all for that (over-large) group.
        """
        if group in self._ctg:
            return self._ctg[group]
        idx = sorted(group)
        k = len(idx)
        gsub = tuple(self.goal[i] for i in idx)
        full = (True,) * k
        H = {(gsub, full): 0}
        pq = [(0, gsub, full)]
        while pq:
            if len(H) > self.GROUP_CTG_CAP:
                self._ctg[group] = None
                return None
            d, vc, vs = heapq.heappop(pq)
            if d > H[(vc, vs)]:
                continue
            per = []
            ok = True
            for p in range(k):
                opts = []
                if vs[p]:
                    # settled at goal in v: stayed settled, or just settled now
                    if vc[p] == gsub[p]:
                        opts.append((gsub[p], True, 0))
                        opts.append((gsub[p], False, 0))
                else:
                    # unsettled in v: reached by a unit move/wait (cost 1)
                    for u in self.grid.neighbors(vc[p]):
                        opts.append((u, False, 1))
                if not opts:
                    ok = False
                    break
                per.append(opts)
            if not ok:
                continue
            for combo in itertools.product(*per):
                uc = tuple(c[0] for c in combo)
                if not _legal(uc, vc):
                    continue
                us = tuple(c[1] for c in combo)
                nd = d + sum(c[2] for c in combo)
                key = (uc, us)
                if nd < H.get(key, float("inf")):
                    H[key] = nd
                    heapq.heappush(pq, (nd, uc, us))
        self._ctg[group] = H
        return H

    def _group_succ(self, idx, gsub, subcfg, subsettled):
        """All legal settle-aware joint successors (coupled: branch all moves)."""
        per = []
        for p in range(len(idx)):
            gi = gsub[p]
            if subsettled[p]:
                per.append([(gi, True, 0)])
                continue
            on_goal = subcfg[p] == gi
            opts = []
            for m in self.grid.neighbors(subcfg[p]):
                if on_goal and m == gi:
                    opts.append((gi, True, 0))     # settle for good
                    opts.append((gi, False, 1))    # wait on goal, may yet vacate
                else:
                    opts.append((m, False, 1))
            per.append(opts)
        for combo in itertools.product(*per):
            v = tuple(c[0] for c in combo)
            if _legal(subcfg, v):
                yield v, tuple(c[1] for c in combo), sum(c[2] for c in combo)

    def _group_policy(self, group, subcfg, subsettled):
        """Optimal-policy successors of a coupled group at ``(subcfg, subsettled)``.

        Above :data:`MAX_EXACT_GROUP` this degrades to branch-all (every legal
        joint move); at or below it, only successors that lie on an optimal joint
        path (``step + cost-to-go == cost-to-go``) survive — the refinement that
        makes rM* generate far fewer joint configurations than basic M*.
        """
        key = (group, subcfg, subsettled)
        cached = self._pol.get(key)
        if cached is not None:
            return cached
        idx = sorted(group)
        gsub = tuple(self.goal[i] for i in idx)
        all_succ = list(self._group_succ(idx, gsub, subcfg, subsettled))
        H = None if len(group) > self.MAX_EXACT_GROUP else self._group_ctg(group)
        if H is None:           # over MAX_EXACT_GROUP, or cost-to-go too large
            self._pol[key] = all_succ
            return all_succ
        base = H.get((subcfg, subsettled))
        if base is None:
            self._pol[key] = []
            return []
        out = [(v, vs, cost) for (v, vs, cost) in all_succ
               if H.get((v, vs)) is not None and cost + H[(v, vs)] == base]
        self._pol[key] = out
        return out

    # -- top-level neighbour generation -----------------------------------
    def _neighbors(self, cfg: tuple, settled: frozenset, part: frozenset):
        """Generate ``(next_config, next_settled, cost)`` successors.

        Each partition unit contributes independently: a singleton follows its
        individual optimal policy (settling for free on its goal); a coupled
        group contributes its joint optimal policy. The Cartesian product over
        units stays small precisely because independent groups never merge.
        """
        units = sorted(part, key=lambda g: min(g))
        opts = []
        for g in units:
            idx = sorted(g)
            uopts = []
            if len(g) == 1:
                i = idx[0]
                gi = self.goal[i]
                if i in settled:
                    uopts.append((idx, (gi,), (True,), 0))
                else:
                    on_goal = cfg[i] == gi
                    for m in self._steps1(i, cfg[i]):
                        if on_goal and m == gi:
                            uopts.append((idx, (gi,), (True,), 0))
                        else:
                            uopts.append((idx, (m,), (False,), 1))
            else:
                sub = tuple(cfg[i] for i in idx)
                subset = tuple(i in settled for i in idx)
                for (v, vs, cost) in self._group_policy(g, sub, subset):
                    uopts.append((idx, v, vs, cost))
            opts.append(uopts)
        for combo in itertools.product(*opts):
            v = list(cfg)
            nset = set(settled)
            cost = 0
            for (idxs, assign, sett, c) in combo:
                cost += c
                for p, i in enumerate(idxs):
                    v[i] = assign[p]
                    if sett[p]:
                        nset.add(i)
                    else:
                        nset.discard(i)
            yield tuple(v), frozenset(nset), cost

    # -- search -----------------------------------------------------------
    def solve(self, stats: dict | None = None):
        for i in range(self.n):
            if self.dist[i].get(self.start[i]) is None:
                return None

        start = (self.start, frozenset())
        goal_cfg = self.goal
        INF = float("inf")
        g: dict = {start: 0}
        parent: dict = {}
        singleton = frozenset(frozenset((i,)) for i in range(self.n))
        part: dict = {start: singleton}
        back: dict = {start: set()}
        in_open: set = set()
        counter = itertools.count()
        heap: list = []

        def push(node):
            if node in in_open:
                return
            in_open.add(node)
            heapq.heappush(heap, (g[node] + self._h(node[0]), next(counter), node))

        def merged(p, pairs):
            changed = False
            for pair in pairs:
                i, j = tuple(pair)
                gi = next(s for s in p if i in s)
                gj = next(s for s in p if j in s)
                if gi is not gj:
                    p = frozenset((p - {gi, gj}) | {gi | gj})
                    changed = True
            return p, changed

        def backprop(vk, pairs):
            stack = [(vk, pairs)]
            while stack:
                v, ps = stack.pop()
                p = part.get(v)
                if p is None:
                    continue
                np, changed = merged(p, ps)
                if changed:
                    part[v] = np
                    if g.get(v, INF) < INF:
                        push(v)
                    for vm in back.get(v, ()):
                        stack.append((vm, ps))

        push(start)
        expansions = 0
        peak = 1
        while heap:
            _, _, vk = heapq.heappop(heap)
            in_open.discard(vk)
            expansions += 1
            peak = max(peak, max(len(s) for s in part[vk]))
            if expansions > self.max_expansions:
                if stats is not None:
                    stats["expansions"] = expansions
                    stats["max_group"] = peak
                return None
            if vk[0] == goal_cfg:
                if stats is not None:
                    stats["expansions"] = expansions
                    stats["max_group"] = peak
                return self._reconstruct(parent, vk, g[vk])

            cfg, settled = vk
            for v, vsettled, ec in self._neighbors(cfg, settled, part[vk]):
                vl = (v, vsettled)
                back.setdefault(vl, set()).add(vk)
                pairs = _collision_pairs(cfg, v)
                if pairs:
                    backprop(vk, pairs)
                    continue  # illegal step — never relax through a collision
                ng = g[vk] + ec
                if vl not in part:
                    part[vl] = singleton
                if ng < g.get(vl, INF):
                    g[vl] = ng
                    parent[vl] = vk
                    push(vl)

        if stats is not None:
            stats["expansions"] = expansions
            stats["max_group"] = peak
        return None

    def _reconstruct(self, parent: dict, goal: tuple, cost: int) -> Solution:
        nodes = [goal]
        cur = goal
        while cur in parent:
            cur = parent[cur]
            nodes.append(cur)
        nodes.reverse()
        paths: dict = {}
        for idx, a in enumerate(self.ids):
            seq = [node[0][idx] for node in nodes]
            gc = self.goal[idx]
            while len(seq) > 1 and seq[-1] == gc and seq[-2] == gc:
                seq.pop()
            paths[a] = seq
        return Solution(paths=paths, cost=cost)


def rmstar(grid: GridWorld, agents: dict, *, max_expansions: int = 200_000,
           stats: dict | None = None):
    """Solve a MAPF instance optimally (sum-of-costs) via recursive M*.

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns a
    :class:`mrn_coord.mapf.solution.Solution` whose paths are collision-free and
    minimal in sum-of-costs — the **same** optimum
    :func:`mrn_coord.mapf.mstar.mstar` and :func:`mrn_coord.mapf.cbs.cbs` find —
    or ``None`` if the instance is infeasible or the expansion budget is
    exhausted.

    rM* keeps a *partition* of the agents into independent collision groups and
    couples only agents that genuinely collide, so its peak coupling dimension is
    the largest irreducible interacting group rather than basic M*'s union of all
    collisions. If ``stats`` is given, ``stats["expansions"]`` is the number of
    joint configurations popped and ``stats["max_group"]`` the largest collision
    group the partition ever formed; compare against
    :func:`mrn_coord.mapf.mstar.mstar`'s ``max_collision_set`` and ``expansions``
    on collisions that decompose to see the recursive decomposition pay off.
    """
    return _RMStar(grid, agents, max_expansions).solve(stats)
