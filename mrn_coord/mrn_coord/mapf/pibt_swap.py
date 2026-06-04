"""Swap-enhanced PIBT — the successor generator of LaCAM2 (Okumura, IJCAI 2023,
"Improving LaCAM for Scalable Eventually Optimal Multi-Agent Pathfinding").

PIBT (:func:`pibt_solve <mrn_coord.lifelong.pibt_solve>`) builds one collision-free
step at a time by priority inheritance: a high-priority agent claims its best
neighbour and *pushes* whoever sits there to move aside, recursively. It is fast
and complete on biconnected graphs, but on a narrow corridor where two agents must
**exchange** ends it livelocks — the pushed agent has nowhere to go but back, and
the pair oscillates forever. The repo's :func:`pibt_solve` papers over this with a
deterministic *escape salt* that perturbs equal-distance ties until the symmetry
breaks. LaCAM2 fixes it at the source with the canonical **swap** operation, and
that is what this module reproduces.

The idea: before agent ``i`` commits to its best neighbour ``v``, check whether a
swap with the agent ``j`` sitting on ``v`` is both *required* (``j`` cannot get out
of ``i``'s way — it would be pushed into a dead end or back onto ``i``'s goal) and
*possible* (somewhere along the corridor there is a vertex of degree ≥ 2, a pocket,
where one of them can step aside). When both hold, ``i`` **reverses** its candidate
order — it moves *away* from its goal, vacating the corridor — and *pulls* ``j``
into the cell it just left. The exchange happens over the pocket instead of
deadlocking. Both predicates are short corridor *emulations* walking the puller
away from the pusher and counting escape routes by vertex degree; the paper notes
the detector is implementation-defined (it "do[es] not aim at designing complete
detectors"), so this is a faithful port of Okumura's reference C++
(``Kei18/lacam2``: ``funcPIBT`` / ``swap_possible_and_required`` /
``is_swap_required`` / ``is_swap_possible``), with the random tie-break replaced by
a deterministic coordinate tie-break so runs are reproducible.

Collision-free by construction (an agent only ever moves into a cell free at its
turn); incomplete in general (PIBT always is), but the swap makes it finish the
corridor exchanges that base PIBT cannot. ``swap=False`` recovers plain PIBT.
"""

from __future__ import annotations

from collections import deque
from math import floor

from .grid import Cell, GridWorld

INF = float("inf")


def _bfs_dist(grid: GridWorld, goal: Cell) -> dict:
    """4-connected BFS distance from every free cell to ``goal``."""
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


def _pibt_swap_step(grid, config, order, dist, goals, *, swap: bool):
    """One PIBT step with the optional swap operation.

    ``config`` maps agent -> current cell; ``order`` is the priority order to
    decide agents in (highest first); ``dist`` maps agent -> {cell: dist-to-goal};
    ``goals`` maps agent -> goal cell. Returns ``{agent: next_cell}`` (always a
    full, collision-free assignment — a stuck agent stays put).

    Faithful port of ``funcPIBT`` with swap from Okumura's reference: candidates
    are the neighbours *plus* the current cell, sorted by distance-to-goal with a
    deterministic coordinate tie-break; if a swap is required-and-possible the
    candidate order is reversed and the partner pulled in at ``k == 0``.
    """
    # Adjacency *excluding* the cell itself (grid.neighbors includes it), and the
    # vertex degree used by the swap emulations.
    def adj(v):
        return [u for u in grid.neighbors(v) if u != v]

    occupied_now = {config[a]: a for a in config}
    next_pos: dict = {}
    next_occ: dict = {}

    def candidates(a):
        here = config[a]
        return sorted(grid.neighbors(here),
                      key=lambda u: (dist[a].get(u, INF), u))

    def is_swap_required(pusher, puller, v_pusher, v_puller):
        # Emulate pushing `puller` away from `pusher` down the corridor; a swap is
        # required only if the puller cannot escape (degree stays <= 1) and the
        # distances say the exchange genuinely helps.
        dp = dist[pusher]
        while dp.get(v_puller, INF) < dp.get(v_pusher, INF):
            neigh = adj(v_puller)
            n = len(neigh)
            tmp = None
            for u in neigh:
                a = occupied_now.get(u)
                if u == v_pusher or (len(adj(u)) == 1 and a is not None
                                     and goals[a] == u):
                    n -= 1                       # not an escape route
                else:
                    tmp = u
            if n >= 2:
                return False                     # puller can step aside: no swap
            if n <= 0:
                break                            # dead end: swap required
            v_pusher, v_puller = v_puller, tmp
        dl = dist[puller]
        return (dl.get(v_pusher, INF) < dl.get(v_puller, INF)
                and (dp.get(v_pusher, INF) == 0
                     or dp.get(v_puller, INF) < dp.get(v_pusher, INF)))

    def is_swap_possible(v_pusher_origin, v_puller_origin):
        # Reverse emulation: can the puller be pulled toward the pusher's origin,
        # i.e. is there a pocket (degree >= 2) to swap through?
        v_pusher, v_puller = v_pusher_origin, v_puller_origin
        while v_puller != v_pusher_origin:
            neigh = adj(v_puller)
            n = len(neigh)
            tmp = None
            for u in neigh:
                a = occupied_now.get(u)
                if u == v_pusher or (len(adj(u)) == 1 and a is not None
                                     and goals[a] == u):
                    n -= 1
                else:
                    tmp = u
            if n >= 2:
                return True
            if n <= 0:
                return False
            v_pusher, v_puller = v_puller, tmp
        return False

    def swap_partner(ai, cand):
        # `cand` = ai's sorted candidate list. Returns the agent ai should swap
        # with, or None. Mirrors swap_possible_and_required (cases a/b and c).
        here = config[ai]
        if cand[0] == here:
            return None                          # ai prefers to stay: no swap
        aj = occupied_now.get(cand[0])
        if (aj is not None and aj not in next_pos
                and is_swap_required(ai, aj, here, config[aj])
                and is_swap_possible(config[aj], here)):
            return aj
        for u in adj(here):                      # "clear" operation (case c)
            ak = occupied_now.get(u)
            if ak is None or cand[0] == config[ak]:
                continue
            if (is_swap_required(ak, ai, here, cand[0])
                    and is_swap_possible(cand[0], here)):
                return ak
        return None

    def decide(ai):
        here = config[ai]
        cand = candidates(ai)
        partner = swap_partner(ai, cand) if swap else None
        if partner is not None:
            cand = list(reversed(cand))          # move ai away, vacate the corridor
        for k, u in enumerate(cand):
            if u in next_occ:
                continue                         # vertex conflict
            ak = occupied_now.get(u)
            if ak is not None and next_pos.get(ak) == here:
                continue                         # swap conflict (ak moving into ai)
            next_pos[ai] = u
            next_occ[u] = ai
            if (ak is not None and ak != ai and ak not in next_pos
                    and not decide(ak)):
                # ak failed and reclaimed its own cell (== u) inside its decide;
                # next_occ[u] is now ak, next_pos[ai] is stale — overwritten below.
                continue
            if (k == 0 and partner is not None and partner not in next_pos
                    and here not in next_occ):
                next_pos[partner] = here         # pull the swap partner in
                next_occ[here] = partner
            return True
        next_pos[ai] = here                      # stuck: stay put
        next_occ[here] = ai
        return False

    for a in order:
        if a not in next_pos:
            decide(a)
    return next_pos


def pibt_swap(grid: GridWorld, agents: dict, *, swap: bool = True,
              max_timestep: int = 1000):
    """Run swap-enhanced PIBT until every agent reaches its goal or time runs out.

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns
    ``{agent: [cell, ...]}`` (one path per agent, all the same length and
    collision-free) once every agent is on its goal, or ``None`` if the
    ``max_timestep`` budget is exhausted (PIBT is incomplete). With ``swap=False``
    this is plain PIBT — useful to expose the corridor livelocks the swap fixes.

    Priorities follow Okumura's scheme: start at the goal distance scaled by the
    map size and, each step, increment for any agent not yet on its goal (reset the
    fractional part on arrival) so a stuck agent eventually wins right of way.
    """
    ids = sorted(agents)
    starts = {a: agents[a][0] for a in ids}
    goals = {a: agents[a][1] for a in ids}
    for a in ids:
        if not grid.is_free(starts[a]) or not grid.is_free(goals[a]):
            return None
    dist = {a: _bfs_dist(grid, goals[a]) for a in ids}
    for a in ids:
        if starts[a] not in dist[a]:
            return None                          # goal unreachable

    size = grid.width * grid.height
    config = {a: starts[a] for a in ids}
    paths = {a: [starts[a]] for a in ids}
    prio = {a: dist[a].get(starts[a], INF) / size for a in ids}

    for _ in range(max_timestep):
        order = sorted(ids, key=lambda a: (-prio[a], a))
        config = _pibt_swap_step(grid, config, order, dist, goals, swap=swap)
        for a in ids:
            paths[a].append(config[a])
        done = all(config[a] == goals[a] for a in ids)
        for a in ids:
            if config[a] != goals[a]:
                prio[a] += 1
            else:
                prio[a] -= floor(prio[a])
        if done:
            return paths
    return None
