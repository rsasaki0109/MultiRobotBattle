"""Offline TSWAP: fast complete anonymous MAPF by target swapping.

A reproduction of **Offline TSWAP**, Keisuke Okumura & Xavier Defago,
*"Solving Simultaneous Target Assignment and Path Planning Efficiently with
Time-Independent Execution"* (ICAPS 2022; extended in AIJ 2023). Like
:mod:`flow`, it solves the **anonymous** (unlabeled / target-interchangeable)
problem — any agent may fill any goal — but from the opposite end of the
speed/optimality trade-off:

- :mod:`flow` (Yu & LaValle 2013) is makespan-**optimal** but heavy: it builds a
  time-expanded network and binary-searches the horizon with max-flow.
- **TSWAP** is **sub-optimal but complete and cheap**: it takes an *arbitrary*
  initial target assignment and repeatedly does one-timestep path planning with
  **target swapping** until every agent sits on a target. No search tree, no
  flow network — collision-free *by construction*, complete *by a potential
  argument*. This is the anonymous-MAPF analogue of :mod:`push_and_rotate`'s
  constructive stance for the labeled problem.

**The algorithm (Okumura & Defago, Algorithm 1).** Each agent ``a`` holds a
current location ``a.v`` and a current target ``a.g`` (from the initial
assignment). Until all agents are on their targets, one timestep is planned by
processing the agents in a fixed order and, *updating positions and targets in
place*, applying::

    u = nextNode(a.v, a.g)                 # neighbour (incl. waiting) closest to a.g
    if u is free:           a.v <- u       # move into the empty cell
    elif u == b.g (b sits on u, its own target):
                            swap a.g, b.g  # TARGET SWAP  (line 10)
    elif a lies on a deadlock cycle A':
                            rotate targets of A'   # ROTATION  (line 12)
    else:                   a waits

where ``nextNode(v, g) = argmin over Neigh(v) u {v} of dist(., g)`` with a
deterministic tie-break, and a **deadlock** is a set ``A' = (a_1, .., a_j)`` with
``nextNode(a_1.v, a_1.g) = a_2.v, .., nextNode(a_j.v, a_j.g) = a_1.v`` (the
"wants" pointers form a cycle); rotating shifts each member's target one step
around the cycle (``a_1.g <- a_j.g, a_2.g <- a_1.g, ..``).

**Why it is collision-free by construction.** An agent moves *only* into a cell
that is empty at the instant it is processed, and it vacates its own cell as it
goes. A vertex conflict would need two agents to enter one cell — the second
sees it occupied and does not move. A swap (head-on) conflict ``x->y`` /
``y->x`` would need ``y`` empty when the ``x->y`` mover is processed while the
other agent still sits on ``y`` — impossible. So the per-timestep snapshots are
always a bijection of distinct cells with only legal moves between them.

**Why it is complete (potential).** With ``Pi(u, u')`` the interior of a
shortest ``u->u'`` path, the potential ::

    phi = sum over a { dist(a.v, a.g) + #{ b : b.g in Pi(a.v, a.g) } }

is non-increasing and **strictly decreases** every timestep while ``phi > 0``:
if no agent moved (first term unchanged) and no swap fired (second term
unchanged) then the blocked "wants" pointers must close a cycle, which the
rotation resolves — decreasing the potential. Hence TSWAP terminates on any
solvable instance regardless of the initial assignment, with makespan bounded by
``O(|A| . diam(G))``.

**Honest scope** (see ``docs/coordination.md``): TSWAP is *not* makespan-optimal
— its makespan lower-bounds to but never beats :mod:`flow`'s. Paired with a good
initial assignment it is empirically near-optimal at a fraction of the cost; the
``assignment`` argument lets a caller plug in any matching (the default is a
deterministic greedy nearest-target one). The value here is the constructive
*completeness* and the target-swap mechanism, not optimality.
"""

from __future__ import annotations

from collections import deque

from .grid import Cell, GridWorld


# --------------------------------------------------------------------------- #
# Shortest-path distance fields (one BFS per distinct target cell, cached)     #
# --------------------------------------------------------------------------- #
def _dist_field(grid: GridWorld, target: Cell) -> dict:
    """BFS distance from every free cell to ``target`` (4-connected)."""
    dist = {target: 0}
    q = deque([target])
    while q:
        v = q.popleft()
        for u in grid.neighbors(v):
            if u not in dist:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


def _next_node(grid: GridWorld, fields: dict, v: Cell, g: Cell) -> Cell:
    """``argmin`` over ``neighbors(v) u {v}`` of ``dist(., g)`` — the cell that
    steps ``v`` closest to ``g``. Deterministic tie-break by ``neighbors`` order
    (waiting in ``v`` is the first candidate, so ties prefer not moving)."""
    field = fields[g]
    best = v
    best_d = field.get(v)
    for u in grid.neighbors(v):
        d = field.get(u)
        if d is not None and (best_d is None or d < best_d):
            best, best_d = u, d
    return best


# --------------------------------------------------------------------------- #
# Initial assignment (any matching works; default is greedy nearest)          #
# --------------------------------------------------------------------------- #
def _greedy_assignment(grid: GridWorld, starts, goals, fields) -> list:
    """A deterministic greedy assignment: pair each start (in order) with its
    nearest still-free goal by shortest-path distance. Returns ``assign`` where
    ``assign[i]`` is the goal index given to agent ``i``."""
    n = len(starts)
    taken = [False] * n
    assign = [None] * n
    for i in range(n):
        best_j, best_d = None, None
        for j in range(n):
            if taken[j]:
                continue
            d = fields[goals[j]].get(starts[i])
            key = (float("inf") if d is None else d, j)
            if best_d is None or key < best_d:
                best_j, best_d = j, key
        taken[best_j] = True
        assign[i] = best_j
    return assign


# --------------------------------------------------------------------------- #
# Deadlock (rotation) detection: follow the "wants" pointers from agent i      #
# --------------------------------------------------------------------------- #
def _deadlock_cycle(grid, fields, loc, tgt, occ, i) -> list | None:
    """Return the agents on the deadlock cycle through ``i`` (``a_1=i, a_2, ..``
    with ``nextNode(a_k.v, a_k.g)`` occupied by ``a_{k+1}`` and the chain looping
    back to ``i``), or ``None`` if ``i`` is not on such a cycle."""
    chain = [i]
    pos = {i: 0}
    cur = i
    while True:
        if loc[cur] == tgt[cur]:
            return None                      # settled agent — not a moving cycle
        u = _next_node(grid, fields, loc[cur], tgt[cur])
        nxt = occ.get(u)
        if nxt is None or nxt == cur:
            return None                      # cur can move / points to itself
        if nxt == i:
            return chain                     # closed the loop back to i
        if nxt in pos:
            return None                      # cycle that excludes i (i on a tail)
        pos[nxt] = len(chain)
        chain.append(nxt)
        cur = nxt


# --------------------------------------------------------------------------- #
# Offline TSWAP                                                                #
# --------------------------------------------------------------------------- #
def tswap(grid: GridWorld, starts, goals, *, assignment=None,
          max_timesteps: int | None = None, stats: dict | None = None):
    """Plan collision-free anonymous paths by target swapping.

    ``starts`` and ``goals`` are equal-length lists of distinct free cells; any
    agent may fill any goal. ``assignment`` (optional) is a permutation list with
    ``assignment[i]`` the goal index initially given to agent ``i`` — *any*
    assignment works (TSWAP repairs it); the default is a deterministic greedy
    nearest matching. Returns a ``dict`` ``{i: [cell, ...]}`` of per-agent paths
    (each starting at ``starts[i]`` and ending on some goal, collectively a
    perfect matching onto the goal set and pairwise collision-free), or ``None``
    if a start/goal is blocked, a goal is unreachable, or the timestep cap is hit
    (only on degenerate, slack-free instances). ``stats`` records ``swaps``,
    ``rotations``, ``timesteps``, ``makespan`` and ``final_assignment``."""
    starts = list(starts)
    goals = list(goals)
    if len(starts) != len(goals):
        raise ValueError("starts and goals must have equal length")
    n = len(starts)
    if n == 0:
        if stats is not None:
            stats.update(swaps=0, rotations=0, timesteps=0, makespan=0,
                         final_assignment=[])
        return {}
    for v in starts + goals:
        if not grid.is_free(v):
            return None

    # One distance field per (distinct) goal cell; every start must reach its
    # goals — if any goal is unreachable from any start the instance is broken.
    fields = {g: _dist_field(grid, g) for g in goals}
    for g in goals:
        for s in starts:
            if s not in fields[g]:
                return None

    if assignment is None:
        assignment = _greedy_assignment(grid, starts, goals, fields)

    loc = {i: starts[i] for i in range(n)}
    tgt = {i: goals[assignment[i]] for i in range(n)}
    occ = {starts[i]: i for i in range(n)}
    paths = {i: [starts[i]] for i in range(n)}

    if max_timesteps is None:
        diam = grid.width + grid.height
        max_timesteps = 2 * n * diam + 8

    swaps = rotations = 0
    t = 0
    while any(loc[i] != tgt[i] for i in range(n)):
        if t >= max_timesteps:
            return None                      # did not converge (degenerate map)
        for i in range(n):
            if loc[i] == tgt[i]:
                continue
            u = _next_node(grid, fields, loc[i], tgt[i])
            b = occ.get(u)
            if b is None:                    # free — move
                del occ[loc[i]]
                loc[i] = u
                occ[u] = i
            elif u == tgt[b]:                # b sits on its own target — SWAP
                tgt[i], tgt[b] = tgt[b], tgt[i]
                swaps += 1
            else:                            # blocked by a mover — maybe ROTATE
                cycle = _deadlock_cycle(grid, fields, loc, tgt, occ, i)
                if cycle is not None:
                    old = [tgt[a] for a in cycle]
                    for k, a in enumerate(cycle):
                        tgt[a] = old[k - 1]  # shift one step around the cycle
                    rotations += 1
                # otherwise i waits this timestep
        t += 1
        for i in range(n):
            paths[i].append(loc[i])

    if stats is not None:
        stats.update(
            swaps=swaps,
            rotations=rotations,
            timesteps=t,
            makespan=max(len(p) - 1 for p in paths.values()),
            final_assignment=[goals.index(tgt[i]) for i in range(n)],
        )
    return paths
