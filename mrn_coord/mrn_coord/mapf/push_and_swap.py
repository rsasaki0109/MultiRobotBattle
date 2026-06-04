"""Push and Swap: the swap-only ancestor of Push and Rotate.

A Python reproduction of Luna & Bekris's *"Push and Swap: Fast Cooperative
Path-Finding with Completeness Guarantees"* (IJCAI 2011) — the algorithm
:mod:`push_and_rotate` descends from, kept here as its own solver so the exact
completeness gap that motivated Push and Rotate is visible and gated.

Push and Swap places agents one at a time in priority order with **two**
movement primitives only:

- **push** — advance an agent one step along its shortest path to its goal,
  shoving blocking (not-yet-placed) agents into the nearest free space.
- **swap** — when two agents must pass and pushing cannot, bring them to a
  vertex of degree >= 3, clear two neighbours, and rotate the pair around that
  hub (six moves), then reverse the approach so every displaced agent is
  restored — exchanging only the two.

It has **no rotate primitive** and no packed-grid reduction. That is the whole
difference from :mod:`push_and_rotate`: Luna & Bekris's paper claimed
completeness for instances with >= 2 empty vertices, but de Wilde, ter Mors &
Witteveen (JAIR 2014) showed the bare push/swap core gets stuck on *cyclic,
slack-free* regions — a fully occupied ring, or a packed rectangle — where the
only way to advance an agent past another is to rotate a whole cycle by one.
Push and Rotate adds exactly that primitive (and a subproblem decomposition) to
close the gap. So this solver:

- **succeeds**, collision-free and on-goal *by construction*, wherever there is
  enough slack for push and swap — the same sparse, tree-like maps
  :mod:`push_and_rotate` solves; and
- **fails** (returns ``None``) on the cyclic-packed instances where
  :mod:`push_and_rotate`'s ``rotate`` succeeds.

Internally it reuses :mod:`push_and_rotate`'s ``_Solver`` — the *same* push and
swap primitives, byte-for-byte — but runs the order-sweep with ``rotate`` and
the residual-pocket BFS turned off (``allow_rotate=False,
allow_residual=False``) and never dispatches the packed-grid reduction. The
contrast is therefore clean: identical push/swap machinery, the rotate
completion the only thing removed.
"""

from __future__ import annotations

from .grid import GridWorld
from .push_and_rotate import _Solver, _moves_to_paths, _neighbors4
from .solution import Solution, sum_of_costs


def push_and_swap(grid: GridWorld, agents: dict, *, max_moves: int = 100_000,
                  stats: dict | None = None):
    """Solve a MAPF instance with the push/swap primitives only.

    ``agents`` maps an id to ``(start, goal)``. Returns a :class:`Solution`
    (collision-free, every agent on its goal — suboptimal), or ``None`` if push
    and swap alone could not place every agent (the cyclic, slack-free regime
    where :func:`push_and_rotate`'s rotate is needed). ``stats`` records
    ``stats["moves"]``."""
    if not agents:
        return Solution(paths={}, cost=0)
    for a, (s, g) in agents.items():
        if not grid.is_free(s) or not grid.is_free(g):
            return None

    def gdeg(a):
        return len(_neighbors4(grid, agents[a][1]))

    def sdeg(a):
        return len(_neighbors4(grid, agents[a][0]))

    # Push and Swap is order-sensitive (a bad order shoves an agent into a
    # dead-end a later placement then seals); place goal-corner agents first and
    # sweep a few alternative orders, exactly as the push_and_rotate core does —
    # the only difference here is that rotate and the residual BFS are off.
    orders = [
        sorted(agents, key=lambda a: (gdeg(a), agents[a][1])),
        sorted(agents, key=lambda a: (gdeg(a), -sdeg(a), agents[a][1])),
        sorted(agents, key=lambda a: (-sdeg(a), gdeg(a), agents[a][0])),
        sorted(agents, key=lambda a: (agents[a][1],)),
        sorted(agents, reverse=True, key=lambda a: (gdeg(a), agents[a][1])),
    ]

    for order in orders:
        solver = _Solver(grid, agents, max_moves)
        moves = solver.solve(order, allow_rotate=False, allow_residual=False)
        if moves is None:
            continue
        paths = _moves_to_paths(agents, moves)
        if stats is not None:
            stats["moves"] = len(moves)
        return Solution(paths=paths, cost=sum_of_costs(paths))
    return None
