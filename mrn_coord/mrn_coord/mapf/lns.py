"""MAPF-LNS: anytime MAPF by Large Neighborhood Search (Li et al., 2021).

CBS/ECBS *search* for a good solution from scratch; LaCAM finds *a* solution
fast. LNS takes a different, anytime route: start from any feasible solution
(here prioritized planning, falling back to complete LaCAM), then repeatedly
**destroy** a small neighborhood — rip out the paths of a handful of agents —
and **repair** it by replanning just those agents around everyone else's fixed
paths. Keep the repair if it does not raise the sum-of-costs. Each round is
cheap (it replans a few agents, not all), the cost decreases monotonically, and
you can stop whenever the budget runs out — so a rough initial solution is
polished toward the optimum over time, on team sizes far beyond CBS's reach.

Three destroy heuristics:

- **random** — a random set of agents.
- **worst** — the most *delayed* agent (largest gap between its current path
  cost and its obstacle-aware shortest path) plus the agents whose paths cross
  it; replanning this cluster together is what unsticks a bad detour.
- **map** — the agents passing through a congested high-degree vertex (an
  intersection, picked degree-weighted); replanning them together relieves the
  junction. Only used in adaptive mode.

By default the round picks random/worst with a fair coin and uses a fixed
neighborhood size — the original Li et al. ensemble. With ``adaptive=True`` the
round instead uses a **bi-level Thompson-Sampling bandit** (BALANCE, Phan et
al., AAAI 2024): a top bandit learns which of the three destroy heuristics pays
off, and a per-heuristic bottom bandit learns the neighborhood size from
``{2,4,8,16,32}``. The reward is the realized cost improvement
``max(0, c(P) - c(P+))``, so the search shifts effort toward whatever is
actually biting on *this* instance instead of a fixed 50/50 coin and a fixed k.

Pure and deterministic given the seed (the bandit samples from the same seeded
RNG). Repair is collision-free by construction (prioritized replanning over the
frozen paths), so every accepted solution stays valid.
"""

from __future__ import annotations

import random
from collections import deque

from .grid import Cell, GridWorld
from .lacam import lacam
from .prioritized import prioritized_planning
from .solution import Solution, sum_of_costs
from .space_time_astar import plan_path


def _bfs_dist_from(grid: GridWorld, goal: Cell) -> dict:
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


def _reservations(paths, subset, horizon):
    """Vertex/edge reservations from every agent *not* in ``subset``."""
    vertex: set = set()
    edge: set = set()
    for agent, path in paths.items():
        if agent in subset:
            continue
        for t, cell in enumerate(path):
            vertex.add((cell, t))
        goal_cell = path[-1]
        for t in range(len(path) - 1, horizon + 1):
            vertex.add((goal_cell, t))
        for t in range(len(path) - 1):
            edge.add((path[t + 1], path[t], t + 1))
    return vertex, edge


def _repair(grid, agents, paths, subset, horizon):
    """Replan ``subset`` around the frozen paths; return new paths or ``None``.

    Reservations (including each frozen agent's goal hold) extend to ``horizon``
    and every replan is capped at ``max_time=horizon``, so a repaired path can
    never run past the reserved window and slip through a held goal cell.
    """
    vertex, edge = _reservations(paths, subset, horizon)
    # replan the longest-detoured agents first (id breaks ties so the order is
    # independent of the set's iteration order — keeps the search deterministic)
    order = sorted(subset, key=lambda a: (-(len(paths[a]) - 1), a))
    new_paths = {}
    for agent in order:
        start, goal = agents[agent]
        path = plan_path(grid, start, goal, frozenset(vertex), frozenset(edge),
                         max_time=horizon)
        if path is None:
            return None
        new_paths[agent] = path
        for t, cell in enumerate(path):
            vertex.add((cell, t))
        for t in range(len(path) - 1, horizon + 1):
            vertex.add((path[-1], t))
        for t in range(len(path) - 1):
            edge.add((path[t + 1], path[t], t + 1))
    return new_paths


def _worst_neighborhood(paths, shortest, rng, k):
    """Most-delayed agent plus the agents whose paths cross it (padded random)."""
    delay = {a: (len(paths[a]) - 1) - shortest[a] for a in paths}
    seed_agent = max(delay, key=lambda a: (delay[a], a))
    seed_cells = set(paths[seed_agent])
    crossing = [a for a in paths
                if a != seed_agent and any(c in seed_cells for c in paths[a])]
    rng.shuffle(crossing)
    chosen = [seed_agent] + crossing
    if len(chosen) < k:
        rest = [a for a in paths if a not in chosen]
        rng.shuffle(rest)
        chosen += rest
    return set(chosen[:k])


def _vertex_degrees(grid: GridWorld) -> dict:
    """Free-neighbor count per free cell — high degree marks an intersection."""
    deg = {}
    for x in range(grid.width):
        for y in range(grid.height):
            cell = (x, y)
            if grid.is_free(cell):
                deg[cell] = sum(1 for _ in grid.neighbors(cell))
    return deg


def _map_neighborhood(paths, degrees, rng, k):
    """Agents through a congested high-degree vertex (degree-weighted pick)."""
    on_path: dict = {}
    for agent, path in paths.items():
        for cell in path:
            on_path.setdefault(cell, set()).add(agent)
    if not on_path:
        return set(list(paths)[:k])
    cells = sorted(on_path)                      # deterministic candidate order
    weights = [degrees.get(c, 1) for c in cells]
    pick = rng.choices(cells, weights=weights, k=1)[0]
    chosen = sorted(on_path[pick])
    rng.shuffle(chosen)
    if len(chosen) < k:
        rest = [a for a in paths if a not in chosen]
        rng.shuffle(rest)
        chosen += rest
    return set(chosen[:k])


class _NormalGammaBandit:
    """Thompson Sampling over a fixed arm set with Normal-Gamma posteriors.

    Each arm's reward is modeled Normal with unknown mean and precision; the
    Normal-Gamma prior is conjugate, so an arm only needs running
    ``(n, mean, M2)`` (Welford). To pick, we draw a plausible mean from each
    arm's posterior and take the argmax — exploration falls out of posterior
    width, so a barely-tried arm can still win. Deterministic given ``rng``.
    """

    def __init__(self, arms, rng, *, mu0=0.0, lambda0=1.0, alpha0=1.0,
                 beta0=1.0):
        self.arms = list(arms)
        self.rng = rng
        self.mu0, self.lambda0 = mu0, lambda0
        self.alpha0, self.beta0 = alpha0, beta0
        self.n = {a: 0 for a in self.arms}
        self.mean = {a: 0.0 for a in self.arms}
        self.m2 = {a: 0.0 for a in self.arms}

    def _sample(self, arm) -> float:
        n = self.n[arm]
        lam = self.lambda0 + n
        alpha = self.alpha0 + n / 2.0
        mu = (self.lambda0 * self.mu0 + n * self.mean[arm]) / lam
        beta = (self.beta0 + 0.5 * self.m2[arm]
                + (self.lambda0 * n * (self.mean[arm] - self.mu0) ** 2)
                / (2.0 * lam))
        tau = self.rng.gammavariate(alpha, 1.0 / beta)   # sampled precision
        sigma = (1.0 / (lam * tau)) ** 0.5
        return self.rng.gauss(mu, sigma)

    def select(self):
        best, best_val = self.arms[0], None
        for arm in self.arms:                            # arm order breaks ties
            val = self._sample(arm)
            if best_val is None or val > best_val:
                best, best_val = arm, val
        return best

    def update(self, arm, reward: float) -> None:
        self.n[arm] += 1
        delta = reward - self.mean[arm]
        self.mean[arm] += delta / self.n[arm]
        self.m2[arm] += delta * (reward - self.mean[arm])


def mapf_lns(
    grid: GridWorld,
    agents: dict,
    *,
    neighborhood_size: int | None = None,
    iterations: int = 100,
    seed: int = 0,
    init: Solution | None = None,
    adaptive: bool = False,
    stats: dict | None = None,
):
    """Improve a MAPF solution by Large Neighborhood Search (anytime).

    ``agents`` maps an agent id to a ``(start, goal)`` tuple. Returns the best
    :class:`Solution` found (collision-free), or ``None`` if no initial solution
    exists. ``init`` seeds the search (default: prioritized planning, then LaCAM
    if that fails).

    With ``adaptive=True`` the destroy heuristic and neighborhood size are
    chosen each round by a bi-level Thompson-Sampling bandit (BALANCE) instead
    of a fixed coin/size — it learns online which destroy pays off on this
    instance. The default (``adaptive=False``) is the original fixed ensemble
    and is byte-for-byte unchanged.

    If ``stats`` is given, it records ``initial_cost``, ``final_cost``,
    ``iterations``, and ``accepted`` (rounds that improved); in adaptive mode it
    also records ``arm_pulls`` (per-heuristic selection counts).
    """
    ids = sorted(agents)
    if init is None:
        init = prioritized_planning(grid, agents) or lacam(grid, agents)
    if init is None:
        return None

    paths = {a: list(init.paths[a]) for a in ids}
    cur_cost = sum_of_costs(paths)
    initial_cost = cur_cost

    k = neighborhood_size or max(2, min(len(ids), 8))
    k = min(k, len(ids))
    rng = random.Random(seed)
    shortest = {a: _bfs_dist_from(grid, agents[a][1]).get(agents[a][0], 0)
                for a in ids}
    # Generous, fixed reservation/replan horizon: at least the prioritized
    # feasibility bound, and never shorter than the longest initial path, so
    # every frozen agent's goal hold and every capped repair fit inside the
    # reserved window (a repaired path can never run past a held goal cell).
    formula = (2 * grid.width * grid.height + len(ids)
               + grid.width + grid.height + 5)
    horizon = max(formula, max(len(p) for p in paths.values())) + len(ids) + 5
    accepted = 0

    heuristics = ("random", "worst", "map")
    if adaptive:
        degrees = _vertex_degrees(grid)
        sizes = [s for s in (2, 4, 8, 16, 32) if 2 <= s <= len(ids)]
        if not sizes:                          # tiny instance: only one size fits
            sizes = [max(2, len(ids))]
        h_bandit = _NormalGammaBandit(heuristics, rng)
        size_bandits = {h: _NormalGammaBandit(sizes, rng) for h in heuristics}

    def _destroy(name, size):
        size = min(size, len(ids))
        if size >= len(ids):
            return set(ids)
        if name == "random":
            return set(rng.sample(ids, size))
        if name == "worst":
            return _worst_neighborhood(paths, shortest, rng, size)
        return _map_neighborhood(paths, degrees, rng, size)

    for _ in range(iterations):
        if adaptive:
            heuristic = h_bandit.select()
            size = size_bandits[heuristic].select()
            subset = _destroy(heuristic, size)
        elif k >= len(ids):
            subset = set(ids)
        elif rng.random() < 0.5:
            subset = set(rng.sample(ids, k))                  # random destroy
        else:
            subset = _worst_neighborhood(paths, shortest, rng, k)  # worst destroy

        repaired = _repair(grid, agents, paths, subset, horizon)
        reward = 0.0
        if repaired is not None:
            candidate = dict(paths)
            candidate.update(repaired)
            new_cost = sum_of_costs(candidate)
            if new_cost <= cur_cost:
                if new_cost < cur_cost:
                    accepted += 1
                    reward = float(cur_cost - new_cost)   # realized improvement
                paths = candidate
                cur_cost = new_cost
        if adaptive:
            h_bandit.update(heuristic, reward)
            size_bandits[heuristic].update(size, reward)

    if stats is not None:
        stats["initial_cost"] = initial_cost
        stats["final_cost"] = cur_cost
        stats["iterations"] = iterations
        stats["accepted"] = accepted
        if adaptive:
            stats["arm_pulls"] = dict(h_bandit.n)
    return Solution(paths=paths, cost=cur_cost)
