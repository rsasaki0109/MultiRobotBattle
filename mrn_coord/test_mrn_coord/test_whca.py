"""Tests for Windowed Hierarchical Cooperative A* (Silver, AIIDE 2005).

WHCA* layers a true-distance heuristic (RRA*) and a rolling cooperation window
onto prioritized planning. The contracts: the RRA* oracle returns the exact
shortest-path distance; the cooperative search is collision-free by construction
and reaches every goal on solvable instances; and the rolling window with a
rotating priority order resolves transient deadlocks that a single fixed priority
order (plain prioritized planning) livelocks on.
"""

import random
import unittest
from collections import deque

from mrn_coord.mapf import GridWorld
from mrn_coord.mapf.conflicts import detect_first_conflict
from mrn_coord.mapf.grid import manhattan
from mrn_coord.mapf.prioritized import prioritized_planning
from mrn_coord.mapf.whca import RRAStar, _segment_search, whca_star


def _bfs(grid, goal):
    d = {goal: 0}
    q = deque([goal])
    while q:
        c = q.popleft()
        for nb in grid.neighbors(c):
            if nb != c and nb not in d:
                d[nb] = d[c] + 1
                q.append(nb)
    return d


def _inst(seed, n, w, h):
    rng = random.Random(seed)
    free = [(x, y) for x in range(w) for y in range(h)]
    cells = rng.sample(free, 2 * n)
    return GridWorld(w, h), {i: (cells[i], cells[n + i]) for i in range(n)}


class TestRRAStar(unittest.TestCase):
    def test_true_distance_matches_bfs(self):
        grid = GridWorld(8, 8, blocked=frozenset((3, y) for y in range(0, 6)))
        bfs = _bfs(grid, (7, 7))
        rra = RRAStar(grid, (7, 7), (0, 0))
        for cell, d in bfs.items():
            self.assertEqual(rra.distance(cell), d)

    def test_unreachable_is_none(self):
        # A walled-off pocket has no path to the goal.
        blocked = {(1, 0), (1, 1), (0, 1)}
        grid = GridWorld(4, 4, blocked=frozenset(blocked))
        rra = RRAStar(grid, (3, 3), (0, 0))
        self.assertIsNone(rra.distance((0, 0)))


class TestHierarchicalHeuristic(unittest.TestCase):
    def test_true_distance_prunes_vs_manhattan(self):
        # A wall the agent must detour around makes Manhattan badly misleading;
        # the true-distance heuristic expands far fewer states.
        grid = GridWorld(11, 11, blocked=frozenset((5, y) for y in range(0, 9)))
        start, goal = (0, 0), (10, 0)
        rra = RRAStar(grid, goal, start)
        st_true: dict = {}
        _segment_search(grid, start, goal, 0, 999, {}, set(), rra.distance,
                        stats=st_true)
        st_man: dict = {}
        _segment_search(grid, start, goal, 0, 999, {}, set(),
                        lambda c: manhattan(c, goal), stats=st_man)
        self.assertLess(st_true["expansions"], st_man["expansions"])


class TestWhca(unittest.TestCase):
    def test_collision_free_and_reaches_goals(self):
        for win in (4, 8, 16):
            for seed in range(40):
                grid, agents = _inst(seed, 5, 8, 8)
                sol = whca_star(grid, agents, window=win)
                self.assertIsNotNone(sol, f"seed={seed} window={win}")
                self.assertIsNone(detect_first_conflict(sol.paths))
                for a, (start, goal) in agents.items():
                    self.assertEqual(sol.paths[a][0], start)
                    self.assertEqual(sol.paths[a][-1], goal)

    def test_rolling_window_solves_where_prioritized_fails(self):
        # A frozen congested instance: plain prioritized planning fails, and so
        # does full-horizon non-rotating WHCA* (prioritized + true distance); the
        # rolling window resolves it.
        rng = random.Random(20)
        free = [(x, y) for x in range(7) for y in range(7)]
        blocked: set = set()
        while len(blocked) < int(len(free) * 0.18):
            blocked.add(rng.choice(free))
        free2 = [c for c in free if c not in blocked]
        cells = rng.sample(free2, 12)
        grid = GridWorld(7, 7, blocked=frozenset(blocked))
        agents = {i: (cells[i], cells[6 + i]) for i in range(6)}

        self.assertIsNone(prioritized_planning(grid, agents))
        self.assertIsNone(whca_star(grid, agents, window=400,
                                    rotate_priority=False))
        sol = whca_star(grid, agents, window=8)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_window_recovers_full_horizon_hca(self):
        # A window >= the makespan is plain HCA*: one reservation of every whole
        # path, collision-free, reaching all goals on an easy instance.
        grid, agents = _inst(3, 4, 8, 8)
        sol = whca_star(grid, agents, window=500)
        self.assertIsNotNone(sol)
        self.assertIsNone(detect_first_conflict(sol.paths))

    def test_deterministic(self):
        grid, agents = _inst(11, 5, 8, 8)
        a = whca_star(grid, agents, window=8)
        b = whca_star(grid, agents, window=8)
        self.assertEqual(a.cost, b.cost)
        self.assertEqual(a.paths, b.paths)

    def test_empty_instance(self):
        sol = whca_star(GridWorld(5, 5), {})
        self.assertEqual(sol.cost, 0)


if __name__ == "__main__":
    unittest.main()
