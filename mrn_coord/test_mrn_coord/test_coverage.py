"""Tests for cooperative coverage: occupancy, frontiers, and allocation."""

import itertools
import unittest

from mrn_coord.coverage import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyGrid,
    allocate_frontiers,
    bfs_free_distances,
    cluster_frontiers,
    frontier_cells,
    greedy_auction,
    hungarian_assignment,
    is_frontier,
    min_cost_assignment,
)


def _brute_force_assignment_cost(cost):
    """Optimal total cost by trying every assignment (small matrices only)."""
    n = len(cost)
    m = len(cost[0])
    rows = min(n, m)
    best = None
    if n <= m:
        for cols in itertools.permutations(range(m), n):
            total = sum(cost[i][cols[i]] for i in range(n))
            best = total if best is None else min(best, total)
    else:
        for r in itertools.permutations(range(n), m):
            total = sum(cost[r[j]][j] for j in range(m))
            best = total if best is None else min(best, total)
    return best, rows


class TestOccupancy(unittest.TestCase):
    def test_from_rows_states(self):
        grid = OccupancyGrid.from_rows(["?#.", "..."])
        # row 0 is the top (y=1)
        self.assertEqual(grid.state((0, 1)), UNKNOWN)
        self.assertEqual(grid.state((1, 1)), OCCUPIED)
        self.assertEqual(grid.state((2, 1)), FREE)
        self.assertTrue(grid.is_free((0, 0)))

    def test_rejects_ragged_rows(self):
        with self.assertRaises(ValueError):
            OccupancyGrid.from_rows(["..", "..."])

    def test_rejects_bad_char(self):
        with self.assertRaises(ValueError):
            OccupancyGrid.from_rows(["x"])

    def test_free_neighbors(self):
        grid = OccupancyGrid.from_rows([".#", ".."])
        self.assertEqual(set(grid.free_neighbors((0, 0))), {(1, 0), (0, 1)})


class TestFrontier(unittest.TestCase):
    def test_is_frontier(self):
        # rows: top "..", bottom ".?"  => (1,0) is UNKNOWN, the rest free.
        grid = OccupancyGrid.from_rows(["..", ".?"])
        self.assertTrue(is_frontier(grid, (0, 0)))    # free, borders unknown (1,0)
        self.assertTrue(is_frontier(grid, (1, 1)))    # free, borders unknown (1,0)
        self.assertFalse(is_frontier(grid, (0, 1)))   # free, only free neighbors
        self.assertFalse(is_frontier(grid, (1, 0)))   # the unknown cell itself

    def test_frontier_cells_set(self):
        grid = OccupancyGrid.from_rows(["???", ".?.", "..."])
        cells = set(frontier_cells(grid))
        # free cells adjacent to an unknown
        self.assertIn((0, 1), cells)
        self.assertIn((2, 1), cells)
        self.assertIn((1, 0), cells)      # borders the central unknown (1,1)
        self.assertNotIn((0, 0), cells)   # only free neighbors, no unknown

    def test_cluster_groups_adjacent(self):
        # one connected frontier strip along the top free row under unknowns
        grid = OccupancyGrid.from_rows(["????", "....", "...."])
        clusters = cluster_frontiers(grid)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].size, 4)
        self.assertIn(clusters[0].representative, clusters[0].cells)

    def test_two_separate_clusters(self):
        # two unknown pockets separated by a wall -> two frontier clusters
        grid = OccupancyGrid.from_rows([
            "?.#.?",
            "...#.",
            ".....",
        ])
        clusters = cluster_frontiers(grid)
        self.assertGreaterEqual(len(clusters), 2)


class TestBfsDistance(unittest.TestCase):
    def test_distances_over_free_space(self):
        grid = OccupancyGrid.from_rows(["...", "...", "..."])
        dist = bfs_free_distances(grid, (0, 0))
        self.assertEqual(dist[(0, 0)], 0)
        self.assertEqual(dist[(2, 0)], 2)
        self.assertEqual(dist[(2, 2)], 4)

    def test_walls_block_and_unreachable_absent(self):
        grid = OccupancyGrid.from_rows([
            ".#.",
            ".#.",
            ".#.",
        ])
        dist = bfs_free_distances(grid, (0, 0))
        self.assertIn((0, 2), dist)
        self.assertNotIn((2, 0), dist)    # walled off


class TestHungarian(unittest.TestCase):
    def test_matches_brute_force_square(self):
        matrices = [
            [[4, 2, 8], [4, 3, 7], [3, 1, 6]],
            [[1, 2], [2, 1]],
            [[9, 9, 1], [1, 9, 9], [9, 1, 9]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
        for cost in matrices:
            pairs = hungarian_assignment(cost)
            total = sum(cost[i][j] for i, j in pairs)
            best, rows = _brute_force_assignment_cost(cost)
            self.assertEqual(len(pairs), rows)
            self.assertEqual(total, best)
            # a valid assignment: distinct rows and distinct cols
            self.assertEqual(len({i for i, _ in pairs}), len(pairs))
            self.assertEqual(len({j for _, j in pairs}), len(pairs))

    def test_rectangular_more_cols(self):
        cost = [[5, 1, 4, 2], [3, 6, 1, 7]]
        pairs = hungarian_assignment(cost)
        total = sum(cost[i][j] for i, j in pairs)
        best, rows = _brute_force_assignment_cost(cost)
        self.assertEqual(len(pairs), rows)
        self.assertEqual(total, best)

    def test_min_cost_assignment_more_rows(self):
        # 3 robots, 2 frontiers -> only 2 matched, optimal pair chosen
        cost = [[2, 9], [9, 2], [5, 5]]
        pairs = min_cost_assignment(cost)
        total = sum(cost[i][j] for i, j in pairs)
        best, rows = _brute_force_assignment_cost(cost)
        self.assertEqual(len(pairs), rows)
        self.assertEqual(total, best)


class TestGreedyAuction(unittest.TestCase):
    def test_valid_one_to_one(self):
        cost = {("r1", "f1"): 1, ("r1", "f2"): 5,
                ("r2", "f1"): 4, ("r2", "f2"): 2}
        assignment = greedy_auction(["r1", "r2"], ["f1", "f2"],
                                    lambda r, f: cost[(r, f)])
        self.assertEqual(assignment, {"r1": "f1", "r2": "f2"})

    def test_skips_unreachable(self):
        INF = float("inf")
        cost = {("r1", "f1"): INF, ("r1", "f2"): 3,
                ("r2", "f1"): 2, ("r2", "f2"): INF}
        assignment = greedy_auction(["r1", "r2"], ["f1", "f2"],
                                    lambda r, f: cost[(r, f)])
        self.assertEqual(assignment, {"r1": "f2", "r2": "f1"})


class TestAllocateFrontiers(unittest.TestCase):
    def setUp(self):
        # open room with unknown pockets on the left and right edges
        self.grid = OccupancyGrid.from_rows([
            "?.......?",
            "?.......?",
            "?.......?",
        ])
        # two robots near each side
        self.robots = {"L": (1, 1), "R": (7, 1)}
        clusters = cluster_frontiers(self.grid)
        self.targets = [c.representative for c in clusters]

    def test_finds_two_targets(self):
        self.assertEqual(len(self.targets), 2)

    def test_each_robot_gets_nearest_side(self):
        for method in ("hungarian", "greedy"):
            assignment = allocate_frontiers(
                self.grid, self.robots, self.targets, method=method
            )
            self.assertEqual(len(assignment), 2)
            # the left robot should be sent to the lower-x frontier
            left_target = min(self.targets, key=lambda c: c[0])
            right_target = max(self.targets, key=lambda c: c[0])
            self.assertEqual(assignment["L"], left_target)
            self.assertEqual(assignment["R"], right_target)

    def test_empty_inputs(self):
        self.assertEqual(allocate_frontiers(self.grid, {}, self.targets), {})
        self.assertEqual(allocate_frontiers(self.grid, self.robots, []), {})


if __name__ == "__main__":
    unittest.main()
