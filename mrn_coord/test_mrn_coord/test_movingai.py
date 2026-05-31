"""Tests for the MovingAI MAPF benchmark loader and runner."""

import os
import unittest

from mrn_coord.mapf.movingai import (
    parse_map,
    parse_scen,
    run_mapf_benchmark,
)

_MAP = """type octile
height 4
width 5
map
.....
..@..
..@..
.....
"""

_SCEN = """version 1
0\tm.map\t5\t4\t0\t0\t4\t3\t7
0\tm.map\t5\t4\t4\t0\t0\t3\t7
"""


class TestMovingAI(unittest.TestCase):
    def test_parse_map(self):
        grid = parse_map(_MAP)
        self.assertEqual((grid.width, grid.height), (5, 4))
        self.assertFalse(grid.is_free((2, 1)))   # '@'
        self.assertFalse(grid.is_free((2, 2)))   # '@'
        self.assertTrue(grid.is_free((0, 0)))

    def test_parse_scen(self):
        tasks = parse_scen(_SCEN)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].start, (0, 0))
        self.assertEqual(tasks[0].goal, (4, 3))
        self.assertAlmostEqual(tasks[0].optimal_length, 7.0)

    def test_run_benchmark_solves(self):
        grid = parse_map(_MAP)
        tasks = parse_scen(_SCEN)
        res = run_mapf_benchmark(grid, tasks, solver="cbs")
        self.assertTrue(res["solved"])
        self.assertEqual(res["num_agents"], 2)
        self.assertGreater(res["makespan"], 0)

    def test_bundled_example_solves(self):
        here = os.path.dirname(os.path.abspath(__file__))
        bench = os.path.join(here, "..", "benchmarks")
        if not os.path.exists(os.path.join(bench, "example.map")):
            self.skipTest("bundled example not present")
        from mrn_coord.mapf.movingai import load_map, load_scen
        grid = load_map(os.path.join(bench, "example.map"))
        tasks = load_scen(os.path.join(bench, "example.scen"))
        res = run_mapf_benchmark(grid, tasks, solver="cbs", max_expansions=20000)
        self.assertTrue(res["solved"])
        self.assertEqual(res["num_agents"], 3)


if __name__ == "__main__":
    unittest.main()
