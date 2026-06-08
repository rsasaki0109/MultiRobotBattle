"""RoboMaster terrain presets — layout sanity."""

from __future__ import annotations

import math
import unittest

from mrn_coord.battle import battle_scenario, simulate
from mrn_coord.battle_terrain import (
    arena_terrain,
    chokepoint_terrain,
    chokepoint_walls,
    cover_along_segment_rect,
    kingdom_terrain,
    objective_terrain,
    point_clearance_rect,
    push_out_of_walls,
    segment_intersects_rect,
    total_war_terrain,
)


def _disc_inside(ox, oy, r, width, height, margin=0.5):
    return (margin + r <= ox <= width - margin - r
            and margin + r <= oy <= height - margin - r)


def _rect_inside(cx, cy, hw, hh, width, height, margin=0.5):
    return (margin + hw <= cx <= width - margin - hw
            and margin + hh <= cy <= height - margin - hh)


class TestBattleTerrain(unittest.TestCase):
    def test_presets_fit_default_arena(self):
        w, h = 40.0, 24.0
        for bundle in (chokepoint_terrain(), arena_terrain(), objective_terrain()):
            self.assertGreater(len(bundle["obstacles"]), 2)
            for ox, oy, r in bundle["obstacles"]:
                self.assertTrue(_disc_inside(ox, oy, r, w, h), (ox, oy, r))
            for cx, cy, hw, hh in bundle.get("walls", ()):
                self.assertTrue(_rect_inside(cx, cy, hw, hh, w, h), (cx, cy, hw, hh))

    def test_total_war_preset_scales(self):
        bundle = total_war_terrain(width=140.0, height=72.0)
        self.assertGreaterEqual(len(bundle["obstacles"]), 8)
        self.assertGreaterEqual(len(bundle["walls"]), 6)
        for ox, oy, r in bundle["obstacles"]:
            self.assertTrue(_disc_inside(ox, oy, r, 140.0, 72.0))
        for cx, cy, hw, hh in bundle["walls"]:
            self.assertTrue(_rect_inside(cx, cy, hw, hh, 140.0, 72.0))

    def test_kingdom_preset_scales(self):
        bundle = kingdom_terrain(width=100.0, height=56.0)
        self.assertGreaterEqual(len(bundle["obstacles"]), 5)
        self.assertGreaterEqual(len(bundle["walls"]), 4)

    def test_grand_alliance_with_terrain_resolves(self):
        bots, cfg, _ = battle_scenario("grand_alliance")
        self.assertGreater(len(cfg.obstacles) + len(cfg.walls), 10)
        res = simulate(bots, cfg, max_ticks=120, frame_stride=10)
        self.assertGreater(res.ticks, 0)

    def test_chokepoint_lane_has_vertical_gaps(self):
        walls = chokepoint_walls()
        mid = 20.0
        slabs = [w for w in walls if abs(w[0] - mid) < 0.5 and w[3] > 0.9]
        self.assertEqual(len(slabs), 3)
        ys = sorted(p[1] for p in slabs)
        for a, b in zip(ys, ys[1:]):
            self.assertGreater(b - a, 3.5)

    def test_wall_blocks_line_of_fire(self):
        self.assertEqual(
            cover_along_segment_rect(0.0, 0.0, 10.0, 0.0, 5.0, 0.0, 1.0, 2.0, 0.5),
            0.0,
        )
        self.assertGreater(
            cover_along_segment_rect(0.0, 5.0, 10.0, 5.0, 5.0, 0.0, 1.0, 2.0, 0.5),
            0.9,
        )

    def test_push_out_of_walls(self):
        walls = ((5.0, 5.0, 1.0, 1.0),)
        x, y = push_out_of_walls(5.0, 5.0, walls, body=0.5)
        self.assertFalse(segment_intersects_rect(x, y, x, y, 5.0, 5.0, 1.0, 1.0))
        self.assertGreater(point_clearance_rect(x, y, 5.0, 5.0, 1.0, 1.0), 0.4)


if __name__ == "__main__":
    unittest.main()
