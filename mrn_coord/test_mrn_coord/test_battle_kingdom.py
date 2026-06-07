"""Kingdom-scale battle — spatial hash + grand army deployment."""

import time
import unittest

from mrn_coord.battle import (
    BattleConfig,
    battle_scenario,
    kingdom_config,
    make_grand_army,
    run_battle,
    simulate,
)
from mrn_coord.battle import RED, BLUE


class TestKingdomBattle(unittest.TestCase):
    def test_grand_army_count(self):
        cfg = kingdom_config()
        red = make_grand_army(cfg, RED, (10, 20), rows=5, cols=6)
        self.assertEqual(len(red), 30)

    def test_kingdom_scenario_resolves(self):
        bots, cfg, title = battle_scenario("kingdom")
        self.assertEqual(len(bots), 160)
        self.assertGreater(cfg.width, 80)
        res = simulate(bots, cfg, max_ticks=1000, frame_stride=4)
        self.assertIsNotNone(res.winner)
        self.assertIn("Kingdom", title)

    def test_spatial_large_battle_under_budget(self):
        cfg = kingdom_config()
        t0 = time.time()
        res = run_battle(40, cfg, seed=3, max_ticks=700)
        elapsed = time.time() - t0
        self.assertTrue(res.winner is not None or res.ticks >= 699)
        self.assertLess(elapsed, 45.0, f"too slow: {elapsed:.1f}s")


if __name__ == "__main__":
    unittest.main()
