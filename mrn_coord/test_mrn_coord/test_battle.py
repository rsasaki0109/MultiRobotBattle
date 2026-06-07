"""Tests for the swarm battle — invariants of the team-combat simulation."""

import unittest

from mrn_coord.battle import (
    BLUE,
    RED,
    BattleConfig,
    Bot,
    battle_step,
    make_armies,
    run_battle,
)


class TestArmies(unittest.TestCase):
    def test_counts_and_sides(self):
        cfg = BattleConfig()
        bots = make_armies(10, cfg, seed=0)
        self.assertEqual(len(bots), 20)
        red = [b for b in bots if b.team == RED]
        blue = [b for b in bots if b.team == BLUE]
        self.assertEqual(len(red), 10)
        self.assertEqual(len(blue), 10)
        # everyone starts alive at full health
        self.assertTrue(all(b.alive and b.hp == b.max_hp for b in bots))
        # red clusters left of centre, blue right of centre
        self.assertTrue(all(b.x < cfg.width / 2 for b in red))
        self.assertTrue(all(b.x > cfg.width / 2 for b in blue))


class TestCombatRules(unittest.TestCase):
    def test_damage_only_within_range(self):
        cfg = BattleConfig(attack_range=3.0, dps=10.0, dt=0.1)
        # one red, one blue, far apart (> attack_range) → nobody fires
        far = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
               Bot(20.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(far, cfg)
        self.assertTrue(all(b.hp == 100 for b in far))
        # close (< attack_range) → both take damage
        near = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(1.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(near, cfg)
        self.assertTrue(all(b.hp < 100 for b in near))

    def test_focus_fire_stacks(self):
        # a 1v1 vs a 1v3: the lone target should take ~3x the damage
        cfg = BattleConfig(attack_range=4.0, dps=10.0, dt=0.1, max_speed=0.0)
        duel = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(1.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(duel, cfg)
        solo_dmg = 100 - duel[1].hp
        swarmed = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                   Bot(0.0, 1.0, 0, 0, BLUE, 100, 100),
                   Bot(0.0, -1.0, 0, 0, BLUE, 100, 100),
                   Bot(1.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(swarmed, cfg)
        focus_dmg = 100 - swarmed[0].hp
        self.assertAlmostEqual(focus_dmg, 3 * solo_dmg, places=6)

    def test_zero_hp_is_eliminated(self):
        cfg = BattleConfig(attack_range=4.0, dps=1000.0, dt=1.0)
        bots = [Bot(0.0, 0.0, 0, 0, RED, 100, 100),
                Bot(1.0, 0.0, 0, 0, BLUE, 100, 100)]
        battle_step(bots, cfg)
        self.assertTrue(all(not b.alive and b.hp == 0.0 for b in bots))


class TestEngagement(unittest.TestCase):
    def test_default_battles_are_decisive(self):
        # fight-to-the-death default → every battle reaches a winner
        for seed in range(6):
            res = run_battle(14, seed=seed)
            self.assertIsNotNone(res.winner, f"seed {seed} stalled")
            # the loser is wiped out; the winner has survivors
            self.assertEqual(res.survivors[res.winner] > 0, True)
            loser = BLUE if res.winner == RED else RED
            self.assertEqual(res.survivors[loser], 0)

    def test_counts_never_increase(self):
        res = run_battle(12, seed=3)
        reds = [c[0] for c in res.counts]
        blues = [c[1] for c in res.counts]
        self.assertEqual(reds, sorted(reds, reverse=True))
        self.assertEqual(blues, sorted(blues, reverse=True))

    def test_total_bots_conserved(self):
        res = run_battle(12, seed=5)
        # alive + dead is constant across the whole history
        for frame in res.frames:
            self.assertEqual(len(frame), 24)
        last = res.frames[-1]
        alive = sum(1 for b in last if b[4])
        self.assertEqual(alive, res.survivors[RED] + res.survivors[BLUE])

    def test_deterministic(self):
        a = run_battle(13, seed=7)
        b = run_battle(13, seed=7)
        self.assertEqual(a.winner, b.winner)
        self.assertEqual(a.ticks, b.ticks)
        self.assertEqual(a.survivors, b.survivors)
        self.assertEqual(a.frames[-1], b.frames[-1])

    def test_seed_changes_outcome_history(self):
        a = run_battle(13, seed=1)
        b = run_battle(13, seed=2)
        # different seeds → different engagements (not identical histories)
        self.assertNotEqual(a.frames[5], b.frames[5])


class TestRetreatOption(unittest.TestCase):
    def test_retreat_pulls_wounded_away(self):
        # a wounded red bot with retreat enabled should steer away from its enemy
        cfg = BattleConfig(retreat_frac=0.5, w_retreat=5.0, w_pursue=2.0,
                           w_flock=0.0, w_sep=0.0, attack_range=0.0, max_speed=10.0)
        bots = [Bot(0.0, 0.0, 0, 0, RED, 10, 100),    # red at 10% hp (wounded)
                Bot(3.0, 0.0, 0, 0, BLUE, 100, 100)]  # blue to the +x
        battle_step(bots, cfg)
        # the wounded red bot should have moved in -x (away from the blue enemy)
        self.assertLess(bots[0].x, 0.0)


if __name__ == "__main__":
    unittest.main()
