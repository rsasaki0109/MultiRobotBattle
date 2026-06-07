"""Tests for the distilled battle transformer."""

import unittest

from mrn_coord.battle import BattleConfig, run_battle
from mrn_coord.battle_policy.count_aware import CountAwarePolicy, policy_for_name
from mrn_coord.battle_policy.transformer import (
    TransformerPolicy,
    TransformerWeights,
    encode_features,
    forward,
    build_observation,
)


class TestTransformer(unittest.TestCase):
    def test_forward_is_deterministic(self):
        policy = TransformerPolicy.default()
        w = policy.weights
        live = __import__("mrn_coord.battle", fromlist=["make_armies"]).make_armies(
            6, BattleConfig(), seed=0)
        obs = build_observation(live, 0)
        feat = encode_features(obs)
        a = forward(w, *feat[:3])
        b = forward(w, *feat[:3])
        self.assertEqual(a, b)

    def test_distilled_policy_tracks_teacher_on_snapshots(self):
        teacher = CountAwarePolicy("auto")
        policy = TransformerPolicy.default()
        cfg = BattleConfig()
        live = __import__("mrn_coord.battle", fromlist=["make_armies"]).make_armies(
            10, cfg, seed=3)
        matches = 0
        n = 0
        for i in range(len(live)):
            t = teacher.decide(live, i, cfg)
            p = policy.decide(live, i, cfg)
            if t is None or p is None:
                continue
            n += 1
            if t.target_index == p.target_index:
                matches += 1
        self.assertGreater(n, 0)
        self.assertGreaterEqual(matches / n, 0.85)

    def test_weights_load_and_save_roundtrip(self):
        policy = TransformerPolicy.default()
        path = "/tmp/battle_weights_test.json"
        policy.weights.save(path)
        loaded = TransformerWeights.load(path)
        self.assertEqual(loaded.d_model, policy.weights.d_model)
        self.assertEqual(len(loaded.w_out), 12)


class TestTransformerBattle(unittest.TestCase):
    def test_transformer_vs_nearest_both_decisive(self):
        for tactics in ("nearest", "transformer"):
            res = run_battle(10, BattleConfig(tactics=tactics), seed=5)
            self.assertIsNotNone(res.winner, tactics)


if __name__ == "__main__":
    unittest.main()
