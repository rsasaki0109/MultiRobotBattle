import unittest
from pathlib import Path
import tempfile

from mrn_netem.loss_models import RandomLossModel
from mrn_netem.profile import load_network_profile


class TestRandomLossModel(unittest.TestCase):
    def test_random_loss_is_deterministic(self):
        first = RandomLossModel(loss_rate=0.25, seed=42).mask(20)
        second = RandomLossModel(loss_rate=0.25, seed=42).mask(20)
        self.assertEqual(first, second)

    def test_random_loss_rejects_invalid_rate(self):
        with self.assertRaises(ValueError):
            RandomLossModel(loss_rate=1.5)

    def test_load_network_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            path.write_text(
                "\n".join(
                    [
                        "network:",
                        "  packet_loss_percent: 20",
                        "  latency_ms_mean: 80",
                        "  jitter_ms: 15",
                    ]
                ),
                encoding="utf-8",
            )
            profile = load_network_profile(path)
        self.assertEqual(profile.packet_loss_percent, 20.0)
        self.assertEqual(profile.loss_rate, 0.2)
        self.assertEqual(profile.latency_ms_mean, 80.0)

    def test_profile_rejects_invalid_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            path.write_text("network:\n  packet_loss_percent: 120\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_network_profile(path)


if __name__ == "__main__":
    unittest.main()
