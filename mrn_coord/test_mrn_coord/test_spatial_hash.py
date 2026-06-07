"""Tests for the battle spatial hash."""

import math
import unittest

from mrn_coord.spatial_hash import SpatialHash


class TestSpatialHash(unittest.TestCase):
    def test_matches_brute_force_disk(self):
        pts = [(i * 1.7, (i * 0.9) % 11.0) for i in range(80)]
        sh = SpatialHash(cell_size=3.0)
        sh.build(pts)
        for i, (x, y) in enumerate(pts):
            r = 4.5
            got = set(sh.query_disk(x, y, r, pts))
            want = {j for j, (px, py) in enumerate(pts)
                    if math.hypot(x - px, y - py) <= r}
            self.assertEqual(got, want, msg=f"point {i}")


if __name__ == "__main__":
    unittest.main()
