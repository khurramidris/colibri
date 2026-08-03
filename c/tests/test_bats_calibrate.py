import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bats_calibrate", ROOT / "tools" / "bats_calibrate.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BatsCalibrateTest(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(MODULE.parse_size("64KiB"), 65536)
        self.assertEqual(MODULE.parse_size("1.5MB"), 1_500_000)
        with self.assertRaises(ValueError):
            MODULE.parse_size("zero")

    def test_fit_known_model(self):
        # fixed=80us, bandwidth=4GB/s => slope=1/(4*1000) us/byte.
        points = [
            (64_000, 80.0 + 64_000 / 4000.0),
            (1_000_000, 80.0 + 1_000_000 / 4000.0),
            (8_000_000, 80.0 + 8_000_000 / 4000.0),
        ]
        fixed, gbps = MODULE.fit_latency_model(points)
        self.assertAlmostEqual(fixed, 80.0, places=8)
        self.assertAlmostEqual(gbps, 4.0, places=8)

    def test_negative_intercept_is_constrained(self):
        fixed, gbps = MODULE.fit_latency_model([(100, 1.0), (200, 1.5), (300, 2.0)])
        self.assertGreaterEqual(fixed, 0.0)
        self.assertGreater(gbps, 0.0)

    def test_percentile(self):
        self.assertEqual(MODULE.percentile([1.0, 2.0, 3.0], 0.5), 2.0)


if __name__ == "__main__":
    unittest.main()
