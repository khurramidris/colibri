from __future__ import annotations

import unittest

from lattice.stats import bootstrap_speedup, effective_throughput, geometric_mean, score_candidate


class StatsTests(unittest.TestCase):
    def test_geometric_mean(self):
        self.assertAlmostEqual(geometric_mean([1.0, 4.0]), 2.0)

    def test_effective_throughput_is_weighted_harmonic(self):
        self.assertAlmostEqual(effective_throughput([1.0, 2.0], [1.0, 1.0]), 4.0 / 3.0)

    def test_bootstrap_is_deterministic(self):
        pairs = [(1.1, 1.0), (1.2, 1.0), (1.15, 2.0)]
        self.assertEqual(bootstrap_speedup(pairs, confidence=0.9, seed=42, samples=200),
                         bootstrap_speedup(pairs, confidence=0.9, seed=42, samples=200))

    def test_candidate_clears_gates(self):
        score = score_candidate("fast", {"a": 1.0, "b": 2.0},
            {"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]},
            {"a": [1.2, 1.2, 1.2], "b": [2.4, 2.4, 2.4]}, min_runs=3,
            min_gain=0.03, max_regression=0.05, confidence=0.9,
            require_confidence=True, hourly_cost_usd=2.0)
        self.assertTrue(score.eligible)
        self.assertAlmostEqual(score.weighted_speedup or 0, 1.2)
        self.assertIsNotNone(score.cost_per_million_usd)

    def test_single_workload_regression_blocks_aggregate_gain(self):
        score = score_candidate("mixed", {"a": 1.0, "b": 1.0},
            {"a": [1.0, 1.0], "b": [1.0, 1.0]}, {"a": [2.0, 2.0], "b": [0.8, 0.8]},
            min_runs=2, min_gain=0.03, max_regression=0.05, confidence=0.9,
            require_confidence=False, hourly_cost_usd=None)
        self.assertFalse(score.eligible)
        self.assertIn("regression", score.reason)


if __name__ == "__main__":
    unittest.main()
