from __future__ import annotations

import unittest

from lattice.common import LatticeError
from lattice.stats import (
    bonferroni_confidence,
    bootstrap_speedup,
    effective_throughput,
    geometric_mean,
    score_candidate,
)


class StatsTests(unittest.TestCase):
    def test_geometric_mean(self):
        self.assertAlmostEqual(geometric_mean([1.0, 4.0]), 2.0)

    def test_effective_throughput_is_weighted_harmonic(self):
        self.assertAlmostEqual(effective_throughput([1.0, 2.0], [1.0, 1.0]), 4.0 / 3.0)

    def test_bonferroni_confidence_controls_candidate_family(self):
        self.assertAlmostEqual(bonferroni_confidence(0.90, 5), 0.98)
        self.assertAlmostEqual(bonferroni_confidence(0.95, 1), 0.95)

    def test_stratified_bootstrap_is_deterministic(self):
        ratios = {"interactive": [1.1, 1.2, 1.15], "long": [1.05, 1.08, 1.06]}
        weights = {"interactive": 3.0, "long": 1.0}
        self.assertEqual(
            bootstrap_speedup(ratios, weights, confidence=0.9, seed=42, samples=500),
            bootstrap_speedup(ratios, weights, confidence=0.9, seed=42, samples=500),
        )

    def test_bootstrap_refuses_underpowered_workload(self):
        with self.assertRaisesRegex(LatticeError, "at least 3 paired runs"):
            bootstrap_speedup(
                {"case": [1.1, 1.2]}, {"case": 1.0},
                confidence=0.9, seed=42, samples=500,
            )

    def test_stratification_preserves_workload_weights(self):
        ratios = {
            "important": [1.20, 1.20, 1.20],
            "rare": [0.80] * 20,
        }
        weights = {"important": 9.0, "rare": 1.0}
        low, high = bootstrap_speedup(ratios, weights, confidence=0.9, seed=7, samples=500)
        expected = geometric_mean([1.20, 0.80], [9.0, 1.0])
        self.assertAlmostEqual(low, expected)
        self.assertAlmostEqual(high, expected)
        self.assertGreater(low, 1.0)

    def test_candidate_clears_confidence_screening(self):
        score = score_candidate(
            "fast", {"a": 1.0, "b": 2.0},
            {"a": {0: 1.0, 1: 1.0, 2: 1.0}, "b": {0: 2.0, 1: 2.0, 2: 2.0}},
            {"a": {0: 1.2, 1: 1.2, 2: 1.2}, "b": {0: 2.4, 1: 2.4, 2: 2.4}},
            min_runs=3, min_gain=0.03, max_regression=0.05,
            confidence=0.98, require_confidence=True, hourly_cost_usd=2.0,
        )
        self.assertTrue(score.eligible)
        self.assertTrue(score.confidence_evaluated)
        self.assertIn("confidence-screened", score.reason)
        self.assertAlmostEqual(score.weighted_speedup or 0, 1.2)
        self.assertIsNotNone(score.cost_per_million_usd)
        self.assertEqual(score.per_case["a"]["paired_runs"], 3)
        self.assertEqual(score.per_case["a"]["paired_repeat_ids"], [0, 1, 2])

    def test_underpowered_candidate_is_exploratory_or_ineligible(self):
        baseline = {"case": {0: 1.0, 1: 1.0}}
        candidate = {"case": {0: 1.2, 1: 1.2}}
        exploratory = score_candidate(
            "fast", {"case": 1.0}, baseline, candidate,
            min_runs=2, min_gain=0.03, max_regression=0.05,
            confidence=0.9, require_confidence=False, hourly_cost_usd=None,
        )
        self.assertTrue(exploratory.eligible)
        self.assertIsNone(exploratory.ci_low)
        self.assertFalse(exploratory.confidence_evaluated)
        screened = score_candidate(
            "fast", {"case": 1.0}, baseline, candidate,
            min_runs=2, min_gain=0.03, max_regression=0.05,
            confidence=0.9, require_confidence=True, hourly_cost_usd=None,
        )
        self.assertFalse(screened.eligible)
        self.assertIn("underpowered", screened.reason)
        self.assertIsNone(screened.ci_low)

    def test_point_estimator_uses_paired_ratios_not_ratio_of_medians(self):
        score = score_candidate(
            "paired", {"case": 1.0},
            {"case": {0: 1.0, 1: 2.0, 2: 100.0}},
            {"case": {0: 2.0, 1: 100.0, 2: 101.0}},
            min_runs=3, min_gain=0.0, max_regression=1.0,
            confidence=0.9, require_confidence=False, hourly_cost_usd=None,
        )
        self.assertAlmostEqual(score.weighted_speedup or 0.0, 2.0)
        self.assertAlmostEqual(score.per_case["case"]["candidate_tok_s"], 100.0)
        self.assertAlmostEqual(score.per_case["case"]["baseline_tok_s"], 2.0)

    def test_pairing_uses_exact_repeat_identity_when_failures_differ(self):
        score = score_candidate(
            "paired", {"case": 1.0},
            {"case": {0: 1.0, 2: 100.0}},
            {"case": {1: 100.0, 2: 101.0}},
            min_runs=1, min_gain=0.0, max_regression=1.0,
            confidence=0.9, require_confidence=False, hourly_cost_usd=None,
        )
        self.assertAlmostEqual(score.weighted_speedup or 0.0, 1.01)
        self.assertEqual(score.per_case["case"]["paired_repeat_ids"], [2])
        self.assertEqual(score.per_case["case"]["paired_runs"], 1)
        self.assertIsNone(score.ci_low)

    def test_single_workload_regression_blocks_aggregate_gain(self):
        score = score_candidate(
            "mixed", {"a": 1.0, "b": 1.0},
            {"a": {0: 1.0, 1: 1.0}, "b": {0: 1.0, 1: 1.0}},
            {"a": {0: 2.0, 1: 2.0}, "b": {0: 0.8, 1: 0.8}},
            min_runs=2, min_gain=0.03, max_regression=0.05,
            confidence=0.9, require_confidence=False, hourly_cost_usd=None,
        )
        self.assertFalse(score.eligible)
        self.assertIn("regression", score.reason)


if __name__ == "__main__":
    unittest.main()
