#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from progressive_expert import (
    COLIBRI_E8_IQ3_BPW,
    decode_progressive,
    encode_progressive,
    evaluate,
    make_synthetic,
)


class ProgressiveExpertTests(unittest.TestCase):
    def test_shape_and_finite_roundtrip(self) -> None:
        weights = make_synthetic(7, 131, seed=7)
        encoded = encode_progressive(
            weights, group_size=64, correction_fraction=0.03
        )
        decoded = decode_progressive(encoded)
        self.assertEqual(decoded.shape, weights.shape)
        self.assertTrue(np.isfinite(decoded).all())

    def test_payload_accounting_is_exact(self) -> None:
        weights = np.zeros((2, 64), dtype=np.float32)
        encoded = encode_progressive(
            weights, group_size=64, correction_fraction=0.0
        )
        # Per group: 8 sign bytes + 2 base-scale + 1 count + 2 correction-scale.
        self.assertEqual(encoded.payload_bytes, 2 * 13)
        self.assertAlmostEqual(encoded.effective_bpw, 26 * 8 / 128)

    def test_tail_group_correction_indices_are_valid(self) -> None:
        weights = make_synthetic(3, 70, seed=11)
        encoded = encode_progressive(
            weights, group_size=64, correction_fraction=0.20
        )
        decode_progressive(encoded)  # raises on an invalid tail index

    def test_more_refinement_does_not_increase_reconstruction_error(self) -> None:
        weights = make_synthetic(16, 256, seed=3)
        errors = []
        for fraction in (0.0, 0.01, 0.03, 0.07):
            decoded = decode_progressive(
                encode_progressive(
                    weights,
                    group_size=64,
                    correction_fraction=fraction,
                )
            )
            errors.append(float(np.linalg.norm(weights - decoded)))
        for before, after in zip(errors, errors[1:]):
            self.assertLessEqual(after, before + 1.0e-5)

    def test_screening_report_has_physical_baseline(self) -> None:
        weights = make_synthetic(8, 128, seed=5)
        report = evaluate(
            weights,
            correction_fractions=[0.01, 0.03],
            activation_samples=8,
            seed=5,
        )
        self.assertAlmostEqual(
            report["strongest_physical_baseline_bpw"],
            COLIBRI_E8_IQ3_BPW,
        )
        self.assertEqual(len(report["progressive"]), 2)
        for row in report["progressive"]:
            self.assertGreater(row["payload_bytes"], 0)
            self.assertGreater(row["effective_bpw"], 0)


if __name__ == "__main__":
    unittest.main()
