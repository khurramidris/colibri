from __future__ import annotations

import copy
import unittest

from lattice.common import LatticeError
from lattice.oracle import ORACLE_POLICY, ORACLE_SCHEMA, compare_oracles, validate_oracle


def fixture() -> dict:
    return {
        "schema": ORACLE_SCHEMA,
        "policy": dict(ORACLE_POLICY),
        "steps": [{
            "step": 0,
            "forced": 3,
            "top1": 2,
            "top2": 4,
            "nonfinite": 0,
            "top1_logit": 3.0,
            "forced_logit": 1.25,
            "margin": 0.5,
            "mean": 0.833333333,
            "rms": 1.9,
            "projection_0": 1.0,
            "projection_1": -2.0,
            "projection_2": 3.0,
            "projection_3": -4.0,
            "topk_ids_hash": "0123456789abcdef",
        }],
    }


class OracleTests(unittest.TestCase):
    def test_identical_oracles_match(self):
        value = fixture()
        self.assertIsNone(compare_oracles(value, copy.deepcopy(value)))

    def test_small_numeric_drift_is_tolerated(self):
        base = fixture()
        trial = copy.deepcopy(base)
        trial["steps"][0]["projection_2"] += 0.001
        self.assertIsNone(compare_oracles(base, trial))

    def test_identity_or_large_numeric_drift_is_rejected(self):
        base = fixture()
        trial = copy.deepcopy(base)
        trial["steps"][0]["top1"] = 9
        self.assertIn("top1 differs", compare_oracles(base, trial) or "")
        trial = copy.deepcopy(base)
        trial["steps"][0]["forced_logit"] += 0.1
        self.assertIn("beyond tolerance", compare_oracles(base, trial) or "")

    def test_nonfinite_or_malformed_oracle_is_rejected(self):
        value = fixture()
        value["steps"][0]["nonfinite"] = 1
        self.assertIn("non-finite", compare_oracles(value, value) or "")
        value = fixture()
        value["steps"][0]["topk_ids_hash"] = "bad"
        with self.assertRaisesRegex(LatticeError, "top-k identity"):
            validate_oracle(value)


if __name__ == "__main__":
    unittest.main()
