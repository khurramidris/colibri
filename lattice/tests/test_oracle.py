from __future__ import annotations

import copy
import unittest

from lattice.common import LatticeError
from lattice.oracle import ORACLE_POLICY, ORACLE_SCHEMA, compare_oracles, validate_oracle


def row(*, top1: int = 2, forced_logit: float = 1.25, topk: list[int] | None = None) -> list:
    return [
        3, top1, 4, 0,
        3.0, forced_logit, 0.5, 0.8, 1.9,
        1.0, -2.0, 3.0, -4.0,
        topk or [top1, 4, 3, 1, 5, 6, 7, 8],
    ]


def fixture() -> dict:
    return {"schema": ORACLE_SCHEMA, "policy": dict(ORACLE_POLICY), "steps": [row()]}


class OracleTests(unittest.TestCase):
    def test_identical_oracles_match(self):
        value = fixture()
        self.assertIsNone(compare_oracles(value, copy.deepcopy(value)))

    def test_small_numeric_drift_is_tolerated(self):
        base = fixture(); trial = copy.deepcopy(base)
        trial["steps"][0][11] += 0.001
        self.assertIsNone(compare_oracles(base, trial))

    def test_exact_topk_or_large_numeric_drift_is_rejected(self):
        base = fixture(); trial = copy.deepcopy(base)
        trial["steps"][0][13][3] = 9
        self.assertIn("topk_ids differs", compare_oracles(base, trial) or "")
        trial = copy.deepcopy(base)
        trial["steps"][0][5] += 0.1
        self.assertIn("beyond tolerance", compare_oracles(base, trial) or "")

    def test_nonfinite_or_malformed_oracle_is_rejected(self):
        value = fixture(); value["steps"][0][3] = 1
        self.assertIn("non-finite", compare_oracles(value, value) or "")
        value = fixture(); value["steps"][0][13] = [2, 4]
        with self.assertRaisesRegex(LatticeError, "top-k identity"):
            validate_oracle(value)


if __name__ == "__main__":
    unittest.main()
