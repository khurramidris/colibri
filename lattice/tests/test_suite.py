from __future__ import annotations

import unittest

from lattice.common import LatticeError
from lattice.suite import parse_suite


class SuiteTests(unittest.TestCase):
    def test_valid_suite_is_canonical(self):
        suite = parse_suite({"schema_version": 1, "name": "demo", "default_context": 2048,
            "default_tokens": 8, "hourly_cost_usd": 1.25,
            "cases": [{"id": "coding", "prompt": "write code", "weight": 2},
                      {"id": "legal", "prompt": "summarize", "context": 4096, "tokens": 16}]})
        self.assertEqual(suite.cases[0].context, 2048)
        self.assertEqual(suite.cases[1].tokens, 16)
        self.assertEqual(suite.hourly_cost_usd, 1.25)
        self.assertEqual(len(suite.fingerprint), 64)

    def test_duplicate_case_rejected(self):
        with self.assertRaisesRegex(LatticeError, "duplicate"):
            parse_suite({"schema_version": 1, "name": "bad",
                "cases": [{"id": "same", "prompt": "a"}, {"id": "same", "prompt": "b"}]})

    def test_path_like_id_rejected(self):
        with self.assertRaises(LatticeError):
            parse_suite({"schema_version": 1, "name": "bad", "cases": [{"id": "../escape", "prompt": "x"}]})

    def test_non_finite_weight_rejected(self):
        with self.assertRaises(LatticeError):
            parse_suite({"schema_version": 1, "name": "bad", "cases": [{"id": "x", "prompt": "x", "weight": float("inf")}]})


if __name__ == "__main__":
    unittest.main()
