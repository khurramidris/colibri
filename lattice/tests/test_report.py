from __future__ import annotations

import unittest

from lattice.report import render_report


class ReportContractTests(unittest.TestCase):
    def test_report_source_states_replay_not_quality_equivalence(self):
        constants = " ".join(
            str(value) for value in render_report.__code__.co_consts if isinstance(value, str)
        )
        self.assertIn("Deterministic replay consistency", constants)
        self.assertIn("does not prove equal logits", constants)
        self.assertIn("Session evidence root", constants)


if __name__ == "__main__":
    unittest.main()
