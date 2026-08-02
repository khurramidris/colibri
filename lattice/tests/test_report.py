from __future__ import annotations

import unittest

from lattice.report import render_report


class ReportContractTests(unittest.TestCase):
    def test_report_source_states_numerical_sketch_not_quality_equivalence(self):
        constants = " ".join(
            str(value) for value in render_report.__code__.co_consts if isinstance(value, str)
        )
        self.assertIn("Numerical replay consistency", constants)
        self.assertIn("not complete logit equality", constants)
        self.assertIn("current tolerances still require calibration", constants)
        self.assertIn("Session evidence root", constants)
        self.assertIn("Absolute tolerance", constants)


if __name__ == "__main__":
    unittest.main()
