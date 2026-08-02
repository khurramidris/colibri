from __future__ import annotations

import unittest

from lattice.report import render_report


class ReportContractTests(unittest.TestCase):
    def test_report_discloses_numerical_and_statistical_boundaries(self):
        constants = " ".join(
            str(value) for value in render_report.__code__.co_consts if isinstance(value, str)
        )
        self.assertIn("Numerical replay consistency", constants)
        self.assertIn("not complete logit equality", constants)
        self.assertIn("Stratified", constants)
        self.assertIn("Bonferroni", constants)
        self.assertIn("family-wise confidence", constants)
        self.assertIn("Session evidence root", constants)
        self.assertIn("screening methodology", constants)


if __name__ == "__main__":
    unittest.main()
