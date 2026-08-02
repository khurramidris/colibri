from __future__ import annotations

import inspect
import unittest

from lattice.report import render_report


class ReportContractTests(unittest.TestCase):
    def test_report_discloses_numerical_and_statistical_boundaries(self):
        source = inspect.getsource(render_report)
        self.assertIn("Numerical replay consistency", source)
        self.assertIn("not complete logit equality", source)
        self.assertIn("Stratified", source)
        self.assertIn("Bonferroni", source)
        self.assertIn("family-wise confidence", source)
        self.assertIn("Session evidence root", source)
        self.assertIn("screening methodology", source)


if __name__ == "__main__":
    unittest.main()
