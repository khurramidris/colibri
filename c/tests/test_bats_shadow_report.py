import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bats_shadow_report", ROOT / "tools" / "bats_shadow_report.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BatsShadowReportTest(unittest.TestCase):
    def test_summary_separates_hits_and_misses(self):
        rows = [
            {
                "layer": 0, "row": 0, "misses": 2, "marginal_bytes": 8_000_000,
                "predicted_transfer_us": 2000.0, "miss_get_us": 2500.0,
            },
            {
                "layer": 0, "row": 1, "misses": 0, "marginal_bytes": 0,
                "predicted_transfer_us": 0.0, "miss_get_us": 0.0,
            },
            {
                "layer": 1, "row": 0, "misses": 1, "marginal_bytes": 4_000_000,
                "predicted_transfer_us": 1000.0, "miss_get_us": 1250.0,
            },
        ]
        result = MODULE.report(rows)
        overall = result["overall"]
        self.assertEqual(overall["rows"], 3)
        self.assertEqual(overall["miss_rows"], 2)
        self.assertEqual(overall["total_misses"], 3)
        self.assertEqual(overall["marginal_bytes"], 12_000_000)
        self.assertAlmostEqual(overall["prediction_to_actual_ratio"], 0.8)
        self.assertAlmostEqual(overall["mae_us"], 375.0)
        self.assertEqual(set(result["by_layer"]), {"0", "1"})

    def test_empty_input_is_explicit(self):
        result = MODULE.report([])
        self.assertEqual(result["overall"]["rows"], 0)
        self.assertIsNone(result["overall"]["mape"])


if __name__ == "__main__":
    unittest.main()
