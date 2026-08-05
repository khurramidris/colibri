import math
import unittest

from traffic_model import Inputs, evaluate


class TrafficModelTests(unittest.TestCase):
    def test_k3_planning_point(self):
        result = evaluate(
            Inputs(
                compressed_trunk_gib=35.0,
                usable_ram_gib=25.0,
                reserved_nontrunk_gib=7.0,
                expert_traffic_gib_per_token=24.1,
                exact_trunk_traffic_gib_per_token=101.3,
            )
        )
        self.assertEqual(result["resident_trunk_gib"], 18.0)
        self.assertEqual(result["streamed_trunk_gib_per_token"], 17.0)
        self.assertAlmostEqual(result["storage_traffic_speedup"], 125.4 / 41.1)
        self.assertGreater(result["amdahl_predicted_speedup"], 2.0)

    def test_full_trunk_resident(self):
        result = evaluate(
            Inputs(
                compressed_trunk_gib=20.0,
                usable_ram_gib=32.0,
                reserved_nontrunk_gib=8.0,
                expert_traffic_gib_per_token=10.0,
                exact_trunk_traffic_gib_per_token=40.0,
            )
        )
        self.assertEqual(result["streamed_trunk_gib_per_token"], 0.0)
        self.assertEqual(result["candidate_total_storage_gib_per_token"], 10.0)

    def test_no_ram_for_trunk(self):
        result = evaluate(
            Inputs(
                compressed_trunk_gib=35.0,
                usable_ram_gib=6.0,
                reserved_nontrunk_gib=7.0,
                expert_traffic_gib_per_token=20.0,
                exact_trunk_traffic_gib_per_token=100.0,
            )
        )
        self.assertEqual(result["resident_trunk_gib"], 0.0)
        self.assertEqual(result["streamed_trunk_gib_per_token"], 35.0)

    def test_prefetch_cannot_hurt_modelled_speed(self):
        base = Inputs(35.0, 25.0, 7.0, 24.1, 101.3)
        no_overlap = evaluate(base)["amdahl_predicted_speedup"]
        overlap = evaluate(
            Inputs(35.0, 25.0, 7.0, 24.1, 101.3, prefetch_overlap_fraction=0.5)
        )["amdahl_predicted_speedup"]
        self.assertGreaterEqual(overlap, no_overlap)

    def test_rejects_invalid_fraction(self):
        with self.assertRaises(ValueError):
            evaluate(Inputs(35.0, 25.0, 7.0, 24.1, 101.3, io_fraction_of_baseline=1.1))


if __name__ == "__main__":
    unittest.main()
