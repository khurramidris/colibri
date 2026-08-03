import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bats_replay", ROOT / "tools" / "bats_replay.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BatsReplayTest(unittest.TestCase):
    def test_resident_candidate_beats_cold_candidate(self):
        record = next(MODULE.records(ROOT / "tests" / "bats_trace.jsonl"))
        result = MODULE.choose(record)
        self.assertEqual(result["selected"], [0])
        self.assertEqual(result["marginal_bytes"], 0)
        self.assertAlmostEqual(result["predicted_us"], 50.0)

    def test_union_cost_is_incremental(self):
        record = {
            "hardware": {"nvme": {"bandwidth_gbps": 4.0, "fixed_us": 80.0, "queue_us": 20.0}},
            "experts": [
                {"id": 0, "bytes": 1_000_000, "tier": "nvme"},
                {"id": 1, "bytes": 1_000_000, "tier": "nvme"},
                {"id": 2, "bytes": 1_000_000, "tier": "nvme"},
            ],
            "candidates": [
                {"id": 1, "expected_accepted_tokens": 2.0, "experts": [0, 1]},
                {"id": 2, "expected_accepted_tokens": 1.9, "experts": [1, 2]},
            ],
            "max_candidates": 2,
            "budget_us": 1_100.0,
        }
        result = MODULE.choose(record)
        self.assertEqual(result["selected"], [0, 1])
        self.assertEqual(result["marginal_bytes"], 3_000_000)
        self.assertAlmostEqual(result["predicted_us"], 1_050.0)


if __name__ == "__main__":
    unittest.main()
