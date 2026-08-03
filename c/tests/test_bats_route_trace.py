import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bats_route_trace", ROOT / "tools" / "bats_route_trace.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BatsRouteTraceTest(unittest.TestCase):
    def test_parse_and_globalize_experts(self):
        routes = [
            MODULE.parse_route_line("0 0 2 3:0.7000 5:0.3000"),
            MODULE.parse_route_line("0 1 2 3:0.8000 7:0.2000"),
        ]
        records = list(
            MODULE.build_records(
                routes,
                n_experts=8,
                expert_bytes=4_000_000,
                default_tier="nvme",
                hardware={"nvme": {"bandwidth_gbps": 4.0}, "exec": {"bandwidth_gbps": 1.0}},
                residency={(2, 3): {"tier": "exec", "resident_in_exec": True}},
                max_candidates=1,
            )
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["candidates"][0]["experts"], [19, 21])
        self.assertEqual(record["candidates"][1]["experts"], [19, 23])
        self.assertTrue(next(e for e in record["experts"] if e["id"] == 19)["resident_in_exec"])
        self.assertTrue(record["_meta"]["default_acceptance"])

    def test_acceptance_annotations(self):
        route = MODULE.parse_route_line("4 9 1 0:1.0000")
        records = list(
            MODULE.build_records(
                [route],
                n_experts=4,
                expert_bytes=100,
                default_tier="ram",
                hardware={"ram": {"bandwidth_gbps": 10.0}, "exec": {"bandwidth_gbps": 1.0}},
                acceptance={(4, 9): {
                    "expected_accepted_tokens": 1.75,
                    "verify_compute_us": 11.0,
                    "kv_us": 3.0,
                }},
            )
        )
        candidate = records[0]["candidates"][0]
        self.assertEqual(candidate["expected_accepted_tokens"], 1.75)
        self.assertEqual(candidate["verify_compute_us"], 11.0)
        self.assertFalse(records[0]["_meta"]["default_acceptance"])

    def test_residency_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "resident.txt"
            path.write_text("2 3 exec\n2 5 nvme 17.5\n", encoding="utf-8")
            result = MODULE.read_residency(path)
        self.assertTrue(result[(2, 3)]["resident_in_exec"])
        self.assertTrue(result[(2, 5)]["in_flight"])
        self.assertEqual(result[(2, 5)]["remaining_us"], 17.5)

    def test_writer_emits_jsonl(self):
        out = io.StringIO()
        MODULE.write_records([{"a": 1}, {"b": 2}], out, pretty=False)
        rows = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(rows, [{"a": 1}, {"b": 2}])


if __name__ == "__main__":
    unittest.main()
