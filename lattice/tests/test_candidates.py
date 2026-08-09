from __future__ import annotations

import unittest

from lattice.candidates import default_candidates


class CandidateTests(unittest.TestCase):
    def test_candidates_duplicate_of_effective_baseline_are_removed(self):
        plan = {
            "cpu": {"physical_cores": 4, "sockets": 1},
            "tiers": {"disk": {"cold_expert_bytes": 1024}, "vram": {"devices": []}},
        }
        candidates = default_candidates(plan, {"PIPE": "1"})
        identifiers = {candidate.id for candidate in candidates}
        self.assertNotIn("io-pipeline", identifiers)
        self.assertIn("direct-pipeline", identifiers)


if __name__ == "__main__":
    unittest.main()
