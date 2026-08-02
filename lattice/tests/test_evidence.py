from __future__ import annotations

import unittest

from lattice.common import LatticeError
from lattice.evidence import seal_record, session_evidence_root, verify_record_digest


class EvidenceTests(unittest.TestCase):
    def test_record_digest_detects_edit(self):
        record = seal_record({"schema_version": 1, "id": "run-a", "metrics": {"tok_s": 1.0}})
        verify_record_digest(record, "fixture")
        record["metrics"]["tok_s"] = 9.0
        with self.assertRaisesRegex(LatticeError, "digest mismatch"):
            verify_record_digest(record, "fixture")

    def test_session_root_binds_runs_and_replays(self):
        run = seal_record({"schema_version": 1, "id": "run-a", "status": "success"})
        session = {
            "schema_version": 1,
            "id": "session-a",
            "status": "completed",
            "run_ids": ["run-a"],
            "replays": {"case": {"sha256": "a" * 64}},
        }
        first = session_evidence_root(session, [run])
        changed = dict(session)
        changed["replays"] = {"case": {"sha256": "b" * 64}}
        self.assertNotEqual(first, session_evidence_root(changed, [run]))


if __name__ == "__main__":
    unittest.main()
