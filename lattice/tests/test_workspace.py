from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lattice.common import LatticeError
from lattice.workspace import Workspace


class WorkspaceTests(unittest.TestCase):
    def test_immutable_run_is_idempotent_but_not_mutable(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(Path(directory))
            ws.initialize({"schema_version": 1})
            run = {"schema_version": 1, "id": "run-abc"}
            ws.write_run(run)
            self.assertRegex(run["record_sha256"], r"^[0-9a-f]{64}$")
            ws.write_run(run)
            with self.assertRaises(LatticeError):
                changed = dict(run)
                changed["status"] = "changed"
                ws.write_run(changed)

    def test_run_digest_is_verified_on_read(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(Path(directory))
            ws.initialize({"schema_version": 1})
            run = {"schema_version": 1, "id": "run-abc", "metrics": {"tok_s": 1.0}}
            path = ws.write_run(run)
            text = path.read_text(encoding="utf-8").replace('"tok_s": 1.0', '"tok_s": 9.0')
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(LatticeError, "digest mismatch"):
                ws.list_runs()

    def test_persisted_ids_are_validated_before_path_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(Path(directory))
            ws.initialize({"schema_version": 1})
            with self.assertRaisesRegex(LatticeError, "invalid session id"):
                ws.load_session("../../escape")
            with self.assertRaisesRegex(LatticeError, "invalid profile id"):
                ws.load_profile("../escape")
            with self.assertRaisesRegex(LatticeError, "invalid run id"):
                ws.write_run({"schema_version": 1, "id": "../escape"})
            self.assertFalse((Path(directory).parent / "escape.json").exists())

    def test_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(Path(directory))
            ws.initialize({"schema_version": 1})
            with ws.acquire_lock("x"):
                with self.assertRaises(LatticeError):
                    with ws.acquire_lock("x"):
                        pass

    def test_profile_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(Path(directory))
            ws.initialize({"schema_version": 1})
            profile = {"schema_version": 1, "id": "profile-a"}
            ws.write_profile(profile)
            self.assertEqual(ws.load_profile()["id"], "profile-a")

    def test_project_loader_rejects_incomplete_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(Path(directory))
            ws.initialize({"schema_version": 1})
            with self.assertRaisesRegex(LatticeError, "missing required fields"):
                ws.load_project()

    def test_force_cannot_orphan_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(Path(directory))
            ws.initialize({"schema_version": 1, "name": "old"})
            ws.write_run({"schema_version": 1, "id": "run-a"})
            with self.assertRaisesRegex(LatticeError, "contains evidence"):
                ws.initialize({"schema_version": 1, "name": "new"}, force=True)


if __name__ == "__main__":
    unittest.main()
