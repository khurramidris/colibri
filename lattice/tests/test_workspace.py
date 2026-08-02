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
            ws.write_run(run)
            with self.assertRaises(LatticeError):
                ws.write_run({**run, "status": "changed"})

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
