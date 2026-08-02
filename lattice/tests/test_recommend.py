from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lattice.common import atomic_write_json, canonical_json, sha256_bytes
from lattice.recommend import recommend
from lattice.workspace import Workspace


class RecommendTests(unittest.TestCase):
    def _workspace(self, root: Path):
        ws = Workspace(root); ws.initialize({"schema_version": 1, "model_fingerprint": "m", "runtime_fingerprint": "r",
            "suite_fingerprint": "placeholder", "suite": {"schema_version": 1, "name": "demo",
            "cases": [{"id": "case", "prompt": "x", "weight": 1, "context": 4096, "tokens": 8}]}})
        project = ws.load_project(); from lattice.suite import parse_suite
        project["suite_fingerprint"] = parse_suite(project["suite"]).fingerprint; atomic_write_json(ws.project_path, project)
        replay = {"prompt_ids": [1, 2], "full_ids": [1, 2, 3, 4]}; digest = sha256_bytes(canonical_json(replay))
        atomic_write_json(ws.replays_dir / "case.json", replay); return ws, digest

    def _session(self, ws, digest, repeats):
        return {"schema_version": 1, "id": "session-a", "status": "completed", "completed_at": "2026-08-02T00:00:00Z",
            "suite_fingerprint": ws.load_project()["suite_fingerprint"], "model_fingerprint": "m", "runtime_fingerprint": "r",
            "repeats": repeats, "candidates": [{"id": "baseline", "description": "base", "environment": {}},
            {"id": "fast", "description": "fast", "environment": {"PIPE": "1"}}],
            "replays": {"case": {"path": "replays/case.json", "sha256": digest, "prompt_tokens": 2, "continuation_tokens": 2}}, "run_ids": []}

    def _run(self, candidate, repeat, tok_s, digest, suffix=""):
        return {"schema_version": 1, "id": f"run-{candidate}-{repeat}{suffix}", "session_id": "session-a",
            "candidate_id": candidate, "case_id": "case", "repeat": repeat, "replay_sha256": digest,
            "candidate_environment": {} if candidate == "baseline" else {"PIPE": "1"}, "status": "success",
            "metrics": {"tok_s": tok_s}, "returncode": 0, "output_truncated": False}

    def test_fast_candidate_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            ws, digest = self._workspace(Path(directory)); session = self._session(ws, digest, 2)
            for candidate, values in (("baseline", [1.0, 1.0]), ("fast", [1.2, 1.2])):
                for repeat, value in enumerate(values):
                    run = self._run(candidate, repeat, value, digest); ws.write_run(run); session["run_ids"].append(run["id"])
            ws.write_session(session); profile, _ = recommend(ws, "session-a", min_runs=2, require_confidence=True)
            self.assertEqual(profile["winner"]["id"], "fast")

    def test_duplicate_success_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ws, digest = self._workspace(Path(directory)); session = self._session(ws, digest, 1)
            for run in [self._run("baseline", 0, 1.0, digest), self._run("baseline", 0, 1.0, digest, "-duplicate"), self._run("fast", 0, 1.2, digest)]:
                ws.write_run(run); session["run_ids"].append(run["id"])
            ws.write_session(session)
            with self.assertRaisesRegex(Exception, "duplicate successful run task"): recommend(ws, "session-a", min_runs=1)


if __name__ == "__main__": unittest.main()
