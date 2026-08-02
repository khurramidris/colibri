from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lattice.common import atomic_write_json, canonical_json, sha256_bytes
from lattice.recommend import recommend
from lattice.workspace import Workspace


class RecommendTests(unittest.TestCase):
    def _workspace(self, root: Path) -> tuple[Workspace, str]:
        ws = Workspace(root)
        ws.initialize({
            "schema_version": 1,
            "repo_root": "/fixture/repo",
            "model_path": "/fixture/model",
            "engine_path": "/fixture/engine",
            "model_family": "colibri",
            "model_fingerprint": "m",
            "runtime_fingerprint": "r",
            "hardware_fingerprint": "h",
            "execution_fingerprint": "e",
            "qualification_context": 4096,
            "qualification_environment": {},
            "storage_topology": {},
            "plan": {},
            "doctor": {"status": "ok"},
            "suite_fingerprint": "placeholder",
            "suite": {
                "schema_version": 1,
                "name": "demo",
                "cases": [{"id": "case", "prompt": "x", "weight": 1, "context": 4096, "tokens": 8}],
            },
        })
        project = ws.load_project()
        from lattice.suite import parse_suite
        project["suite_fingerprint"] = parse_suite(project["suite"]).fingerprint
        atomic_write_json(ws.project_path, project)
        replay = {"prompt_ids": [1, 2], "full_ids": [1, 2, 3, 4]}
        replay_hash = sha256_bytes(canonical_json(replay))
        replay_path = ws.replays_dir / "case.json"
        atomic_write_json(replay_path, replay)
        return ws, replay_hash

    def _session(self, ws: Workspace, replay_hash: str, repeats: int) -> dict:
        return {
            "schema_version": 1,
            "id": "session-a",
            "status": "completed",
            "completed_at": "2026-08-02T00:00:00Z",
            "suite_fingerprint": ws.load_project()["suite_fingerprint"],
            "model_fingerprint": "m",
            "runtime_fingerprint": "r",
            "hardware_fingerprint": "h",
            "execution_fingerprint": "e",
            "qualification_context": 4096,
            "repeats": repeats,
            "candidates": [
                {"id": "baseline", "description": "base", "environment": {}},
                {"id": "fast", "description": "fast", "environment": {"PIPE": "1"}},
            ],
            "replays": {
                "case": {
                    "path": "replays/case.json",
                    "sha256": replay_hash,
                    "prompt_tokens": 2,
                    "continuation_tokens": 2,
                }
            },
            "run_ids": [],
        }

    def _run(self, candidate: str, repeat: int, tok_s: float, replay_hash: str, suffix: str = "") -> dict:
        environment = {} if candidate == "baseline" else {"PIPE": "1"}
        return {
            "schema_version": 1,
            "id": f"run-{candidate}-{repeat}{suffix}",
            "session_id": "session-a",
            "candidate_id": candidate,
            "case_id": "case",
            "repeat": repeat,
            "replay_sha256": replay_hash,
            "execution_fingerprint": "e",
            "candidate_environment": environment,
            "status": "success",
            "metrics": {"tok_s": tok_s},
            "returncode": 0,
            "output_truncated": False,
        }

    def test_fast_candidate_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            ws, replay_hash = self._workspace(Path(directory))
            session = self._session(ws, replay_hash, repeats=2)
            for candidate, values in (("baseline", [1.0, 1.0]), ("fast", [1.2, 1.2])):
                for repeat, value in enumerate(values):
                    run = self._run(candidate, repeat, value, replay_hash)
                    ws.write_run(run)
                    session["run_ids"].append(run["id"])
            ws.write_session(session)
            profile, _ = recommend(ws, "session-a", min_runs=2, require_confidence=True)
            self.assertEqual(profile["winner"]["id"], "fast")

    def test_duplicate_success_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ws, replay_hash = self._workspace(Path(directory))
            session = self._session(ws, replay_hash, repeats=1)
            runs = [
                self._run("baseline", 0, 1.0, replay_hash),
                self._run("baseline", 0, 1.0, replay_hash, suffix="-duplicate"),
                self._run("fast", 0, 1.2, replay_hash),
            ]
            for run in runs:
                ws.write_run(run)
                session["run_ids"].append(run["id"])
            ws.write_session(session)
            with self.assertRaisesRegex(Exception, "duplicate successful run task"):
                recommend(ws, "session-a", min_runs=1)


if __name__ == "__main__":
    unittest.main()
