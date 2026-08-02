from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lattice.common import atomic_write_json, canonical_json, sha256_bytes
from lattice.evidence import session_evidence_root
from lattice.oracle import ORACLE_POLICY, ORACLE_SCHEMA
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
            "oracle_policy": dict(ORACLE_POLICY),
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

    def _finalize(self, ws: Workspace, session: dict) -> None:
        session["evidence_root_sha256"] = session_evidence_root(
            session, ws.list_runs(session["id"])
        )
        ws.write_session(session)

    def _oracle(self, *, top1: int = 2, forced_logit: float = 1.25) -> dict:
        return {
            "schema": ORACLE_SCHEMA,
            "policy": dict(ORACLE_POLICY),
            "steps": [{
                "step": 0, "forced": 3, "top1": top1, "top2": 4, "nonfinite": 0,
                "top1_logit": 3.0, "forced_logit": forced_logit, "margin": 0.5,
                "mean": 0.8, "rms": 1.9,
                "projection_0": 1.0, "projection_1": -2.0,
                "projection_2": 3.0, "projection_3": -4.0,
                "topk_ids_hash": "0123456789abcdef",
            }],
        }

    def _run(self, candidate: str, repeat: int, tok_s: float, replay_hash: str, suffix: str = "", *, top1: int = 2) -> dict:
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
            "metrics": {"tok_s": tok_s, "oracle": self._oracle(top1=top1)},
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
            self._finalize(ws, session)
            profile, _ = recommend(ws, "session-a", min_runs=2, require_confidence=True)
            self.assertEqual(profile["winner"]["id"], "fast")

    def test_numerical_oracle_mismatch_retains_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            ws, replay_hash = self._workspace(Path(directory))
            session = self._session(ws, replay_hash, repeats=2)
            for repeat in range(2):
                baseline = self._run("baseline", repeat, 1.0, replay_hash)
                fast = self._run("fast", repeat, 2.0, replay_hash, top1=9)
                for run in (baseline, fast):
                    ws.write_run(run)
                    session["run_ids"].append(run["id"])
            self._finalize(ws, session)
            profile, scores = recommend(ws, "session-a", min_runs=2)
            self.assertTrue(profile["baseline_retained"])
            self.assertEqual(profile["winner"]["id"], "baseline")
            self.assertIn("numerical oracle mismatch", scores[0].reason)

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
            self._finalize(ws, session)
            with self.assertRaisesRegex(Exception, "duplicate successful run task"):
                recommend(ws, "session-a", min_runs=1)


if __name__ == "__main__":
    unittest.main()
