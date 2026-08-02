from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lattice.candidates import Candidate
from lattice.colibri import ColibriContext
from lattice.common import LatticeError
from lattice.experiment import create_project, resume_experiment, run_experiment
from lattice.process import ProcessResult
from lattice.oracle import ORACLE_POLICY, ORACLE_SCHEMA
from lattice.suite import parse_suite
from lattice.workspace import Workspace


def result(stdout: str = "") -> ProcessResult:
    return ProcessResult(("fixture",), 0, stdout, "", False, 0.01, False)


class ResumeTests(unittest.TestCase):
    def test_resume_continues_partial_calibration_and_trials(self):
        suite = parse_suite({
            "schema_version": 1,
            "name": "resume",
            "cases": [
                {"id": "first", "prompt": "one", "tokens": 4},
                {"id": "second", "prompt": "two", "tokens": 4},
            ],
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = Workspace(root / "workspace")
            context = ColibriContext(
                repo_root=root, c_dir=root, coli=root / "coli", engine=root / "engine",
                model=root / "model", model_family="colibri",
                runtime_fingerprint="runtime", model_fingerprint="model",
                plan={"cpu": {"physical_cores": 1, "sockets": 1}, "tiers": {"disk": {}, "ram": {}, "vram": {}}},
                doctor={"status": "ok"}, base_environment={},
                qualification_context=4096, qualification_environment={},
                execution_fingerprint="execution", hardware_fingerprint="hardware",
                storage_topology={},
            )
            workspace.initialize(create_project(context, suite))
            calls = []

            def calibrate(_context, *, prompt, tokens, ctx, timeout):
                calls.append(prompt)
                if prompt == "two" and calls.count("two") == 1:
                    raise LatticeError("fixture interruption")
                replay = {"prompt_ids": [1, 2], "full_ids": [1, 2, 3, 4]}
                return replay, result("calibration")

            def replay(_context, *, replay_path, candidate, ctx, timeout):
                oracle = {
                    "schema": ORACLE_SCHEMA,
                    "policy": dict(ORACLE_POLICY),
                    "steps": [{
                        "step": 0, "forced": 3, "top1": 2, "top2": 4, "nonfinite": 0,
                        "top1_logit": 3.0, "forced_logit": 1.25, "margin": 0.5,
                        "mean": 0.8, "rms": 1.9,
                        "projection_0": 1.0, "projection_1": -2.0,
                        "projection_2": 3.0, "projection_3": -4.0,
                        "topk_ids_hash": "0123456789abcdef",
                    }],
                }
                return {"tok_s": 1.0, "hit_pct": 50.0, "p50_ms": 1.0, "p99_ms": 2.0, "oracle": oracle}, result("replay")

            with patch("lattice.experiment.calibrate_case", side_effect=calibrate), \
                    patch("lattice.experiment.run_replay", side_effect=replay):
                with self.assertRaisesRegex(LatticeError, "fixture interruption"):
                    run_experiment(
                        workspace, context, suite, repeats=1, timeout=5,
                        candidates=(Candidate("baseline", "baseline", {}),),
                    )
                sessions = list(workspace.sessions_dir.glob("*.json"))
                self.assertEqual(len(sessions), 1)
                session_id = sessions[0].stem
                interrupted = workspace.load_session(session_id)
                self.assertEqual(interrupted["status"], "interrupted")
                self.assertEqual(set(interrupted["replays"]), {"first"})
                self.assertEqual(interrupted["run_ids"], [])

                completed = resume_experiment(
                    workspace, context, suite, session_id, timeout=5,
                )
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(set(completed["replays"]), {"first", "second"})
                self.assertEqual(len(completed["run_ids"]), 2)
                self.assertEqual(calls.count("one"), 1)
                self.assertEqual(calls.count("two"), 2)


if __name__ == "__main__":
    unittest.main()
