from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lattice.cli import main
from lattice.common import atomic_write_json
from lattice.workspace import Workspace


def write_safetensors(path: Path):
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")


class IntegrationTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "protocol fixture uses a POSIX executable script; Windows is covered by unit tests")
    def test_full_qualification_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            c_dir = repo / "c"
            model = root / "model"
            workspace_path = root / "workspace"
            c_dir.mkdir(parents=True)
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"glm_moe"}', encoding="utf-8")
            (model / "tokenizer.json").write_text('{}', encoding="utf-8")
            write_safetensors(model / "model-00001-of-00001.safetensors")
            (c_dir / "version.py").write_text('__version__="test"\n', encoding="utf-8")
            (c_dir / "autotune.py").write_text('# fixture\n', encoding="utf-8")
            (c_dir / "resource_plan.py").write_text(
                """
def build_plan(model, context=4096, policy='quality'):
    return {
      'cpu': {'physical_cores': 4, 'sockets': 1},
      'tiers': {
        'disk': {'cold_expert_bytes': 1024, 'available_bytes': 999999},
        'ram': {'cache_slots_per_layer': 2, 'budget_bytes': 1024},
        'vram': {'devices': []}
      },
      'expected_bottleneck': 'disk expert misses',
      'requested_context': context
    }
def environment_for_plan(plan, env=None, cuda_enabled=True):
    out=dict(env or {})
    out['RAM_GB']='8'
    return out
""".lstrip(), encoding="utf-8")
            (c_dir / "doctor.py").write_text(
                """
def run_doctor(model, ram, ctx, devices, vram, engine_path=None, deep=False, mirror_dir=None):
    return {'schema_version': 1, 'status': 'ok', 'checks': [{'id':'fixture','status':'pass'}]}
""".lstrip(), encoding="utf-8")
            coli = c_dir / "coli"
            coli.write_text(
                """#!/usr/bin/env python3
print('[PROMPT_TOKENS] 3: 1 2 3')
print('[TOKENS] 8 generated: 4 5 6 7 8 9 10 11')
""", encoding="utf-8")
            engine = c_dir / "colibri"
            engine.write_text(
                """#!/usr/bin/env python3
import os, sys
if os.environ.get('URING') == '1':
    print('fixture: io_uring unavailable', file=sys.stderr)
    raise SystemExit(2)
speed=1.0
if os.environ.get('PIPE') == '1': speed=1.15
if os.environ.get('DIRECT') == '1': speed=1.35
if os.environ.get('PILOT_REAL') == '1': speed=1.25
if os.environ.get('OMP_NUM_THREADS') == '2': speed=0.90
oracle_path=os.environ.get('REPLAY_ORACLE_OUT')
if not oracle_path:
    print('missing REPLAY_ORACLE_OUT', file=sys.stderr)
    raise SystemExit(2)
with open(oracle_path, 'w', encoding='utf-8') as oracle:
    for forced in range(4, 12):
        print(
            f'STEP\\tv2\\t{forced}\\t2\\t4\\t0\\t3\\t1.25\\t0.5\\t0.8\\t1.9',
            '\\t1\\t-2\\t3\\t-4\\t2,4,3,1,5,6,7,8',
            sep='',
            file=oracle,
        )
    print('SUMMARY\\tv2\\t8\\t8\\tseparate_replay_pass\\tprivate_file', file=oracle)
print('REPLAY_ORACLE_WRITTEN v2 steps=8 topk=8 measurement=separate_replay_pass transport=private_file')
print(f'REPLAY decode: 8 tokens | {speed:.2f} tok/s')
print('expert hit 70.0%')
print('latency p50 10.0 ms p99 20.0 ms')
""", encoding="utf-8")
            engine.chmod(0o755)
            suite_path = root / "suite.json"
            suite = {
                "schema_version": 1,
                "name": "investor-demo",
                "hourly_cost_usd": 2.0,
                "cases": [
                    {"id": "coding", "prompt": "Write a safe parser.", "tokens": 8, "weight": 2},
                    {"id": "analysis", "prompt": "Compare two systems.", "tokens": 8, "weight": 1, "context": 8192},
                ],
            }
            suite_path.write_text(json.dumps(suite), encoding="utf-8")

            self.assertEqual(main([
                "init", "--repo", str(repo), "--model", str(model), "--suite", str(suite_path),
                "--workspace", str(workspace_path),
            ]), 0)
            self.assertEqual(main([
                "qualify", "--workspace", str(workspace_path), "--repeats", "3", "--timeout", "10",
            ]), 0)
            workspace = Workspace(workspace_path)
            project = workspace.load_project()
            self.assertEqual(project["qualification_context"], 8192)
            self.assertEqual(project["plan"]["requested_context"], 8192)
            sessions = list(workspace.sessions_dir.glob("*.json"))
            self.assertEqual(len(sessions), 1)
            session = json.loads(sessions[0].read_text(encoding="utf-8"))
            self.assertEqual(session["status"], "completed")
            self.assertRegex(session["evidence_root_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(session["oracle_policy"]["schema"], "coli-replay-oracle/2")
            runs = workspace.list_runs(session["id"])
            self.assertEqual(len(runs), len(session["candidates"]) * 2 * 3)
            self.assertTrue(all("record_sha256" in run for run in runs))
            self.assertTrue(all(
                run["status"] != "success" or "oracle" in run["metrics"]
                for run in runs
            ))
            has_io_uring = any(candidate["id"] == "io-uring" for candidate in session["candidates"])
            expected_failures = 6 if has_io_uring else 0
            self.assertEqual(
                sum(run["status"] == "failed" for run in runs),
                expected_failures,
                [run.get("error") for run in runs if run["status"] == "failed"],
            )

            self.assertEqual(main([
                "recommend", "--workspace", str(workspace_path), "--session", session["id"],
                "--min-runs", "3", "--min-gain", "0.03", "--require-confidence",
            ]), 0)
            profile = workspace.load_profile()
            self.assertEqual(profile["winner"]["id"], "direct-pipeline")
            self.assertEqual(profile["evidence_root_sha256"], session["evidence_root_sha256"])
            self.assertEqual(profile["oracle_policy"], session["oracle_policy"])
            self.assertEqual(profile["statistics_policy"]["schema"], "lattice-statistics/2")
            self.assertEqual(profile["statistics_policy"]["bootstrap"], "stratified_paired_case_medians")
            self.assertGreater(profile["statistics_policy"]["per_candidate_confidence"], 0.90)
            report_path = root / "report.md"
            self.assertEqual(main([
                "report", "--workspace", str(workspace_path), "--output", str(report_path),
            ]), 0)
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Promoted `direct-pipeline`", report)
            self.assertIn("per million generated tokens", report)
            self.assertIn("Numerical replay consistency", report)
            self.assertIn("Stratified", report)
            self.assertIn("Bonferroni", report)
            self.assertIn(session["evidence_root_sha256"], report)
            self.assertEqual(main(["verify", "--workspace", str(workspace_path)]), 0)
            original_project = workspace.load_project()
            tampered = dict(original_project)
            tampered["qualification_environment"] = dict(original_project["qualification_environment"])
            tampered["qualification_environment"]["COLI_METAL"] = "1"
            atomic_write_json(workspace.project_path, tampered)
            self.assertEqual(main(["verify", "--workspace", str(workspace_path)]), 2)
            atomic_write_json(workspace.project_path, original_project)
            self.assertEqual(main(["verify", "--workspace", str(workspace_path)]), 0)
            self.assertEqual(main(["env", "--workspace", str(workspace_path), "--format", "json"]), 0)
            self.assertEqual(main(["launch", "--workspace", str(workspace_path), "--", "info"]), 0)


if __name__ == "__main__":
    unittest.main()
