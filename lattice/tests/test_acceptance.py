from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from lattice.acceptance import (
    load_reference,
    parse_olmoe_output,
    prepare_olmoe_acceptance,
    render_acceptance_report,
    run_olmoe_acceptance,
    write_acceptance_outputs,
)
from lattice.common import LatticeError
from lattice.evidence import verify_record_digest


def write_safetensors(path: Path) -> None:
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")


def valid_output(*, matching: int = 3, total: int = 3) -> str:
    return (
        "resident weights loaded in 1.5s | RSS after load: 2.25 GB\n"
        f"Matching tokens: {matching}/{total}\n"
        "PEAK RSS: 3.50 GB\n"
        "Expert cache hit rate: 75.0%  (hit=30 miss=10)\n"
        f"Speed: 2.00 tok/s (1.5s for {total} tokens)\n"
    )


class AcceptanceTests(unittest.TestCase):
    def test_reference_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ref.json"
            path.write_text(json.dumps({"prompt_ids": [1, 2], "full_ids": [1, 2, 3]}), encoding="utf-8")
            self.assertEqual(load_reference(path, vocab_size=10)["full_ids"], [1, 2, 3])
            path.write_text(json.dumps({"prompt_ids": [1, 2], "full_ids": [1, 9, 3]}), encoding="utf-8")
            with self.assertRaisesRegex(LatticeError, "begin with prompt_ids"):
                load_reference(path, vocab_size=10)

    def test_output_parser_requires_complete_consistent_telemetry(self):
        metrics = parse_olmoe_output(valid_output())
        self.assertEqual(metrics["matching_tokens"], 3)
        self.assertEqual(metrics["tok_s"], 2.0)
        self.assertEqual(metrics["peak_rss_gb"], 3.5)
        with self.assertRaisesRegex(LatticeError, "speed token count"):
            parse_olmoe_output(valid_output(total=2).replace("for 2 tokens", "for 3 tokens"))

    @unittest.skipIf(os.name == "nt", "fake executable fixture uses a POSIX script")
    def test_acceptance_lifecycle_preserves_exact_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            c_dir = repo / "c"
            model = root / "model"
            c_dir.mkdir(parents=True)
            model.mkdir()
            (c_dir / "coli").write_text("# fixture\n", encoding="utf-8")
            engine = c_dir / "olmoe"
            engine.write_text(
                """#!/usr/bin/env python3
import os
assert os.environ['PILOT'] == '0'
assert os.environ['HOT'] == '0'
print('resident weights loaded in 1.5s | RSS after load: 2.25 GB')
print('Matching tokens: 3/3')
print('PEAK RSS: 3.50 GB')
print('Expert cache hit rate: 75.0%  (hit=30 miss=10)')
print('Speed: 2.00 tok/s (1.5s for 3 tokens)')
""",
                encoding="utf-8",
            )
            engine.chmod(0o755)
            (model / "config.json").write_text(
                json.dumps({"model_type": "olmoe", "vocab_size": 128}), encoding="utf-8"
            )
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            write_safetensors(model / "model-00000.safetensors")
            reference = root / "reference.json"
            reference.write_text(
                json.dumps({"prompt_ids": [1, 2], "full_ids": [1, 2, 3, 4, 5]}), encoding="utf-8"
            )
            prepared = prepare_olmoe_acceptance(
                repo,
                model,
                reference,
                repeats=3,
                timeout=10,
                cache_cap=4,
                quant_bits=8,
                threads=2,
            )
            record = run_olmoe_acceptance(prepared)
            self.assertEqual(record["status"], "accepted")
            self.assertEqual(record["summary"]["successful_runs"], 3)
            self.assertRegex(record["evidence_root_sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(all(verify_record_digest(run) for run in record["runs"]))
            report = render_acceptance_report(record)
            self.assertIn("TOKEN-EXACT-REFERENCE-REPLAY", report.upper())
            self.assertIn("Exact successful runs: 3/3", report)
            output = root / "acceptance.json"
            markdown = root / "acceptance.md"
            write_acceptance_outputs(record, output, markdown)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "accepted")
            self.assertIn("Scientific boundary", markdown.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(LatticeError, "overwrite immutable"):
                changed = dict(record)
                changed["status"] = "rejected"
                write_acceptance_outputs(changed, output)

    @unittest.skipIf(os.name == "nt", "fake executable fixture uses a POSIX script")
    def test_token_mismatch_is_rejected_and_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            c_dir = repo / "c"
            model = root / "model"
            c_dir.mkdir(parents=True)
            model.mkdir()
            (c_dir / "coli").write_text("# fixture\n", encoding="utf-8")
            engine = c_dir / "olmoe"
            engine.write_text(
                """#!/usr/bin/env python3
print('resident weights loaded in 1.0s | RSS after load: 2.0 GB')
print('Matching tokens: 2/3')
print('PEAK RSS: 3.0 GB')
print('Expert cache hit rate: 50.0%  (hit=10 miss=10)')
print('Speed: 1.0 tok/s (3.0s for 3 tokens)')
""",
                encoding="utf-8",
            )
            engine.chmod(0o755)
            (model / "config.json").write_text(
                json.dumps({"model_type": "olmoe", "vocab_size": 128}), encoding="utf-8"
            )
            (model / "tokenizer.json").write_text("{}", encoding="utf-8")
            write_safetensors(model / "model-00000.safetensors")
            reference = root / "reference.json"
            reference.write_text(
                json.dumps({"prompt_ids": [1, 2], "full_ids": [1, 2, 3, 4, 5]}), encoding="utf-8"
            )
            record = run_olmoe_acceptance(
                prepare_olmoe_acceptance(repo, model, reference, repeats=1, timeout=10)
            )
            self.assertEqual(record["status"], "rejected")
            self.assertIn("token mismatch", record["runs"][0]["error"])
            self.assertTrue(verify_record_digest(record["runs"][0]))


if __name__ == "__main__":
    unittest.main()
