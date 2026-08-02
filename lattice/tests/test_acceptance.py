from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from lattice.acceptance import (
    load_and_verify_olmoe_acceptance,
    load_reference,
    parse_olmoe_output,
    prepare_olmoe_acceptance,
    render_acceptance_report,
    run_olmoe_acceptance,
    verify_olmoe_acceptance,
    write_acceptance_outputs,
)
from lattice.common import LatticeError, atomic_write_json
from lattice.evidence import seal_record, verify_record_digest


def write_safetensors(path: Path) -> None:
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")


def reference_payload(
    *,
    prompt_ids: list[int] | None = None,
    continuation: list[int] | None = None,
) -> dict:
    prompt_ids = list(prompt_ids or [1, 2])
    continuation = list(continuation or [3, 4, 5])
    return {
        "schema_version": 2,
        "generator": "transformers-greedy",
        "model": "allenai/fixture-olmoe",
        "source_kind": "huggingface",
        "requested_revision": "main",
        "resolved_revision": "a" * 40,
        "source_fingerprint": None,
        "versions": {
            "python": "3.13.5",
            "torch": "2.9.0",
            "transformers": "4.57.0",
        },
        "platform": {"system": "fixture", "release": "fixture", "machine": "x86_64"},
        "device_request": "cpu",
        "dtype_request": "float32",
        "template_mode": "plain_prompt",
        "prompt_sha256": "b" * 64,
        "requested_new_tokens": len(continuation),
        "actual_new_tokens": len(continuation),
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": len(continuation),
            "pad_token_id": 0,
        },
        "prompt_ids": prompt_ids,
        "full_ids": prompt_ids + continuation,
    }


def valid_output(
    *,
    reference: list[int] | None = None,
    engine: list[int] | None = None,
    matching: int | None = None,
    hit_pct: float = 75.0,
    hits: int = 30,
    misses: int = 10,
    tok_s: float = 2.0,
    seconds: float = 1.5,
    load_rss: float = 2.25,
    peak_rss: float = 3.5,
    include_pin_marker: bool = True,
) -> str:
    reference = list(reference or [3, 4, 5])
    engine = list(engine or reference)
    matching = (
        sum(left == right for left, right in zip(reference, engine))
        if matching is None else matching
    )
    lines = [
        f"resident weights loaded in 1.5s | RSS after load: {load_rss:.2f} GB",
    ]
    if include_pin_marker:
        lines.append("OLMOE_PERSISTED_PINS_DISABLED")
    lines.extend([
        "Reference: " + " ".join(map(str, reference)),
        "C engine : " + " ".join(map(str, engine)),
        f"Matching tokens: {matching}/{len(reference)}",
        f"PEAK RSS: {peak_rss:.2f} GB",
        f"Expert cache hit rate: {hit_pct:.1f}%  (hit={hits} miss={misses})",
        f"Speed: {tok_s:.2f} tok/s ({seconds:.1f}s for {len(reference)} tokens)",
    ])
    return "\n".join(lines) + "\n"


def prepare_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    repo = root / "repo"
    c_dir = repo / "c"
    model = root / "model"
    c_dir.mkdir(parents=True)
    model.mkdir()
    (c_dir / "coli").write_text("# fixture\n", encoding="utf-8")
    engine = c_dir / "olmoe"
    engine.write_text("fixture", encoding="utf-8")
    (model / "config.json").write_text(
        json.dumps({"model_type": "olmoe", "vocab_size": 128}), encoding="utf-8"
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    write_safetensors(model / "model-00000.safetensors")
    reference = root / "reference.json"
    atomic_write_json(reference, reference_payload())
    return repo, model, engine, reference


class AcceptanceTests(unittest.TestCase):
    def test_reference_requires_versioned_transformers_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            atomic_write_json(path, reference_payload())
            loaded = load_reference(path, vocab_size=128)
            self.assertEqual(loaded["full_ids"], [1, 2, 3, 4, 5])
            self.assertEqual(loaded["provenance"]["resolved_revision"], "a" * 40)

            old = reference_payload()
            old["schema_version"] = 1
            atomic_write_json(path, old)
            with self.assertRaisesRegex(LatticeError, "version-2 Transformers"):
                load_reference(path, vocab_size=128)

            unpinned = reference_payload()
            unpinned["resolved_revision"] = "main"
            atomic_write_json(path, unpinned)
            with self.assertRaisesRegex(LatticeError, "40-character Hub commit"):
                load_reference(path, vocab_size=128)

            sampled = reference_payload()
            sampled["generation"]["do_sample"] = True
            atomic_write_json(path, sampled)
            with self.assertRaisesRegex(LatticeError, "greedy contract"):
                load_reference(path, vocab_size=128)

    def test_local_reference_requires_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            payload = reference_payload()
            payload.update({
                "source_kind": "local",
                "resolved_revision": None,
                "source_fingerprint": "c" * 64,
            })
            atomic_write_json(path, payload)
            self.assertEqual(
                load_reference(path)["provenance"]["source_fingerprint"],
                "c" * 64,
            )
            payload["source_fingerprint"] = None
            atomic_write_json(path, payload)
            with self.assertRaisesRegex(LatticeError, "source fingerprint"):
                load_reference(path)

    def test_output_parser_requires_complete_consistent_telemetry(self):
        metrics = parse_olmoe_output(valid_output())
        self.assertEqual(metrics["matching_tokens"], 3)
        self.assertEqual(metrics["engine_tokens"], [3, 4, 5])
        self.assertEqual(metrics["tok_s"], 2.0)
        self.assertTrue(metrics["persisted_pins_disabled"])

        with self.assertRaisesRegex(LatticeError, "persisted-pin isolation marker"):
            parse_olmoe_output(valid_output(include_pin_marker=False))
        with self.assertRaisesRegex(LatticeError, "match counter disagrees"):
            parse_olmoe_output(
                valid_output(engine=[3, 4, 9], matching=3)
            )
        with self.assertRaisesRegex(LatticeError, "hit percentage disagrees"):
            parse_olmoe_output(valid_output(hit_pct=50.0, hits=30, misses=10))
        with self.assertRaisesRegex(LatticeError, "throughput disagrees"):
            parse_olmoe_output(valid_output(tok_s=9.0, seconds=1.5))
        with self.assertRaisesRegex(LatticeError, "peak RSS is below"):
            parse_olmoe_output(valid_output(load_rss=3.0, peak_rss=2.0))
        duplicate = valid_output() + "Speed: 2.00 tok/s (1.5s for 3 tokens)\n"
        with self.assertRaisesRegex(LatticeError, "exactly one speed telemetry"):
            parse_olmoe_output(duplicate)

    def test_prepare_rejects_zero_cache_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, model, engine, reference = prepare_fixture(Path(directory))
            with self.assertRaisesRegex(LatticeError, "between 1 and 512"):
                prepare_olmoe_acceptance(
                    repo, model, reference, engine=engine, cache_cap=0
                )

    def test_reference_drift_stops_before_engine_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, model, engine, reference = prepare_fixture(Path(directory))
            prepared = prepare_olmoe_acceptance(
                repo, model, reference, engine=engine, repeats=1, timeout=10
            )
            atomic_write_json(reference, reference_payload(continuation=[8, 9]))
            with self.assertRaisesRegex(LatticeError, "identity changed.*reference"):
                run_olmoe_acceptance(prepared)

    @unittest.skipIf(os.name == "nt", "fake executable fixture uses a POSIX script")
    def test_acceptance_lifecycle_is_sealed_and_offline_verifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, model, engine, reference = prepare_fixture(root)
            engine.write_text(
                """#!/usr/bin/env python3
import os
assert os.environ['PILOT'] == '0'
assert os.environ['HOT'] == '0'
assert os.environ['IDOT'] == '0'
assert os.environ['OLMOE_IGNORE_PERSISTED_PINS'] == '1'
print('resident weights loaded in 1.5s | RSS after load: 2.25 GB')
print('OLMOE_PERSISTED_PINS_DISABLED')
print('Reference: 3 4 5')
print('C engine : 3 4 5')
print('Matching tokens: 3/3')
print('PEAK RSS: 3.50 GB')
print('Expert cache hit rate: 75.0%  (hit=30 miss=10)')
print('Speed: 2.00 tok/s (1.5s for 3 tokens)')
""",
                encoding="utf-8",
            )
            engine.chmod(0o755)
            record = run_olmoe_acceptance(
                prepare_olmoe_acceptance(
                    repo, model, reference, repeats=3, timeout=10,
                    cache_cap=4, quant_bits=8, threads=2,
                )
            )
            self.assertEqual(record["status"], "accepted")
            self.assertEqual(record["summary"]["successful_runs"], 3)
            self.assertRegex(record["record_sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(verify_record_digest(record))
            self.assertTrue(all(verify_record_digest(run) for run in record["runs"]))
            self.assertTrue(all(run["stdout_bytes"] > 0 for run in record["runs"]))
            verification = verify_olmoe_acceptance(record)
            self.assertEqual(verification["status"], "ok")
            self.assertEqual(verification["result"], "accepted")

            report = render_acceptance_report(record)
            self.assertIn("single-reference-token-exact-execution", report)
            self.assertIn("not a production-readiness", report)
            self.assertIn("Persisted pin state disabled", report)
            self.assertIn("Reference provenance", report)

            output = root / "acceptance.json"
            markdown = root / "acceptance.md"
            write_acceptance_outputs(record, output, markdown)
            loaded_verification = load_and_verify_olmoe_acceptance(output)
            self.assertEqual(loaded_verification["record_sha256"], record["record_sha256"])
            self.assertIn("Scientific boundary", markdown.read_text(encoding="utf-8"))
            write_acceptance_outputs(record, output, markdown)

            changed = copy.deepcopy(record)
            changed["status"] = "rejected"
            with self.assertRaisesRegex(LatticeError, "digest mismatch"):
                verify_olmoe_acceptance(changed)
            changed = copy.deepcopy(record)
            changed["runs"][0]["metrics"]["engine_tokens"][-1] = 9
            changed["runs"][0] = seal_record(changed["runs"][0])
            changed = seal_record(changed)
            with self.assertRaisesRegex(LatticeError, "not token-exact|evidence root mismatch"):
                verify_olmoe_acceptance(changed)

    @unittest.skipIf(os.name == "nt", "fake executable fixture uses a POSIX script")
    def test_token_mismatch_is_rejected_and_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            repo, model, engine, reference = prepare_fixture(Path(directory))
            engine.write_text(
                """#!/usr/bin/env python3
print('resident weights loaded in 1.0s | RSS after load: 2.0 GB')
print('OLMOE_PERSISTED_PINS_DISABLED')
print('Reference: 3 4 5')
print('C engine : 3 4 9')
print('Matching tokens: 2/3')
print('PEAK RSS: 3.0 GB')
print('Expert cache hit rate: 50.0%  (hit=10 miss=10)')
print('Speed: 1.0 tok/s (3.0s for 3 tokens)')
""",
                encoding="utf-8",
            )
            engine.chmod(0o755)
            record = run_olmoe_acceptance(
                prepare_olmoe_acceptance(repo, model, reference, repeats=1, timeout=10)
            )
            self.assertEqual(record["status"], "rejected")
            self.assertIn("token mismatch", record["runs"][0]["error"])
            self.assertIsNone(record["runs"][0]["metrics"])
            self.assertTrue(verify_record_digest(record["runs"][0]))
            self.assertEqual(verify_olmoe_acceptance(record)["result"], "rejected")


if __name__ == "__main__":
    unittest.main()
