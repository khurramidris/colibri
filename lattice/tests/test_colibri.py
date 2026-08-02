from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lattice.colibri import (
    clean_environment,
    create_context,
    detect_family,
    fingerprint_model,
    fingerprint_runtime,
    fingerprint_storage_topology,
    qualification_environment,
    parse_calibration,
    parse_replay_metrics,
    parse_replay_oracle,
    read_replay_oracle_file,
)
from lattice.common import LatticeError


def write_safetensors(path: Path, tensors=None):
    tensors = tensors or {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    header = json.dumps(tensors, separators=(",", ":")).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00")


class ColibriTests(unittest.TestCase):
    def test_family_detection(self):
        self.assertEqual(detect_family({"model_type": "kimi_k3"}), "kimi_k3")
        self.assertEqual(detect_family({"architectures": ["InklingForCausalLM"]}), "inkling")
        self.assertEqual(detect_family({"model_type": "olmoe"}), "olmoe")
        self.assertEqual(detect_family({"model_type": "glm_moe"}), "colibri")
        with self.assertRaisesRegex(LatticeError, "unsupported or unrecognized"):
            detect_family({"model_type": "mystery_model"})

    def test_non_glm_qualification_is_refused_until_adapter_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c_dir = root / "c"
            model = root / "model"
            c_dir.mkdir(); model.mkdir()
            (c_dir / "coli").write_text("# fixture\n")
            engine = c_dir / "kimi_k3"
            engine.write_text("fixture")
            (model / "config.json").write_text('{"model_type":"kimi_k3"}')
            with self.assertRaisesRegex(LatticeError, "GLM/colibri"):
                create_context(root, model, engine=engine, deep=False)

    def test_model_fingerprint_uses_headers_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"glm"}', encoding="utf-8")
            (root / "tokenizer.json").write_text('{}', encoding="utf-8")
            shard = root / "a.safetensors"
            write_safetensors(shard)
            before = fingerprint_model(root)
            data = shard.read_bytes()
            shard.write_bytes(data[:-1] + b"\x01")
            self.assertNotEqual(before, fingerprint_model(root))
            changed = fingerprint_model(root)
            shard.write_bytes(data + b"\x00")
            self.assertNotEqual(changed, fingerprint_model(root))

    def test_storage_topology_fingerprint_covers_mirror_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            mirror = root / "mirror"
            primary.mkdir(); mirror.mkdir()
            (primary / "config.json").write_text('{"model_type":"glm"}')
            (primary / "tokenizer.json").write_text('{}')
            write_safetensors(primary / "a.safetensors")
            write_safetensors(mirror / "a.safetensors")
            before = fingerprint_storage_topology(primary, {"COLI_MODEL_MIRROR": str(mirror)})
            shard = mirror / "a.safetensors"
            payload = shard.read_bytes()
            shard.write_bytes(payload[:-1] + b"\x01")
            after = fingerprint_storage_topology(primary, {"COLI_MODEL_MIRROR": str(mirror)})
            self.assertNotEqual(before["fingerprint"], after["fingerprint"])

    def test_quality_environment_is_stripped_but_backend_is_attested(self):
        env = clean_environment({
            "PATH": "/safe/bin",
            "COLI_TEMP": "0.8",
            "IDOT": "0",
            "COLI_METAL": "1",
            "COLI_UNREVIEWED_APPROX": "1",
            "LD_PRELOAD": "/tmp/inject.so",
            "PYTHONPATH": "/tmp/import-inject",
            "OPENAI_API_KEY": "secret",
            "KMP_AFFINITY": "scatter",
        })
        self.assertNotIn("COLI_TEMP", env)
        self.assertNotIn("IDOT", env)
        self.assertNotIn("COLI_UNREVIEWED_APPROX", env)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("KMP_AFFINITY", env)
        snapshot = qualification_environment(env)
        self.assertEqual(snapshot["COLI_METAL"], "1")
        self.assertEqual(snapshot["PATH"], "/safe/bin")

    def test_unknown_or_semantic_override_is_rejected(self):
        with self.assertRaisesRegex(LatticeError, "invalid qualification environment override"):
            clean_environment({}, {"COLI_UNREVIEWED_APPROX": "1"})
        with self.assertRaisesRegex(LatticeError, "DRAFT must remain"):
            clean_environment({}, {"DRAFT": "3"})
        env = clean_environment({}, {"DRAFT": "0", "PIN_GB": "all", "COLI_CUDA": "1"})
        snapshot = qualification_environment(env)
        self.assertEqual(snapshot["DRAFT"], "0")
        self.assertEqual(snapshot["PIN_GB"], "all")

    def test_out_of_bounds_tensor_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{}', encoding="utf-8")
            (root / "tokenizer.json").write_text('{}', encoding="utf-8")
            write_safetensors(root / "a.safetensors", {"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 999]}})
            with self.assertRaises(LatticeError):
                fingerprint_model(root)

    def test_clean_environment_removes_quality_and_adaptive_keys(self):
        env = clean_environment({"PATH": "/bin", "TOPK": "4", "PIPE": "1", "PIN": "auto"})
        self.assertNotIn("TOPK", env)
        self.assertNotIn("PIPE", env)
        self.assertNotIn("PIN", env)
        self.assertEqual(env["COLI_POLICY"], "quality")

    def test_parse_calibration_and_metrics(self):
        replay = parse_calibration("[PROMPT_TOKENS] 2: 1 2\n[TOKENS] 3 generated: 3 4 5")
        self.assertEqual(replay["full_ids"], [1, 2, 3, 4, 5])
        with self.assertRaisesRegex(LatticeError, "declared 4 token IDs"):
            parse_calibration("[PROMPT_TOKENS] 2: 1 2\n[TOKENS] 4 generated: 3 4 5")
        artifact = (
            "STEP\tv2\t3\t2\t4\t0\t3\t1.25\t0.5\t0.8\t1.9"
            "\t1\t-2\t3\t-4\t2,4,3,1,5,6,7,8\n"
            "SUMMARY\tv2\t1\t8\tseparate_replay_pass\tprivate_file\n"
        )
        output = (
            "REPLAY_ORACLE_WRITTEN v2 steps=1 topk=8 measurement=separate_replay_pass transport=private_file\n"
            "REPLAY decode: 1 tokens in 0.400s | 2.50 tok/s\nexpert hit 70.5%\nlatency p50 10.2 ms p99 18.4 ms"
        )
        metrics = parse_replay_metrics(output, artifact, expected_forced=[3])
        self.assertEqual(metrics["tok_s"], 2.5)
        self.assertEqual(metrics["hit_pct"], 70.5)
        self.assertEqual(metrics["oracle"]["steps"][0][1], 2)
        self.assertEqual(metrics["decode_tokens"], 1)
        with self.assertRaisesRegex(LatticeError, "internally inconsistent"):
            parse_replay_metrics(output.replace("2.50 tok/s", "9.00 tok/s"), artifact)

    def test_runtime_fingerprint_covers_all_support_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coli = root / "coli"
            engine = root / "colibri"
            support = root / "future_support.py"
            coli.write_text("launcher", encoding="utf-8")
            engine.write_text("engine", encoding="utf-8")
            support.write_text("VALUE = 1\n", encoding="utf-8")
            before = fingerprint_runtime(root, coli, engine)
            support.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(before, fingerprint_runtime(root, coli, engine))

    def test_long_oracle_uses_separate_bounded_artifact(self):
        step = (
            "STEP\tv2\t3\t2\t4\t0\t3\t1.25\t0.5\t0.8\t1.9"
            "\t1\t-2\t3\t-4\t2,4,3,1,5,6,7,8\n"
        )
        artifact = step * 2048 + "SUMMARY\tv2\t2048\t8\tseparate_replay_pass\tprivate_file\n"
        self.assertGreater(len(artifact.encode()), 65536)
        parsed = parse_replay_oracle(artifact)
        self.assertEqual(len(parsed["steps"]), 2048)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.tsv"
            path.write_bytes(artifact.encode("utf-8"))
            self.assertEqual(read_replay_oracle_file(path), artifact)


if __name__ == "__main__":
    unittest.main()
