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
    fingerprint_storage_topology,
    qualification_environment,
    parse_calibration,
    parse_replay_metrics,
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
            "COLI_TEMP": "0.8",
            "IDOT": "0",
            "COLI_METAL": "1",
            "COLI_UNREVIEWED_APPROX": "1",
        })
        self.assertNotIn("COLI_TEMP", env)
        self.assertNotIn("IDOT", env)
        self.assertNotIn("COLI_UNREVIEWED_APPROX", env)
        self.assertEqual(qualification_environment(env)["COLI_METAL"], "1")

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
        oracle_line = (
            "REPLAY_ORACLE_STEP v1 step=0 forced=3 top1=2 top2=4 "
            "top1_logit=3 forced_logit=1.25 margin=0.5 mean=0.8 rms=1.9 "
            "p0=1 p1=-2 p2=3 p3=-4 topk_ids=0123456789abcdef nonfinite=0"
        )
        metrics = parse_replay_metrics(
            oracle_line + "\nREPLAY_ORACLE_SUMMARY v1 steps=1 topk=8 measurement=separate_replay_pass\n"
            "REPLAY decode: 16 tokens | 2.50 tok/s\nexpert hit 70.5%\nlatency p50 10.2 ms p99 18.4 ms"
        )
        self.assertEqual(metrics["tok_s"], 2.5)
        self.assertEqual(metrics["hit_pct"], 70.5)
        self.assertEqual(metrics["oracle"]["steps"][0]["top1"], 2)


if __name__ == "__main__":
    unittest.main()
