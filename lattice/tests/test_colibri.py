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
        metrics = parse_replay_metrics("REPLAY decode: 16 tokens | 2.50 tok/s\nexpert hit 70.5%\nlatency p50 10.2 ms p99 18.4 ms")
        self.assertEqual(metrics["tok_s"], 2.5)
        self.assertEqual(metrics["hit_pct"], 70.5)


if __name__ == "__main__":
    unittest.main()
