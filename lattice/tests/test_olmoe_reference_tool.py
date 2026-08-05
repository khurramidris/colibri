from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

from lattice.common import LatticeError


TOOL_PATH = Path(__file__).resolve().parents[2] / "c" / "tools" / "make_olmoe_reference.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("lattice_olmoe_reference_tool", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load OLMoE reference tool")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OlmoeReferenceToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()

    def test_tool_import_does_not_require_torch(self):
        self.assertTrue(callable(self.tool.resolve_source))
        self.assertTrue(callable(self.tool.fingerprint_local_model))

    def test_local_fingerprint_is_deterministic_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_text('{"model_type":"olmoe"}\n', encoding="utf-8")
            (model / "tokenizer.json").write_text('{}\n', encoding="utf-8")
            shard = model / "model-00001.safetensors"
            shard.write_bytes(b"a" * 200000)
            first = self.tool.fingerprint_local_model(model)
            self.assertEqual(first, self.tool.fingerprint_local_model(model))
            shard.write_bytes(b"a" * 100000 + b"b" + b"a" * 99999)
            self.assertNotEqual(first, self.tool.fingerprint_local_model(model))

    def test_local_source_rejects_revision_and_records_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"fixture")
            source = self.tool.resolve_source(str(model), None, lambda *_args, **_kwargs: None)
            self.assertEqual(source["source_kind"], "local")
            self.assertRegex(source["source_fingerprint"], r"^[0-9a-f]{64}$")
            with self.assertRaisesRegex(ValueError, "not valid for a local"):
                self.tool.resolve_source(str(model), "main", lambda *_args, **_kwargs: None)

    def test_remote_source_is_resolved_to_immutable_commit(self):
        calls = []

        def fake_info(model, revision=None):
            calls.append((model, revision))
            return types.SimpleNamespace(sha="a" * 40)

        source = self.tool.resolve_source("owner/model", "release", fake_info)
        self.assertEqual(calls, [("owner/model", "release")])
        self.assertEqual(source["source_kind"], "huggingface")
        self.assertEqual(source["resolved_revision"], "a" * 40)
        self.assertIsNone(source["source_fingerprint"])

        with self.assertRaisesRegex(ValueError, "valid immutable"):
            self.tool.resolve_source(
                "owner/model", None,
                lambda *_args, **_kwargs: types.SimpleNamespace(sha="main"),
            )

    def test_write_once_json_is_idempotent_and_refuses_change(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            value = {"schema_version": 2, "prompt_ids": [1], "full_ids": [1, 2]}
            self.tool.write_once_json(output, value)
            self.tool.write_once_json(output, value)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), value)
            with self.assertRaisesRegex(FileExistsError, "write-once"):
                self.tool.write_once_json(output, {**value, "full_ids": [1, 3]})


if __name__ == "__main__":
    unittest.main()
