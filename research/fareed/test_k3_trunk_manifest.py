from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from k3_trunk_manifest import ManifestError, build_manifest, pack_manifest
from synthetic_trunk import TrunkReader, read_index


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> dict:
    header = {}
    offset = 0
    payloads = []
    for name, dtype, shape, payload in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-(8 + len(raw))) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for payload in payloads:
            handle.write(payload)
    return header


def layer_tensors(layer: int) -> list[tuple[str, str, list[int], bytes]]:
    prefix = f"language_model.model.layers.{layer}"
    return [
        (f"{prefix}.self_attn.q_proj.weight", "U8", [2, 4], bytes([layer + 1]) * 8),
        (f"{prefix}.self_attn.q_proj.weight.qs", "F32", [2], bytes([20 + layer]) * 8),
        (f"{prefix}.input_layernorm.weight", "BF16", [4], bytes([40 + layer]) * 8),
        (
            f"{prefix}.block_sparse_moe.experts.0.w1.weight_packed",
            "U8",
            [4, 4],
            bytes([60 + layer]) * 16,
        ),
        (
            f"{prefix}.block_sparse_moe.experts.0.w1.weight_scale",
            "U8",
            [4, 1],
            bytes([80 + layer]) * 4,
        ),
    ]


class K3TrunkManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="k3-manifest-")
        self.root = Path(self.tmp.name)
        for layer in range(3):
            write_safetensors(
                self.root / f"model-{layer + 1:05d}-of-000094.safetensors",
                layer_tensors(layer),
            )
        write_safetensors(
            self.root / "model-00094-of-000094.safetensors",
            [
                ("language_model.model.embed_tokens.weight", "BF16", [4, 4], b"E" * 32),
                ("language_model.model.norm.weight", "BF16", [4], b"N" * 8),
                ("language_model.lm_head.weight", "U8", [4, 4], b"H" * 16),
                ("language_model.lm_head.weight.qs", "F32", [4], b"S" * 16),
            ],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_builds_exact_contiguous_layer_spans(self) -> None:
        manifest = build_manifest(self.root)
        self.assertEqual("cerno-k3-trunk-manifest-v1", manifest["format"])
        self.assertEqual(3, manifest["summary"]["layers"])
        self.assertEqual(3 * (8 + 8 + 8), manifest["summary"]["trunk_payload_bytes"])
        self.assertEqual(3 * (16 + 4), manifest["summary"]["expert_payload_bytes"])
        self.assertEqual(32 + 8 + 16 + 16, manifest["summary"]["global_payload_bytes"])

        for layer_id, record in enumerate(manifest["layers"]):
            self.assertEqual(layer_id, record["layer_id"])
            self.assertEqual(24, record["payload_bytes"])
            names = [tensor["name"] for tensor in record["tensors"]]
            self.assertIn(
                f"language_model.model.layers.{layer_id}.self_attn.q_proj.weight.qs",
                names,
            )
            self.assertEqual([0, 8, 16], [t["relative_start"] for t in record["tensors"]])
            self.assertEqual([8, 16, 24], [t["relative_end"] for t in record["tensors"]])

    def test_pack_copies_source_bytes_without_reencoding(self) -> None:
        manifest = build_manifest(self.root)
        output_bin = self.root / "trunk.bin"
        output_json = self.root / "trunk.json"
        packed = pack_manifest(
            manifest,
            source_dir=self.root,
            output_bin=output_bin,
            output_json=output_json,
        )
        self.assertEqual(3, packed["summary"]["layers"])
        _, entries = read_index(output_bin)
        self.assertEqual([24, 24, 24], [entry.length for entry in entries])

        with TrunkReader(output_bin, resident_budget_bytes=0) as reader:
            payloads = [payload for _, payload in reader.iter_layers()]
        for layer, payload in enumerate(payloads):
            self.assertEqual(
                bytes([layer + 1]) * 8
                + bytes([20 + layer]) * 8
                + bytes([40 + layer]) * 8,
                payload,
            )
        saved = json.loads(output_json.read_text())
        self.assertEqual(packed, saved)

    def test_rejects_expert_interleaving(self) -> None:
        bad = self.root / "model-00001-of-000094.safetensors"
        prefix = "language_model.model.layers.0"
        write_safetensors(
            bad,
            [
                (f"{prefix}.self_attn.q_proj.weight", "U8", [1], b"A"),
                (
                    f"{prefix}.block_sparse_moe.experts.0.w1.weight_packed",
                    "U8",
                    [1],
                    b"X",
                ),
                (f"{prefix}.input_layernorm.weight", "BF16", [1], b"B"),
            ],
        )
        with self.assertRaisesRegex(ManifestError, "interleaved"):
            build_manifest(self.root)

    def test_rejects_two_layers_in_one_shard(self) -> None:
        bad_root = self.root / "two-layers"
        write_safetensors(
            bad_root / "model-00001-of-000094.safetensors",
            [
                ("language_model.model.layers.0.input_layernorm.weight", "BF16", [1], b"A"),
                ("language_model.model.layers.1.input_layernorm.weight", "BF16", [1], b"B"),
            ],
        )
        with self.assertRaisesRegex(ManifestError, "multiple layers"):
            build_manifest(bad_root)

    def test_rejects_noncontiguous_layers(self) -> None:
        bad_root = self.root / "gap"
        write_safetensors(
            bad_root / "model-00001-of-000094.safetensors",
            [("language_model.model.layers.0.input_layernorm.weight", "BF16", [1], b"A")],
        )
        write_safetensors(
            bad_root / "model-00002-of-000094.safetensors",
            [("language_model.model.layers.2.input_layernorm.weight", "BF16", [1], b"B")],
        )
        with self.assertRaisesRegex(ManifestError, "not contiguous"):
            build_manifest(bad_root)


if __name__ == "__main__":
    unittest.main()
