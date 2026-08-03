from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKER = ROOT / "tools" / "pack_expert_blocks.py"
MAGIC = b"LTBLK1\0\0"
HEADER_BYTES = 4096


def tensor_bytes(seed: int, size: int) -> bytes:
    return bytes((seed + i) & 0xFF for i in range(size))


def write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    offset = 0
    header: dict[str, object] = {}
    payload = bytearray()
    for name, data in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def w2_slice(data: bytes, rows: int, row_bytes: int, start: int, width: int) -> bytes:
    return b"".join(data[row * row_bytes + start:row * row_bytes + start + width] for row in range(rows))


class PackExpertBlocksTest(unittest.TestCase):
    def make_model(self, root: Path) -> dict[tuple[int, str, str], bytes]:
        latent, intermediate, experts = 64, 128, 2
        sizes = {
            ("w1", "packed"): intermediate * latent // 2,
            ("w1", "scale"): intermediate * latent // 32,
            ("w2", "packed"): latent * intermediate // 2,
            ("w2", "scale"): latent * intermediate // 32,
            ("w3", "packed"): intermediate * latent // 2,
            ("w3", "scale"): intermediate * latent // 32,
        }
        tensors: dict[str, bytes] = {}
        source: dict[tuple[int, str, str], bytes] = {}
        seed = 3
        for expert in range(experts):
            for matrix, suffix in sizes:
                data = tensor_bytes(seed, sizes[(matrix, suffix)])
                seed += 17
                name = (
                    f"model.layers.0.block_sparse_moe.experts.{expert}."
                    f"{matrix}.weight_{suffix}"
                )
                tensors[name] = data
                source[(expert, matrix, suffix)] = data
        write_safetensors(root / "model.safetensors", tensors)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "routed_expert_hidden_size": latent,
                    "moe_intermediate_size": intermediate,
                    "num_experts": experts,
                }
            ),
            encoding="utf-8",
        )
        return source

    def test_record_is_byte_exact_and_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model, output = root / "model", root / "blocks"
            model.mkdir()
            source = self.make_model(model)
            result = subprocess.run(
                [
                    sys.executable,
                    str(PACKER),
                    str(model),
                    str(output),
                    "--layers", "0",
                    "--block-channels", "64",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            packed = output / "layer-000.ltblk"
            raw = packed.read_bytes()
            self.assertEqual(raw[:8], MAGIC)
            self.assertEqual(len(raw), 4096 + 2 * 2 * 8192)
            self.assertEqual(struct.unpack_from("<I", raw, 16)[0], 0)
            self.assertEqual(struct.unpack_from("<Q", raw, 48)[0], 8192)

            w1p = source[(0, "w1", "packed")]
            w1s = source[(0, "w1", "scale")]
            w2p = source[(0, "w2", "packed")]
            w2s = source[(0, "w2", "scale")]
            w3p = source[(0, "w3", "packed")]
            w3s = source[(0, "w3", "scale")]
            expected_payload = b"".join(
                [
                    w1p[: 64 * 32],
                    w1s[: 64 * 2],
                    w3p[: 64 * 32],
                    w3s[: 64 * 2],
                    w2_slice(w2p, 64, 64, 0, 32),
                    w2_slice(w2s, 64, 4, 0, 2),
                ]
            )
            record = raw[HEADER_BYTES:HEADER_BYTES + 8192]
            self.assertEqual(record[: len(expected_payload)], expected_payload)
            self.assertEqual(record[len(expected_payload):], bytes(8192 - len(expected_payload)))
            receipt = json.loads((output / "layer-000.ltblk.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema"], "lattice.expert-blocks.v1")
            self.assertEqual(receipt["layout"]["blocks"], 2)

    def test_valid_output_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model, output = root / "model", root / "blocks"
            model.mkdir()
            self.make_model(model)
            command = [
                sys.executable,
                str(PACKER),
                str(model),
                str(output),
                "--layers", "0",
                "--block-channels", "64",
            ]
            first = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(first.returncode, 0, first.stderr)
            packed = output / "layer-000.ltblk"
            before = packed.stat().st_mtime_ns
            time.sleep(0.01)
            second = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(packed.stat().st_mtime_ns, before)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["receipts"][0]["status"], "reused")

    def test_missing_tensor_fails_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model, output = root / "model", root / "blocks"
            model.mkdir()
            self.make_model(model)
            # Rebuild a deliberately incomplete one-expert index under a new directory.
            bad = root / "bad"
            bad.mkdir()
            write_safetensors(
                bad / "model.safetensors",
                {"model.layers.0.block_sparse_moe.experts.0.w1.weight_packed": bytes(4096)},
            )
            (bad / "config.json").write_text(
                json.dumps({"routed_expert_hidden_size": 64, "moe_intermediate_size": 128, "num_experts": 1}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(PACKER), str(bad), str(output), "--layers", "0", "--block-channels", "64"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((output / "layer-000.ltblk").exists())
            self.assertFalse(any(output.glob("*.tmp.*")))


if __name__ == "__main__":
    unittest.main()
