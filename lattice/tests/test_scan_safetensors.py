from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "tools" / "scan_safetensors.py"


def write_safetensors(path: Path, tensors: list[tuple[str, int]]) -> None:
    offset = 0
    header: dict[str, object] = {}
    for name, size in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


class ScanSafetensorsTest(unittest.TestCase):
    def test_manifest_geometry_and_roles(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model = Path(td)
            (model / "config.json").write_text(
                json.dumps({"text_config": {"num_experts": 4, "num_experts_per_token": 2}}),
                encoding="utf-8",
            )
            write_safetensors(
                model / "model-00001-of-00001.safetensors",
                [
                    ("model.layers.0.self_attn.q_proj.weight", 16),
                    ("model.layers.0.block_sparse_moe.experts.0.w1.weight", 8),
                    ("model.layers.0.block_sparse_moe.experts.1.w1.weight", 8),
                    ("model.norm.weight", 4),
                ],
            )
            result = subprocess.run(
                [sys.executable, str(SCANNER), str(model)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [line.split("\t") for line in result.stdout.splitlines() if line and not line.startswith("#")]
            by_role: dict[str, list[list[str]]] = {}
            for row in rows:
                by_role.setdefault(row[1], []).append(row)
            self.assertIn("attention", by_role)
            self.assertIn("routed_expert", by_role)
            self.assertIn("norm", by_role)
            routed = by_role["routed_expert"][0]
            self.assertEqual(routed[2], "8")
            self.assertEqual(routed[3], "2")
            self.assertAlmostEqual(float(routed[4]), 1.0)
            self.assertEqual(routed[8], "streamable,gpu")
            self.assertEqual(by_role["norm"][0][8], "ram_required,gpu")

    def test_missing_expert_geometry_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model = Path(td)
            write_safetensors(
                model / "model.safetensors",
                [("model.layers.0.block_sparse_moe.experts.0.w1.weight", 8)],
            )
            result = subprocess.run(
                [sys.executable, str(SCANNER), str(model)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("num_experts/top_k", result.stderr)

    def test_header_bounds_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model = Path(td)
            (model / "model.safetensors").write_bytes(struct.pack("<Q", 10_000) + b"{}")
            result = subprocess.run(
                [sys.executable, str(SCANNER), str(model), "--num-experts", "1", "--top-k", "1"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("extends beyond end of file", result.stderr)


if __name__ == "__main__":
    unittest.main()
