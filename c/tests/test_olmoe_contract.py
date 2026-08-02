#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
BINARY = C_DIR / ("olmoe.exe" if os.name == "nt" else "olmoe")
SOURCE = C_DIR / "olmoe.c"


class OlmoeNativeContractTests(unittest.TestCase):
    def run_engine(self, *arguments: str, snap: Path | None = None):
        self.assertTrue(BINARY.is_file(), f"OLMoE binary was not built: {BINARY}")
        environment = dict(os.environ)
        environment["SNAP"] = str((snap or C_DIR).resolve())
        return subprocess.run(
            [str(BINARY), *arguments],
            cwd=C_DIR,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
        )

    def test_zero_cache_capacity_fails_before_model_loading(self):
        result = self.run_engine("0", "8", "missing-reference.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cache_cap must be 1..512", result.stderr)
        self.assertNotIn("config.json", result.stderr)

    def test_invalid_quantization_fails_before_model_loading(self):
        result = self.run_engine("16", "9", "missing-reference.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("quant_bits must be 2..8", result.stderr)
        self.assertNotIn("config.json", result.stderr)

    def test_oversized_reference_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.json"
            reference.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
            result = self.run_engine("16", "8", str(reference))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reference JSON must be", result.stderr)
        self.assertNotIn("config.json", result.stderr)

    def test_persisted_pin_load_and_save_are_both_guarded(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("OLMOE_IGNORE_PERSISTED_PINS", source)
        self.assertIn("OLMOE_PERSISTED_PINS_DISABLED", source)
        self.assertIn("if (!g_ignore_persisted_pins) {", source)
        self.assertIn("if (!g_ignore_persisted_pins && m.hot_pinned) {", source)


if __name__ == "__main__":
    unittest.main()
