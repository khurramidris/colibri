from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lattice.common import (
    LatticeError,
    atomic_write_json,
    canonical_json,
    load_json,
    strict_json_loads,
)


class CommonTests(unittest.TestCase):
    def test_nonfinite_json_is_rejected_on_read_and_write(self):
        for text in ('{"value": NaN}', '{"value": Infinity}', '{"value": -Infinity}'):
            with self.assertRaisesRegex(LatticeError, "non-standard JSON constant"):
                strict_json_loads(text)
        with self.assertRaisesRegex(LatticeError, "non-finite"):
            canonical_json({"value": float("nan")})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            with self.assertRaisesRegex(LatticeError, "non-finite"):
                atomic_write_json(path, {"value": float("inf")})
            self.assertFalse(path.exists())

    def test_write_once_json_is_idempotent_but_not_replaceable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.json"
            atomic_write_json(path, {"schema_version": 1, "value": 7}, exclusive=True)
            atomic_write_json(path, {"schema_version": 1, "value": 7}, exclusive=True)
            self.assertEqual(load_json(path)["value"], 7)
            with self.assertRaisesRegex(LatticeError, "write-once"):
                atomic_write_json(path, {"schema_version": 1, "value": 8}, exclusive=True)
            self.assertEqual(load_json(path)["value"], 7)

    def test_symlink_json_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text('{"ok": true}\n', encoding="utf-8")
            link = root / "link.json"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(LatticeError, "regular file"):
                load_json(link)


if __name__ == "__main__":
    unittest.main()
