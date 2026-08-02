#!/usr/bin/env python3
"""Correct one deterministic acceptance-test assertion before publication."""
from __future__ import annotations

from pathlib import Path

path = Path(__file__).resolve().parents[1] / "lattice" / "tests" / "test_acceptance.py"
text = path.read_text(encoding="utf-8")
old = '            self.assertIn("token mismatch", record["runs"][0]["error"])\n'
new = '            self.assertIn("token mismatch", record["runs"][0]["error"].lower())\n'
if text.count(old) != 1:
    raise SystemExit(f"expected one token-mismatch assertion, found {text.count(old)}")
path.write_text(text.replace(old, new), encoding="utf-8", newline="\n")
print("OLMoE token-mismatch assertion corrected")
