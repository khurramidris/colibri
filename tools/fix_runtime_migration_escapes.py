#!/usr/bin/env python3
"""Correct literal newline escaping in the one-time runtime migration source."""
from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("materialize_runtime_isolation.py")
text = path.read_text(encoding="utf-8")

old_output = 'output = f"{result.stdout}\\n{result.stderr}"'
new_output = 'output = f"{result.stdout}\\\\n{result.stderr}"'
if text.count(old_output) != 2:
    raise SystemExit(f"expected two subprocess output escapes, found {text.count(old_output)}")
text = text.replace(old_output, new_output)

for value in (1, 2):
    old = f'support.write_text("VALUE = {value}\\n", encoding="utf-8")'
    new = f'support.write_text("VALUE = {value}\\\\n", encoding="utf-8")'
    if text.count(old) != 1:
        raise SystemExit(f"expected one support-module escape for VALUE={value}")
    text = text.replace(old, new)

path.write_text(text, encoding="utf-8", newline="\n")
print("runtime migration escapes corrected")
