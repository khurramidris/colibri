#!/usr/bin/env python3
"""Upgrade synthetic tests to the execution-identity schema under migration."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = "p" * 64


def replace_exact(path: str, old: str, new: str, expected: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} replacements, found {count}: {old!r}")
    target.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


replace_exact(
    "lattice/tests/test_recommend.py",
    '            "execution_fingerprint": "e",\n            "qualification_context": 4096,\n',
    f'            "execution_fingerprint": "e",\n'
    f'            "plan_fingerprint": "{PLAN}",\n'
    f'            "replay_cap": 0,\n'
    f'            "qualification_context": 4096,\n',
    expected=2,
)
replace_exact(
    "lattice/tests/test_recommend.py",
    '            "execution_fingerprint": "e",\n            "candidate_environment": environment,\n',
    f'            "execution_fingerprint": "e",\n'
    f'            "plan_fingerprint": "{PLAN}",\n'
    f'            "replay_cap": 0,\n'
    f'            "candidate_environment": environment,\n',
)
replace_exact(
    "lattice/tests/test_resume.py",
    '                execution_fingerprint="execution", hardware_fingerprint="hardware",\n'
    '                storage_topology={},\n',
    f'                execution_fingerprint="execution", hardware_fingerprint="hardware",\n'
    f'                storage_topology={{}}, plan_fingerprint="{PLAN}", replay_cap=0,\n',
)

print("execution-identity fixtures materialized")
