from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import canonical_json, sha256_bytes, sha256_file, strict_loads
from .model import AtlasError


def _load_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AtlasError(f"not a regular file: {path}")
    value = strict_loads(path.read_text("utf-8"), label=str(path))
    if not isinstance(value, dict):
        raise AtlasError(f"expected JSON object: {path}")
    return value


def verify_output(output_dir: Path, *, require_inputs: bool = False) -> dict[str, Any]:
    evidence_path = output_dir / "atlas-evidence.json"
    report_path = output_dir / "atlas-report.md"
    receipt_path = output_dir / "atlas-receipt.json"
    evidence = _load_object(evidence_path)
    receipt = _load_object(receipt_path)
    if evidence.get("schema") != "lattice.atlas.evidence.v1":
        raise AtlasError("unexpected evidence schema")
    expected_record = evidence.get("record_sha256")
    if not isinstance(expected_record, str):
        raise AtlasError("evidence is missing record_sha256")
    body = dict(evidence)
    body.pop("record_sha256", None)
    actual_record = sha256_bytes(canonical_json(body))
    if actual_record != expected_record:
        raise AtlasError("evidence record digest mismatch")
    checks = {
        "record_sha256": True,
        "evidence_file_sha256": receipt.get("evidence_sha256") == sha256_file(evidence_path),
        "report_file_sha256": report_path.is_file() and not report_path.is_symlink() and receipt.get("report_sha256") == sha256_file(report_path),
    }
    if not all(checks.values()):
        raise AtlasError("output receipt verification failed")

    input_checks: dict[str, Any] = {}
    missing: list[str] = []
    for name in ("manifest", "trace", "ternary"):
        path_text = evidence.get("inputs", {}).get(f"{name}_path")
        expected = evidence.get("inputs", {}).get(f"{name}_sha256")
        if path_text is None or expected is None:
            if name != "ternary":
                missing.append(name)
            continue
        path = Path(path_text)
        if not path.is_file() or path.is_symlink():
            missing.append(name)
            input_checks[name] = {"available": False, "match": None}
            continue
        match = sha256_file(path) == expected
        input_checks[name] = {"available": True, "match": match}
        if not match:
            raise AtlasError(f"{name} input digest mismatch")
    if require_inputs and missing:
        raise AtlasError("required source inputs are unavailable: " + ", ".join(sorted(missing)))
    return {
        "status": "verified",
        "output_dir": str(output_dir),
        "record_sha256": actual_record,
        "checks": checks,
        "inputs": input_checks,
        "missing_inputs": sorted(missing),
    }
