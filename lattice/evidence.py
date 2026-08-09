from __future__ import annotations

from typing import Any

from .common import LatticeError, canonical_json, sha256_bytes

DIGEST_FIELD = "record_sha256"


def record_digest(record: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        raise LatticeError("evidence record must be an object")
    payload = {key: value for key, value in record.items() if key != DIGEST_FIELD}
    return sha256_bytes(canonical_json(payload))


def seal_record(record: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(record)
    sealed[DIGEST_FIELD] = record_digest(sealed)
    return sealed


def verify_record_digest(record: dict[str, Any], label: str = "record") -> str:
    expected = record.get(DIGEST_FIELD)
    if not isinstance(expected, str) or len(expected) != 64:
        raise LatticeError(f"{label} has no valid record digest")
    actual = record_digest(record)
    if actual != expected:
        raise LatticeError(f"{label} digest mismatch")
    return expected


def session_evidence_root(session: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    run_entries = []
    for run in runs:
        digest = verify_record_digest(run, f"run {run.get('id', '<unknown>')}")
        run_entries.append({"id": run.get("id"), "sha256": digest})
    replays = session.get("replays") or {}
    replay_entries = [
        {"case_id": case_id, "sha256": record.get("sha256")}
        for case_id, record in sorted(replays.items())
    ]
    session_core = {
        key: value
        for key, value in session.items()
        if key != "evidence_root_sha256"
    }
    payload = {
        "schema_version": 1,
        "session": session_core,
        "runs": sorted(run_entries, key=lambda item: (str(item["id"]), item["sha256"])),
        "replays": replay_entries,
    }
    return sha256_bytes(canonical_json(payload))
