from __future__ import annotations

from pathlib import Path
from typing import Any

from .candidates import Candidate, validate_candidate
from .common import LatticeError, canonical_json, ensure_under, load_json, sha256_bytes, validate_id
from .suite import WorkloadSuite
from .workspace import Workspace


def _candidate_definitions(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise LatticeError("session candidate matrix is missing")
    definitions: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LatticeError(f"session candidate[{index}] must be an object")
        candidate = validate_candidate(Candidate(
            id=item.get("id"),
            description=item.get("description"),
            environment=item.get("environment") if isinstance(item.get("environment"), dict) else {},
        ))
        if candidate.id in definitions:
            raise LatticeError(f"duplicate candidate id in session: {candidate.id}")
        definitions[candidate.id] = candidate.as_dict()
    if next(iter(definitions)) != "baseline" or definitions["baseline"]["environment"]:
        raise LatticeError("session candidate matrix must start with an empty baseline")
    return definitions


def validate_session_evidence(
    workspace: Workspace,
    project: dict[str, Any],
    suite: WorkloadSuite,
    session: dict[str, Any],
    runs: list[dict[str, Any]],
    *,
    require_complete: bool = True,
) -> dict[str, dict[str, Any]]:
    if session.get("schema_version") != 1:
        raise LatticeError("unsupported session schema")
    session_id = validate_id(session.get("id"), "session id")
    status = session.get("status")
    if status not in {"running", "interrupted", "completed"}:
        raise LatticeError(f"invalid session status: {status}")
    if require_complete and status != "completed":
        raise LatticeError(f"session is not completed: {session_id}")
    if status == "completed" and not isinstance(session.get("completed_at"), str):
        raise LatticeError(f"completed session has no completion timestamp: {session_id}")
    if session.get("suite_fingerprint") != suite.fingerprint:
        raise LatticeError("session workload fingerprint does not match project")
    if session.get("model_fingerprint") != project.get("model_fingerprint"):
        raise LatticeError("session model fingerprint does not match project")
    if session.get("runtime_fingerprint") != project.get("runtime_fingerprint"):
        raise LatticeError("session runtime fingerprint does not match project")
    repeats = session.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 20:
        raise LatticeError("session repeats is invalid")
    candidates = _candidate_definitions(session.get("candidates"))
    replays = session.get("replays")
    expected_case_ids = {case.id for case in suite.cases}
    if not isinstance(replays, dict):
        raise LatticeError("session replay set is invalid")
    replay_case_ids = set(replays)
    if (require_complete and replay_case_ids != expected_case_ids) or not replay_case_ids.issubset(expected_case_ids):
        raise LatticeError("session replay set does not match workload suite")
    replay_hashes: dict[str, str] = {}
    for case in suite.cases:
        if case.id not in replays:
            continue
        record = replays[case.id]
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise LatticeError(f"invalid replay record for {case.id}")
        replay_path = ensure_under(workspace.root, workspace.root / record["path"])
        replay = load_json(replay_path)
        if not isinstance(replay, dict):
            raise LatticeError(f"invalid replay payload for {case.id}")
        prompt_ids, full_ids = replay.get("prompt_ids"), replay.get("full_ids")
        if (not isinstance(prompt_ids, list) or not isinstance(full_ids, list)
                or not prompt_ids or len(full_ids) <= len(prompt_ids)
                or full_ids[:len(prompt_ids)] != prompt_ids
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in full_ids)):
            raise LatticeError(f"invalid replay token sequence for {case.id}")
        digest = sha256_bytes(canonical_json(replay))
        if digest != record.get("sha256"):
            raise LatticeError(f"replay hash mismatch for {case.id}")
        replay_hashes[case.id] = digest
    run_ids = session.get("run_ids")
    if not isinstance(run_ids, list) or any(not isinstance(value, str) for value in run_ids):
        raise LatticeError("session run_ids is invalid")
    if len(run_ids) != len(set(run_ids)):
        raise LatticeError("session run_ids contains duplicates")
    actual_ids = {run.get("id") for run in runs}
    if set(run_ids) != actual_ids:
        raise LatticeError("session run set does not match immutable run records")
    case_ids = {case.id for case in suite.cases}
    expected_tasks = {
        (candidate_id, case_id, repeat)
        for repeat in range(repeats)
        for case_id in case_ids
        for candidate_id in candidates
    }
    actual_tasks: set[tuple[str, str, int]] = set()
    successful_tasks: set[tuple[str, str, int]] = set()
    for run in runs:
        if run.get("schema_version") != 1:
            raise LatticeError("unsupported run schema")
        validate_id(run.get("id"), "run id")
        if run.get("session_id") != session_id:
            raise LatticeError(f"run belongs to another session: {run.get('id')}")
        candidate_id, case_id, repeat = run.get("candidate_id"), run.get("case_id"), run.get("repeat")
        if candidate_id not in candidates or case_id not in case_ids or not isinstance(repeat, int) or not 0 <= repeat < repeats:
            raise LatticeError(f"run has an invalid task identity: {run.get('id')}")
        task = (candidate_id, case_id, repeat)
        actual_tasks.add(task)
        if run.get("candidate_environment") != candidates[candidate_id]["environment"]:
            raise LatticeError(f"candidate environment mismatch in run {run.get('id')}")
        if case_id not in replay_hashes:
            raise LatticeError(f"run references an uncalibrated workload: {run.get('id')}")
        if run.get("replay_sha256") != replay_hashes[case_id]:
            raise LatticeError(f"replay identity mismatch in run {run.get('id')}")
        status = run.get("status")
        if status == "success":
            if task in successful_tasks:
                raise LatticeError(f"duplicate successful run task: {candidate_id}/{case_id}/repeat-{repeat}")
            successful_tasks.add(task)
            metrics = run.get("metrics")
            tok_s = metrics.get("tok_s") if isinstance(metrics, dict) else None
            if (not isinstance(tok_s, (int, float)) or isinstance(tok_s, bool) or tok_s <= 0
                    or run.get("returncode") != 0 or run.get("output_truncated") is not False):
                raise LatticeError(f"successful run is incomplete: {run.get('id')}")
        elif status == "failed":
            if not isinstance(run.get("error"), str) or not run["error"]:
                raise LatticeError(f"failed run has no error evidence: {run.get('id')}")
        else:
            raise LatticeError(f"invalid run status: {run.get('id')}")
    if not actual_tasks.issubset(expected_tasks):
        raise LatticeError("session contains tasks outside its declared matrix")
    if require_complete and actual_tasks != expected_tasks:
        missing = expected_tasks - actual_tasks
        raise LatticeError(f"session task matrix is incomplete ({len(missing)} missing)")
    return candidates
