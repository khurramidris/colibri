from __future__ import annotations

import math
from typing import Any

from .candidates import Candidate, validate_candidate
from .common import LatticeError, canonical_json, ensure_under, load_json, sha256_bytes, validate_id
from .evidence import session_evidence_root, verify_record_digest
from .oracle import ORACLE_POLICY, validate_oracle
from .suite import WorkloadSuite
from .workspace import Workspace


def _candidate_definitions(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise LatticeError("session candidate matrix is missing")
    definitions: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise LatticeError(f"session candidate[{index}] must be an object")
        if not isinstance(item.get("environment"), dict):
            raise LatticeError(f"session candidate[{index}].environment must be an object")
        candidate = validate_candidate(Candidate(
            id=item.get("id"),
            description=item.get("description"),
            environment=item["environment"],
        ))
        if candidate.id in definitions:
            raise LatticeError(f"duplicate candidate id in session: {candidate.id}")
        definitions[candidate.id] = candidate.as_dict()
    if next(iter(definitions)) != "baseline" or definitions["baseline"]["environment"]:
        raise LatticeError("session candidate matrix must start with an empty baseline")
    return definitions


def _valid_token_list(value: Any, label: str) -> list[int]:
    if (not isinstance(value, list) or not value
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value)):
        raise LatticeError(f"{label} is not a valid token list")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LatticeError(f"{label} must be a finite number")
    number = float(value)
    if positive and number <= 0:
        raise LatticeError(f"{label} must be positive")
    if nonnegative and number < 0:
        raise LatticeError(f"{label} must be non-negative")
    return number


def _validate_process_evidence(record: Any, label: str, *, require_success: bool) -> None:
    if not isinstance(record, dict):
        raise LatticeError(f"{label} process evidence must be an object")
    if not isinstance(record.get("stdout"), str) or not isinstance(record.get("stderr"), str):
        raise LatticeError(f"{label} output evidence is invalid")
    if not isinstance(record.get("timed_out"), bool) or not isinstance(record.get("output_truncated"), bool):
        raise LatticeError(f"{label} process flags are invalid")
    _finite_number(record.get("duration_seconds"), f"{label} duration", nonnegative=True)
    for field in ("stdout_bytes", "stderr_bytes"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LatticeError(f"{label} {field} is invalid")
    returncode = record.get("returncode")
    if returncode is not None and (isinstance(returncode, bool) or not isinstance(returncode, int)):
        raise LatticeError(f"{label} return code is invalid")
    if require_success and (
        returncode != 0 or record.get("timed_out") is not False
        or record.get("output_truncated") is not False
    ):
        raise LatticeError(f"{label} process evidence is not a complete success")


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
    session_status = session.get("status")
    if session_status not in {"running", "interrupted", "completed"}:
        raise LatticeError(f"invalid session status: {session_status}")
    if require_complete and session_status != "completed":
        raise LatticeError(f"session is not completed: {session_id}")
    if session_status == "completed" and not isinstance(session.get("completed_at"), str):
        raise LatticeError(f"completed session has no completion timestamp: {session_id}")
    for field in (
        "suite_fingerprint", "model_fingerprint", "runtime_fingerprint",
        "hardware_fingerprint", "execution_fingerprint", "plan_fingerprint",
        "replay_cap", "qualification_context",
    ):
        expected = suite.fingerprint if field == "suite_fingerprint" else project.get(field)
        if session.get(field) != expected:
            raise LatticeError(f"session {field.replace('_', ' ')} does not match project")
    if session.get("oracle_policy") != ORACLE_POLICY:
        raise LatticeError("session replay numerical oracle policy does not match Lattice")
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
    replay_continuations: dict[str, list[int]] = {}
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
        prompt_ids = _valid_token_list(replay.get("prompt_ids"), f"replay prompt for {case.id}")
        full_ids = _valid_token_list(replay.get("full_ids"), f"replay path for {case.id}")
        if len(full_ids) <= len(prompt_ids) or full_ids[:len(prompt_ids)] != prompt_ids:
            raise LatticeError(f"replay path does not extend its prompt for {case.id}")
        continuation = full_ids[len(prompt_ids):]
        if len(full_ids) > case.context:
            raise LatticeError(f"replay path exceeds declared context for {case.id}")
        if len(continuation) > case.tokens:
            raise LatticeError(f"replay continuation exceeds requested token count for {case.id}")
        if record.get("prompt_tokens") != len(prompt_ids):
            raise LatticeError(f"replay prompt count mismatch for {case.id}")
        if record.get("continuation_tokens") != len(continuation):
            raise LatticeError(f"replay continuation count mismatch for {case.id}")
        _validate_process_evidence(
            record.get("calibration"), f"calibration for {case.id}", require_success=True
        )
        digest = sha256_bytes(canonical_json(replay))
        if digest != record.get("sha256"):
            raise LatticeError(f"replay hash mismatch for {case.id}")
        replay_hashes[case.id] = digest
        replay_continuations[case.id] = continuation

    run_ids = session.get("run_ids")
    if not isinstance(run_ids, list):
        raise LatticeError("session run_ids is invalid")
    validated_run_ids = [validate_id(value, "session run id") for value in run_ids]
    if len(validated_run_ids) != len(set(validated_run_ids)):
        raise LatticeError("session run_ids contains duplicates")
    for run in runs:
        verify_record_digest(run, f"run {run.get('id', '<unknown>')}")
    actual_ids = {validate_id(run.get("id"), "run id") for run in runs}
    if set(validated_run_ids) != actual_ids:
        raise LatticeError("session run set does not match write-once run records")

    case_ids = {case.id for case in suite.cases}
    expected_tasks = {
        (candidate_id, case_id, repeat)
        for repeat in range(repeats)
        for case_id in case_ids
        for candidate_id in candidates
    }
    actual_tasks: set[tuple[str, str, int]] = set()
    actual_attempts: set[tuple[str, str, int, int]] = set()
    successful_tasks: set[tuple[str, str, int]] = set()
    for run in runs:
        if run.get("schema_version") != 1:
            raise LatticeError("unsupported run schema")
        run_id = validate_id(run.get("id"), "run id")
        if run.get("session_id") != session_id:
            raise LatticeError(f"run belongs to another session: {run_id}")
        candidate_id, case_id = run.get("candidate_id"), run.get("case_id")
        repeat, attempt = run.get("repeat"), run.get("attempt")
        if (candidate_id not in candidates or case_id not in case_ids
                or isinstance(repeat, bool) or not isinstance(repeat, int) or not 0 <= repeat < repeats
                or isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0):
            raise LatticeError(f"run has an invalid task or attempt identity: {run_id}")
        task = (candidate_id, case_id, repeat)
        attempt_key = (*task, attempt)
        if attempt_key in actual_attempts:
            raise LatticeError(
                f"duplicate run attempt: {candidate_id}/{case_id}/repeat-{repeat}/attempt-{attempt}"
            )
        actual_attempts.add(attempt_key)
        actual_tasks.add(task)
        if run.get("candidate_environment") != candidates[candidate_id]["environment"]:
            raise LatticeError(f"candidate environment mismatch in run {run_id}")
        if case_id not in replay_hashes:
            raise LatticeError(f"run references an uncalibrated workload: {run_id}")
        if run.get("replay_sha256") != replay_hashes[case_id]:
            raise LatticeError(f"replay identity mismatch in run {run_id}")
        for field in ("execution_fingerprint", "plan_fingerprint", "replay_cap"):
            if run.get(field) != project.get(field):
                raise LatticeError(f"{field.replace('_', ' ')} mismatch in run {run_id}")
        run_status = run.get("status")
        _validate_process_evidence(run, f"run {run_id}", require_success=run_status == "success")
        if run_status == "success":
            if task in successful_tasks:
                raise LatticeError(
                    f"multiple successful attempts for {candidate_id}/{case_id}/repeat-{repeat}"
                )
            successful_tasks.add(task)
            metrics = run.get("metrics")
            if not isinstance(metrics, dict):
                raise LatticeError(f"successful run has no metrics: {run_id}")
            _finite_number(metrics.get("tok_s"), f"throughput for {run_id}", positive=True)
            validate_oracle(
                metrics.get("oracle"),
                expected_forced=replay_continuations[case_id],
            )
            hit_pct = metrics.get("hit_pct")
            if hit_pct is not None and not 0 <= _finite_number(hit_pct, f"hit rate for {run_id}") <= 100:
                raise LatticeError(f"hit rate is outside 0..100 for {run_id}")
            p50, p99 = metrics.get("p50_ms"), metrics.get("p99_ms")
            if (p50 is None) != (p99 is None):
                raise LatticeError(f"latency evidence is incomplete for {run_id}")
            if p50 is not None:
                p50_value = _finite_number(p50, f"p50 latency for {run_id}", nonnegative=True)
                p99_value = _finite_number(p99, f"p99 latency for {run_id}", nonnegative=True)
                if p99_value < p50_value:
                    raise LatticeError(f"p99 latency is below p50 for {run_id}")
            if run.get("error") is not None:
                raise LatticeError(f"successful run contains an error message: {run_id}")
        elif run_status == "failed":
            if not isinstance(run.get("error"), str) or not run["error"].strip():
                raise LatticeError(f"failed run has no error evidence: {run_id}")
            if run.get("metrics") is not None:
                raise LatticeError(f"failed run unexpectedly contains accepted metrics: {run_id}")
        else:
            raise LatticeError(f"invalid run status: {run_id}")
    if not actual_tasks.issubset(expected_tasks):
        raise LatticeError("session contains tasks outside its declared matrix")
    if require_complete and actual_tasks != expected_tasks:
        missing = expected_tasks - actual_tasks
        raise LatticeError(f"session task matrix is incomplete ({len(missing)} missing)")
    if session_status == "completed":
        expected_root = session.get("evidence_root_sha256")
        if not isinstance(expected_root, str) or len(expected_root) != 64:
            raise LatticeError("completed session has no valid evidence root")
        if session_evidence_root(session, runs) != expected_root:
            raise LatticeError("completed session evidence root mismatch")
    return candidates
