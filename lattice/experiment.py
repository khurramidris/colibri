from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .candidates import Candidate, default_candidates, validate_candidate
from .colibri import ColibriContext, calibrate_case, run_replay
from .common import LatticeError, atomic_write_json, canonical_json, sha256_bytes, short_id, utc_now
from .evidence import session_evidence_root
from .integrity import validate_session_evidence
from .oracle import ORACLE_POLICY
from .suite import WorkloadCase, WorkloadSuite
from .workspace import Workspace

Progress = Callable[[str], None]


def _task_order(cases: list[WorkloadCase], candidates: list[Candidate], repeats: int):
    for repeat in range(repeats):
        for case_index, case in enumerate(cases):
            offset = (repeat + case_index) % len(candidates)
            rotated = candidates[offset:] + candidates[:offset]
            for candidate in rotated:
                yield repeat, case, candidate


def create_project(context: ColibriContext, suite: WorkloadSuite) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "repo_root": str(context.repo_root),
        "model_path": str(context.model),
        "engine_path": str(context.engine),
        "model_family": context.model_family,
        "model_fingerprint": context.model_fingerprint,
        "runtime_fingerprint": context.runtime_fingerprint,
        "hardware_fingerprint": context.hardware_fingerprint,
        "execution_fingerprint": context.execution_fingerprint,
        "plan_fingerprint": context.plan_fingerprint,
        "replay_cap": context.replay_cap,
        "qualification_context": context.qualification_context,
        "qualification_environment": context.qualification_environment,
        "storage_topology": context.storage_topology,
        "suite": suite.as_dict(),
        "suite_fingerprint": suite.fingerprint,
        "plan": context.plan,
        "doctor": context.doctor,
    }


def build_replays(
    workspace: Workspace,
    context: ColibriContext,
    suite: WorkloadSuite,
    *,
    timeout: int,
    progress: Progress,
    existing: dict[str, dict[str, Any]] | None = None,
    checkpoint: Callable[[dict[str, dict[str, Any]]], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Calibrate missing workload cases and checkpoint each replay atomically.

    Frontier-model calibration can itself take hours.  ``existing`` allows a
    previously interrupted session to retain completed replays, while
    ``checkpoint`` publishes every newly completed case before the next engine
    invocation begins.
    """
    replays: dict[str, dict[str, Any]] = dict(existing or {})
    case_ids = {case.id for case in suite.cases}
    unknown = set(replays) - case_ids
    if unknown:
        raise LatticeError(
            "session contains replay records outside the workload suite: "
            + ", ".join(sorted(unknown))
        )
    for case in suite.cases:
        if case.id in replays:
            progress(f"reusing calibrated workload {case.id}")
            continue
        progress(f"calibrating workload {case.id}")
        replay, result = calibrate_case(
            context,
            prompt=case.prompt,
            tokens=case.tokens,
            ctx=case.context,
            timeout=timeout,
        )
        replay_hash = sha256_bytes(canonical_json(replay))
        path = workspace.replays_dir / f"{case.id}-{replay_hash[:16]}.json"
        atomic_write_json(path, replay, exclusive=True)
        combined_output = f"{result.stdout}\n{result.stderr}"
        if case.expected_contains is not None and case.expected_contains not in combined_output:
            raise LatticeError(f"calibration for {case.id} did not contain required text")
        replays[case.id] = {
            "path": str(path.relative_to(workspace.root)),
            "sha256": replay_hash,
            "prompt_tokens": len(replay["prompt_ids"]),
            "continuation_tokens": len(replay["full_ids"]) - len(replay["prompt_ids"]),
            "calibration_duration_seconds": result.duration_seconds,
        }
        if checkpoint is not None:
            checkpoint(replays)
    return replays


def _execute_task(
    workspace: Workspace,
    context: ColibriContext,
    session: dict[str, Any],
    case: WorkloadCase,
    candidate: Candidate,
    repeat: int,
    *,
    timeout: int,
    progress: Progress,
) -> dict[str, Any]:
    replays = session["replays"]
    task = {
        "session_id": session["id"],
        "candidate_id": candidate.id,
        "case_id": case.id,
        "repeat": repeat,
        "replay_sha256": replays[case.id]["sha256"],
        "execution_fingerprint": context.execution_fingerprint,
        "plan_fingerprint": context.plan_fingerprint,
        "replay_cap": context.replay_cap,
    }
    run_id = short_id("run", {**task, "attempt_ns": time.time_ns()})
    progress(f"{case.id}: {candidate.id} ({repeat + 1}/{session['repeats']})")
    started_at = utc_now()
    status = "failed"
    metrics = None
    error = None
    result_data: dict[str, Any] = {}
    try:
        metrics, result = run_replay(
            context,
            replay_path=workspace.root / replays[case.id]["path"],
            candidate=candidate.environment,
            ctx=case.context,
            timeout=timeout,
        )
        status = "success"
        result_data = {
            "returncode": result.returncode,
            "duration_seconds": result.duration_seconds,
            "stdout_tail": result.stdout,
            "stderr_tail": result.stderr,
            "output_truncated": result.output_truncated,
        }
    except Exception as exc:  # every failed attempt remains immutable evidence
        error = str(exc)
    run = {
        "schema_version": 1,
        "id": run_id,
        **task,
        "candidate_environment": dict(sorted(candidate.environment.items())),
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": status,
        "metrics": metrics,
        "error": error,
        **result_data,
    }
    workspace.write_run(run)
    session["run_ids"].append(run_id)
    workspace.write_session(session)
    return run


def _run_pending_tasks(
    workspace: Workspace,
    context: ColibriContext,
    suite: WorkloadSuite,
    session: dict[str, Any],
    candidates: list[Candidate],
    *,
    timeout: int,
    retry_failed: bool,
    progress: Progress,
) -> dict[str, Any]:
    existing = workspace.list_runs(session["id"])
    successes = {(run["candidate_id"], run["case_id"], run["repeat"]) for run in existing if run.get("status") == "success"}
    attempts = {(run["candidate_id"], run["case_id"], run["repeat"]) for run in existing}
    session["status"] = "running"
    session["completed_at"] = None
    session.pop("evidence_root_sha256", None)
    workspace.write_session(session)
    try:
        for repeat, case, candidate in _task_order(list(suite.cases), candidates, session["repeats"]):
            task = (candidate.id, case.id, repeat)
            if task in successes or (task in attempts and not retry_failed):
                continue
            _execute_task(
                workspace, context, session, case, candidate, repeat,
                timeout=timeout, progress=progress,
            )
        session["status"] = "completed"
        session["completed_at"] = utc_now()
        session["evidence_root_sha256"] = session_evidence_root(
            session, workspace.list_runs(session["id"])
        )
        workspace.write_session(session)
        return session
    except BaseException:
        session["status"] = "interrupted"
        session["completed_at"] = utc_now()
        session.pop("evidence_root_sha256", None)
        workspace.write_session(session)
        raise


def run_experiment(
    workspace: Workspace,
    context: ColibriContext,
    suite: WorkloadSuite,
    *,
    repeats: int = 2,
    timeout: int = 900,
    candidates: tuple[Candidate, ...] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    if not 1 <= repeats <= 20:
        raise LatticeError("repeats must be between 1 and 20")
    if not 1 <= timeout <= 86400:
        raise LatticeError("timeout must be between 1 and 86400 seconds")
    progress = progress or (lambda _message: None)
    candidate_tuple = candidates or default_candidates(context.plan, context.base_environment)
    if not candidate_tuple or candidate_tuple[0].id != "baseline":
        raise LatticeError("candidate matrix must begin with baseline")
    session_seed = {
        "project": workspace.load_project().get("created_at"),
        "suite": suite.fingerprint,
        "candidates": [candidate.as_dict() for candidate in candidate_tuple],
        "repeats": repeats,
        "started_at_ns": time.time_ns(),
    }
    session_id = short_id("session", session_seed)
    with workspace.acquire_lock("operation"):
        session: dict[str, Any] = {
            "schema_version": 1,
            "id": session_id,
            "status": "running",
            "started_at": utc_now(),
            "completed_at": None,
            "suite_fingerprint": suite.fingerprint,
            "model_fingerprint": context.model_fingerprint,
            "runtime_fingerprint": context.runtime_fingerprint,
            "hardware_fingerprint": context.hardware_fingerprint,
            "execution_fingerprint": context.execution_fingerprint,
            "plan_fingerprint": context.plan_fingerprint,
            "replay_cap": context.replay_cap,
            "qualification_context": context.qualification_context,
            "oracle_policy": dict(ORACLE_POLICY),
            "repeats": repeats,
            "timeout_seconds": timeout,
            "candidates": [candidate.as_dict() for candidate in candidate_tuple],
            "replays": {},
            "run_ids": [],
        }
        workspace.write_session(session)

        def checkpoint(replays: dict[str, dict[str, Any]]) -> None:
            session["replays"] = dict(replays)
            workspace.write_session(session)

        try:
            session["replays"] = build_replays(
                workspace, context, suite, timeout=timeout, progress=progress,
                existing=session["replays"], checkpoint=checkpoint,
            )
            workspace.write_session(session)
            return _run_pending_tasks(
                workspace, context, suite, session, list(candidate_tuple),
                timeout=timeout, retry_failed=True, progress=progress,
            )
        except BaseException:
            session["status"] = "interrupted"
            session["completed_at"] = utc_now()
            session.pop("evidence_root_sha256", None)
            workspace.write_session(session)
            raise


def resume_experiment(
    workspace: Workspace,
    context: ColibriContext,
    suite: WorkloadSuite,
    session_id: str,
    *,
    timeout: int | None = None,
    retry_failed: bool = True,
    progress: Progress | None = None,
) -> dict[str, Any]:
    progress = progress or (lambda _message: None)
    with workspace.acquire_lock("operation"):
        project = workspace.load_project()
        session = workspace.load_session(session_id)
        if session.get("status") == "completed":
            raise LatticeError(f"session is already completed: {session_id}")
        runs = workspace.list_runs(session_id)
        candidate_defs = validate_session_evidence(
            workspace, project, suite, session, runs, require_complete=False,
        )
        candidates = [validate_candidate(Candidate(
            item["id"], item["description"], item["environment"],
        )) for item in candidate_defs.values()]
        effective_timeout = int(timeout or session.get("timeout_seconds") or 900)
        if not 1 <= effective_timeout <= 86400:
            raise LatticeError("timeout must be between 1 and 86400 seconds")

        session["status"] = "running"
        session["completed_at"] = None
        workspace.write_session(session)

        def checkpoint(replays: dict[str, dict[str, Any]]) -> None:
            session["replays"] = dict(replays)
            workspace.write_session(session)

        try:
            session["replays"] = build_replays(
                workspace, context, suite, timeout=effective_timeout, progress=progress,
                existing=session.get("replays") or {}, checkpoint=checkpoint,
            )
            workspace.write_session(session)
            return _run_pending_tasks(
                workspace, context, suite, session, candidates,
                timeout=effective_timeout, retry_failed=retry_failed, progress=progress,
            )
        except BaseException:
            session["status"] = "interrupted"
            session["completed_at"] = utc_now()
            session.pop("evidence_root_sha256", None)
            workspace.write_session(session)
            raise
