from __future__ import annotations

import math
from typing import Any

from .common import LatticeError, short_id, utc_now
from .integrity import validate_session_evidence
from .stats import CandidateScore, score_candidate
from .suite import WorkloadSuite, parse_suite
from .workspace import Workspace


def _successful_samples(runs: list[dict[str, Any]], candidate_id: str) -> dict[str, list[float]]:
    values: dict[str, list[tuple[int, float]]] = {}
    seen: set[tuple[str, int]] = set()
    for run in runs:
        if run.get("candidate_id") != candidate_id or run.get("status") != "success":
            continue
        case_id = run.get("case_id")
        repeat = run.get("repeat")
        metrics = run.get("metrics") or {}
        tok_s = metrics.get("tok_s")
        if not isinstance(case_id, str) or not isinstance(repeat, int) or not isinstance(tok_s, (int, float)) or tok_s <= 0:
            continue
        key = (case_id, repeat)
        if key in seen:
            raise LatticeError(f"duplicate successful evidence for {candidate_id}/{case_id}/repeat-{repeat}")
        seen.add(key)
        values.setdefault(case_id, []).append((repeat, float(tok_s)))
    return {case_id: [value for _, value in sorted(samples)] for case_id, samples in values.items()}


def evaluate_session(
    workspace: Workspace,
    session_id: str,
    *,
    min_runs: int = 2,
    min_gain: float = 0.03,
    max_regression: float = 0.05,
    confidence: float = 0.90,
    require_confidence: bool = False,
    hourly_cost_usd: float | None = None,
) -> dict[str, Any]:
    if not 1 <= min_runs <= 20:
        raise LatticeError("min_runs must be between 1 and 20")
    if not 0 <= min_gain <= 1:
        raise LatticeError("min_gain must be between 0 and 1")
    if not 0 <= max_regression <= 1:
        raise LatticeError("max_regression must be between 0 and 1")
    if not 0.5 < confidence < 1:
        raise LatticeError("confidence must be between 0.5 and 1")
    if hourly_cost_usd is not None and (not math.isfinite(hourly_cost_usd) or hourly_cost_usd < 0):
        raise LatticeError("hourly_cost_usd must be finite and non-negative")
    project = workspace.load_project()
    suite: WorkloadSuite = parse_suite(project["suite"])
    session = workspace.load_session(session_id)
    if session.get("repeats", 0) < min_runs:
        raise LatticeError("promotion requires more runs than the session contains")
    runs = workspace.list_runs(session_id)
    candidate_defs = validate_session_evidence(workspace, project, suite, session, runs)
    baseline = _successful_samples(runs, "baseline")
    weights = {case.id: case.weight for case in suite.cases}
    hourly = suite.hourly_cost_usd if hourly_cost_usd is None else hourly_cost_usd
    scores: list[CandidateScore] = []
    for candidate_id in candidate_defs:
        if candidate_id == "baseline":
            continue
        scores.append(score_candidate(
            candidate_id,
            weights,
            baseline,
            _successful_samples(runs, candidate_id),
            min_runs=min_runs,
            min_gain=min_gain,
            max_regression=max_regression,
            confidence=confidence,
            require_confidence=require_confidence,
            hourly_cost_usd=hourly,
        ))
    eligible = [score for score in scores if score.eligible and score.weighted_speedup is not None]
    winner_score = max(eligible, key=lambda score: score.weighted_speedup or 0.0) if eligible else None
    winner_id = winner_score.candidate_id if winner_score else "baseline"
    policy = {
        "min_runs": min_runs,
        "min_gain": min_gain,
        "max_regression": max_regression,
        "confidence": confidence,
        "require_confidence": require_confidence,
        "hourly_cost_usd": hourly,
    }
    return {
        "session_id": session_id,
        "winner": candidate_defs[winner_id],
        "winner_score": None if winner_score is None else winner_score.as_dict(),
        "baseline_retained": winner_score is None,
        "selection_policy": policy,
        "model_fingerprint": project["model_fingerprint"],
        "runtime_fingerprint": project["runtime_fingerprint"],
        "suite_fingerprint": suite.fingerprint,
        "scores": [score.as_dict() for score in scores],
    }


def recommend(workspace: Workspace, session_id: str, **policy: Any) -> tuple[dict[str, Any], list[CandidateScore]]:
    evaluation = evaluate_session(workspace, session_id, **policy)
    seed_policy = evaluation["selection_policy"]
    profile_seed = {
        "session_id": session_id,
        "winner": evaluation["winner"]["id"],
        "policy": seed_policy,
    }
    profile = {
        "schema_version": 1,
        "id": short_id("profile", profile_seed),
        "created_at": utc_now(),
        **evaluation,
    }
    workspace.write_profile(profile)
    scores = [CandidateScore(
        candidate_id=item["candidate_id"],
        eligible=item["eligible"],
        reason=item["reason"],
        weighted_speedup=item["weighted_speedup"],
        effective_tok_s=item["effective_tok_s"],
        cost_per_million_usd=item["cost_per_million_usd"],
        worst_case_regression=item["worst_case_regression"],
        ci_low=None if item["confidence_interval"] is None else item["confidence_interval"][0],
        ci_high=None if item["confidence_interval"] is None else item["confidence_interval"][1],
        per_case=item["per_case"],
    ) for item in evaluation["scores"]]
    return profile, scores
