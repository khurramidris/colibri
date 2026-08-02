from __future__ import annotations

import math
from typing import Any

from .common import LatticeError, short_id, utc_now
from .integrity import validate_session_evidence
from .oracle import ORACLE_POLICY, compare_oracles, validate_oracle
from .stats import (
    STATISTICS_POLICY,
    CandidateScore,
    bonferroni_confidence,
    score_candidate,
)
from .suite import WorkloadSuite, parse_suite
from .workspace import Workspace

EXPLORATORY_ASSURANCE = "exploratory-point-estimate"
SCREENED_ASSURANCE = "confidence-screened-uncalibrated"
DEPLOYMENT_BLOCKER = (
    "numerical tolerances and statistical interval coverage have not been "
    "calibrated on real supported backends"
)


def _successful_samples(runs: list[dict[str, Any]], candidate_id: str) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {}
    for run in runs:
        if run.get("candidate_id") != candidate_id or run.get("status") != "success":
            continue
        case_id = run.get("case_id")
        repeat = run.get("repeat")
        metrics = run.get("metrics") or {}
        tok_s = metrics.get("tok_s")
        if (not isinstance(case_id, str) or isinstance(repeat, bool) or not isinstance(repeat, int)
                or isinstance(tok_s, bool) or not isinstance(tok_s, (int, float))
                or not math.isfinite(float(tok_s)) or tok_s <= 0):
            raise LatticeError(f"invalid successful throughput evidence for {candidate_id}")
        case_values = values.setdefault(case_id, {})
        if repeat in case_values:
            raise LatticeError(f"duplicate successful evidence for {candidate_id}/{case_id}/repeat-{repeat}")
        case_values[repeat] = float(tok_s)
    return values


def _successful_run_map(runs: list[dict[str, Any]], candidate_id: str) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        if run.get("candidate_id") != candidate_id or run.get("status") != "success":
            continue
        key = (run["case_id"], run["repeat"])
        if key in result:
            raise LatticeError(f"duplicate successful evidence for {candidate_id}/{key[0]}/repeat-{key[1]}")
        validate_oracle((run.get("metrics") or {}).get("oracle"))
        result[key] = run
    return result


def _baseline_oracle_stability(baseline_runs: dict[tuple[str, int], dict[str, Any]]) -> None:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for (case_id, _repeat), run in sorted(baseline_runs.items()):
        by_case.setdefault(case_id, []).append(run)
    for case_id, runs in by_case.items():
        reference = (runs[0]["metrics"] or {})["oracle"]
        for run in runs[1:]:
            mismatch = compare_oracles(reference, (run["metrics"] or {})["oracle"])
            if mismatch:
                raise LatticeError(f"baseline numerical oracle is unstable for {case_id}: {mismatch}")


def _candidate_oracle_mismatch(
    baseline_runs: dict[tuple[str, int], dict[str, Any]],
    candidate_runs: dict[tuple[str, int], dict[str, Any]],
) -> str | None:
    for key, run in sorted(candidate_runs.items()):
        baseline = baseline_runs.get(key)
        if baseline is None:
            continue
        mismatch = compare_oracles((baseline["metrics"] or {})["oracle"], (run["metrics"] or {})["oracle"])
        if mismatch:
            return f"numerical oracle mismatch for {key[0]}/repeat-{key[1]}: {mismatch}"
    return None


def _ineligible_oracle_score(candidate_id: str, reason: str) -> CandidateScore:
    return CandidateScore(candidate_id, False, reason, None, None, None, None, None, None, {})


def _finite_fraction(value: Any, label: str, *, lower: float, upper: float, inclusive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LatticeError(f"{label} must be finite")
    number = float(value)
    valid = lower <= number <= upper if inclusive else lower < number < upper
    if not valid:
        brackets = "between" if inclusive else "strictly between"
        raise LatticeError(f"{label} must be {brackets} {lower} and {upper}")
    return number


def evaluate_session(
    workspace: Workspace,
    session_id: str,
    *,
    min_runs: int = 3,
    min_gain: float = 0.03,
    max_regression: float = 0.05,
    confidence: float = 0.90,
    require_confidence: bool = True,
    hourly_cost_usd: float | None = None,
) -> dict[str, Any]:
    if isinstance(min_runs, bool) or not isinstance(min_runs, int) or not 1 <= min_runs <= 20:
        raise LatticeError("min_runs must be an integer between 1 and 20")
    if not isinstance(require_confidence, bool):
        raise LatticeError("require_confidence must be a boolean")
    minimum_confidence = int(STATISTICS_POLICY["minimum_confidence_runs_per_workload"])
    if require_confidence and min_runs < minimum_confidence:
        raise LatticeError(
            f"confidence-gated promotion requires at least {minimum_confidence} paired runs"
        )
    min_gain = _finite_fraction(min_gain, "min_gain", lower=0.0, upper=1.0)
    max_regression = _finite_fraction(max_regression, "max_regression", lower=0.0, upper=1.0)
    confidence = _finite_fraction(confidence, "confidence", lower=0.5, upper=1.0, inclusive=False)
    if hourly_cost_usd is not None:
        if (isinstance(hourly_cost_usd, bool) or not isinstance(hourly_cost_usd, (int, float))
                or not math.isfinite(float(hourly_cost_usd)) or hourly_cost_usd < 0):
            raise LatticeError("hourly_cost_usd must be finite and non-negative")
        hourly_cost_usd = float(hourly_cost_usd)
    project = workspace.load_project()
    suite: WorkloadSuite = parse_suite(project["suite"])
    session = workspace.load_session(session_id)
    if session.get("repeats", 0) < min_runs:
        raise LatticeError("promotion requires more runs than the session contains")
    runs = workspace.list_runs(session_id)
    candidate_defs = validate_session_evidence(workspace, project, suite, session, runs)
    candidate_count = len(candidate_defs) - 1
    per_candidate_confidence = (
        bonferroni_confidence(confidence, candidate_count)
        if candidate_count > 0 else None
    )
    statistics_policy = {
        **STATISTICS_POLICY,
        "familywise_confidence": confidence,
        "candidate_comparisons": candidate_count,
        "per_candidate_confidence": per_candidate_confidence,
        "confidence_required": require_confidence,
    }
    baseline = _successful_samples(runs, "baseline")
    baseline_runs = _successful_run_map(runs, "baseline")
    _baseline_oracle_stability(baseline_runs)
    weights = {case.id: case.weight for case in suite.cases}
    hourly = suite.hourly_cost_usd if hourly_cost_usd is None else hourly_cost_usd
    scores: list[CandidateScore] = []
    for candidate_id in candidate_defs:
        if candidate_id == "baseline":
            continue
        candidate_runs = _successful_run_map(runs, candidate_id)
        mismatch = _candidate_oracle_mismatch(baseline_runs, candidate_runs)
        if mismatch:
            scores.append(_ineligible_oracle_score(candidate_id, mismatch))
            continue
        assert per_candidate_confidence is not None
        scores.append(score_candidate(
            candidate_id,
            weights,
            baseline,
            _successful_samples(runs, candidate_id),
            min_runs=min_runs,
            min_gain=min_gain,
            max_regression=max_regression,
            confidence=per_candidate_confidence,
            require_confidence=require_confidence,
            hourly_cost_usd=hourly,
        ))
    eligible = [score for score in scores if score.eligible and score.weighted_speedup is not None]
    winner_score = max(eligible, key=lambda score: score.weighted_speedup or 0.0) if eligible else None
    winner_id = winner_score.candidate_id if winner_score else "baseline"
    assurance_level = SCREENED_ASSURANCE if require_confidence else EXPLORATORY_ASSURANCE
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
        "assurance_level": assurance_level,
        "deployable": False,
        "deployment_blocker": DEPLOYMENT_BLOCKER,
        "selection_policy": policy,
        "statistics_policy": statistics_policy,
        "oracle_policy": dict(ORACLE_POLICY),
        "model_fingerprint": project["model_fingerprint"],
        "runtime_fingerprint": project["runtime_fingerprint"],
        "hardware_fingerprint": project["hardware_fingerprint"],
        "execution_fingerprint": project["execution_fingerprint"],
        "plan_fingerprint": project["plan_fingerprint"],
        "replay_cap": project["replay_cap"],
        "qualification_context": project["qualification_context"],
        "suite_fingerprint": suite.fingerprint,
        "evidence_root_sha256": session["evidence_root_sha256"],
        "scores": [score.as_dict() for score in scores],
    }


def recommend(workspace: Workspace, session_id: str, **policy: Any) -> tuple[dict[str, Any], list[CandidateScore]]:
    evaluation = evaluate_session(workspace, session_id, **policy)
    profile_seed = {
        "session_id": session_id,
        "winner": evaluation["winner"]["id"],
        "policy": evaluation["selection_policy"],
        "statistics_policy": evaluation["statistics_policy"],
        "oracle_policy": evaluation["oracle_policy"],
        "assurance_level": evaluation["assurance_level"],
        "deployable": evaluation["deployable"],
        "evidence_root_sha256": evaluation["evidence_root_sha256"],
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
