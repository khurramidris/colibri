from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable

from .common import LatticeError

STATISTICS_POLICY = {
    "schema": "lattice-statistics/2",
    "point_estimator": "weighted_geometric_of_paired_case_medians",
    "bootstrap": "stratified_paired_case_medians",
    "bootstrap_samples": 5000,
    "multiple_comparisons": "bonferroni_candidates",
    "minimum_confidence_runs": 3,
}


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    eligible: bool
    reason: str
    weighted_speedup: float | None
    effective_tok_s: float | None
    cost_per_million_usd: float | None
    worst_case_regression: float | None
    ci_low: float | None
    ci_high: float | None
    per_case: dict[str, dict[str, float | int]]

    def as_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "eligible": self.eligible,
            "reason": self.reason,
            "weighted_speedup": self.weighted_speedup,
            "effective_tok_s": self.effective_tok_s,
            "cost_per_million_usd": self.cost_per_million_usd,
            "worst_case_regression": self.worst_case_regression,
            "confidence_interval": None if self.ci_low is None else [self.ci_low, self.ci_high],
            "per_case": self.per_case,
        }


def geometric_mean(values: Iterable[float], weights: Iterable[float] | None = None) -> float:
    vals = list(values)
    if not vals or any(value <= 0 or not math.isfinite(value) for value in vals):
        raise LatticeError("geometric mean requires finite positive values")
    if weights is None:
        return math.exp(sum(math.log(value) for value in vals) / len(vals))
    ws = list(weights)
    if len(ws) != len(vals) or any(weight <= 0 or not math.isfinite(weight) for weight in ws):
        raise LatticeError("weights must be finite, positive and match values")
    total = sum(ws)
    return math.exp(sum(weight * math.log(value) for value, weight in zip(vals, ws)) / total)


def effective_throughput(case_tps: Iterable[float], weights: Iterable[float]) -> float:
    values, ws = list(case_tps), list(weights)
    if len(values) != len(ws) or not values:
        raise LatticeError("effective throughput requires matching non-empty values and weights")
    if any(v <= 0 or not math.isfinite(v) for v in values) or any(w <= 0 or not math.isfinite(w) for w in ws):
        raise LatticeError("throughput and weights must be finite and positive")
    return sum(ws) / sum(weight / value for value, weight in zip(values, ws))


def bonferroni_confidence(familywise_confidence: float, comparisons: int) -> float:
    if not 0.5 < familywise_confidence < 1.0:
        raise LatticeError("confidence must be between 0.5 and 1")
    if isinstance(comparisons, bool) or not isinstance(comparisons, int) or comparisons < 1:
        raise LatticeError("comparisons must be a positive integer")
    return 1.0 - (1.0 - familywise_confidence) / comparisons


def bootstrap_speedup(
    paired_by_case: dict[str, list[float]],
    case_weights: dict[str, float],
    *,
    confidence: float,
    seed: int,
    samples: int = 5000,
) -> tuple[float, float]:
    """Stratified paired bootstrap of the workload-weighted speedup.

    Repeats are resampled independently within each workload. Each bootstrap
    draw computes the median paired ratio for every workload and then applies
    the declared workload weights. This preserves workload composition instead
    of allowing workloads with more observations to dominate the interval.
    """
    if not paired_by_case or set(paired_by_case) != set(case_weights):
        raise LatticeError("bootstrap requires matching workload ratios and weights")
    if not 0.5 < confidence < 1.0:
        raise LatticeError("confidence must be between 0.5 and 1")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 100:
        raise LatticeError("bootstrap samples must be an integer of at least 100")
    ordered_cases = list(case_weights)
    for case_id in ordered_cases:
        ratios = paired_by_case[case_id]
        weight = case_weights[case_id]
        if (not ratios or any(ratio <= 0 or not math.isfinite(ratio) for ratio in ratios)
                or weight <= 0 or not math.isfinite(weight)):
            raise LatticeError("bootstrap ratios and weights must be finite and positive")
    rng = random.Random(seed)
    estimates: list[float] = []
    weights = [case_weights[case_id] for case_id in ordered_cases]
    for _ in range(samples):
        case_medians: list[float] = []
        for case_id in ordered_cases:
            ratios = paired_by_case[case_id]
            draw = [ratios[rng.randrange(len(ratios))] for _ in ratios]
            case_medians.append(statistics.median(draw))
        estimates.append(geometric_mean(case_medians, weights))
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(len(estimates) - 1, math.floor(tail * len(estimates))))
    high_index = max(0, min(len(estimates) - 1, math.ceil((1.0 - tail) * len(estimates)) - 1))
    return estimates[low_index], estimates[high_index]


def score_candidate(
    candidate_id: str,
    case_weights: dict[str, float],
    baseline: dict[str, list[float]],
    candidate: dict[str, list[float]],
    *,
    min_runs: int,
    min_gain: float,
    max_regression: float,
    confidence: float,
    require_confidence: bool,
    hourly_cost_usd: float | None,
) -> CandidateScore:
    per_case: dict[str, dict[str, float | int]] = {}
    case_speedups: list[float] = []
    weights_for_score: list[float] = []
    paired_by_case: dict[str, list[float]] = {}
    candidate_medians: list[float] = []
    throughput_weights: list[float] = []
    for case_id, weight in case_weights.items():
        base_values = baseline.get(case_id, [])
        trial_values = candidate.get(case_id, [])
        if len(base_values) < min_runs or len(trial_values) < min_runs:
            return CandidateScore(candidate_id, False, f"insufficient successful runs for {case_id}", None, None, None, None, None, None, per_case)
        count = min(len(base_values), len(trial_values))
        if count < min_runs:
            return CandidateScore(candidate_id, False, f"insufficient paired runs for {case_id}", None, None, None, None, None, None, per_case)
        pairs = []
        for index in range(count):
            base_value, trial_value = base_values[index], trial_values[index]
            if (base_value <= 0 or trial_value <= 0 or not math.isfinite(base_value)
                    or not math.isfinite(trial_value)):
                raise LatticeError(f"non-positive or non-finite throughput for {case_id}")
            pairs.append(trial_value / base_value)
        speedup = statistics.median(pairs)
        base_median = statistics.median(base_values[:count])
        trial_median = statistics.median(trial_values[:count])
        per_case[case_id] = {
            "baseline_tok_s": base_median,
            "candidate_tok_s": trial_median,
            "speedup": speedup,
            "regression": min(0.0, speedup - 1.0),
            "paired_runs": count,
        }
        paired_by_case[case_id] = pairs
        case_speedups.append(speedup)
        weights_for_score.append(weight)
        candidate_medians.append(trial_median)
        throughput_weights.append(weight)
    worst = min(case_speedups) - 1.0
    weighted = geometric_mean(case_speedups, weights_for_score)
    effective = effective_throughput(candidate_medians, throughput_weights)
    cost = None
    if hourly_cost_usd is not None and effective > 0:
        cost = hourly_cost_usd * 1_000_000.0 / (effective * 3600.0)
    ci_low, ci_high = bootstrap_speedup(
        paired_by_case,
        case_weights,
        confidence=confidence,
        seed=int.from_bytes(candidate_id.encode("utf-8"), "little", signed=False) % (2**32),
        samples=int(STATISTICS_POLICY["bootstrap_samples"]),
    )
    if worst < -max_regression:
        return CandidateScore(candidate_id, False, f"workload regression {worst:.2%} exceeds {max_regression:.2%}", weighted, effective, cost, worst, ci_low, ci_high, per_case)
    if weighted < 1.0 + min_gain:
        return CandidateScore(candidate_id, False, f"weighted gain {weighted - 1.0:.2%} is below {min_gain:.2%}", weighted, effective, cost, worst, ci_low, ci_high, per_case)
    if require_confidence and ci_low <= 1.0:
        return CandidateScore(candidate_id, False, f"adjusted confidence lower bound {ci_low - 1.0:.2%} does not clear zero", weighted, effective, cost, worst, ci_low, ci_high, per_case)
    return CandidateScore(candidate_id, True, "cleared all promotion gates", weighted, effective, cost, worst, ci_low, ci_high, per_case)
