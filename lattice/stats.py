from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Iterable

from .common import LatticeError


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
    per_case: dict[str, dict[str, float]]

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
    if any(v <= 0 for v in values) or any(w <= 0 for w in ws):
        raise LatticeError("throughput and weights must be positive")
    return sum(ws) / sum(weight / value for value, weight in zip(values, ws))


def bootstrap_speedup(
    paired_ratios: list[tuple[float, float]],
    *,
    confidence: float,
    seed: int,
    samples: int = 2000,
) -> tuple[float, float]:
    """Bootstrap weighted geometric speedup from (ratio, case-weight) pairs."""
    if not paired_ratios:
        raise LatticeError("bootstrap requires paired ratios")
    if not 0.5 < confidence < 1.0:
        raise LatticeError("confidence must be between 0.5 and 1")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        draw = [paired_ratios[rng.randrange(len(paired_ratios))] for _ in paired_ratios]
        estimates.append(geometric_mean((ratio for ratio, _ in draw), (weight for _, weight in draw)))
    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(len(estimates) - 1, int(tail * len(estimates))))
    high_index = max(0, min(len(estimates) - 1, int((1.0 - tail) * len(estimates)) - 1))
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
    per_case: dict[str, dict[str, float]] = {}
    ratios_for_score: list[float] = []
    weights_for_score: list[float] = []
    paired: list[tuple[float, float]] = []
    candidate_medians: list[float] = []
    weights: list[float] = []
    for case_id, weight in case_weights.items():
        base_values = baseline.get(case_id, [])
        trial_values = candidate.get(case_id, [])
        if len(base_values) < min_runs or len(trial_values) < min_runs:
            return CandidateScore(candidate_id, False, f"insufficient successful runs for {case_id}", None, None, None, None, None, None, per_case)
        count = min(len(base_values), len(trial_values))
        if count < min_runs:
            return CandidateScore(candidate_id, False, f"insufficient paired runs for {case_id}", None, None, None, None, None, None, per_case)
        base_median = statistics.median(base_values)
        trial_median = statistics.median(trial_values)
        ratio = trial_median / base_median
        per_case[case_id] = {
            "baseline_tok_s": base_median,
            "candidate_tok_s": trial_median,
            "speedup": ratio,
            "regression": min(0.0, ratio - 1.0),
        }
        ratios_for_score.append(ratio)
        weights_for_score.append(weight)
        candidate_medians.append(trial_median)
        weights.append(weight)
        for index in range(count):
            paired.append((trial_values[index] / base_values[index], weight))
    worst = min(ratios_for_score) - 1.0
    weighted = geometric_mean(ratios_for_score, weights_for_score)
    effective = effective_throughput(candidate_medians, weights)
    cost = None
    if hourly_cost_usd is not None and effective > 0:
        cost = hourly_cost_usd * 1_000_000.0 / (effective * 3600.0)
    ci_low, ci_high = bootstrap_speedup(
        paired,
        confidence=confidence,
        seed=int.from_bytes(candidate_id.encode("utf-8"), "little", signed=False) % (2**32),
    )
    if worst < -max_regression:
        return CandidateScore(candidate_id, False, f"workload regression {worst:.2%} exceeds {max_regression:.2%}", weighted, effective, cost, worst, ci_low, ci_high, per_case)
    if weighted < 1.0 + min_gain:
        return CandidateScore(candidate_id, False, f"weighted gain {weighted - 1.0:.2%} is below {min_gain:.2%}", weighted, effective, cost, worst, ci_low, ci_high, per_case)
    if require_confidence and ci_low <= 1.0:
        return CandidateScore(candidate_id, False, f"confidence lower bound {ci_low - 1.0:.2%} does not clear zero", weighted, effective, cost, worst, ci_low, ci_high, per_case)
    return CandidateScore(candidate_id, True, "cleared all promotion gates", weighted, effective, cost, worst, ci_low, ci_high, per_case)
