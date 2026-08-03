from __future__ import annotations

from typing import Any

DEFAULT_GATES = {
    "cache_practical_hit_rate": 0.60,
    "cache_oracle_hit_rate": 0.75,
    "transfer_reduction": 0.40,
    "prefetch_recall": 0.85,
    "prefetch_max_overfetch": 0.20,
    "ternary_safe_fraction": 0.50,
}


def _gate_values(overrides: dict[str, float]) -> dict[str, float]:
    result = dict(DEFAULT_GATES)
    result.update(overrides)
    return result


def evaluate_decision(
    cache_rows: list[dict],
    transfer_rows: list[dict],
    prefetch_rows: list[dict],
    ternary_summary: dict[str, Any] | None,
    gate_overrides: dict[str, float],
    *,
    target_capacity: int,
) -> dict:
    gates = _gate_values(gate_overrides)
    practical = [row for row in cache_rows if row["policy"] != "belady_oracle" and row["capacity"] == target_capacity]
    oracle = [row for row in cache_rows if row["policy"] == "belady_oracle" and row["capacity"] == target_capacity]
    best_practical = max(practical, key=lambda row: row["hit_rate"])
    best_oracle = max(oracle, key=lambda row: row["hit_rate"])

    transfer_by_key = {(row["policy"], row["capacity"], row["profile"]): row for row in transfer_rows}
    reductions: list[dict] = []
    profiles = sorted({row["profile"] for row in transfer_rows})
    for profile in profiles:
        baseline_candidates = [row for row in transfer_rows if row["profile"] == profile and row["capacity"] == 0]
        if not baseline_candidates:
            continue
        baseline = max(item["exposed_transfer_us"]["mean"] for item in baseline_candidates)
        for row in practical:
            transfer = transfer_by_key[(row["policy"], row["capacity"], profile)]
            current = transfer["exposed_transfer_us"]["mean"]
            reduction = 1.0 - current / baseline if baseline > 0 else 0.0
            reductions.append({"profile": profile, "policy": row["policy"], "capacity": row["capacity"], "reduction": reduction})
    best_reduction = max(reductions, key=lambda row: row["reduction"]) if reductions else {"reduction": 0.0}

    predictors = [row for row in prefetch_rows if row["predictor"] not in {"oracle"}]
    best_predictor = max(predictors, key=lambda row: (row["recall"], row["precision"]))

    cache_checks = {
        "practical_hit_rate": best_practical["hit_rate"] >= gates["cache_practical_hit_rate"],
        "oracle_hit_rate": best_oracle["hit_rate"] >= gates["cache_oracle_hit_rate"],
        "transfer_reduction": best_reduction["reduction"] >= gates["transfer_reduction"],
        "prefetch_recall": best_predictor["recall"] >= gates["prefetch_recall"],
        "prefetch_overfetch": best_predictor["overfetch_fraction"] <= gates["prefetch_max_overfetch"],
    }
    cache_pass = all(cache_checks.values())
    if ternary_summary is None:
        ternary = {"status": "not_measured", "pass": None}
    else:
        ternary = {
            "status": "measured",
            "safe_fraction": ternary_summary["safe_fraction"],
            "pass": ternary_summary["safe_fraction"] >= gates["ternary_safe_fraction"],
        }
    return {
        "gates": gates,
        "target_cache_capacity": target_capacity,
        "cache_prefetch": {
            "decision": "go" if cache_pass else "no-go",
            "checks": cache_checks,
            "best_practical_cache": best_practical,
            "best_oracle_cache": best_oracle,
            "best_transfer_reduction": best_reduction,
            "best_prefetch_predictor": best_predictor,
        },
        "ternary": ternary,
        "overall": "conditional-go" if cache_pass and ternary.get("pass") is not False else "hold",
    }
