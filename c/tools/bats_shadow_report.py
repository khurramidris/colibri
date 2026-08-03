#!/usr/bin/env python3
"""Summarize and calibrate OLMoE BATS shadow JSONL output."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED = {
    "layer",
    "row",
    "misses",
    "marginal_bytes",
    "predicted_transfer_us",
    "miss_get_us",
}


def records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            missing = REQUIRED - record.keys()
            if missing:
                raise SystemExit(f"{path}:{line_number}: missing fields: {', '.join(sorted(missing))}")
            yield record


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom <= 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def summarize(items: Iterable[dict]) -> dict:
    rows = list(items)
    miss_rows = [r for r in rows if int(r["misses"]) > 0 and float(r["miss_get_us"]) > 0.0]
    predicted = [float(r["predicted_transfer_us"]) for r in miss_rows]
    actual = [float(r["miss_get_us"]) for r in miss_rows]
    errors = [p - a for p, a in zip(predicted, actual)]
    abs_errors = [abs(e) for e in errors]
    sq_errors = [e * e for e in errors]

    total_bytes = sum(int(r["marginal_bytes"]) for r in miss_rows)
    total_actual = sum(actual)
    total_predicted = sum(predicted)
    total_misses = sum(int(r["misses"]) for r in miss_rows)

    return {
        "rows": len(rows),
        "miss_rows": len(miss_rows),
        "cache_only_rows": len(rows) - len(miss_rows),
        "total_misses": total_misses,
        "marginal_bytes": total_bytes,
        "predicted_transfer_us": total_predicted,
        "actual_miss_get_us": total_actual,
        "prediction_to_actual_ratio": (
            total_predicted / total_actual if total_actual > 0 else None
        ),
        "mean_error_us": sum(errors) / len(errors) if errors else None,
        "mae_us": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "rmse_us": math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else None,
        "mape": (
            sum(abs(p - a) / a for p, a in zip(predicted, actual)) / len(actual)
            if actual
            else None
        ),
        "pearson": pearson(predicted, actual),
        "observed_us_per_miss": total_actual / total_misses if total_misses else None,
        "effective_gbps_without_fixed_cost": (
            total_bytes / (total_actual * 1000.0)
            if total_bytes > 0 and total_actual > 0
            else None
        ),
    }


def report(items: Iterable[dict]) -> dict:
    rows = list(items)
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(row)
    return {
        "overall": summarize(rows),
        "by_layer": {
            str(layer): summarize(layer_rows)
            for layer, layer_rows in sorted(by_layer.items())
        },
        "interpretation": {
            "prediction_to_actual_ratio": "1.0 is calibrated; below 1 underpredicts exposed miss time.",
            "effective_gbps_without_fixed_cost": (
                "Descriptive only: folds fixed latency, queueing, page cache and overlap into one rate."
            ),
            "scope": (
                "Shadow data measures current OLMoE expert_get behavior. It does not establish "
                "speculative-decoding speedup until acceptance annotations and an A/B policy run exist."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shadow_log", type=Path)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--fail-mape-above", type=float)
    args = parser.parse_args()

    result = report(records(args.shadow_log))
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    mape = result["overall"]["mape"]
    if args.fail_mape_above is not None and mape is not None and mape > args.fail_mape_above:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
