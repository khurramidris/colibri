#!/usr/bin/env python3
"""Accounting model for Fareed-inspired compressed trunk streaming.

This predicts bytes and Amdahl-style speedup only. It is not a benchmark.
All quantities are GiB or fractions in [0, 1].
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Inputs:
    compressed_trunk_gib: float
    usable_ram_gib: float
    reserved_nontrunk_gib: float
    expert_traffic_gib_per_token: float
    exact_trunk_traffic_gib_per_token: float
    io_fraction_of_baseline: float = 0.75
    prefetch_overlap_fraction: float = 0.0


def evaluate(x: Inputs) -> dict[str, float]:
    values = asdict(x)
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if not 0.0 <= x.io_fraction_of_baseline <= 1.0:
        raise ValueError("io_fraction_of_baseline must be in [0, 1]")
    if not 0.0 <= x.prefetch_overlap_fraction <= 1.0:
        raise ValueError("prefetch_overlap_fraction must be in [0, 1]")

    available_for_trunk = max(0.0, x.usable_ram_gib - x.reserved_nontrunk_gib)
    resident_trunk = min(x.compressed_trunk_gib, available_for_trunk)
    streamed_trunk = max(0.0, x.compressed_trunk_gib - resident_trunk)

    exact_total = (
        x.exact_trunk_traffic_gib_per_token
        + x.expert_traffic_gib_per_token
    )
    candidate_total = streamed_trunk + x.expert_traffic_gib_per_token

    if candidate_total == 0.0:
        traffic_speedup = float("inf") if exact_total > 0.0 else 1.0
    else:
        traffic_speedup = exact_total / candidate_total

    visible_candidate_io = (
        streamed_trunk * (1.0 - x.prefetch_overlap_fraction)
        + x.expert_traffic_gib_per_token
    )
    io_ratio = visible_candidate_io / exact_total if exact_total else 0.0
    f = x.io_fraction_of_baseline
    predicted_speedup = 1.0 / ((1.0 - f) + f * io_ratio)

    return {
        "resident_trunk_gib": resident_trunk,
        "streamed_trunk_gib_per_token": streamed_trunk,
        "exact_total_storage_gib_per_token": exact_total,
        "candidate_total_storage_gib_per_token": candidate_total,
        "storage_traffic_speedup": traffic_speedup,
        "visible_candidate_io_ratio": io_ratio,
        "amdahl_predicted_speedup": predicted_speedup,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compressed-trunk-gib", type=float, required=True)
    parser.add_argument("--usable-ram-gib", type=float, required=True)
    parser.add_argument("--reserved-nontrunk-gib", type=float, required=True)
    parser.add_argument("--expert-traffic-gib", type=float, required=True)
    parser.add_argument("--exact-trunk-traffic-gib", type=float, required=True)
    parser.add_argument("--io-fraction", type=float, default=0.75)
    parser.add_argument("--prefetch-overlap", type=float, default=0.0)
    args = parser.parse_args()

    inputs = Inputs(
        compressed_trunk_gib=args.compressed_trunk_gib,
        usable_ram_gib=args.usable_ram_gib,
        reserved_nontrunk_gib=args.reserved_nontrunk_gib,
        expert_traffic_gib_per_token=args.expert_traffic_gib,
        exact_trunk_traffic_gib_per_token=args.exact_trunk_traffic_gib,
        io_fraction_of_baseline=args.io_fraction,
        prefetch_overlap_fraction=args.prefetch_overlap,
    )
    print(json.dumps({"inputs": asdict(inputs), "result": evaluate(inputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
