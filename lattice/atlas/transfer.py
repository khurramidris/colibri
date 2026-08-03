from __future__ import annotations

import math
from statistics import mean, median
from typing import Iterable

from .model import TransferProfile


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return values[low]
    fraction = position - low
    return values[low] * (1 - fraction) + values[high] * fraction


def evaluate_transfer(cache_rows: Iterable[dict], profiles: Iterable[TransferProfile]) -> list[dict]:
    results: list[dict] = []
    for row in cache_rows:
        token_bytes = row.get("token_miss_bytes", {})
        token_layers = row.get("token_miss_layers", {})
        for profile in profiles:
            raw_values: list[float] = []
            exposed_values: list[float] = []
            for token, byte_count in token_bytes.items():
                layer_count = token_layers.get(token, 0)
                bandwidth_us = byte_count / (profile.bandwidth_gbps * 1e9) * 1e6
                raw = bandwidth_us + layer_count * profile.fixed_us_per_nonempty_layer
                exposed = raw * (1.0 - profile.overlap_fraction)
                raw_values.append(raw)
                exposed_values.append(exposed)
            results.append(
                {
                    "policy": row["policy"],
                    "capacity": row["capacity"],
                    "profile": profile.name,
                    "tokens_evaluated": len(raw_values),
                    "tokens_with_misses": sum(1 for value in raw_values if value > 0.0),
                    "raw_transfer_us": {
                        "mean": mean(raw_values) if raw_values else 0.0,
                        "median": median(raw_values) if raw_values else 0.0,
                        "p95": _percentile(raw_values, 0.95) or 0.0,
                    },
                    "exposed_transfer_us": {
                        "mean": mean(exposed_values) if exposed_values else 0.0,
                        "median": median(exposed_values) if exposed_values else 0.0,
                        "p95": _percentile(exposed_values, 0.95) or 0.0,
                    },
                }
            )
    return results
