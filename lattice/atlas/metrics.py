from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import mean, median
from typing import Iterable, Sequence

from .model import AtlasManifest, RouteRecord


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def normalized_entropy(counts: Counter[int], n_experts: int) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    maximum = math.log(n_experts)
    return entropy / maximum if maximum else 0.0


def gini(counts: Counter[int], n_experts: int) -> float:
    values = sorted(counts.get(index, 0) for index in range(n_experts))
    total = sum(values)
    if total == 0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(values))
    return (2.0 * weighted) / (n_experts * total) - (n_experts + 1.0) / n_experts


def route_metrics(records: Iterable[RouteRecord], manifest: AtlasManifest) -> dict:
    records = list(records)
    per_layer: dict[int, Counter[int]] = defaultdict(Counter)
    global_counts: Counter[int] = Counter()
    by_request_token_layer: dict[tuple[str, str, str, int, int], RouteRecord] = {}
    router_us: list[float] = []
    expert_us: list[float] = []
    for record in records:
        per_layer[record.layer].update(record.experts)
        global_counts.update(record.experts)
        by_request_token_layer[(record.run_id, record.request_id, record.phase, record.token, record.layer)] = record
        if record.router_ns is not None:
            router_us.append(record.router_ns / 1000.0)
        if record.expert_ns is not None:
            expert_us.append(record.expert_ns / 1000.0)

    coverage: dict[str, float] = {}
    for capacity in manifest.cache_capacities:
        hits = 0
        total = 0
        for layer in range(manifest.n_layers):
            hot = {expert for expert, _ in per_layer[layer].most_common(capacity)}
            hits += sum(count for expert, count in per_layer[layer].items() if expert in hot)
            total += sum(per_layer[layer].values())
        coverage[str(capacity)] = hits / total if total else 0.0

    overlaps: list[float] = []
    reuse_distances: list[int] = []
    grouped: dict[tuple[str, str, str, int], list[RouteRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.run_id, record.request_id, record.phase, record.layer)].append(record)
    for values in grouped.values():
        values.sort(key=lambda item: item.token)
        previous: set[int] | None = None
        positions: dict[int, int] = {}
        access_index = 0
        for record in values:
            current = set(record.experts)
            if previous is not None:
                union = current | previous
                overlaps.append(len(current & previous) / len(union) if union else 1.0)
            previous = current
            for expert in record.experts:
                if expert in positions:
                    reuse_distances.append(access_index - positions[expert])
                positions[expert] = access_index
                access_index += 1

    layer_rows = []
    for layer in range(manifest.n_layers):
        counts = per_layer[layer]
        total = sum(counts.values())
        layer_rows.append(
            {
                "layer": layer,
                "selections": total,
                "normalized_entropy": normalized_entropy(counts, manifest.n_experts),
                "gini": gini(counts, manifest.n_experts),
                "top_8_coverage": sum(value for _, value in counts.most_common(min(8, manifest.n_experts))) / total if total else 0.0,
                "top_32_coverage": sum(value for _, value in counts.most_common(min(32, manifest.n_experts))) / total if total else 0.0,
                "hottest": [{"expert": expert, "count": count} for expert, count in counts.most_common(8)],
            }
        )

    return {
        "records": len(records),
        "requests": len({(r.run_id, r.request_id) for r in records}),
        "tokens": len({(r.run_id, r.request_id, r.phase, r.token) for r in records}),
        "global_normalized_entropy": normalized_entropy(global_counts, manifest.n_experts),
        "global_gini": gini(global_counts, manifest.n_experts),
        "static_top_capacity_coverage": coverage,
        "adjacent_token_jaccard": {
            "mean": mean(overlaps) if overlaps else None,
            "median": median(overlaps) if overlaps else None,
            "p10": _percentile(overlaps, 0.10),
            "p90": _percentile(overlaps, 0.90),
            "samples": len(overlaps),
        },
        "reuse_distance": {
            "median": median(reuse_distances) if reuse_distances else None,
            "p90": _percentile(reuse_distances, 0.90),
            "p99": _percentile(reuse_distances, 0.99),
            "samples": len(reuse_distances),
        },
        "timing_us": {
            "router_median": median(router_us) if router_us else None,
            "router_p95": _percentile(router_us, 0.95),
            "expert_median": median(expert_us) if expert_us else None,
            "expert_p95": _percentile(expert_us, 0.95),
        },
        "layers": layer_rows,
    }
