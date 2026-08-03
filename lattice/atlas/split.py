from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from .model import AtlasError, RouteRecord


def stable_request_split(
    records: Iterable[RouteRecord], train_fraction: float
) -> tuple[list[RouteRecord], list[RouteRecord], dict[str, str]]:
    by_request: dict[tuple[str, str], list[RouteRecord]] = defaultdict(list)
    for record in records:
        by_request[(record.run_id, record.request_id)].append(record)
    if len(by_request) < 2:
        raise AtlasError("at least two requests are required for held-out evaluation")
    ranked = sorted(
        by_request,
        key=lambda key: hashlib.sha256(f"{key[0]}\0{key[1]}".encode()).digest(),
    )
    train_count = round(len(ranked) * train_fraction)
    train_count = min(max(train_count, 1), len(ranked) - 1)
    train_keys = set(ranked[:train_count])
    assignment = {
        f"{run_id}/{request_id}": ("train" if (run_id, request_id) in train_keys else "test")
        for run_id, request_id in ranked
    }
    train: list[RouteRecord] = []
    test: list[RouteRecord] = []
    for key, values in by_request.items():
        (train if key in train_keys else test).extend(values)
    order = lambda r: (r.run_id, r.request_id, r.phase, r.token, r.layer)
    return sorted(train, key=order), sorted(test, key=order), assignment
