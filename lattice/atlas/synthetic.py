from __future__ import annotations

import random
from typing import Iterable

from .model import RouteRecord


def generate_routes(
    *,
    seed: int,
    requests: int,
    tokens: int,
    n_layers: int,
    n_experts: int,
    top_k: int,
    hot_experts: int,
    hot_probability: float,
    include_prefill: bool = True,
) -> Iterable[RouteRecord]:
    rng = random.Random(seed)
    hot_experts = max(top_k, min(hot_experts, n_experts))
    for request in range(requests):
        request_hot = {
            layer: rng.sample(range(n_experts), hot_experts)
            for layer in range(n_layers)
        }
        phases = [("prefill", max(2, tokens // 4))] if include_prefill else []
        phases.append(("decode", tokens))
        for phase, phase_tokens in phases:
            for token in range(phase_tokens):
                for layer in range(n_layers):
                    chosen: set[int] = set()
                    while len(chosen) < top_k:
                        if rng.random() < hot_probability:
                            chosen.add(rng.choice(request_hot[layer]))
                        else:
                            chosen.add(rng.randrange(n_experts))
                    experts = tuple(sorted(chosen))
                    scores = tuple(1.0 / top_k for _ in experts)
                    yield RouteRecord(
                        run_id="synthetic",
                        request_id=f"request-{request:04d}",
                        phase=phase,
                        token=token,
                        layer=layer,
                        experts=experts,
                        scores=scores,
                    )
