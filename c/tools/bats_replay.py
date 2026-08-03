#!/usr/bin/env python3
"""Replay BATS candidate decisions from dependency-free JSONL traces."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TIERS = {"exec", "vram", "pinned", "ram", "nvme", "remote"}


@dataclass(frozen=True)
class TierCost:
    bandwidth_gbps: float
    fixed_us: float = 0.0
    queue_us: float = 0.0
    overlap: float = 0.0


def transfer_us(expert: dict, tiers: dict[str, TierCost]) -> float:
    if expert.get("resident_in_exec") or expert["tier"] == "exec":
        return 0.0
    if expert.get("in_flight"):
        return max(0.0, float(expert.get("remaining_us", 0.0)))
    tier = expert["tier"]
    if tier not in tiers or tiers[tier].bandwidth_gbps <= 0:
        return math.inf
    cfg = tiers[tier]
    payload = float(expert["bytes"]) / (cfg.bandwidth_gbps * 1000.0)
    return cfg.fixed_us + cfg.queue_us + payload * (1.0 - min(1.0, max(0.0, cfg.overlap)))


def choose(record: dict) -> dict:
    experts = {int(e["id"]): e for e in record["experts"]}
    tiers = {
        name: TierCost(**cfg)
        for name, cfg in record["hardware"].items()
        if name in TIERS
    }
    selected: list[int] = []
    union: set[int] = set()
    total_us = 0.0
    total_bytes = 0
    total_tokens = 0.0
    budget = float(record.get("budget_us", 0.0))
    limit = min(int(record.get("max_candidates", len(record["candidates"]))), len(record["candidates"]))

    while len(selected) < limit:
        best = None
        for index, candidate in enumerate(record["candidates"]):
            if index in selected:
                continue
            new = set(map(int, candidate["experts"])) - union
            cost = float(candidate.get("verify_compute_us", 0.0)) + float(candidate.get("kv_us", 0.0))
            byte_cost = 0
            for eid in new:
                expert = experts[eid]
                cost += transfer_us(expert, tiers)
                if not expert.get("resident_in_exec") and expert["tier"] != "exec":
                    byte_cost += int(expert["bytes"])
            if budget > 0 and total_us + cost > budget:
                continue
            expected = float(candidate["expected_accepted_tokens"])
            score = expected / (cost + 1e-9)
            key = (score, -int(candidate.get("id", index)), -index)
            if best is None or key > best[0]:
                best = (key, index, cost, byte_cost, expected, new)
        if best is None:
            break
        _, index, cost, byte_cost, expected, new = best
        selected.append(index)
        union.update(new)
        total_us += cost
        total_bytes += byte_cost
        total_tokens += expected

    return {
        "selected": selected,
        "predicted_us": total_us,
        "marginal_bytes": total_bytes,
        "expected_accepted_tokens": total_tokens,
        "objective_tokens_per_us": total_tokens / (total_us + 1e-9),
    }


def records(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    for index, record in enumerate(records(args.trace)):
        output = {"record": index, **choose(record)}
        print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
