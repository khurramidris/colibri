#!/usr/bin/env python3
"""Convert Colibri ROUTE_TRACE output into BATS JSONL shadow-planning records."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO

TIERS = {"exec", "vram", "pinned", "ram", "nvme", "remote"}


@dataclass(frozen=True)
class RouteRow:
    call: int
    row: int
    layer: int
    expert_ids: tuple[int, ...]
    gates: tuple[float, ...]


def parse_route_line(line: str, *, source: str = "<stream>", line_number: int = 0) -> RouteRow:
    parts = line.split()
    where = f"{source}:{line_number}" if line_number else source
    if len(parts) < 4:
        raise ValueError(f"{where}: expected '<call> <row> <layer> <id>:<gate> ...'")
    try:
        call, row, layer = map(int, parts[:3])
    except ValueError as exc:
        raise ValueError(f"{where}: call, row and layer must be integers") from exc

    ids: list[int] = []
    gates: list[float] = []
    seen: set[int] = set()
    for token in parts[3:]:
        if ":" not in token:
            raise ValueError(f"{where}: malformed expert token {token!r}")
        eid_text, gate_text = token.split(":", 1)
        try:
            eid = int(eid_text)
            gate = float(gate_text)
        except ValueError as exc:
            raise ValueError(f"{where}: malformed expert token {token!r}") from exc
        if eid < 0:
            raise ValueError(f"{where}: expert id must be non-negative")
        if eid in seen:
            raise ValueError(f"{where}: duplicate expert id {eid}")
        seen.add(eid)
        ids.append(eid)
        gates.append(gate)
    return RouteRow(call, row, layer, tuple(ids), tuple(gates))


def read_routes(path: Path) -> Iterable[RouteRow]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield parse_route_line(line, source=str(path), line_number=line_number)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc


def read_residency(path: Path | None) -> dict[tuple[int, int], dict]:
    """Read '<layer> <expert> <tier> [remaining_us]' records."""
    if path is None:
        return {}
    result: dict[tuple[int, int], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) not in (3, 4):
                raise SystemExit(
                    f"{path}:{line_number}: expected '<layer> <expert> <tier> [remaining_us]'"
                )
            try:
                layer, expert = int(parts[0]), int(parts[1])
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_number}: layer/expert must be integers") from exc
            tier = parts[2].lower()
            if tier not in TIERS:
                raise SystemExit(f"{path}:{line_number}: unknown tier {tier!r}")
            state = {"tier": tier}
            if tier == "exec":
                state["resident_in_exec"] = True
            if len(parts) == 4:
                try:
                    remaining_us = float(parts[3])
                except ValueError as exc:
                    raise SystemExit(f"{path}:{line_number}: remaining_us must be numeric") from exc
                if remaining_us < 0:
                    raise SystemExit(f"{path}:{line_number}: remaining_us must be non-negative")
                state["in_flight"] = True
                state["remaining_us"] = remaining_us
            result[(layer, expert)] = state
    return result


def read_acceptance(path: Path | None) -> dict[tuple[int, int], dict]:
    """Read optional JSONL annotations keyed by call and row."""
    if path is None:
        return {}
    result: dict[tuple[int, int], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
                key = (int(record["call"]), int(record["row"]))
                expected = float(record["expected_accepted_tokens"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"{path}:{line_number}: expected JSON with call, row, expected_accepted_tokens"
                ) from exc
            if expected < 0:
                raise SystemExit(f"{path}:{line_number}: expected_accepted_tokens must be non-negative")
            result[key] = {
                "expected_accepted_tokens": expected,
                "verify_compute_us": float(record.get("verify_compute_us", 0.0)),
                "kv_us": float(record.get("kv_us", 0.0)),
            }
    return result


def build_records(
    routes: Iterable[RouteRow],
    *,
    n_experts: int,
    expert_bytes: int,
    default_tier: str,
    hardware: dict,
    residency: dict[tuple[int, int], dict] | None = None,
    acceptance: dict[tuple[int, int], dict] | None = None,
    max_candidates: int = 0,
    budget_us: float = 0.0,
) -> Iterable[dict]:
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    if expert_bytes <= 0:
        raise ValueError("expert_bytes must be positive")
    if default_tier not in TIERS:
        raise ValueError(f"unknown default tier {default_tier!r}")
    residency = residency or {}
    acceptance = acceptance or {}

    grouped: dict[int, list[RouteRow]] = {}
    order: list[int] = []
    for route in routes:
        if route.call not in grouped:
            grouped[route.call] = []
            order.append(route.call)
        grouped[route.call].append(route)

    for call in order:
        rows = sorted(grouped[call], key=lambda item: item.row)
        expert_states: dict[int, dict] = {}
        candidates: list[dict] = []
        used_default_acceptance = False
        for route in rows:
            global_ids: list[int] = []
            for local_eid in route.expert_ids:
                if local_eid >= n_experts:
                    raise ValueError(
                        f"call {call} row {route.row}: expert {local_eid} >= n_experts {n_experts}"
                    )
                gid = route.layer * n_experts + local_eid
                global_ids.append(gid)
                state = {
                    "id": gid,
                    "bytes": expert_bytes,
                    "tier": default_tier,
                    "layer": route.layer,
                    "local_expert": local_eid,
                }
                state.update(residency.get((route.layer, local_eid), {}))
                expert_states[gid] = state

            annotation = acceptance.get((call, route.row))
            if annotation is None:
                used_default_acceptance = True
                annotation = {
                    "expected_accepted_tokens": 1.0,
                    "verify_compute_us": 0.0,
                    "kv_us": 0.0,
                }
            candidates.append(
                {
                    "id": route.row,
                    "expected_accepted_tokens": annotation["expected_accepted_tokens"],
                    "verify_compute_us": annotation["verify_compute_us"],
                    "kv_us": annotation["kv_us"],
                    "experts": global_ids,
                    "gate_mass": sum(route.gates),
                    "layer": route.layer,
                }
            )

        record = {
            "hardware": hardware,
            "experts": [expert_states[eid] for eid in sorted(expert_states)],
            "candidates": candidates,
            "max_candidates": max_candidates or len(candidates),
            "budget_us": budget_us,
            "_meta": {
                "source": "colibri-route-trace",
                "call": call,
                "n_experts": n_experts,
                "default_acceptance": used_default_acceptance,
                "note": (
                    "Default acceptance=1.0 is suitable only for transfer-cost shadow analysis; "
                    "provide --acceptance for speculative-decoding experiments."
                ),
            },
        }
        yield record


def hardware_from_args(args: argparse.Namespace) -> dict:
    return {
        args.default_tier: {
            "bandwidth_gbps": args.bandwidth_gbps,
            "fixed_us": args.fixed_us,
            "queue_us": args.queue_us,
            "overlap": args.overlap,
        },
        "exec": {"bandwidth_gbps": 1.0},
    }


def write_records(records: Iterable[dict], handle: TextIO, *, pretty: bool) -> None:
    for record in records:
        if pretty:
            handle.write(json.dumps(record, sort_keys=True, indent=2))
            handle.write("\n")
        else:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert ROUTE_TRACE rows into BATS JSONL shadow-planning records."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--expert-bytes", type=int, required=True)
    parser.add_argument("--default-tier", choices=sorted(TIERS), default="nvme")
    parser.add_argument("--bandwidth-gbps", type=float, required=True)
    parser.add_argument("--fixed-us", type=float, default=0.0)
    parser.add_argument("--queue-us", type=float, default=0.0)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--residency", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--budget-us", type=float, default=0.0)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    if args.bandwidth_gbps <= 0:
        parser.error("--bandwidth-gbps must be positive")
    if not 0.0 <= args.overlap <= 1.0:
        parser.error("--overlap must be between 0 and 1")
    if args.fixed_us < 0 or args.queue_us < 0 or args.budget_us < 0:
        parser.error("latency and budget values must be non-negative")

    converted = build_records(
        read_routes(args.trace),
        n_experts=args.n_experts,
        expert_bytes=args.expert_bytes,
        default_tier=args.default_tier,
        hardware=hardware_from_args(args),
        residency=read_residency(args.residency),
        acceptance=read_acceptance(args.acceptance),
        max_candidates=args.max_candidates,
        budget_us=args.budget_us,
    )
    write_records(converted, handle=__import__("sys").stdout, pretty=args.pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
