from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .colibri import SAFE_TUNABLE_KEYS
from .common import LatticeError, validate_id


@dataclass(frozen=True)
class Candidate:
    id: str
    description: str
    environment: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "environment": dict(sorted(self.environment.items())),
        }


def validate_candidate(candidate: Candidate) -> Candidate:
    validate_id(candidate.id, "candidate id")
    if not candidate.description or len(candidate.description) > 240:
        raise LatticeError(f"candidate {candidate.id} requires a description up to 240 characters")
    unknown = set(candidate.environment) - SAFE_TUNABLE_KEYS
    if unknown:
        raise LatticeError(f"candidate {candidate.id} uses forbidden keys: {', '.join(sorted(unknown))}")
    clean: dict[str, str] = {}
    for key, value in candidate.environment.items():
        text = str(value)
        if not text or len(text) > 128 or any(ch in text for ch in "\r\n\x00"):
            raise LatticeError(f"candidate {candidate.id} has invalid value for {key}")
        clean[key] = text
    return Candidate(candidate.id, candidate.description, clean)


def default_candidates(plan: dict[str, Any], base_environment: dict[str, str] | None = None) -> tuple[Candidate, ...]:
    candidates: list[Candidate] = [
        Candidate("baseline", "Colibri's generated quality-preserving plan", {}),
    ]
    cores = max(1, int(plan.get("cpu", {}).get("physical_cores") or os.cpu_count() or 1))
    if cores > 1:
        candidates.append(Candidate(
            "half-threads",
            "Use half of the detected physical cores to test memory-bandwidth saturation",
            {"OMP_NUM_THREADS": str(max(1, cores // 2))},
        ))
    sockets = max(1, int(plan.get("cpu", {}).get("sockets") or 1))
    if sockets > 1:
        candidates.append(Candidate(
            "numa-interleave",
            "Interleave expert slabs across NUMA nodes",
            {"COLI_NUMA": "1"},
        ))
    disk = plan.get("tiers", {}).get("disk", {})
    if int(disk.get("cold_expert_bytes") or 0) > 0:
        candidates.extend([
            Candidate("io-pipeline", "Overlap expert reads with resident computation", {"PIPE": "1"}),
            Candidate(
                "direct-pipeline",
                "Combine queued expert loading with direct/unbuffered storage reads",
                {"PIPE": "1", "DIRECT": "1"},
            ),
            Candidate(
                "pilot-real",
                "Use value-preserving cross-layer prefetch with real expert loads",
                {"PIPE": "1", "PILOT": "1", "PILOT_REAL": "1"},
            ),
        ])
        if os.name != "nt" and os.uname().sysname.lower() == "linux":
            candidates.append(Candidate(
                "io-uring",
                "Use Linux queued expert I/O with direct reads",
                {"PIPE": "1", "DIRECT": "1", "URING": "1"},
            ))
    devices = plan.get("tiers", {}).get("vram", {}).get("devices") or []
    if devices:
        candidates.extend([
            Candidate("cuda-pipe-1", "Single-GPU resident pipeline", {"COLI_CUDA_PIPE": "1"}),
            Candidate("cuda-pipe-2", "Persistent residual pipeline across GPU layers", {"COLI_CUDA_PIPE": "2"}),
            Candidate("cuda-sync", "Disable asynchronous CUDA scheduling for an A/B control", {"COLI_CUDA_ASYNC": "0"}),
        ])
    unique: list[Candidate] = []
    seen_env: set[tuple[tuple[str, str], ...]] = set()
    baseline = {key: str(value) for key, value in (base_environment or {}).items() if key in SAFE_TUNABLE_KEYS}
    for candidate in candidates:
        checked = validate_candidate(candidate)
        effective = dict(baseline)
        effective.update(checked.environment)
        signature = tuple(sorted(effective.items()))
        if signature in seen_env:
            continue
        seen_env.add(signature)
        unique.append(checked)
    return tuple(unique)
