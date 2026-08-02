from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import LatticeError, canonical_json, load_json, sha256_bytes, validate_id

MAX_CASES = 64
MAX_PROMPT_CHARS = 32768
MAX_METADATA_BYTES = 65536


@dataclass(frozen=True)
class WorkloadCase:
    id: str
    prompt: str
    weight: float
    context: int
    tokens: int
    expected_contains: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "prompt": self.prompt,
            "weight": self.weight,
            "context": self.context,
            "tokens": self.tokens,
        }
        if self.expected_contains is not None:
            result["expected_contains"] = self.expected_contains
        return result


@dataclass(frozen=True)
class WorkloadSuite:
    name: str
    cases: tuple[WorkloadCase, ...]
    hourly_cost_usd: float | None
    metadata: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json(self.as_dict()))

    @property
    def total_weight(self) -> float:
        return sum(case.weight for case in self.cases)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "name": self.name,
            "cases": [case.as_dict() for case in self.cases],
        }
        if self.hourly_cost_usd is not None:
            result["hourly_cost_usd"] = self.hourly_cost_usd
        if self.metadata:
            result["metadata"] = self.metadata
        return result


def _as_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LatticeError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise LatticeError(f"{label} must be between {minimum} and {maximum}")
    return value


def _as_positive_float(value: Any, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LatticeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise LatticeError(f"{label} must be finite")
    if number < 0 or (number == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise LatticeError(f"{label} must be {comparator}")
    return number


def parse_suite(data: Any) -> WorkloadSuite:
    if not isinstance(data, dict):
        raise LatticeError("workload suite must be a JSON object")
    if data.get("schema_version") != 1:
        raise LatticeError("unsupported workload schema_version (expected 1)")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 120:
        raise LatticeError("workload name must be a non-empty string up to 120 characters")
    default_context = _as_int(data.get("default_context", 4096), "default_context", 128, 262144)
    default_tokens = _as_int(data.get("default_tokens", 16), "default_tokens", 4, 2048)
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise LatticeError("workload suite requires at least one case")
    if len(raw_cases) > MAX_CASES:
        raise LatticeError(f"workload suite supports at most {MAX_CASES} cases")
    cases: list[WorkloadCase] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases):
        label = f"cases[{index}]"
        if not isinstance(raw, dict):
            raise LatticeError(f"{label} must be an object")
        case_id = validate_id(raw.get("id"), f"{label}.id")
        if case_id in seen:
            raise LatticeError(f"duplicate workload case id: {case_id}")
        seen.add(case_id)
        prompt = raw.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise LatticeError(f"{label}.prompt must be non-empty")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise LatticeError(f"{label}.prompt exceeds {MAX_PROMPT_CHARS} characters")
        weight = _as_positive_float(raw.get("weight", 1.0), f"{label}.weight")
        context = _as_int(raw.get("context", default_context), f"{label}.context", 128, 262144)
        tokens = _as_int(raw.get("tokens", default_tokens), f"{label}.tokens", 4, 2048)
        expected = raw.get("expected_contains")
        if expected is not None and (not isinstance(expected, str) or not expected or len(expected) > 4096):
            raise LatticeError(f"{label}.expected_contains must be a non-empty string up to 4096 characters")
        cases.append(WorkloadCase(case_id, prompt, weight, context, tokens, expected))
    hourly = data.get("hourly_cost_usd")
    hourly_cost = None if hourly is None else _as_positive_float(hourly, "hourly_cost_usd", allow_zero=True)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise LatticeError("metadata must be an object")
    metadata_bytes = canonical_json(metadata)
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise LatticeError(f"metadata exceeds {MAX_METADATA_BYTES} canonical JSON bytes")
    return WorkloadSuite(name.strip(), tuple(cases), hourly_cost, metadata)


def load_suite(path: Path) -> WorkloadSuite:
    return parse_suite(load_json(path))
