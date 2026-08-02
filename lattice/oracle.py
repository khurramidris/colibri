from __future__ import annotations

import math
import re
from typing import Any

from .common import LatticeError

ORACLE_SCHEMA = "coli-replay-oracle/1"
ORACLE_POLICY = {
    "schema": ORACLE_SCHEMA,
    "topk": 8,
    "absolute_tolerance": 0.005,
    "relative_tolerance": 0.0005,
}

_INTEGER_FIELDS = ("step", "forced", "top1", "top2", "nonfinite")
_FLOAT_FIELDS = (
    "top1_logit", "forced_logit", "margin", "mean", "rms",
    "projection_0", "projection_1", "projection_2", "projection_3",
)
_HASH_RE = re.compile(r"^[0-9a-f]{16}$")


def validate_oracle(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LatticeError("replay numerical oracle is missing")
    if value.get("schema") != ORACLE_SCHEMA:
        raise LatticeError("unsupported replay numerical oracle schema")
    if value.get("policy") != ORACLE_POLICY:
        raise LatticeError("replay numerical oracle policy mismatch")
    steps = value.get("steps")
    if not isinstance(steps, list) or not steps:
        raise LatticeError("replay numerical oracle has no steps")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise LatticeError(f"replay numerical oracle step {index} is invalid")
        for field in _INTEGER_FIELDS:
            field_value = step.get(field)
            if isinstance(field_value, bool) or not isinstance(field_value, int):
                raise LatticeError(f"replay numerical oracle step {index} has invalid {field}")
        if step["step"] != index or min(step["forced"], step["top1"], step["top2"], step["nonfinite"]) < 0:
            raise LatticeError(f"replay numerical oracle step {index} has invalid identity")
        for field in _FLOAT_FIELDS:
            field_value = step.get(field)
            if (isinstance(field_value, bool) or not isinstance(field_value, (int, float))
                    or not math.isfinite(float(field_value))):
                raise LatticeError(f"replay numerical oracle step {index} has invalid {field}")
        if not isinstance(step.get("topk_ids_hash"), str) or not _HASH_RE.fullmatch(step["topk_ids_hash"]):
            raise LatticeError(f"replay numerical oracle step {index} has invalid top-k identity")
    return value


def _close(reference: float, candidate: float, *, absolute: float, relative: float) -> bool:
    return math.isclose(reference, candidate, abs_tol=absolute, rel_tol=relative)


def compare_oracles(reference: Any, candidate: Any) -> str | None:
    """Return a human-readable mismatch, or ``None`` when sketches agree.

    Identity fields are exact. Numerical fields use the versioned tolerance
    policy recorded in every run. This is a compact consistency check, not a
    proof that every logit is equal.
    """
    reference = validate_oracle(reference)
    candidate = validate_oracle(candidate)
    if len(reference["steps"]) != len(candidate["steps"]):
        return "step count differs"
    absolute = float(ORACLE_POLICY["absolute_tolerance"])
    relative = float(ORACLE_POLICY["relative_tolerance"])
    for index, (base, trial) in enumerate(zip(reference["steps"], candidate["steps"])):
        if base["nonfinite"] or trial["nonfinite"]:
            return f"step {index} contains non-finite logits"
        for field in ("forced", "top1", "top2", "topk_ids_hash"):
            if base[field] != trial[field]:
                return f"step {index} {field} differs"
        for field in _FLOAT_FIELDS:
            if not _close(float(base[field]), float(trial[field]), absolute=absolute, relative=relative):
                return f"step {index} {field} differs beyond tolerance"
    return None
