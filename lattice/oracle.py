from __future__ import annotations

import math
from typing import Any

from .common import LatticeError

ORACLE_SCHEMA = "coli-replay-oracle/2"
ORACLE_POLICY = {
    "schema": ORACLE_SCHEMA,
    "topk": 8,
    "measurement": "separate_replay_pass",
    "transport": "private_file",
    "representation": "compact_rows",
    "absolute_tolerance": 0.005,
    "relative_tolerance": 0.0005,
}

FORCED, TOP1, TOP2, NONFINITE = 0, 1, 2, 3
TOP1_LOGIT, FORCED_LOGIT, MARGIN, MEAN, RMS = 4, 5, 6, 7, 8
PROJECTION_0, PROJECTION_1, PROJECTION_2, PROJECTION_3 = 9, 10, 11, 12
TOPK_IDS = 13
ROW_LENGTH = 14
FLOAT_INDEXES = tuple(range(TOP1_LOGIT, PROJECTION_3 + 1))


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
    for index, row in enumerate(steps):
        if not isinstance(row, list) or len(row) != ROW_LENGTH:
            raise LatticeError(f"replay numerical oracle row {index} is invalid")
        for field in (FORCED, TOP1, TOP2, NONFINITE):
            item = row[field]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise LatticeError(f"replay numerical oracle row {index} has invalid identity")
        for field in FLOAT_INDEXES:
            item = row[field]
            if (isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))):
                raise LatticeError(f"replay numerical oracle row {index} has invalid numeric sketch")
        topk = row[TOPK_IDS]
        if (not isinstance(topk, list) or len(topk) != ORACLE_POLICY["topk"]
                or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in topk)
                or len(set(topk)) != len(topk)):
            raise LatticeError(f"replay numerical oracle row {index} has invalid top-k identity")
        if row[TOP1] != topk[0] or row[TOP2] != topk[1]:
            raise LatticeError(f"replay numerical oracle row {index} top-k order is inconsistent")
    return value


def _close(reference: float, candidate: float, *, absolute: float, relative: float) -> bool:
    return math.isclose(reference, candidate, abs_tol=absolute, rel_tol=relative)


def compare_oracles(reference: Any, candidate: Any) -> str | None:
    """Compare compact numerical sketches under the versioned policy."""
    reference = validate_oracle(reference)
    candidate = validate_oracle(candidate)
    if len(reference["steps"]) != len(candidate["steps"]):
        return "step count differs"
    absolute = float(ORACLE_POLICY["absolute_tolerance"])
    relative = float(ORACLE_POLICY["relative_tolerance"])
    for index, (base, trial) in enumerate(zip(reference["steps"], candidate["steps"])):
        if base[NONFINITE] or trial[NONFINITE]:
            return f"step {index} contains non-finite logits"
        for field, label in ((FORCED, "forced"), (TOP1, "top1"), (TOP2, "top2"), (TOPK_IDS, "topk_ids")):
            if base[field] != trial[field]:
                return f"step {index} {label} differs"
        for field in FLOAT_INDEXES:
            if not _close(float(base[field]), float(trial[field]), absolute=absolute, relative=relative):
                return f"step {index} numeric field {field} differs beyond tolerance"
    return None
