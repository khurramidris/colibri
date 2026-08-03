from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from pathlib import Path

from .io import strict_loads
from .model import AtlasError

SCHEMA = "lattice.ternary.probe.v1"


def load_ternary_probes(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip() or raw.lstrip().startswith(b"#"):
                continue
            value = strict_loads(raw.decode("utf-8"), label=f"{path}:{line_number}")
            if not isinstance(value, dict) or value.get("schema") != SCHEMA:
                raise AtlasError(f"{path}:{line_number}: invalid ternary probe schema")
            required = ("layer", "expert", "samples", "output_cosine", "nrmse")
            if any(key not in value for key in required):
                raise AtlasError(f"{path}:{line_number}: missing required ternary field")
            cosine = float(value["output_cosine"])
            nrmse = float(value["nrmse"])
            if not -1.0 <= cosine <= 1.0 or nrmse < 0:
                raise AtlasError(f"{path}:{line_number}: invalid ternary metrics")
            rows.append(value)
    if not rows:
        raise AtlasError("ternary probe file contains no rows")
    return rows


def summarize_ternary(rows: list[dict], *, min_cosine: float = 0.98, max_nrmse: float = 0.15) -> dict:
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(row)
    safe = [row for row in rows if float(row["output_cosine"]) >= min_cosine and float(row["nrmse"]) <= max_nrmse]
    layer_rows = []
    for layer, values in sorted(by_layer.items()):
        layer_safe = [row for row in values if float(row["output_cosine"]) >= min_cosine and float(row["nrmse"]) <= max_nrmse]
        layer_rows.append(
            {
                "layer": layer,
                "experts": len(values),
                "safe_fraction": len(layer_safe) / len(values),
                "median_cosine": median(float(row["output_cosine"]) for row in values),
                "mean_nrmse": mean(float(row["nrmse"]) for row in values),
            }
        )
    return {
        "status": "measured",
        "probes": len(rows),
        "thresholds": {"min_output_cosine": min_cosine, "max_nrmse": max_nrmse},
        "safe_fraction": len(safe) / len(rows),
        "layers": layer_rows,
    }
