#!/usr/bin/env python3
"""Probe one SwiGLU expert from NPZ matrices/activations with ternary weights.

NPZ keys: x [samples,d_model], w_gate [d_ff,d_model], w_up [d_ff,d_model],
w_down [d_model,d_ff]. This is a local operator probe, not a model-quality test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("ternary_probe.py requires numpy") from error
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    args = parser.parse_args()
    data = np.load(args.input)
    x = np.asarray(data["x"], dtype=np.float32)
    matrices = [np.asarray(data[key], dtype=np.float32) for key in ("w_gate", "w_up", "w_down")]

    def ternary(weight):
        scale = float(np.mean(np.abs(weight)))
        if scale == 0:
            return weight.copy()
        threshold = 0.7 * scale
        return np.where(weight > threshold, scale, np.where(weight < -threshold, -scale, 0.0)).astype(np.float32)

    def silu(value):
        return value / (1.0 + np.exp(-value))

    def forward(weights):
        gate, up, down = weights
        return (silu(x @ gate.T) * (x @ up.T)) @ down.T

    reference = forward(matrices)
    approximation = forward([ternary(item) for item in matrices])
    ref_flat = reference.reshape(-1).astype(np.float64)
    app_flat = approximation.reshape(-1).astype(np.float64)
    denominator = float(np.linalg.norm(ref_flat) * np.linalg.norm(app_flat))
    cosine = float(np.dot(ref_flat, app_flat) / denominator) if denominator else 1.0
    rmse = float(np.sqrt(np.mean((ref_flat - app_flat) ** 2)))
    rms_ref = float(np.sqrt(np.mean(ref_flat**2)))
    nrmse = rmse / rms_ref if rms_ref else 0.0
    record = {
        "schema": "lattice.ternary.probe.v1",
        "layer": args.layer,
        "expert": args.expert,
        "samples": int(x.shape[0]),
        "output_cosine": cosine,
        "nrmse": nrmse,
        "max_abs_error": float(np.max(np.abs(ref_flat - app_flat))),
        "method": "symmetric-mean-abs-threshold-0.7",
        "limitation": "local expert operator only; no downstream residual or model-quality claim",
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
