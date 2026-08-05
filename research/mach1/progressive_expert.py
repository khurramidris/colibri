#!/usr/bin/env python3
"""Offline falsification harness for progressive sub-3-bit MoE experts.

This is an original research prototype inspired by the public systems lesson
that expert representation, storage layout, and execution must be co-designed.
It does not reproduce or claim Syzygy Mach-1's unpublished representation.

Format PBE1 (Progressive Binary Expert v1), group size <= 64:
  - one packed sign bit per weight;
  - one fp16 base scale per group;
  - one uint8 correction count per group;
  - one fp16 correction scale per group;
  - zero or more uint8 correction indices per group;
  - one int8 correction coefficient per selected index.
Each selected residual is quantized with one fp16 scale per group.
The decoded group is alpha*sign(W) plus beta*q at selected entries.

The variable correction stream is sequential per expert, matching streamed
expert execution. Group offsets can be built once after loading from counts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


COLIBRI_E8_IQ3_BPW = 98.0 * 8.0 / 256.0


@dataclass(frozen=True)
class ProgressiveExpert:
    rows: int
    cols: int
    group_size: int
    sign_bits: np.ndarray
    base_scales: np.ndarray
    correction_counts: np.ndarray
    correction_scales: np.ndarray
    correction_indices: np.ndarray
    correction_values: np.ndarray

    @property
    def groups_per_row(self) -> int:
        return math.ceil(self.cols / self.group_size)

    @property
    def n_groups(self) -> int:
        return self.rows * self.groups_per_row

    @property
    def n_weights(self) -> int:
        return self.rows * self.cols

    @property
    def payload_bytes(self) -> int:
        return sum(
            int(a.nbytes)
            for a in (
                self.sign_bits,
                self.base_scales,
                self.correction_counts,
                self.correction_scales,
                self.correction_indices,
                self.correction_values,
            )
        )

    @property
    def effective_bpw(self) -> float:
        return 8.0 * self.payload_bytes / self.n_weights


def _validate_matrix(weights: np.ndarray, group_size: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError(f"weights must be 2-D, got shape={w.shape}")
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    if not np.isfinite(w).all():
        raise ValueError("weights contain NaN or infinity")
    if group_size < 8 or group_size > 64:
        raise ValueError("group_size must be in [8, 64]")
    return w


def encode_progressive(
    weights: np.ndarray,
    *,
    group_size: int = 64,
    correction_fraction: float = 0.03,
) -> ProgressiveExpert:
    """Encode a matrix into PBE1.

    correction_fraction is applied independently to each non-empty group.
    Fractions are converted to an integer count with round(), then clamped.
    """
    w = _validate_matrix(weights, group_size)
    if not 0.0 <= correction_fraction <= 1.0:
        raise ValueError("correction_fraction must be in [0, 1]")

    rows, cols = map(int, w.shape)
    groups_per_row = math.ceil(cols / group_size)
    sign_chunks: list[np.ndarray] = []
    base_scales: list[np.float16] = []
    counts: list[np.uint8] = []
    correction_scales: list[np.float16] = []
    indices: list[np.uint8] = []
    values: list[np.int8] = []

    for row in range(rows):
        for group in range(groups_per_row):
            start = group * group_size
            stop = min(start + group_size, cols)
            x = w[row, start:stop]
            n = int(x.size)

            signs_positive = x >= 0.0
            packed = np.packbits(signs_positive, bitorder="little")
            bytes_per_full_group = math.ceil(group_size / 8)
            if packed.size < bytes_per_full_group:
                packed = np.pad(
                    packed,
                    (0, bytes_per_full_group - packed.size),
                    constant_values=0,
                )
            sign_chunks.append(packed.astype(np.uint8, copy=False))

            alpha = float(np.mean(np.abs(x), dtype=np.float64))
            if alpha < 2.0**-24:
                alpha = 0.0
            alpha16 = np.float16(alpha)
            base_scales.append(alpha16)

            base = np.where(signs_positive, float(alpha16), -float(alpha16))
            residual = x - base.astype(np.float32)

            k = int(round(correction_fraction * n))
            k = max(0, min(k, n, 127))
            if k == 0:
                counts.append(np.uint8(0))
                correction_scales.append(np.float16(0.0))
                continue

            # Stable ordering makes identical weights encode identically.
            chosen = np.argsort(-np.abs(residual), kind="stable")[:k]
            # The maximum residual is always in every top-k set, so the scale is
            # stable as k increases. This makes larger correction fractions nested:
            # existing correction values do not change; only new entries are added.
            beta = max(float(np.max(np.abs(residual[chosen]))) / 127.0, 2.0**-24)
            beta16 = np.float16(beta)
            counts.append(np.uint8(k))
            correction_scales.append(beta16)

            for index in np.sort(chosen):
                q = int(np.clip(np.rint(residual[index] / float(beta16)), -127, 127))
                indices.append(np.uint8(index))
                values.append(np.int8(q))

    sign_bits = np.concatenate(sign_chunks).astype(np.uint8, copy=False)
    return ProgressiveExpert(
        rows=rows,
        cols=cols,
        group_size=group_size,
        sign_bits=sign_bits,
        base_scales=np.asarray(base_scales, dtype=np.float16),
        correction_counts=np.asarray(counts, dtype=np.uint8),
        correction_scales=np.asarray(correction_scales, dtype=np.float16),
        correction_indices=np.asarray(indices, dtype=np.uint8),
        correction_values=np.asarray(values, dtype=np.int8),
    )


def decode_progressive(encoded: ProgressiveExpert) -> np.ndarray:
    """Reference decoder. Production kernels should consume the packed form directly."""
    result = np.empty((encoded.rows, encoded.cols), dtype=np.float32)
    bytes_per_group = math.ceil(encoded.group_size / 8)
    entry_cursor = 0

    for group_index in range(encoded.n_groups):
        row = group_index // encoded.groups_per_row
        group = group_index % encoded.groups_per_row
        start = group * encoded.group_size
        stop = min(start + encoded.group_size, encoded.cols)
        n = stop - start

        packed = encoded.sign_bits[
            group_index * bytes_per_group : (group_index + 1) * bytes_per_group
        ]
        positive = np.unpackbits(packed, bitorder="little")[:n].astype(bool)
        alpha = float(encoded.base_scales[group_index])
        group_values = np.where(positive, alpha, -alpha).astype(np.float32)

        count = int(encoded.correction_counts[group_index])
        beta = float(encoded.correction_scales[group_index])
        group_indices = encoded.correction_indices[entry_cursor : entry_cursor + count]
        group_corrections = encoded.correction_values[entry_cursor : entry_cursor + count]
        for index_u8, q_i8 in zip(group_indices, group_corrections):
            index = int(index_u8)
            if index >= n:
                raise ValueError(
                    f"correction index {index} outside tail group of length {n}"
                )
            group_values[index] += beta * int(q_i8)
        entry_cursor += count
        result[row, start:stop] = group_values

    if (
        entry_cursor != encoded.correction_indices.size
        or entry_cursor != encoded.correction_values.size
    ):
        raise ValueError("unused correction entries: corrupt counts or payload")
    return result


def quantize_int3_g64(weights: np.ndarray) -> np.ndarray:
    """Reference approximation of Colibri's int3-g64 quantize/dequant path."""
    w = _validate_matrix(weights, 64)
    rows, cols = w.shape
    out = np.empty_like(w)
    for row in range(rows):
        for start in range(0, cols, 64):
            stop = min(start + 64, cols)
            x = w[row, start:stop]
            scale = max(float(np.max(np.abs(x))) / 3.0, 1.0e-8)
            q = np.clip(np.rint(x / scale), -4, 3)
            out[row, start:stop] = q * scale
    return out


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return float(np.linalg.norm(candidate))
    return float(np.linalg.norm(reference - candidate) / denom)


def evaluate(
    weights: np.ndarray,
    *,
    correction_fractions: Iterable[float],
    group_size: int = 64,
    activation_samples: int = 64,
    seed: int = 0,
) -> dict:
    w = _validate_matrix(weights, group_size)
    rng = np.random.default_rng(seed)
    activations = rng.standard_normal(
        (activation_samples, w.shape[1]), dtype=np.float32
    )
    reference_output = activations @ w.T

    int3 = quantize_int3_g64(w)
    int3_weight_error = relative_l2(w, int3)
    int3_matvec_error = relative_l2(reference_output, activations @ int3.T)

    rows = []
    for fraction in correction_fractions:
        encoded = encode_progressive(
            w, group_size=group_size, correction_fraction=float(fraction)
        )
        decoded = decode_progressive(encoded)
        bpw = encoded.effective_bpw
        traffic_reduction = 1.0 - bpw / COLIBRI_E8_IQ3_BPW
        rows.append(
            {
                "correction_fraction": float(fraction),
                "effective_bpw": bpw,
                "payload_bytes": encoded.payload_bytes,
                "traffic_reduction_vs_e8_iq3": traffic_reduction,
                "capacity_multiplier_vs_e8_iq3": COLIBRI_E8_IQ3_BPW / bpw,
                "weight_relative_l2": relative_l2(w, decoded),
                "matvec_relative_l2": relative_l2(
                    reference_output, activations @ decoded.T
                ),
                "passes_size_gate": bool(bpw <= 2.2),
            }
        )

    return {
        "shape": list(map(int, w.shape)),
        "group_size": group_size,
        "activation_samples": activation_samples,
        "seed": seed,
        "baseline": {
            "name": "colibri-int3-g64-reference",
            "effective_bpw_nominal": 3.0 + 32.0 / 64.0,
            "weight_relative_l2": int3_weight_error,
            "matvec_relative_l2": int3_matvec_error,
        },
        "strongest_physical_baseline_bpw": COLIBRI_E8_IQ3_BPW,
        "progressive": rows,
        "warning": (
            "Synthetic or isolated-tensor results are screening evidence only. "
            "A scientific pass requires real expert weights, held-out activations, "
            "full-logit gates, and end-to-end physical byte/time measurements."
        ),
    }


def make_synthetic(rows: int, cols: int, seed: int) -> np.ndarray:
    """Heavy-tailed synthetic expert matrix for smoke testing, not validation."""
    rng = np.random.default_rng(seed)
    core = rng.standard_normal((rows, cols), dtype=np.float32) * 0.025
    outlier_mask = rng.random((rows, cols)) < 0.01
    outliers = rng.standard_normal((rows, cols), dtype=np.float32) * 0.35
    return core + outlier_mask * outliers


def _parse_fractions(value: str) -> list[float]:
    try:
        values = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(not 0.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("fractions must be comma-separated values in [0,1]")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="2-D float .npy expert matrix")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--fractions",
        type=_parse_fractions,
        default=[0.0, 0.01, 0.03, 0.05, 0.07],
        help="comma-separated correction fractions",
    )
    parser.add_argument("--activation-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.weights:
        weights = np.load(args.weights)
    else:
        weights = make_synthetic(args.rows, args.cols, args.seed)

    report = evaluate(
        weights,
        correction_fractions=args.fractions,
        group_size=args.group_size,
        activation_samples=args.activation_samples,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
