#!/usr/bin/env python3
"""Deterministic stress screen for Mach-inspired sub-3-bit expert formats.

This is a synthetic falsification harness, not model-quality evidence.
It evaluates representation error, rate-distortion feasibility, I/O rooflines,
and cache-capacity effects before any real checkpoint is used.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

E8_BPW = 98.0 * 8.0 / 256.0
KINDS = (
    "gaussian", "laplace", "student3", "outlier1", "outlier5",
    "row_hetero", "lowrank_noise", "blocky",
)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1.0e-30))


def power_norm(a: np.ndarray, *, iterations: int = 18, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(a.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v) + 1.0e-30
    for _ in range(iterations):
        u = a @ v
        nu = np.linalg.norm(u)
        if nu == 0:
            return 0.0
        u /= nu
        v = a.T @ u
        nv = np.linalg.norm(v)
        if nv == 0:
            return 0.0
        v /= nv
    return float(np.linalg.norm(a @ v))


def int3_qdq(w: np.ndarray, group_size: int = 64) -> np.ndarray:
    rows, cols = w.shape
    x = w.reshape(rows, cols // group_size, group_size)
    scale = np.maximum(np.max(np.abs(x), axis=2, keepdims=True) / 3.0, 1.0e-8)
    q = np.clip(np.rint(x / scale), -4, 3)
    return (q * scale).reshape(rows, cols).astype(np.float32)


def pbe1_qdq(w: np.ndarray, corrections: int, group_size: int = 64) -> tuple[np.ndarray, float]:
    """Binary base plus a fixed number of sparse int8 residuals per group."""
    rows, cols = w.shape
    x = w.reshape(rows, cols // group_size, group_size)
    alpha = np.mean(np.abs(x), axis=2, keepdims=True).astype(np.float16).astype(np.float32)
    base = np.where(x >= 0, alpha, -alpha).astype(np.float32)
    residual = x - base
    if corrections:
        idx = np.argpartition(np.abs(residual), -corrections, axis=2)[:, :, -corrections:]
        mask = np.zeros_like(residual, dtype=bool)
        np.put_along_axis(mask, idx, True, axis=2)
        beta = np.maximum(
            np.max(np.where(mask, np.abs(residual), 0), axis=2, keepdims=True) / 127.0,
            2.0**-24,
        ).astype(np.float16).astype(np.float32)
        q = np.clip(np.rint(residual / beta), -127, 127)
        base += np.where(mask, q * beta, 0)
    bpw = 1.0 + 40.0 / group_size + 16.0 * corrections / group_size
    return base.reshape(rows, cols).astype(np.float32), bpw


def q2_qdq(w: np.ndarray, group_size: int = 64, *, power_of_two_scale: bool) -> tuple[np.ndarray, float]:
    """Four-level 2-bit control with optimized group scales."""
    levels = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float32)
    rows, cols = w.shape
    x = w.reshape(rows, cols // group_size, group_size)
    scale = np.maximum(np.max(np.abs(x), axis=2, keepdims=True), 1.0e-12)
    for _ in range(5):
        idx = np.argmin(np.abs((x / scale)[..., None] - levels), axis=-1)
        selected = levels[idx]
        scale = np.maximum(
            np.sum(x * selected, axis=2, keepdims=True)
            / (np.sum(selected * selected, axis=2, keepdims=True) + 1.0e-30),
            1.0e-12,
        )
    if power_of_two_scale:
        scale = 2.0 ** np.rint(np.log2(scale))
        bpw = 2.0 + 8.0 / group_size
    else:
        scale = scale.astype(np.float16).astype(np.float32)
        bpw = 2.0 + 16.0 / group_size
    idx = np.argmin(np.abs((x / scale)[..., None] - levels), axis=-1)
    return (levels[idx] * scale).reshape(rows, cols).astype(np.float32), bpw


def make_matrix(kind: str, rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        w = rng.normal(0, 0.04, (rows, cols))
    elif kind == "laplace":
        w = rng.laplace(0, 0.03, (rows, cols))
    elif kind == "student3":
        w = rng.standard_t(3, (rows, cols)) * 0.025
    elif kind in {"outlier1", "outlier5"}:
        probability = 0.01 if kind == "outlier1" else 0.05
        magnitude = 0.35 if kind == "outlier1" else 0.18
        w = rng.normal(0, 0.025, (rows, cols))
        mask = rng.random((rows, cols)) < probability
        w += mask * rng.normal(0, magnitude, (rows, cols))
    elif kind == "row_hetero":
        scales = np.exp(rng.normal(-3.3, 0.8, (rows, 1)))
        w = rng.normal(0, 1, (rows, cols)) * scales
    elif kind == "lowrank_noise":
        w = (
            rng.normal(0, 0.12, (rows, 8)) @ rng.normal(0, 0.12, (8, cols))
            + rng.normal(0, 0.01, (rows, cols))
        )
    elif kind == "blocky":
        w = np.empty((rows, cols))
        for start in range(0, cols, 64):
            scales = np.exp(rng.normal(-3.2, 0.9, (rows, 1)))
            w[:, start : start + 64] = rng.normal(0, 1, (rows, 64)) * scales
    else:
        raise ValueError(kind)
    return w.astype(np.float32)


def activation_sets(cols: int, samples: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    sparse = np.zeros((samples, cols), dtype=np.float32)
    mask = rng.random((samples, cols)) < 0.1
    sparse[mask] = rng.normal(0, math.sqrt(10), int(mask.sum()))
    variance = np.exp(np.linspace(math.log(0.1), math.log(10), cols))
    correlated = (
        rng.normal(0, 1, (samples, 16)) @ rng.normal(0, 0.25, (16, cols))
        + 0.2 * rng.normal(0, 1, (samples, cols))
    )
    return {
        "gaussian": rng.normal(0, 1, (samples, cols)).astype(np.float32),
        "laplace": rng.laplace(0, 1 / math.sqrt(2), (samples, cols)).astype(np.float32),
        "sparse10": sparse,
        "anisotropic": (rng.normal(0, 1, (samples, cols)) * np.sqrt(variance)).astype(np.float32),
        "correlated": correlated.astype(np.float32),
    }


def summarize(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "beat_int3_rate": float(np.mean(x <= 1.0)),
    }


def representation_screen(seeds: int, rows: int, cols: int) -> tuple[dict, dict]:
    records: dict[str, list[dict]] = {"pbe1_k2": [], "q2_pow2": [], "q2_fp16": []}
    required_bpw: list[float] = []
    for kind in KINDS:
        for seed in range(seeds):
            w = make_matrix(kind, rows, cols, seed)
            q3 = int3_qdq(w)
            activations = activation_sets(cols, 64, 10_000 + seed)
            reference = {name: x @ w.T for name, x in activations.items()}
            q3_output = {name: x @ q3.T for name, x in activations.items()}
            q3_w = rel_l2(w, q3)
            q3_op = power_norm(w - q3, seed=seed) / (power_norm(w, seed=seed) + 1.0e-30)
            candidates = {
                "pbe1_k2": pbe1_qdq(w, 2),
                "q2_pow2": q2_qdq(w, power_of_two_scale=True),
                "q2_fp16": q2_qdq(w, power_of_two_scale=False),
            }
            for name, (candidate, bpw) in candidates.items():
                row = {
                    "bpw": bpw,
                    "weight_ratio_vs_int3": rel_l2(w, candidate) / q3_w,
                    "operator_ratio_vs_int3": (
                        power_norm(w - candidate, seed=seed)
                        / (power_norm(w, seed=seed) + 1.0e-30)
                    ) / q3_op,
                }
                for act_name, x in activations.items():
                    row[f"{act_name}_ratio_vs_int3"] = (
                        rel_l2(reference[act_name], x @ candidate.T)
                        / rel_l2(reference[act_name], q3_output[act_name])
                    )
                records[name].append(row)

            target = rel_l2(reference["gaussian"], q3_output["gaussian"])
            found = math.nan
            for corrections in range(33):
                candidate, bpw = pbe1_qdq(w, corrections)
                if rel_l2(reference["gaussian"], activations["gaussian"] @ candidate.T) <= target:
                    found = bpw
                    break
            required_bpw.append(found)

    summary = {}
    for name, rows_data in records.items():
        first = rows_data[0]
        result = {
            "cases": len(rows_data),
            "bpw": first["bpw"],
            "traffic_reduction_vs_e8_iq3": 1.0 - first["bpw"] / E8_BPW,
        }
        for key in first:
            if key.endswith("_ratio_vs_int3"):
                result[key] = summarize([row[key] for row in rows_data])
        summary[name] = result

    req = np.asarray(required_bpw, dtype=float)
    required = {
        "cases": int(req.size),
        "matched_within_32_corrections_rate": float(np.mean(np.isfinite(req))),
        "median_required_bpw": float(np.nanmedian(req)),
        "p25_required_bpw": float(np.nanquantile(req, 0.25)),
        "p75_required_bpw": float(np.nanquantile(req, 0.75)),
        "rate_meeting_2_2_bpw_gate": float(np.mean(req <= 2.2)),
        "rate_meeting_2_5_bpw_gate": float(np.mean(req <= 2.5)),
    }
    return summary, required


def rate_distortion_screen(seeds: int, rows: int, cols: int) -> dict:
    result = {}
    for kind in KINDS:
        errors = []
        for seed in range(seeds):
            w = make_matrix(kind, rows, cols, seed)
            errors.append(rel_l2(w, int3_qdq(w)))
        median_error = float(np.median(errors))
        result[kind] = {
            "median_int3_relative_l2": median_error,
            "gaussian_rate_distortion_minimum_bpw_to_match": float(-math.log2(median_error)),
        }
    result["interpretation"] = {
        "ideal_gaussian_relative_rmse_at_2_125_bpw": float(2.0**-2.125),
        "note": "This is an optimistic information-theoretic bound, not a realizable format guarantee.",
    }
    return result


def roofline_screen(bpw: float = 2.125) -> list[dict]:
    rows = []
    for io_fraction in (0.5, 0.7, 0.85, 0.95):
        expert_compute = min(0.2, 1.0 - io_fraction)
        other = 1.0 - io_fraction - expert_compute
        for slowdown in (0.8, 1.0, 1.15, 1.3, 1.5):
            new_time = (
                io_fraction * bpw / E8_BPW
                + expert_compute * slowdown
                + other
            )
            rows.append({
                "io_fraction": io_fraction,
                "kernel_slowdown": slowdown,
                "predicted_speedup": 1.0 / new_time,
            })
    return rows


def lru_miss_rate(cache_size: int, alpha: float, seed: int, accesses: int = 30_000) -> float:
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, 257, dtype=float)
    probabilities = None if alpha == 0 else ranks**(-alpha) / np.sum(ranks**(-alpha))
    sequence = rng.choice(256, size=accesses, p=probabilities)
    cache: OrderedDict[int, None] = OrderedDict()
    misses = 0
    for expert_raw in sequence:
        expert = int(expert_raw)
        if expert in cache:
            cache.move_to_end(expert)
        else:
            misses += 1
            cache[expert] = None
            if len(cache) > cache_size:
                cache.popitem(last=False)
    return misses / accesses


def cache_screen(bpw: float = 2.125) -> list[dict]:
    ratio = bpw / E8_BPW
    rows = []
    for alpha in (0.0, 0.6, 1.0, 1.4):
        for base_cache in (8, 16, 32, 64, 128):
            new_cache = min(256, math.floor(base_cache / ratio))
            old_miss = float(np.mean([lru_miss_rate(base_cache, alpha, seed) for seed in range(3)]))
            new_miss = float(np.mean([lru_miss_rate(new_cache, alpha, seed) for seed in range(3)]))
            rows.append({
                "zipf_alpha": alpha,
                "baseline_cached_experts": base_cache,
                "candidate_cached_experts": new_cache,
                "baseline_miss_rate": old_miss,
                "candidate_miss_rate": new_miss,
                "physical_traffic_reduction": 1.0 - ratio * new_miss / old_miss,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=512)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.cols % 64:
        raise SystemExit("--cols must be divisible by 64")
    representation, required = representation_screen(args.seeds, args.rows, args.cols)
    report = {
        "scope": {
            "synthetic_distributions": list(KINDS),
            "seeds_per_distribution": args.seeds,
            "shape": [args.rows, args.cols],
            "evidence_boundary": "Synthetic screening only; no real-model claim.",
        },
        "representation_screen": representation,
        "pbe1_required_corrections": required,
        "rate_distortion_screen": rate_distortion_screen(args.seeds * 2, args.rows, args.cols),
        "roofline_screen": roofline_screen(),
        "cache_screen": cache_screen(),
        "decision": {
            "pbe1": "KILL_AS_PRIMARY_FORMAT",
            "broader_mach_inspired_hypothesis": "NARROW",
            "reason": (
                "The size and systems gains are robust, but PBE1 does not preserve int3-level "
                "quality across the synthetic stress suite. Continue only with a stronger learned "
                "structured/additive representation and unchanged physical accounting gates."
            ),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
