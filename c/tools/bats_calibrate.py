#!/usr/bin/env python3
"""Calibrate BATS storage cost from timed pread samples on the target model disk."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from pathlib import Path


UNITS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "kib": 1024,
    "m": 1000**2,
    "mb": 1000**2,
    "mib": 1024**2,
    "g": 1000**3,
    "gb": 1000**3,
    "gib": 1024**3,
}


def parse_size(text: str) -> int:
    value = text.strip().lower().replace("_", "")
    split = 0
    while split < len(value) and (value[split].isdigit() or value[split] == "."):
        split += 1
    if split == 0:
        raise ValueError(f"invalid size {text!r}")
    number = float(value[:split])
    unit = value[split:]
    if unit not in UNITS or not math.isfinite(number) or number <= 0:
        raise ValueError(f"invalid size {text!r}")
    result = int(number * UNITS[unit])
    if result <= 0:
        raise ValueError(f"invalid size {text!r}")
    return result


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def fit_latency_model(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Fit time_us = fixed_us + bytes * slope_us_per_byte.

    Returns (fixed_us, bandwidth_gbps). The intercept is constrained non-negative.
    """
    if len(points) < 2:
        raise ValueError("at least two transfer sizes are required")
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    if len(set(xs)) < 2 or any(x <= 0 for x in xs) or any(y <= 0 for y in ys):
        raise ValueError("positive measurements at two distinct sizes are required")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    if intercept < 0:
        intercept = 0.0
        slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    if slope <= 0 or not math.isfinite(slope):
        raise ValueError("measurements do not imply positive transfer cost")
    bandwidth_gbps = 1.0 / (slope * 1000.0)
    return intercept, bandwidth_gbps


def pread_exact(fd: int, size: int, offset: int) -> None:
    remaining = size
    cursor = offset
    while remaining:
        chunk = os.pread(fd, remaining, cursor)
        if not chunk:
            raise OSError(f"short read at offset {cursor}")
        cursor += len(chunk)
        remaining -= len(chunk)


def advise_drop(fd: int, offset: int, size: int) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    try:
        os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False


def sample_size(
    fd: int,
    file_size: int,
    size: int,
    *,
    samples: int,
    warmup: int,
    alignment: int,
    rng: random.Random,
    cold_hint: bool,
) -> dict:
    if size > file_size:
        raise ValueError(f"sample size {size} exceeds file size {file_size}")
    max_slot = (file_size - size) // alignment
    offsets = [rng.randrange(max_slot + 1) * alignment for _ in range(samples + warmup)]
    times: list[float] = []
    drop_supported = True
    for index, offset in enumerate(offsets):
        if cold_hint:
            drop_supported = advise_drop(fd, offset, size) and drop_supported
        start = time.perf_counter_ns()
        pread_exact(fd, size, offset)
        elapsed_us = (time.perf_counter_ns() - start) / 1000.0
        if index >= warmup:
            times.append(elapsed_us)
    median_us = statistics.median(times)
    return {
        "bytes": size,
        "samples": samples,
        "median_us": median_us,
        "p90_us": percentile(times, 0.90),
        "min_us": min(times),
        "max_us": max(times),
        "effective_gbps_at_median": size / (median_us * 1000.0),
        "drop_cache_hint_applied": bool(cold_hint and drop_supported),
    }


def calibrate(
    path: Path,
    sizes: list[int],
    *,
    samples: int,
    warmup: int,
    alignment: int,
    seed: int,
    cold_hint: bool,
) -> dict:
    file_size = path.stat().st_size
    rng = random.Random(seed)
    fd = os.open(path, os.O_RDONLY)
    try:
        measurements = [
            sample_size(
                fd,
                file_size,
                size,
                samples=samples,
                warmup=warmup,
                alignment=alignment,
                rng=rng,
                cold_hint=cold_hint,
            )
            for size in sizes
        ]
    finally:
        os.close(fd)

    fixed_us, bandwidth_gbps = fit_latency_model(
        [(m["bytes"], m["median_us"]) for m in measurements]
    )
    return {
        "path": str(path),
        "file_bytes": file_size,
        "mode": "cold-hint" if cold_hint else "page-cache-allowed",
        "seed": seed,
        "alignment": alignment,
        "measurements": measurements,
        "bats_profile": {
            "nvme": {
                "bandwidth_gbps": bandwidth_gbps,
                "fixed_us": fixed_us,
                "queue_us": 0.0,
                "overlap": 0.0,
            }
        },
        "shell": {
            "BATS_NVME_GBPS": bandwidth_gbps,
            "BATS_NVME_FIXED_US": fixed_us,
            "BATS_NVME_QUEUE_US": 0.0,
            "BATS_NVME_OVERLAP": 0.0,
        },
        "caveats": [
            "This is an effective single-read model, not the drive manufacturer's sequential rating.",
            "queue_us and overlap require runtime shadow traces under realistic concurrency.",
            "POSIX_FADV_DONTNEED is only a hint; verify cold behavior from repeated runs and OS telemetry.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="A large model/expert file on the target storage device.")
    parser.add_argument("--sizes", default="64KiB,1MiB,8MiB,32MiB")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--alignment", type=parse_size, default=4096)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--cold-hint", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    try:
        sizes = sorted({parse_size(part) for part in args.sizes.split(",") if part.strip()})
    except ValueError as exc:
        parser.error(str(exc))
    if len(sizes) < 2:
        parser.error("--sizes must contain at least two distinct sizes")
    if args.samples < 1 or args.warmup < 0:
        parser.error("samples must be positive and warmup non-negative")
    if args.alignment < 1:
        parser.error("alignment must be positive")
    if not args.path.is_file():
        parser.error(f"{args.path} is not a file")

    result = calibrate(
        args.path,
        sizes,
        samples=args.samples,
        warmup=args.warmup,
        alignment=args.alignment,
        seed=args.seed,
        cold_hint=args.cold_hint,
    )
    if args.shell:
        for key, value in result["shell"].items():
            print(f"export {key}={value:.12g}")
    else:
        print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
