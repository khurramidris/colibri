#!/usr/bin/env python3
"""Run lattice-plan over a RAM/VRAM grid and emit a reproducible CSV frontier."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_grid(text: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        try:
            value = float(part.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid grid value: {part!r}") from exc
        if value < 0:
            raise argparse.ArgumentTypeError("grid values must be non-negative")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("grid cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, default=root / "build" / "lattice-plan")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ram-gib", type=parse_grid, default=parse_grid("4,8,16,32,64"))
    parser.add_argument("--vram-gib", type=parse_grid, default=parse_grid("0,4,8,16,24"))
    parser.add_argument("--nvme-gbps", type=float, default=7.0)
    parser.add_argument("--ram-gbps", type=float, default=60.0)
    parser.add_argument("--h2d-gbps", type=float, default=24.0)
    parser.add_argument("--overlap", type=float, default=0.65)
    parser.add_argument("--reserve", type=float, default=0.10)
    parser.add_argument("-o", "--output", type=Path, help="CSV output; default stdout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    planner = args.planner.resolve()
    manifest = args.manifest.resolve()
    if not planner.is_file():
        print(f"error: planner binary not found: {planner}; run make -C lattice", file=sys.stderr)
        return 2
    if not manifest.is_file():
        print(f"error: manifest not found: {manifest}", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="lattice-sweep-") as td:
        tmp = Path(td)
        for ram in args.ram_gib:
            for vram in args.vram_gib:
                plan_path = tmp / f"plan-r{ram:g}-v{vram:g}.json"
                command = [
                    str(planner),
                    "--manifest", str(manifest),
                    "--ram-gib", str(ram),
                    "--vram-gib", str(vram),
                    "--nvme-gbps", str(args.nvme_gbps),
                    "--ram-gbps", str(args.ram_gbps),
                    "--h2d-gbps", str(args.h2d_gbps),
                    "--overlap", str(args.overlap),
                    "--reserve", str(args.reserve),
                    "--quiet",
                    "--json", str(plan_path),
                ]
                result = subprocess.run(command, text=True, capture_output=True, check=False)
                row: dict[str, object] = {
                    "ram_gib": ram,
                    "vram_gib": vram,
                    "feasible": result.returncode == 0,
                    "nvme_bytes_per_token": "",
                    "movement_ms_per_token": "",
                    "compute_ms_per_token": "",
                    "predicted_ms_per_token": "",
                    "predicted_tokens_per_second": "",
                    "ram_used_bytes": "",
                    "vram_used_bytes": "",
                    "error": result.stderr.strip().replace("\n", " | "),
                }
                if result.returncode == 0:
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    summary = plan["summary"]
                    predicted_ms = float(summary["predicted_ms_per_token"])
                    row.update(
                        {
                            "nvme_bytes_per_token": summary["nvme_bytes_per_token"],
                            "movement_ms_per_token": summary["movement_ms_per_token"],
                            "compute_ms_per_token": summary["compute_ms_per_token"],
                            "predicted_ms_per_token": predicted_ms,
                            "predicted_tokens_per_second": 1000.0 / predicted_ms if predicted_ms > 0 else "",
                            "ram_used_bytes": summary["ram_used_bytes"],
                            "vram_used_bytes": summary["vram_used_bytes"],
                            "error": "",
                        }
                    )
                rows.append(row)

    fields = list(rows[0].keys())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fp = args.output.open("w", encoding="utf-8", newline="")
    else:
        fp = sys.stdout
    try:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if fp is not sys.stdout:
            fp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
