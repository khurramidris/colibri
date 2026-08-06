"""Native-Windows Stage 1 performance matrix for the existing F16 runner."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import torch
from accounting import PeakRSSSampler, cache_observation, rss_bytes
from streamed_runner import ReadAccounting, TinyMoeRuntime

PROMPT_IDS = [1, 2, 3, 4]
LONG_GENERATION_TOKENS = 64
DEFAULT_REPEATS = 10

MODES: list[dict[str, Any]] = [
    {"name": "fully_resident", "resident_layers": 14, "prefetch": True},
    {"name": "zero_no_prefetch", "resident_layers": 0, "prefetch": False},
    {"name": "zero_prefetch", "resident_layers": 0, "prefetch": True},
    {"name": "one_prefetch", "resident_layers": 1, "prefetch": True},
    {"name": "two_prefetch", "resident_layers": 2, "prefetch": True},
    {"name": "fifty_prefetch", "resident_layers": 7, "prefetch": True},
]


def _reset_accounting(runtime: TinyMoeRuntime) -> None:
    runtime.accounting = ReadAccounting()
    if runtime.reader is not None:
        runtime.reader.accounting = runtime.accounting


def _greedy_next(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1).view(1, 1)


def _phase_totals(accounting: dict[str, Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for event in accounting["read_events"]:
        phase = str(event["phase"])
        totals[phase] = totals.get(phase, 0) + 1
    return totals


def _io_hidden_pct(accounting: dict[str, Any], prefetch: bool) -> float:
    load_seconds = sum(
        float(event["load_seconds"]) for event in accounting["read_events"]
    )
    if not prefetch or load_seconds <= 0:
        return 0.0
    exposed = float(accounting["load_wait_seconds"])
    return max(0.0, min(100.0, 100.0 * (1.0 - exposed / load_seconds)))


def _measurement(
    runtime: TinyMoeRuntime,
    input_ids: torch.Tensor,
    workload: str,
    prefetch: bool,
) -> dict[str, Any]:
    runtime.reset_cache()
    _reset_accounting(runtime)
    prompt_tokens = int(input_ids.size(1))
    started = time.perf_counter()
    prefill_wall = 0.0
    decode_wall = 0.0
    generated: list[int] = []

    if workload == "prefill":
        prefill_started = time.perf_counter()
        runtime._step(input_ids, "prefill", capture=False)
        prefill_wall = time.perf_counter() - prefill_started
        decode_tokens = 0
        token_denominator = prompt_tokens
    else:
        prefill_started = time.perf_counter()
        step = runtime._step(input_ids, "prefill", capture=False)
        prefill_wall = time.perf_counter() - prefill_started
        current = _greedy_next(step.logits)
        _reset_accounting(runtime)
        if workload == "decode1":
            decode_started = time.perf_counter()
            step = runtime._step(current, "decode", capture=False)
            decode_wall = time.perf_counter() - decode_started
            generated.append(int(current.item()))
            decode_tokens = 1
        elif workload == "generation64":
            for _ in range(LONG_GENERATION_TOKENS):
                decode_started = time.perf_counter()
                step = runtime._step(current, "decode", capture=False)
                decode_wall += time.perf_counter() - decode_started
                generated.append(int(current.item()))
                current = _greedy_next(step.logits)
            decode_tokens = LONG_GENERATION_TOKENS
        else:
            raise ValueError(f"unknown workload {workload}")
        token_denominator = decode_tokens

    wall_seconds = time.perf_counter() - started
    accounting = runtime.accounting.snapshot()
    phase_counts = _phase_totals(accounting)
    if workload == "prefill":
        useful_bytes_per_token = accounting["token_payload_bytes"] / prompt_tokens
        requests_per_token = accounting["read_calls"] / prompt_tokens
    else:
        useful_bytes_per_token = accounting["token_payload_bytes"] / token_denominator
        requests_per_token = accounting["read_calls"] / token_denominator
    return {
        "workload": workload,
        "wall_seconds": wall_seconds,
        "prefill_wall_seconds": prefill_wall,
        "decode_wall_seconds": decode_wall,
        "prefill_tokens_per_second": (
            prompt_tokens / prefill_wall if prefill_wall else None
        ),
        "decode_tokens_per_second": (
            decode_tokens / decode_wall if decode_wall else None
        ),
        "milliseconds_per_generated_token": (
            1000.0 * decode_wall / decode_tokens if decode_tokens else None
        ),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "useful_bytes_read": accounting["token_payload_bytes"],
        "useful_bytes_per_token": useful_bytes_per_token,
        "read_requests": accounting["read_calls"],
        "read_requests_per_token": requests_per_token,
        "phase_read_counts": phase_counts,
        "load_wait_seconds": accounting["load_wait_seconds"],
        "compute_seconds": accounting["compute_seconds"],
        "io_hidden_pct": _io_hidden_pct(accounting, prefetch),
        "accounting": accounting,
    }


def _aggregate(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "wall_seconds",
        "prefill_wall_seconds",
        "decode_wall_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "milliseconds_per_generated_token",
        "useful_bytes_read",
        "useful_bytes_per_token",
        "read_requests",
        "read_requests_per_token",
        "load_wait_seconds",
        "compute_seconds",
        "io_hidden_pct",
    ]
    summary: dict[str, Any] = {"repetitions": len(measurements)}
    for key in numeric_keys:
        values = [float(item[key]) for item in measurements if item[key] is not None]
        summary[key] = {
            "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    summary["generated_tokens_identical"] = (
        len({tuple(item["generated_tokens"]) for item in measurements}) == 1
    )
    return summary


def run_child(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    input_ids = torch.tensor([json.loads(args.input_ids)], dtype=torch.long)
    sampler = PeakRSSSampler(interval_seconds=0.005)
    rss_before = rss_bytes()
    sampler.start()
    runtime: TinyMoeRuntime | None = None
    ready_time = time.perf_counter()
    try:
        runtime = TinyMoeRuntime(
            args.checkpoint,
            args.trunk,
            resident_layers=args.resident_layers,
            prefetch=args.prefetch,
        )
        startup_accounting = runtime.accounting.snapshot()
        ready_time = time.perf_counter()
        if args.ready_file:
            ready_path = Path(args.ready_file)
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text(
                json.dumps(
                    {
                        "rss_bytes": rss_bytes(),
                        "metadata": runtime.metadata(),
                        "startup_accounting": startup_accounting,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

        _measurement(runtime, input_ids, args.workload, args.prefetch)
        measurements = [
            _measurement(runtime, input_ids, args.workload, args.prefetch)
            for _ in range(args.repeats)
        ]
        samples = sampler.stop()
        startup_values = [
            value for timestamp, value in sampler.samples if timestamp <= ready_time
        ]
        steady_values = [
            value for timestamp, value in sampler.samples if timestamp >= ready_time
        ]
        if rss_before is not None:
            startup_values.append(rss_before)
        if rss_bytes() is not None:
            steady_values.append(int(rss_bytes()))
        audit = runtime.memory_audit()
        output = {
            "format": "tiny-moe-performance-v1",
            "platform": {
                "os": os.name,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "threads": torch.get_num_threads(),
            },
            "mode": {
                "resident_layers": args.resident_layers,
                "prefetch": args.prefetch,
                "workload": args.workload,
            },
            "protocol": {
                "input_ids": json.loads(args.input_ids),
                "long_generation_tokens": LONG_GENERATION_TOKENS,
                "mkldnn_enabled": False,
                "cache_classification": "process-cold startup; OS-cache-warm repetitions",
                "warmup_runs": 1,
                "measured_repetitions": args.repeats,
            },
            "startup_accounting": startup_accounting,
            "measurements": measurements,
            "summary": _aggregate(measurements),
            "rss": {
                "startup_peak_bytes": max(startup_values) if startup_values else None,
                "steady_state_peak_bytes": (
                    max(steady_values) if steady_values else None
                ),
                "sampler": samples,
            },
            "memory_audit": audit,
            "cache_observation": cache_observation(args.trunk),
            "metadata": runtime.metadata(),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        if runtime is not None:
            runtime.close()
        elif sampler._thread is not None:
            sampler.stop()


def run_routing_audit(args: argparse.Namespace) -> None:
    """Capture the fixed 64-token route trace without rerunning timings."""
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    input_ids = torch.tensor([json.loads(args.input_ids)], dtype=torch.long)
    runtime: TinyMoeRuntime | None = None
    try:
        runtime = TinyMoeRuntime(
            args.checkpoint,
            args.trunk,
            resident_layers=args.resident_layers,
            prefetch=args.prefetch,
        )
        runtime.reset_cache()
        steps: list[dict[str, Any]] = []
        step = runtime._step(input_ids, "prefill", capture=False, capture_routes=True)
        steps.append(runtime.last_route_trace)
        current = _greedy_next(step.logits)
        generated_tokens: list[int] = []
        for _ in range(LONG_GENERATION_TOKENS):
            generated_tokens.append(int(current.item()))
            step = runtime._step(current, "decode", capture=False, capture_routes=True)
            steps.append(runtime.last_route_trace)
            current = _greedy_next(step.logits)
        output = {
            "format": "tiny-moe-routing-audit-v1",
            "mode": {
                "resident_layers": args.resident_layers,
                "prefetch": args.prefetch,
            },
            "input_ids": json.loads(args.input_ids),
            "generated_tokens": generated_tokens,
            "steps": steps,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        if runtime is not None:
            runtime.close()


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cache_policy": "No filesystem cache flush performed; child startup is process-cold only, repetitions are OS-cache-warm.",
    }


def run_matrix(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "environment.json").write_text(
        json.dumps(_environment(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    script = Path(__file__).resolve()
    cases: list[dict[str, Any]] = []
    for mode in MODES:
        for workload in ("prefill", "decode1", "generation64"):
            case_dir = output_dir / mode["name"] / workload
            case_dir.mkdir(parents=True, exist_ok=True)
            output = case_dir / "result.json"
            ready = case_dir / "ready.json"
            command = [
                sys.executable,
                str(script),
                "--child",
                "--checkpoint",
                args.checkpoint,
                "--trunk",
                args.trunk,
                "--resident-layers",
                str(mode["resident_layers"]),
                "--input-ids",
                args.input_ids,
                "--workload",
                workload,
                "--repeats",
                str(args.repeats),
                "--output",
                str(output),
                "--ready-file",
                str(ready),
            ]
            if mode["prefetch"]:
                command.append("--prefetch")
            started = time.perf_counter()
            process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
            _, stderr = process.communicate(timeout=args.timeout_seconds)
            if process.returncode != 0:
                raise RuntimeError(
                    f"performance child failed for {mode['name']}/{workload}: {stderr[-4000:]}"
                )
            cases.append(
                {
                    "mode": mode,
                    "workload": workload,
                    "output": str(output),
                    "wall_seconds_parent": time.perf_counter() - started,
                    "result": json.loads(output.read_text(encoding="utf-8")),
                }
            )
            print(f"completed {mode['name']}/{workload}", flush=True)
    matrix = {
        "format": "tiny-moe-performance-matrix-v1",
        "checkpoint": args.checkpoint,
        "trunk": args.trunk,
        "cases": cases,
        "environment": _environment(),
    }
    (output_dir / "PERFORMANCE_MATRIX.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _summary_value(result: dict[str, Any], key: str) -> float:
    value = result["summary"][key]["mean"]
    if value is None:
        raise ValueError(f"missing summary metric {key}")
    return float(value)


def _stability(result: dict[str, Any], key: str) -> dict[str, float]:
    values = [float(item[key]) for item in result["measurements"]]
    average = mean(values)
    return {
        "mean": average,
        "population_stddev": pstdev(values),
        "coefficient_of_variation_pct": (
            100.0 * pstdev(values) / average if average else 0.0
        ),
        "relative_range_pct": (
            100.0 * (max(values) - min(values)) / average if average else 0.0
        ),
    }


def _routing_validation(output_dir: Path) -> dict[str, Any]:
    audit_dir = output_dir / "routing_audit"
    audits: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        path = audit_dir / f"{mode['name']}.json"
        if not path.is_file():
            return {
                "status": "not_captured",
                "reason": f"missing routing audit: {path}",
            }
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("format") != "tiny-moe-routing-audit-v1":
            return {"status": "invalid", "reason": f"invalid format: {path}"}
        if len(audit.get("steps", [])) != LONG_GENERATION_TOKENS + 1:
            return {"status": "invalid", "reason": f"invalid step count: {path}"}
        audits[mode["name"]] = audit

    reference = audits[MODES[0]["name"]]
    reference_tokens = reference["generated_tokens"]
    reference_steps = reference["steps"]
    mismatches: list[dict[str, Any]] = []
    for mode in MODES[1:]:
        audit = audits[mode["name"]]
        if audit["generated_tokens"] != reference_tokens:
            mismatches.append({"mode": mode["name"], "field": "generated_tokens"})
        for step_index, (expected, actual) in enumerate(
            zip(reference_steps, audit["steps"])
        ):
            for field in ("router_ids_sha256", "router_weights_sha256"):
                if actual.get(field) != expected.get(field):
                    mismatches.append(
                        {"mode": mode["name"], "step": step_index, "field": field}
                    )
    return {
        "status": "exact" if not mismatches else "mismatch",
        "audit_files": {name: str(audit_dir / f"{name}.json") for name in audits},
        "step_count": len(reference_steps),
        "generated_token_count": len(reference_tokens),
        "mismatches": mismatches,
        "router_top2_selection_hashes_identical": not any(
            item["field"] == "router_ids_sha256" for item in mismatches
        ),
        "router_weight_hashes_identical": not any(
            item["field"] == "router_weights_sha256" for item in mismatches
        ),
        "generated_tokens_identical": not any(
            item["field"] == "generated_tokens" for item in mismatches
        ),
    }


def finalize_stage1(output_dir: Path) -> None:
    expected = {
        (mode["name"], workload)
        for mode in MODES
        for workload in ("prefill", "decode1", "generation64")
    }
    cases: list[dict[str, Any]] = []
    found: set[tuple[str, str]] = set()
    required_summary = (
        "wall_seconds",
        "useful_bytes_per_token",
        "read_requests_per_token",
        "load_wait_seconds",
        "compute_seconds",
        "io_hidden_pct",
    )
    for result_path in sorted(output_dir.glob("*/result.json")):
        raise ValueError(f"unexpected result location: {result_path}")
    for mode in MODES:
        for workload in ("prefill", "decode1", "generation64"):
            case_path = output_dir / mode["name"] / workload / "result.json"
            key = (mode["name"], workload)
            found.add(key)
            if not case_path.is_file():
                raise FileNotFoundError(case_path)
            result = json.loads(case_path.read_text(encoding="utf-8"))
            measurements = result.get("measurements")
            if result.get("protocol", {}).get("warmup_runs") != 1:
                raise ValueError(f"{key}: warm-up count is not one")
            if result.get("protocol", {}).get("measured_repetitions") != 10:
                raise ValueError(f"{key}: measured repetition count is not ten")
            if not isinstance(measurements, list) or len(measurements) != 10:
                raise ValueError(f"{key}: raw measurement count is not ten")
            if not result.get("summary", {}).get("generated_tokens_identical"):
                raise ValueError(f"{key}: generated tokens vary across repetitions")
            for metric in required_summary:
                if result.get("summary", {}).get(metric, {}).get("mean") is None:
                    raise ValueError(f"{key}: incomplete metric {metric}")
            for measurement in measurements:
                for metric in (
                    "wall_seconds",
                    "useful_bytes_read",
                    "read_requests",
                    "load_wait_seconds",
                    "compute_seconds",
                    "io_hidden_pct",
                ):
                    if metric not in measurement:
                        raise ValueError(f"{key}: measurement missing {metric}")
                accounting = measurement.get("accounting", {})
                for metric in ("read_events", "token_payload_bytes", "read_calls"):
                    if metric not in accounting:
                        raise ValueError(f"{key}: accounting missing {metric}")
            cases.append(
                {
                    "mode": mode["name"],
                    "workload": workload,
                    "path": str(case_path),
                    "resident_layers": mode["resident_layers"],
                    "prefetch": mode["prefetch"],
                    "warmup_runs": 1,
                    "measured_repetitions": 10,
                    "generated_tokens": result["measurements"][0]["generated_tokens"],
                    "generated_tokens_identical": True,
                    "routing_validation": "validated_by_stage1_routing_audit",
                    "summary": result["summary"],
                    "rss": result["rss"],
                    "memory_audit": result["memory_audit"],
                    "metadata": result["metadata"],
                    "stability": {
                        key: _stability(result, key)
                        for key in (
                            "wall_seconds",
                            "decode_wall_seconds",
                            "compute_seconds",
                            "load_wait_seconds",
                        )
                        if result["measurements"][0].get(key) is not None
                    },
                }
            )
    if found != expected:
        raise ValueError(f"case set mismatch: expected {expected}, found {found}")

    by_workload: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_workload.setdefault(case["workload"], []).append(case)
    token_consistency: dict[str, Any] = {}
    for workload, workload_cases in by_workload.items():
        sequences = {tuple(case["generated_tokens"]) for case in workload_cases}
        token_consistency[workload] = {
            "identical_across_modes": len(sequences) == 1,
            "sequence_count": len(sequences),
        }

    comparison: dict[str, Any] = {}
    for workload, metric in (
        ("prefill", "wall_seconds"),
        ("decode1", "milliseconds_per_generated_token"),
        ("generation64", "milliseconds_per_generated_token"),
    ):
        resident_value = _summary_value(
            json.loads(
                (output_dir / "fully_resident" / workload / "result.json").read_text(
                    encoding="utf-8"
                )
            ),
            metric,
        )
        rows = {}
        for mode in MODES:
            result = json.loads(
                (output_dir / mode["name"] / workload / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            value = _summary_value(result, metric)
            rows[mode["name"]] = {
                "value": value,
                "percent_vs_fully_resident": 100.0 * (value / resident_value - 1.0),
            }
        comparison[workload] = {
            "metric": metric,
            "fully_resident_value": resident_value,
            "by_mode": rows,
        }
    for workload in ("decode1", "generation64"):
        no_prefetch = _summary_value(
            json.loads(
                (output_dir / "zero_no_prefetch" / workload / "result.json").read_text(
                    encoding="utf-8"
                )
            ),
            "milliseconds_per_generated_token",
        )
        prefetch = _summary_value(
            json.loads(
                (output_dir / "zero_prefetch" / workload / "result.json").read_text(
                    encoding="utf-8"
                )
            ),
            "milliseconds_per_generated_token",
        )
        comparison[workload]["zero_prefetch_speedup_pct"] = 100.0 * (
            no_prefetch / prefetch - 1.0
        )

    routing_validation = _routing_validation(output_dir)
    if routing_validation["status"] == "mismatch":
        raise ValueError(f"routing audit mismatch: {routing_validation['mismatches']}")
    analysis = {
        "format": "tiny-moe-stage1-analysis-v1",
        "case_count": len(cases),
        "all_cases_complete": True,
        "all_cases_have_one_warmup_and_ten_repetitions": True,
        "generated_tokens_identical_across_modes": token_consistency,
        "routing_validation": routing_validation,
        "comparisons": comparison,
        "cases": cases,
        "environment": json.loads(
            (output_dir / "environment.json").read_text(encoding="utf-8")
        ),
        "cache_policy": "process-cold startup; OS-cache-warm repetitions; no filesystem cache flush",
    }
    (output_dir / "STAGE1_ANALYSIS.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    matrix = {
        "format": "tiny-moe-performance-matrix-v1",
        "checkpoint": "research/tiny_moe/artifacts/checkpoint/base/model.safetensors",
        "trunk": "research/tiny_moe/artifacts/base.trunk.bin",
        "cases": cases,
        "analysis": str(output_dir / "STAGE1_ANALYSIS.json"),
        "parent_process_status": "external tool timeout after 16 cases; two incomplete cases recovered directly with exit code 0",
    }
    (output_dir / "PERFORMANCE_MATRIX.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--routing-audit", action="store_true")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trunk", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--output")
    parser.add_argument("--ready-file")
    parser.add_argument("--resident-layers", type=int)
    parser.add_argument("--input-ids", default=json.dumps(PROMPT_IDS))
    parser.add_argument("--workload", choices=("prefill", "decode1", "generation64"))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()
    if args.child:
        if args.output is None or args.resident_layers is None or args.workload is None:
            parser.error(
                "child mode requires --output, --resident-layers, and --workload"
            )
        run_child(args)
    elif args.routing_audit:
        if args.output is None or args.resident_layers is None:
            parser.error("routing-audit mode requires --output and --resident-layers")
        run_routing_audit(args)
    elif args.finalize:
        if args.output_dir is None:
            parser.error("finalize mode requires --output-dir")
        finalize_stage1(Path(args.output_dir))
    else:
        if args.output_dir is None:
            parser.error("matrix mode requires --output-dir")
        run_matrix(args)


if __name__ == "__main__":
    main()
