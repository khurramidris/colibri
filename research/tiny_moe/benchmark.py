"""Run and compare the frozen Tiny-MoE Gate 3 residency matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from accounting import cache_observation, rss_bytes, summarize_read_events


def _trace_comparison(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    reference_trace = reference["trace"]
    candidate_trace = candidate["trace"]
    result: dict[str, Any] = {
        "greedy_tokens_identical": reference_trace["token_ids"]
        == candidate_trace["token_ids"],
        "steps": [],
    }
    for step_id, (expected, actual) in enumerate(
        zip(reference_trace["steps"], candidate_trace["steps"])
    ):
        step = {
            "step": step_id,
            "hidden_states_identical": expected["hidden_sha256"]
            == actual["hidden_sha256"],
            "router_top2_identical": expected["router_ids_sha256"]
            == actual["router_ids_sha256"],
            "router_weights_identical": expected["router_weights_sha256"]
            == actual["router_weights_sha256"],
            "persistent_state_identical": expected["persistent"]
            == actual["persistent"],
            "final_logits_identical": expected["logits_sha256"]
            == actual["logits_sha256"],
            "hidden_mismatches": [
                i
                for i, (left, right) in enumerate(
                    zip(expected["hidden_sha256"], actual["hidden_sha256"])
                )
                if left != right
            ],
            "router_top2_mismatches": [
                i
                for i, (left, right) in enumerate(
                    zip(expected["router_ids_sha256"], actual["router_ids_sha256"])
                )
                if left != right
            ],
            "router_weight_mismatches": [
                i
                for i, (left, right) in enumerate(
                    zip(
                        expected["router_weights_sha256"],
                        actual["router_weights_sha256"],
                    )
                )
                if left != right
            ],
        }
        step["all_identical"] = all(
            step[key]
            for key in (
                "hidden_states_identical",
                "router_top2_identical",
                "router_weights_identical",
                "persistent_state_identical",
                "final_logits_identical",
            )
        )
        result["steps"].append(step)
    result["step_count_identical"] = len(reference_trace["steps"]) == len(
        candidate_trace["steps"]
    )
    result["all_identical"] = (
        result["greedy_tokens_identical"]
        and result["step_count_identical"]
        and all(step["all_identical"] for step in result["steps"])
    )
    return result


def _run_child(
    script: Path,
    checkpoint: Path,
    trunk: Path,
    resident_layers: int,
    input_ids: str,
    new_tokens: int,
    output: Path,
    ready: Path,
    disable_mkldnn: bool = False,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--checkpoint",
        str(checkpoint),
        "--trunk",
        str(trunk),
        "--resident-layers",
        str(resident_layers),
        "--input-ids",
        input_ids,
        "--new-tokens",
        str(new_tokens),
        "--output",
        str(output),
        "--ready-file",
        str(ready),
    ]
    if not disable_mkldnn:
        command.append("--enable-mkldnn")
    started = time.perf_counter()
    # The child emits a deliberately detailed accounting record. Do not pipe
    # it without a concurrent drain: a full read-event report can fill the
    # Windows pipe buffer and deadlock the benchmark parent.
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    startup_peak = 0
    steady_peak = 0
    ready_seen = False
    samples = 0
    while process.poll() is None:
        if ready.exists() and not ready_seen:
            try:
                ready_payload = json.loads(ready.read_text(encoding="utf-8"))
                ready_rss = ready_payload.get("rss_bytes")
                if isinstance(ready_rss, int):
                    startup_peak = max(startup_peak, ready_rss)
            except (OSError, json.JSONDecodeError):
                pass
            ready_seen = True
        current = rss_bytes(process.pid)
        if current is not None:
            samples += 1
            if ready_seen:
                steady_peak = max(steady_peak, current)
            else:
                startup_peak = max(startup_peak, current)
        if time.perf_counter() - started > timeout_seconds:
            process.kill()
            _stdout, stderr = process.communicate()
            raise TimeoutError(f"child timed out; stderr={stderr[-1000:]}")
        time.sleep(0.02)
    _stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"child failed ({process.returncode}); stderr={stderr[-4000:]}"
        )
    if not output.is_file():
        raise RuntimeError("child completed without result artifact")
    result = json.loads(output.read_text(encoding="utf-8"))
    result["process_metrics"] = {
        "wall_seconds": time.perf_counter() - started,
        "startup_peak_rss_bytes": startup_peak or None,
        "steady_state_peak_rss_bytes": steady_peak or None,
        "rss_samples": samples,
        "ready_seen": ready_seen,
        "stderr": stderr,
    }
    result["accounting_summary"] = summarize_read_events(
        result["accounting"]["read_events"]
    )
    return result


def run_matrix(
    checkpoint: Path,
    trunk: Path,
    reference_path: Path,
    output_dir: Path,
    input_ids: str,
    new_tokens: int,
    repeats: int,
    disable_mkldnn: bool = False,
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    script = Path(__file__).with_name("streamed_runner.py")
    modes = {
        "fully_resident": 14,
        "fifty_percent": 7,
        "two_layers": 2,
        "one_layer": 1,
        "zero_layers": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    repeat_baselines: dict[str, dict[str, Any]] = {}
    for mode, resident_layers in modes.items():
        for repeat in range(1, repeats + 1):
            case_dir = output_dir / mode / f"repeat_{repeat}"
            case_dir.mkdir(parents=True, exist_ok=True)
            output_path = case_dir / "run.json"
            ready_path = case_dir / "ready.json"
            result = _run_child(
                script,
                checkpoint,
                trunk,
                resident_layers,
                input_ids,
                new_tokens,
                output_path,
                ready_path,
                disable_mkldnn=disable_mkldnn,
            )
            comparison = _trace_comparison(reference, result)
            repeat_identical = True
            if mode in repeat_baselines:
                repeat_identical = repeat_baselines[mode]["trace"] == result["trace"]
            else:
                repeat_baselines[mode] = result
            accounting = result["accounting"]
            expected_stream_reads = (14 - resident_layers) * (1 + new_tokens)
            expected_startup_reads = resident_layers
            cases.append(
                {
                    "mode": mode,
                    "resident_layers": resident_layers,
                    "repeat": repeat,
                    "comparison": comparison,
                    "repeat_identical_to_first": repeat_identical,
                    "metadata": result["metadata"],
                    "memory_audit": result["memory_audit"],
                    "accounting": accounting,
                    "accounting_summary": result["accounting_summary"],
                    "process_metrics": result["process_metrics"],
                    "read_gate": {
                        "expected_startup_reads": expected_startup_reads,
                        "observed_startup_reads": sum(
                            1
                            for event in accounting["read_events"]
                            if event["phase"] == "startup"
                        ),
                        "expected_stream_reads": expected_stream_reads,
                        "observed_stream_reads": sum(
                            1
                            for event in accounting["read_events"]
                            if event["phase"] in {"prefill", "decode"}
                        ),
                        "each_nonresident_layer_once_per_forward": all(
                            event_count == 1 + new_tokens
                            for layer_id, event_count in (
                                (
                                    layer,
                                    sum(
                                        1
                                        for event in accounting["read_events"]
                                        if event["layer_id"] == layer
                                        and event["phase"] in {"prefill", "decode"}
                                    ),
                                )
                                for layer in range(resident_layers, 14)
                            )
                        ),
                    },
                    "cache_observation": cache_observation(trunk),
                }
            )
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "repeat": repeat,
                        "all_identical": comparison["all_identical"],
                        "wall_seconds": result["process_metrics"]["wall_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "format": "tiny-moe-gate3-results-v1",
        "checkpoint": str(checkpoint),
        "trunk": str(trunk),
        "reference": str(reference_path),
        "input_ids": json.loads(input_ids),
        "new_tokens": new_tokens,
        "repeats": repeats,
        "disable_mkldnn": disable_mkldnn,
        "modes": modes,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--trunk", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--input-ids", default="[1, 2, 3, 4]")
    parser.add_argument("--new-tokens", default=2, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--disable-mkldnn", action="store_true")
    args = parser.parse_args()
    result = run_matrix(
        args.checkpoint,
        args.trunk,
        args.reference,
        args.output_dir,
        args.input_ids,
        args.new_tokens,
        args.repeats,
        args.disable_mkldnn,
    )
    output = args.output_dir / "GATE3_MATRIX.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(output), "cases": len(result["cases"])}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
