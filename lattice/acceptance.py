from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import statistics
from pathlib import Path
from typing import Any

from .colibri import clean_environment, detect_family, fingerprint_model, fingerprint_runtime, resolve_engine
from .common import (
    LatticeError,
    atomic_write_json,
    canonical_json,
    load_json,
    sha256_bytes,
    sha256_file,
    short_id,
    utc_now,
)
from .evidence import seal_record
from .process import ProcessResult, run_bounded

ACCEPTANCE_SCHEMA = "lattice-olmoe-acceptance/1"
ASSURANCE_LEVEL = "token-exact-reference-replay"
MAX_REFERENCE_TOKENS = 4096

LOAD_RE = re.compile(
    r"resident weights loaded in\s+([0-9.]+)s\s*\|\s*RSS after load:\s*([0-9.]+)\s+GB",
    re.IGNORECASE,
)
MATCH_RE = re.compile(r"Matching tokens:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
PEAK_RE = re.compile(r"PEAK RSS:\s*([0-9.]+)\s+GB", re.IGNORECASE)
HIT_RE = re.compile(
    r"Expert cache hit rate:\s*([0-9.]+)%\s*\(hit=(\d+)\s+miss=(\d+)\)",
    re.IGNORECASE,
)
SPEED_RE = re.compile(
    r"Speed:\s*([0-9.]+)\s+tok/s\s*\(([0-9.]+)s\s+for\s+(\d+)\s+tokens\)",
    re.IGNORECASE,
)

OLMOE_AMBIENT_KEYS = frozenset({
    "SNAP", "CHAT", "PPL", "TEMP", "NUCLEUS", "PILOT", "WIDE", "HOT",
    "WARMUP", "SMOOTH", "CONF_LIMIT", "PILOT_EVICT_GUARD", "EXPERT_DROP",
    "IDOT", "OMP_NUM_THREADS",
})


def load_reference(path: Path, *, vocab_size: int | None = None) -> dict[str, list[int]]:
    path = path.expanduser().resolve()
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise LatticeError("OLMoE reference must be a JSON object")
    prompt_ids = raw.get("prompt_ids")
    full_ids = raw.get("full_ids")
    if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
        raise LatticeError("OLMoE reference requires prompt_ids and full_ids arrays")
    if not prompt_ids or len(full_ids) <= len(prompt_ids):
        raise LatticeError("OLMoE reference requires a non-empty prompt and continuation")
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise LatticeError("OLMoE full_ids must begin with prompt_ids")
    if len(full_ids) > MAX_REFERENCE_TOKENS:
        raise LatticeError(f"OLMoE reference exceeds the {MAX_REFERENCE_TOKENS}-token engine limit")
    for label, values in (("prompt_ids", prompt_ids), ("full_ids", full_ids)):
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LatticeError(f"OLMoE reference {label} contains an invalid token ID")
            if vocab_size is not None and value >= vocab_size:
                raise LatticeError(f"OLMoE reference token ID {value} exceeds vocab_size={vocab_size}")
    return {"prompt_ids": list(prompt_ids), "full_ids": list(full_ids)}


def parse_olmoe_output(output: str) -> dict[str, Any]:
    load = LOAD_RE.search(output)
    match = MATCH_RE.search(output)
    peak = PEAK_RE.search(output)
    hit = HIT_RE.search(output)
    speed = SPEED_RE.search(output)
    missing = [
        name for name, value in (
            ("load telemetry", load),
            ("token-match telemetry", match),
            ("peak RSS telemetry", peak),
            ("expert-hit telemetry", hit),
            ("speed telemetry", speed),
        ) if value is None
    ]
    if missing:
        raise LatticeError("OLMoE engine output is missing: " + ", ".join(missing))
    assert load is not None and match is not None and peak is not None and hit is not None and speed is not None
    metrics = {
        "load_seconds": float(load.group(1)),
        "rss_after_load_gb": float(load.group(2)),
        "matching_tokens": int(match.group(1)),
        "continuation_tokens": int(match.group(2)),
        "peak_rss_gb": float(peak.group(1)),
        "expert_hit_pct": float(hit.group(1)),
        "expert_hits": int(hit.group(2)),
        "expert_misses": int(hit.group(3)),
        "tok_s": float(speed.group(1)),
        "decode_seconds": float(speed.group(2)),
        "reported_tokens": int(speed.group(3)),
    }
    numeric_fields = (
        "load_seconds", "rss_after_load_gb", "peak_rss_gb", "expert_hit_pct",
        "tok_s", "decode_seconds",
    )
    if any(not math.isfinite(float(metrics[field])) or float(metrics[field]) < 0 for field in numeric_fields):
        raise LatticeError("OLMoE engine emitted non-finite or negative telemetry")
    if metrics["tok_s"] <= 0 or metrics["continuation_tokens"] <= 0:
        raise LatticeError("OLMoE engine emitted non-positive throughput or token count")
    if metrics["reported_tokens"] != metrics["continuation_tokens"]:
        raise LatticeError("OLMoE speed token count does not match reference continuation")
    if not 0 <= metrics["matching_tokens"] <= metrics["continuation_tokens"]:
        raise LatticeError("OLMoE token-match count is invalid")
    if not 0 <= metrics["expert_hit_pct"] <= 100:
        raise LatticeError("OLMoE expert hit percentage is invalid")
    return metrics


def _hardware_identity(model: Path) -> tuple[dict[str, Any], str]:
    stat = model.stat()
    usage = shutil.disk_usage(model)
    payload = {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "logical_cpu_count": os.cpu_count(),
        "model_storage": {
            "path": str(model.resolve()),
            "device": int(getattr(stat, "st_dev", 0)),
            "volume_total_bytes": int(usage.total),
        },
    }
    return payload, sha256_bytes(canonical_json(payload))


def _acceptance_environment(model: Path, threads: int | None) -> tuple[dict[str, str], dict[str, str]]:
    env = clean_environment(dict(os.environ))
    for key in OLMOE_AMBIENT_KEYS:
        env.pop(key, None)
    controlled = {
        "SNAP": str(model),
        "PILOT": "0",
        "WIDE": "1",
        "HOT": "0",
        "WARMUP": "5",
        "SMOOTH": "0.3",
        "CONF_LIMIT": "0.92",
        "PILOT_EVICT_GUARD": "1",
        "EXPERT_DROP": "0",
        "IDOT": "0",
    }
    if threads is not None:
        controlled["OMP_NUM_THREADS"] = str(threads)
    env.update(controlled)
    return env, controlled


def _execution_fingerprint(
    *,
    model_fingerprint: str,
    runtime_fingerprint: str,
    hardware_fingerprint: str,
    reference_sha256: str,
    configuration: dict[str, Any],
) -> str:
    return sha256_bytes(canonical_json({
        "schema_version": 1,
        "model_fingerprint": model_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "hardware_fingerprint": hardware_fingerprint,
        "reference_sha256": reference_sha256,
        "configuration": configuration,
    }))


def prepare_olmoe_acceptance(
    repo_root: Path,
    model: Path,
    reference_path: Path,
    *,
    engine: Path | None = None,
    cache_cap: int = 16,
    quant_bits: int = 8,
    repeats: int = 3,
    timeout: int = 1800,
    threads: int | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    model = model.expanduser().resolve()
    reference_path = reference_path.expanduser().resolve()
    c_dir = repo_root / "c"
    coli = c_dir / "coli"
    if not coli.is_file():
        raise LatticeError(f"not a Colibri checkout (missing c/coli): {repo_root}")
    resolved_engine, family = resolve_engine(c_dir, model, engine)
    if family != "olmoe":
        raise LatticeError(f"OLMoE acceptance requires an OLMoE checkpoint; detected {family}")
    if not 0 <= cache_cap <= 512:
        raise LatticeError("OLMoE cache cap must be between 0 and 512 experts per layer")
    if not 2 <= quant_bits <= 8:
        raise LatticeError("OLMoE quant bits must be between 2 and 8")
    if not 1 <= repeats <= 20:
        raise LatticeError("OLMoE acceptance repeats must be between 1 and 20")
    if timeout < 1:
        raise LatticeError("OLMoE acceptance timeout must be positive")
    if threads is not None and not 1 <= threads <= 4096:
        raise LatticeError("OLMoE acceptance threads must be between 1 and 4096")
    try:
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LatticeError(f"missing OLMoE config: {model / 'config.json'}") from error
    except json.JSONDecodeError as error:
        raise LatticeError(f"invalid OLMoE config: {error}") from error
    if detect_family(config) != "olmoe":
        raise LatticeError("model config is not OLMoE")
    if not (model / "tokenizer.json").is_file():
        raise LatticeError("converted OLMoE model is missing tokenizer.json")
    vocab_size = config.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        vocab_size = None
    reference = load_reference(reference_path, vocab_size=vocab_size)
    model_fingerprint = fingerprint_model(model)
    runtime_fingerprint = fingerprint_runtime(c_dir, coli, resolved_engine)
    hardware, hardware_fingerprint = _hardware_identity(model)
    _, controlled_environment = _acceptance_environment(model, threads)
    reference_sha256 = sha256_file(reference_path)
    configuration = {
        "cache_cap_per_layer": cache_cap,
        "quant_bits": quant_bits,
        "repeats": repeats,
        "timeout_seconds": timeout,
        "threads": threads,
        "environment": controlled_environment,
    }
    execution_fingerprint = _execution_fingerprint(
        model_fingerprint=model_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        hardware_fingerprint=hardware_fingerprint,
        reference_sha256=reference_sha256,
        configuration=configuration,
    )
    seed = {
        "schema": ACCEPTANCE_SCHEMA,
        "execution_fingerprint": execution_fingerprint,
    }
    return {
        "schema_version": 1,
        "schema": ACCEPTANCE_SCHEMA,
        "id": short_id("acceptance", seed),
        "created_at": utc_now(),
        "assurance_level": ASSURANCE_LEVEL,
        "repo_root": str(repo_root),
        "model_path": str(model),
        "engine_path": str(resolved_engine),
        "reference_path": str(reference_path),
        "model_family": family,
        "model_fingerprint": model_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "hardware_fingerprint": hardware_fingerprint,
        "execution_fingerprint": execution_fingerprint,
        "hardware": hardware,
        "reference_sha256": reference_sha256,
        "reference": {
            "prompt_tokens": len(reference["prompt_ids"]),
            "continuation_tokens": len(reference["full_ids"]) - len(reference["prompt_ids"]),
        },
        "configuration": configuration,
        "runs": [],
    }


def _verify_prepared_identity(prepared: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    if prepared.get("schema") != ACCEPTANCE_SCHEMA or prepared.get("model_family") != "olmoe":
        raise LatticeError("invalid OLMoE acceptance record")
    repo_root = Path(prepared["repo_root"])
    model = Path(prepared["model_path"])
    engine = Path(prepared["engine_path"])
    reference_path = Path(prepared["reference_path"])
    c_dir = repo_root / "c"
    coli = c_dir / "coli"
    config = prepared["configuration"]
    env, controlled = _acceptance_environment(model, config.get("threads"))
    checks = {
        "model": fingerprint_model(model) == prepared.get("model_fingerprint"),
        "runtime": fingerprint_runtime(c_dir, coli, engine) == prepared.get("runtime_fingerprint"),
        "reference": sha256_file(reference_path) == prepared.get("reference_sha256"),
        "environment": controlled == config.get("environment"),
    }
    hardware, hardware_fingerprint = _hardware_identity(model)
    checks["hardware"] = hardware_fingerprint == prepared.get("hardware_fingerprint")
    expected_execution = _execution_fingerprint(
        model_fingerprint=prepared["model_fingerprint"],
        runtime_fingerprint=prepared["runtime_fingerprint"],
        hardware_fingerprint=hardware_fingerprint,
        reference_sha256=prepared["reference_sha256"],
        configuration=config,
    )
    checks["execution"] = expected_execution == prepared.get("execution_fingerprint")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise LatticeError("OLMoE acceptance identity changed before execution: " + ", ".join(failed))
    if hardware != prepared.get("hardware"):
        raise LatticeError("OLMoE acceptance hardware description changed before execution")
    return env, controlled


def run_olmoe_acceptance(prepared: dict[str, Any]) -> dict[str, Any]:
    model = Path(prepared["model_path"])
    engine = Path(prepared["engine_path"])
    reference_path = Path(prepared["reference_path"])
    config = prepared["configuration"]
    env, controlled = _verify_prepared_identity(prepared)
    command = [
        str(engine),
        str(config["cache_cap_per_layer"]),
        str(config["quant_bits"]),
        str(reference_path),
    ]
    runs: list[dict[str, Any]] = []
    for repeat in range(config["repeats"]):
        result: ProcessResult = run_bounded(
            command,
            env=env,
            timeout=config["timeout_seconds"],
            cwd=Path(prepared["repo_root"]) / "c",
        )
        output = f"{result.stdout}\n{result.stderr}"
        status = "success"
        error: str | None = None
        metrics: dict[str, Any] | None = None
        if result.timed_out:
            status, error = "failed", "OLMoE acceptance run timed out"
        elif result.returncode != 0:
            status, error = "failed", f"OLMoE engine exited with code {result.returncode}"
        elif result.output_truncated:
            status, error = "failed", "OLMoE engine output exceeded the evidence limit"
        else:
            try:
                metrics = parse_olmoe_output(output)
                expected = prepared["reference"]["continuation_tokens"]
                if metrics["continuation_tokens"] != expected:
                    raise LatticeError("OLMoE engine continuation length differs from reference")
                if metrics["matching_tokens"] != expected:
                    raise LatticeError(
                        f"OLMoE token mismatch: {metrics['matching_tokens']}/{expected} matched"
                    )
            except LatticeError as exc:
                status, error = "failed", str(exc)
        run_seed = {
            "acceptance_id": prepared["id"],
            "repeat": repeat,
            "command": command,
            "environment": controlled,
        }
        run = {
            "schema_version": 1,
            "id": short_id("accept-run", run_seed),
            "acceptance_id": prepared["id"],
            "repeat": repeat,
            "status": status,
            "command": list(result.command),
            "environment": controlled,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "duration_seconds": result.duration_seconds,
            "output_truncated": result.output_truncated,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "metrics": metrics,
            "error": error,
        }
        runs.append(seal_record(run))
    successful = [run for run in runs if run["status"] == "success"]
    accepted = len(successful) == config["repeats"]
    metric_rows = [run["metrics"] for run in successful]
    summary: dict[str, Any] = {
        "accepted": accepted,
        "successful_runs": len(successful),
        "required_runs": config["repeats"],
        "all_runs_token_exact": accepted,
    }
    if metric_rows:
        for name in ("tok_s", "peak_rss_gb", "rss_after_load_gb", "load_seconds", "expert_hit_pct"):
            values = [float(row[name]) for row in metric_rows]
            summary[name] = {
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
    completed = {
        **prepared,
        "completed_at": utc_now(),
        "status": "accepted" if accepted else "rejected",
        "runs": runs,
        "summary": summary,
        "limitations": [
            "Token-exact agreement is checked against one supplied reference continuation.",
            "This acceptance does not compare complete logit vectors or downstream task quality.",
            "This acceptance does not tune or promote a deployment configuration.",
            "Real-model evidence exists only after this command is run with actual converted OLMoE weights.",
        ],
    }
    completed["evidence_root_sha256"] = sha256_bytes(canonical_json({
        "acceptance": {key: value for key, value in completed.items() if key not in {"runs", "evidence_root_sha256"}},
        "runs": [{"id": run["id"], "sha256": run["record_sha256"]} for run in runs],
    }))
    return completed


def render_acceptance_report(record: dict[str, Any]) -> str:
    summary = record["summary"]
    lines = [
        "# OLMoE real-model acceptance report",
        "",
        f"- Acceptance ID: `{record['id']}`",
        f"- Status: **{record['status'].upper()}**",
        f"- Assurance: `{record['assurance_level']}`",
        f"- Model fingerprint: `{record['model_fingerprint']}`",
        f"- Runtime fingerprint: `{record['runtime_fingerprint']}`",
        f"- Hardware fingerprint: `{record['hardware_fingerprint']}`",
        f"- Execution fingerprint: `{record['execution_fingerprint']}`",
        f"- Reference SHA-256: `{record['reference_sha256']}`",
        f"- Evidence root: `{record['evidence_root_sha256']}`",
        "",
        "## Result",
        "",
        f"- Exact successful runs: {summary['successful_runs']}/{summary['required_runs']}",
        f"- All runs token-exact: {summary['all_runs_token_exact']}",
    ]
    for label, key, unit in (
        ("Decode throughput", "tok_s", "tok/s"),
        ("Peak RSS", "peak_rss_gb", "GB"),
        ("RSS after load", "rss_after_load_gb", "GB"),
        ("Load time", "load_seconds", "s"),
        ("Expert hit rate", "expert_hit_pct", "%"),
    ):
        values = summary.get(key)
        if values:
            lines.append(
                f"- {label}: median {values['median']:.4g} {unit} "
                f"(range {values['minimum']:.4g}–{values['maximum']:.4g})"
            )
    lines.extend([
        "",
        "## Configuration",
        "",
        f"- Cache cap per layer: {record['configuration']['cache_cap_per_layer']}",
        f"- Expert quantization bits: {record['configuration']['quant_bits']}",
        f"- Repeats: {record['configuration']['repeats']}",
        f"- Threads: {record['configuration']['threads'] or 'engine auto-tuning'}",
        "",
        "## Scientific boundary",
        "",
    ])
    lines.extend(f"- {item}" for item in record["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_acceptance_outputs(record: dict[str, Any], output: Path, report: Path | None = None) -> None:
    output = output.expanduser().resolve()
    report_path = report.expanduser().resolve() if report is not None else None
    report_text = render_acceptance_report(record) if report_path is not None else None
    if report_path is not None and report_path.exists():
        if report_path.read_text(encoding="utf-8") != report_text:
            raise LatticeError(f"refusing to overwrite acceptance report: {report_path}")
    atomic_write_json(output, record, exclusive=True)
    if report_path is not None and not report_path.exists():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with report_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(report_text or "")
        except FileExistsError as error:
            raise LatticeError(f"refusing to overwrite acceptance report: {report_path}") from error
