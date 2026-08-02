from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import statistics
import uuid
from pathlib import Path
from typing import Any

from .colibri import clean_environment, detect_family, fingerprint_model, fingerprint_runtime, resolve_engine
from .common import (
    LatticeError,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    load_json,
    sha256_bytes,
    sha256_file,
    short_id,
    strict_json_loads,
    utc_now,
)
from .evidence import record_digest, seal_record, verify_record_digest
from .process import ProcessResult, run_bounded

ACCEPTANCE_SCHEMA = "lattice-olmoe-acceptance/2"
ASSURANCE_LEVEL = "single-reference-token-exact-execution"
MAX_REFERENCE_TOKENS = 2048
MAX_CONTINUATION_TOKENS = 512
MAX_REFERENCE_BYTES = 4 * 1024 * 1024
REMOTE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

LOAD_RE = re.compile(
    r"resident weights loaded in\s+([0-9.]+)s\s*\|\s*RSS after load:\s*([0-9.]+)\s+GB",
    re.IGNORECASE,
)
REFERENCE_RE = re.compile(r"^Reference:\s*([0-9 ]*)$", re.MULTILINE)
ENGINE_RE = re.compile(r"^C engine\s*:\s*([0-9 ]*)$", re.MULTILINE)
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
PERSISTED_PINS_DISABLED_RE = re.compile(
    r"^OLMOE_PERSISTED_PINS_DISABLED$", re.MULTILINE
)

OLMOE_AMBIENT_KEYS = frozenset({
    "SNAP", "CHAT", "PPL", "TEMP", "NUCLEUS", "PILOT", "WIDE", "HOT",
    "WARMUP", "SMOOTH", "CONF_LIMIT", "PILOT_EVICT_GUARD", "EXPERT_DROP",
    "IDOT", "OMP_NUM_THREADS", "OLMOE_IGNORE_PERSISTED_PINS",
})


def _as_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise LatticeError(f"{label} must be an integer between {minimum} and {maximum}")
    return value


def _finite(value: Any, label: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise LatticeError(f"{label} must be finite")
    number = float(value)
    if (strict and number <= minimum) or (not strict and number < minimum):
        relation = "greater than" if strict else "at least"
        raise LatticeError(f"{label} must be {relation} {minimum}")
    return number


def _single(pattern: re.Pattern[str], output: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(output))
    if len(matches) != 1:
        raise LatticeError(f"OLMoE engine must emit exactly one {label}; found {len(matches)}")
    return matches[0]


def _token_list(text: str, label: str) -> list[int]:
    try:
        values = [int(value) for value in text.split()]
    except ValueError as error:
        raise LatticeError(f"OLMoE {label} contains a non-integer token ID") from error
    if not values or any(value < 0 for value in values):
        raise LatticeError(f"OLMoE {label} is empty or contains a negative token ID")
    return values


def _reference_provenance(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("schema_version") != 2 or raw.get("generator") != "transformers-greedy":
        raise LatticeError(
            "OLMoE reference must be produced by the version-2 Transformers reference tool"
        )
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise LatticeError("OLMoE reference has no model identity")
    source_kind = raw.get("source_kind")
    resolved_revision = raw.get("resolved_revision")
    source_fingerprint = raw.get("source_fingerprint")
    if source_kind == "huggingface":
        if not isinstance(resolved_revision, str) or not REMOTE_REVISION_RE.fullmatch(resolved_revision):
            raise LatticeError("remote OLMoE reference is not pinned to a 40-character Hub commit")
        if source_fingerprint is not None:
            raise LatticeError("remote OLMoE reference must not claim a local source fingerprint")
    elif source_kind == "local":
        if not isinstance(source_fingerprint, str) or not DIGEST_RE.fullmatch(source_fingerprint):
            raise LatticeError("local OLMoE reference has no valid source fingerprint")
        if resolved_revision is not None:
            raise LatticeError("local OLMoE reference must not claim a Hub revision")
    else:
        raise LatticeError("OLMoE reference source_kind must be 'huggingface' or 'local'")
    generation = raw.get("generation")
    expected_generation = {
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
    }
    if not isinstance(generation, dict) or any(generation.get(key) != value for key, value in expected_generation.items()):
        raise LatticeError("OLMoE reference generation controls are not the required greedy contract")
    prompt_sha = raw.get("prompt_sha256")
    if not isinstance(prompt_sha, str) or not DIGEST_RE.fullmatch(prompt_sha):
        raise LatticeError("OLMoE reference has no valid prompt digest")
    versions = raw.get("versions")
    if not isinstance(versions, dict) or not all(
        isinstance(versions.get(key), str) and versions[key]
        for key in ("python", "torch", "transformers")
    ):
        raise LatticeError("OLMoE reference library provenance is incomplete")
    return {
        "generator": raw["generator"],
        "model": model,
        "source_kind": source_kind,
        "resolved_revision": resolved_revision,
        "source_fingerprint": source_fingerprint,
        "prompt_sha256": prompt_sha,
        "template_mode": raw.get("template_mode"),
        "device_request": raw.get("device_request"),
        "dtype_request": raw.get("dtype_request"),
        "versions": versions,
        "generation": generation,
    }


def load_reference(path: Path, *, vocab_size: int | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    raw = load_json(path, max_bytes=MAX_REFERENCE_BYTES)
    if not isinstance(raw, dict):
        raise LatticeError("OLMoE reference must be a JSON object")
    provenance = _reference_provenance(raw)
    prompt_ids = raw.get("prompt_ids")
    full_ids = raw.get("full_ids")
    if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
        raise LatticeError("OLMoE reference requires prompt_ids and full_ids arrays")
    if not prompt_ids or len(full_ids) <= len(prompt_ids):
        raise LatticeError("OLMoE reference requires a non-empty prompt and continuation")
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise LatticeError("OLMoE full_ids must begin with prompt_ids")
    continuation = len(full_ids) - len(prompt_ids)
    if len(full_ids) > MAX_REFERENCE_TOKENS:
        raise LatticeError(f"OLMoE reference exceeds {MAX_REFERENCE_TOKENS} total tokens")
    if continuation > MAX_CONTINUATION_TOKENS:
        raise LatticeError(f"OLMoE continuation exceeds {MAX_CONTINUATION_TOKENS} tokens")
    for label, values in (("prompt_ids", prompt_ids), ("full_ids", full_ids)):
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LatticeError(f"OLMoE reference {label} contains an invalid token ID")
            if vocab_size is not None and value >= vocab_size:
                raise LatticeError(f"OLMoE reference token ID {value} exceeds vocab_size={vocab_size}")
    if raw.get("actual_new_tokens") != continuation:
        raise LatticeError("OLMoE reference actual_new_tokens does not match the token arrays")
    return {
        "prompt_ids": list(prompt_ids),
        "full_ids": list(full_ids),
        "provenance": provenance,
    }


def _rounded_consistent(reported: float, exact: float, *, decimals: int) -> bool:
    half_unit = 0.5 * 10 ** (-decimals)
    return abs(reported - exact) <= half_unit + 1e-12


def parse_olmoe_output(output: str) -> dict[str, Any]:
    load = _single(LOAD_RE, output, "load telemetry")
    reference_match = _single(REFERENCE_RE, output, "reference token list")
    engine_match = _single(ENGINE_RE, output, "engine token list")
    match = _single(MATCH_RE, output, "token-match telemetry")
    peak = _single(PEAK_RE, output, "peak RSS telemetry")
    hit = _single(HIT_RE, output, "expert-hit telemetry")
    speed = _single(SPEED_RE, output, "speed telemetry")
    _single(PERSISTED_PINS_DISABLED_RE, output, "persisted-pin isolation marker")
    reference_tokens = _token_list(reference_match.group(1), "reference token list")
    engine_tokens = _token_list(engine_match.group(1), "engine token list")
    matching_tokens = int(match.group(1))
    continuation_tokens = int(match.group(2))
    reported_tokens = int(speed.group(3))
    if len(reference_tokens) != continuation_tokens or len(engine_tokens) != continuation_tokens:
        raise LatticeError("OLMoE printed token-list length does not match continuation length")
    actual_matches = sum(left == right for left, right in zip(reference_tokens, engine_tokens))
    if matching_tokens != actual_matches:
        raise LatticeError("OLMoE match counter disagrees with the printed token arrays")
    if reported_tokens != continuation_tokens:
        raise LatticeError("OLMoE speed token count does not match reference continuation")
    load_seconds = _finite(float(load.group(1)), "OLMoE load time")
    rss_after_load = _finite(float(load.group(2)), "OLMoE load RSS")
    peak_rss = _finite(float(peak.group(1)), "OLMoE peak RSS")
    if peak_rss + 0.01 < rss_after_load:
        raise LatticeError("OLMoE peak RSS is below RSS reported after load")
    hit_pct = _finite(float(hit.group(1)), "OLMoE expert hit percentage")
    hits, misses = int(hit.group(2)), int(hit.group(3))
    if hit_pct > 100:
        raise LatticeError("OLMoE expert hit percentage is above 100")
    total = hits + misses
    expected_hit = 0.0 if total == 0 else 100.0 * hits / total
    if not _rounded_consistent(hit_pct, expected_hit, decimals=1):
        raise LatticeError("OLMoE expert hit percentage disagrees with hit/miss counters")
    tok_s = _finite(float(speed.group(1)), "OLMoE throughput", strict=True)
    decode_seconds = _finite(float(speed.group(2)), "OLMoE decode duration", strict=True)
    exact_tok_s = continuation_tokens / decode_seconds
    # Engine prints duration to 0.1 s and speed to 0.01 tok/s. Compare the
    # reported speed against the interval implied by the rounded duration.
    low_seconds = max(1e-9, decode_seconds - 0.05)
    high_seconds = decode_seconds + 0.05
    low_speed = continuation_tokens / high_seconds - 0.005 - 1e-9
    high_speed = continuation_tokens / low_seconds + 0.005 + 1e-9
    if not low_speed <= tok_s <= high_speed:
        raise LatticeError("OLMoE throughput disagrees with token count and rounded duration")
    return {
        "load_seconds": load_seconds,
        "rss_after_load_gb": rss_after_load,
        "reference_tokens": reference_tokens,
        "engine_tokens": engine_tokens,
        "matching_tokens": matching_tokens,
        "continuation_tokens": continuation_tokens,
        "peak_rss_gb": peak_rss,
        "expert_hit_pct": hit_pct,
        "expert_hits": hits,
        "expert_misses": misses,
        "tok_s": tok_s,
        "decode_seconds": decode_seconds,
        "reported_tokens": reported_tokens,
        "implied_tok_s_from_rounded_duration": exact_tok_s,
        "persisted_pins_disabled": True,
    }


def _hardware_identity(model: Path) -> tuple[dict[str, Any], str]:
    stat = model.stat()
    usage = shutil.disk_usage(model)
    payload = {
        "schema_version": 2,
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
        "OLMOE_IGNORE_PERSISTED_PINS": "1",
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
        "schema_version": 2,
        "model_fingerprint": model_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "hardware_fingerprint": hardware_fingerprint,
        "reference_sha256": reference_sha256,
        "configuration": configuration,
    }))


def _read_config(model: Path) -> dict[str, Any]:
    path = model / "config.json"
    try:
        value = strict_json_loads(path.read_text(encoding="utf-8"), label=f"OLMoE config {path}")
    except FileNotFoundError as error:
        raise LatticeError(f"missing OLMoE config: {path}") from error
    except UnicodeDecodeError as error:
        raise LatticeError(f"OLMoE config is not UTF-8: {path}") from error
    if not isinstance(value, dict):
        raise LatticeError("OLMoE config must be an object")
    return value


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
    cache_cap = _as_int(cache_cap, "OLMoE cache cap", 1, 512)
    quant_bits = _as_int(quant_bits, "OLMoE quant bits", 2, 8)
    repeats = _as_int(repeats, "OLMoE acceptance repeats", 1, 20)
    timeout = _as_int(timeout, "OLMoE acceptance timeout", 1, 86400)
    if threads is not None:
        threads = _as_int(threads, "OLMoE acceptance threads", 1, 4096)
    c_dir = repo_root / "c"
    coli = c_dir / "coli"
    if not coli.is_file() or coli.is_symlink():
        raise LatticeError(f"not a Colibri checkout with a regular c/coli launcher: {repo_root}")
    resolved_engine, family = resolve_engine(c_dir, model, engine)
    if family != "olmoe":
        raise LatticeError(f"OLMoE acceptance requires an OLMoE checkpoint; detected {family}")
    config = _read_config(model)
    if detect_family(config) != "olmoe":
        raise LatticeError("model config is not OLMoE")
    if not (model / "tokenizer.json").is_file():
        raise LatticeError("converted OLMoE model is missing tokenizer.json")
    vocab_size = config.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise LatticeError("OLMoE config has no valid vocab_size")
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
    instance_nonce = uuid.uuid4().hex
    return {
        "schema_version": 2,
        "schema": ACCEPTANCE_SCHEMA,
        "id": short_id("acceptance", {
            "execution_fingerprint": execution_fingerprint,
            "instance_nonce": instance_nonce,
        }),
        "instance_nonce": instance_nonce,
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
            "provenance": reference["provenance"],
        },
        "configuration": configuration,
        "runs": [],
    }


def _verify_prepared_identity(prepared: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    if (prepared.get("schema") != ACCEPTANCE_SCHEMA or prepared.get("schema_version") != 2
            or prepared.get("model_family") != "olmoe"):
        raise LatticeError("invalid OLMoE acceptance record")
    repo_root = Path(prepared["repo_root"])
    model = Path(prepared["model_path"])
    engine = Path(prepared["engine_path"])
    reference_path = Path(prepared["reference_path"])
    c_dir = repo_root / "c"
    coli = c_dir / "coli"
    config = prepared["configuration"]
    env, controlled = _acceptance_environment(model, config.get("threads"))
    hardware, hardware_fingerprint = _hardware_identity(model)
    actual = {
        "model": fingerprint_model(model),
        "runtime": fingerprint_runtime(c_dir, coli, engine),
        "reference": sha256_file(reference_path),
        "hardware": hardware_fingerprint,
    }
    expected = {
        "model": prepared.get("model_fingerprint"),
        "runtime": prepared.get("runtime_fingerprint"),
        "reference": prepared.get("reference_sha256"),
        "hardware": prepared.get("hardware_fingerprint"),
    }
    failed = [name for name in actual if actual[name] != expected[name]]
    if controlled != config.get("environment"):
        failed.append("environment")
    execution = _execution_fingerprint(
        model_fingerprint=actual["model"],
        runtime_fingerprint=actual["runtime"],
        hardware_fingerprint=actual["hardware"],
        reference_sha256=actual["reference"],
        configuration=config,
    )
    if execution != prepared.get("execution_fingerprint"):
        failed.append("execution")
    if hardware != prepared.get("hardware"):
        failed.append("hardware-description")
    if failed:
        raise LatticeError("OLMoE acceptance identity changed before execution: " + ", ".join(sorted(set(failed))))
    return env, controlled


def run_olmoe_acceptance(prepared: dict[str, Any]) -> dict[str, Any]:
    model = Path(prepared["model_path"])
    engine = Path(prepared["engine_path"])
    reference_path = Path(prepared["reference_path"])
    config = prepared["configuration"]
    command = [
        str(engine),
        str(config["cache_cap_per_layer"]),
        str(config["quant_bits"]),
        str(reference_path),
    ]
    runs: list[dict[str, Any]] = []
    for repeat in range(config["repeats"]):
        env, controlled = _verify_prepared_identity(prepared)
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
                expected_count = prepared["reference"]["continuation_tokens"]
                if metrics["continuation_tokens"] != expected_count:
                    raise LatticeError("OLMoE engine continuation length differs from reference")
                reference = load_reference(reference_path)
                expected_tokens = reference["full_ids"][len(reference["prompt_ids"]):]
                if metrics["reference_tokens"] != expected_tokens:
                    raise LatticeError("OLMoE engine printed a reference path different from the supplied file")
                if metrics["engine_tokens"] != expected_tokens:
                    raise LatticeError("OLMoE token mismatch against the supplied Transformers reference")
            except LatticeError as exc:
                status, error = "failed", str(exc)
                metrics = None
        run = {
            "schema_version": 2,
            "id": short_id("accept-run", {
                "acceptance_id": prepared["id"],
                "repeat": repeat,
            }),
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
            "stdout_bytes": result.stdout_bytes,
            "stderr_bytes": result.stderr_bytes,
            "metrics": metrics,
            "error": error,
        }
        runs.append(seal_record(run))
    _verify_prepared_identity(prepared)
    successful = [run for run in runs if run["status"] == "success"]
    accepted = len(successful) == config["repeats"]
    metric_rows = [run["metrics"] for run in successful]
    summary: dict[str, Any] = {
        "accepted": accepted,
        "successful_runs": len(successful),
        "required_runs": config["repeats"],
        "all_runs_token_exact": accepted,
        "all_runs_persisted_pins_disabled": accepted,
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
            "Agreement is checked against one supplied continuation generated by a separate Transformers implementation.",
            "Token agreement does not establish complete logit equality or downstream model quality.",
            "The conservative scalar path is acceptance telemetry, not optimized serving performance.",
            "This acceptance does not tune, compare, promote or authorize a deployment configuration.",
            "Model and hardware fingerprints are practical identities, not remote attestation.",
            "No real-model claim exists until this command is run with actual converted weights and the retained record verifies.",
        ],
    }
    completed["evidence_root_sha256"] = sha256_bytes(canonical_json({
        "acceptance": {key: value for key, value in completed.items() if key not in {"runs", "evidence_root_sha256", "record_sha256"}},
        "runs": [{"id": run["id"], "sha256": run["record_sha256"]} for run in runs],
    }))
    return seal_record(completed)


def verify_olmoe_acceptance(record: dict[str, Any], *, live: bool = False) -> dict[str, Any]:
    if not isinstance(record, dict) or record.get("schema") != ACCEPTANCE_SCHEMA:
        raise LatticeError("unsupported OLMoE acceptance evidence")
    verify_record_digest(record, "OLMoE acceptance record")
    runs = record.get("runs")
    config = record.get("configuration")
    if not isinstance(runs, list) or not isinstance(config, dict):
        raise LatticeError("OLMoE acceptance evidence is incomplete")
    repeats = _as_int(config.get("repeats"), "OLMoE acceptance repeats", 1, 20)
    if len(runs) != repeats:
        raise LatticeError("OLMoE acceptance run count does not match configuration")
    seen: set[int] = set()
    successful = 0
    for run in runs:
        verify_record_digest(run, f"OLMoE run {run.get('id', '<unknown>')}")
        repeat = run.get("repeat")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or not 0 <= repeat < repeats or repeat in seen:
            raise LatticeError("OLMoE acceptance contains an invalid or duplicate repeat")
        seen.add(repeat)
        if run.get("acceptance_id") != record.get("id"):
            raise LatticeError("OLMoE run belongs to another acceptance")
        status = run.get("status")
        if status == "success":
            successful += 1
            metrics = run.get("metrics")
            if not isinstance(metrics, dict) or metrics.get("engine_tokens") != metrics.get("reference_tokens"):
                raise LatticeError("successful OLMoE run is not token-exact")
            if metrics.get("persisted_pins_disabled") is not True:
                raise LatticeError("successful OLMoE run did not prove persisted-pin isolation")
            if run.get("returncode") != 0 or run.get("timed_out") or run.get("output_truncated"):
                raise LatticeError("successful OLMoE run has incomplete process evidence")
            if run.get("error") is not None:
                raise LatticeError("successful OLMoE run contains an error")
        elif status == "failed":
            if run.get("metrics") is not None or not isinstance(run.get("error"), str) or not run["error"]:
                raise LatticeError("failed OLMoE run has invalid failure evidence")
        else:
            raise LatticeError("OLMoE run has an invalid status")
        for field in ("stdout_bytes", "stderr_bytes"):
            value = run.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LatticeError(f"OLMoE run has invalid {field}")
    summary = record.get("summary")
    if not isinstance(summary, dict) or summary.get("successful_runs") != successful or summary.get("required_runs") != repeats:
        raise LatticeError("OLMoE acceptance summary does not match runs")
    accepted = successful == repeats
    if record.get("status") != ("accepted" if accepted else "rejected"):
        raise LatticeError("OLMoE acceptance status does not match runs")
    expected_root = sha256_bytes(canonical_json({
        "acceptance": {key: value for key, value in record.items() if key not in {"runs", "evidence_root_sha256", "record_sha256"}},
        "runs": [{"id": run["id"], "sha256": run["record_sha256"]} for run in runs],
    }))
    if expected_root != record.get("evidence_root_sha256"):
        raise LatticeError("OLMoE acceptance evidence root mismatch")
    if live:
        _verify_prepared_identity(record)
    return {
        "status": "ok",
        "acceptance_id": record["id"],
        "result": record["status"],
        "runs": repeats,
        "live_identity_verified": live,
        "record_sha256": record["record_sha256"],
        "evidence_root_sha256": record["evidence_root_sha256"],
    }


def render_acceptance_report(record: dict[str, Any]) -> str:
    verify_olmoe_acceptance(record, live=False)
    summary = record["summary"]
    lines = [
        "# OLMoE single-reference execution evidence",
        "",
        "> This is a narrow execution-acceptance record, not a production-readiness or performance certificate.",
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
        f"- Record SHA-256: `{record['record_sha256']}`",
        "",
        "## Result",
        "",
        f"- Exact successful runs: {summary['successful_runs']}/{summary['required_runs']}",
        f"- All runs token-exact: {summary['all_runs_token_exact']}",
        f"- Persisted pin state disabled in every successful run: {summary['all_runs_persisted_pins_disabled']}",
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
    provenance = record["reference"]["provenance"]
    lines.extend([
        "",
        "## Reference provenance",
        "",
        f"- Generator: `{provenance['generator']}`",
        f"- Model: `{provenance['model']}`",
        f"- Source kind: `{provenance['source_kind']}`",
        f"- Resolved Hub revision: `{provenance['resolved_revision'] or 'not applicable'}`",
        f"- Local source fingerprint: `{provenance['source_fingerprint'] or 'not applicable'}`",
        f"- Prompt SHA-256: `{provenance['prompt_sha256']}`",
        f"- Template mode: `{provenance['template_mode']}`",
        "",
        "## Configuration",
        "",
        f"- Cache cap per layer: {record['configuration']['cache_cap_per_layer']}",
        f"- Expert quantization bits: {record['configuration']['quant_bits']}",
        f"- Repeats: {record['configuration']['repeats']}",
        f"- Threads: {record['configuration']['threads'] or 'engine auto-tuning'}",
        "- Persisted pins: disabled at native engine load and save boundaries",
        "- Arithmetic path: byte-exact scalar path (`IDOT=0`)",
        "",
        "## Scientific boundary",
        "",
    ])
    lines.extend(f"- {item}" for item in record["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_acceptance_outputs(record: dict[str, Any], output: Path, report: Path | None = None) -> None:
    verify_olmoe_acceptance(record, live=False)
    output = output.expanduser().resolve()
    report_path = report.expanduser().resolve() if report is not None else None
    report_text = render_acceptance_report(record) if report_path is not None else None
    if report_path is not None and report_path.exists():
        if report_path.read_text(encoding="utf-8") != report_text:
            raise LatticeError(f"refusing to overwrite write-once acceptance report: {report_path}")
    atomic_write_json(output, record, exclusive=True)
    if report_path is not None:
        atomic_write_text(report_path, report_text or "", exclusive=True)


def load_and_verify_olmoe_acceptance(path: Path, *, live: bool = False) -> dict[str, Any]:
    record = load_json(path, max_bytes=64 * 1024 * 1024)
    if not isinstance(record, dict):
        raise LatticeError("OLMoE acceptance evidence must be a JSON object")
    return verify_olmoe_acceptance(record, live=live)
