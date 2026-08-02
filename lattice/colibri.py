from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from .common import LatticeError, canonical_json, sha256_bytes, sha256_file
from .process import ProcessResult, run_bounded

PROMPT_RE = re.compile(r"\[PROMPT_TOKENS\]\s+\d+:\s*([0-9 ]+)")
TOKENS_RE = re.compile(r"\[TOKENS\]\s+\d+\s+generated:\s*([0-9 ]+)")
SPEED_RE = re.compile(r"REPLAY decode:\s+\d+\s+tokens.*?\|\s*([0-9.]+)\s+tok/s")
HIT_RE = re.compile(r"expert hit\s+([0-9.]+)%")
LATENCY_RE = re.compile(r"latency p50\s+([0-9.]+)\s*ms.*?p99\s+([0-9.]+)\s*ms")

SAFE_TUNABLE_KEYS = frozenset({
    "OMP_NUM_THREADS",
    "COLI_NUMA",
    "PIPE",
    "PIPE_WORKERS",
    "DIRECT",
    "URING",
    "PREFETCH",
    "PILOT",
    "PILOT_REAL",
    "PILOT_K",
    "COLI_CUDA_PIPE",
    "COLI_CUDA_ASYNC",
    "COLI_NO_OMP_TUNE",
    "MLOCK",
})

FORBIDDEN_AMBIENT_KEYS = frozenset({
    "TOPK", "TOPP", "NUCLEUS", "CACHE_ROUTE", "ROUTE_M", "ROUTE_J",
    "ROUTE_P", "ROUTE_ALPHA", "DRAFT", "MTP", "PIN", "PIN_GB", "REPIN",
    "AUTOPIN", "STATS", "REF", "REF_FORCE", "REPLAY", "TOKENS", "PROMPT",
    "NGEN", "THINK", "COLI_CUDA_MTP", "COLI_TEMP", "IDOT", "ABSORB",
    "I4S", "SPEC_PIN", "TOOL", "COLI_TOOL_SALVAGE",
})

SERVING_ONLY_KEYS = frozenset({
    "COLI_API_KEY", "COLI_ALLOWED_HOSTS", "COLI_MAX_QUEUE", "COLI_QUEUE_TIMEOUT",
    "COLI_MODEL_ID", "COLI_KV_SLOTS", "KV_SLOTS", "SERVE",
})

QUALIFICATION_SCALAR_KEYS = frozenset({
    "RAM_GB", "CTX", "CUDA_EXPERT_GB", "CAP", "CAP_RAISE", "MLOCK",
    "COLI_MMAP", "COLI_SSD_FAST_GBS", "COLI_NO_FUSED_PAIR", "DISK_SPLIT",
    "COLI_RAM_OVERCOMMIT", "DRAFT", "MTP", "IDOT", "ABSORB", "I4S", "SPEC_PIN",
    "PIPE", "PIPE_WORKERS", "DIRECT", "URING", "PREFETCH", "PILOT",
    "PILOT_REAL", "PILOT_K", "SEED", "KVSAVE", "AUTOPIN", "REPIN",
    "SNAP", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES",
    "GOMP_SPINCOUNT", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
})

TOPOLOGY_KEYS = frozenset({"COLI_MODEL_DIRS", "COLI_MODEL_MIRROR", "COLI_DISK_WEIGHTS"})


@dataclass(frozen=True)
class ColibriContext:
    repo_root: Path
    c_dir: Path
    coli: Path
    engine: Path
    model: Path
    model_family: str
    runtime_fingerprint: str
    model_fingerprint: str
    plan: dict[str, Any]
    doctor: dict[str, Any]
    base_environment: dict[str, str]
    qualification_context: int = 4096
    qualification_environment: dict[str, str] = field(default_factory=dict)
    execution_fingerprint: str = ""
    hardware_fingerprint: str = ""
    storage_topology: dict[str, Any] = field(default_factory=dict)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LatticeError(f"cannot import Colibri support module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def detect_family(config: dict[str, Any]) -> str:
    model_type = str(config.get("model_type") or "").lower()
    architectures = " ".join(map(str, config.get("architectures") or [])).lower()
    text = f"{model_type} {architectures}"
    if "inkling" in text:
        return "inkling"
    if "kimi" in text:
        return "kimi_k3"
    if "olmoe" in text or "olmo_moe" in text or "olmoe" in text.replace("-", ""):
        return "olmoe"
    return "colibri"


def resolve_engine(c_dir: Path, model: Path, explicit: Path | None = None) -> tuple[Path, str]:
    config_path = model / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LatticeError(f"missing model config: {config_path}") from error
    except json.JSONDecodeError as error:
        raise LatticeError(f"invalid model config: {error}") from error
    family = detect_family(config)
    if explicit is not None:
        engine = explicit.expanduser().resolve()
    else:
        suffix = ".exe" if os.name == "nt" else ""
        engine = (c_dir / f"{family}{suffix}").resolve()
        if family == "colibri" and not engine.exists():
            legacy = (c_dir / f"glm{suffix}").resolve()
            if legacy.exists():
                engine = legacy
    if not engine.is_file():
        raise LatticeError(f"Colibri engine not found for {family}: {engine}")
    return engine, family


def _safetensors_header(path: Path) -> bytes:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise LatticeError(f"short safetensors header: {path}")
        header_len = int.from_bytes(raw, "little")
        if header_len < 2 or header_len > size - 8 or header_len > 128 * 1024 * 1024:
            raise LatticeError(f"invalid safetensors header length: {path}")
        header = stream.read(header_len)
        if len(header) != header_len:
            raise LatticeError(f"truncated safetensors header: {path}")
        try:
            parsed = json.loads(header)
        except json.JSONDecodeError as error:
            raise LatticeError(f"invalid safetensors JSON header: {path}: {error}") from error
        if not isinstance(parsed, dict):
            raise LatticeError(f"invalid safetensors header object: {path}")
        for name, meta in parsed.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict) or "data_offsets" not in meta:
                raise LatticeError(f"invalid tensor metadata for {name}: {path}")
            offsets = meta["data_offsets"]
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in offsets)):
                raise LatticeError(f"invalid tensor offsets for {name}: {path}")
            start, end = offsets
            if not 0 <= start <= end <= size - 8 - header_len:
                raise LatticeError(f"tensor offsets out of bounds for {name}: {path}")
        return raw + header


def _sample_payload(path: Path, header_bytes: int, sample_size: int = 65536) -> str:
    size = path.stat().st_size
    data_start = min(size, header_bytes)
    positions = {data_start, max(data_start, (size + data_start) // 2 - sample_size // 2), max(data_start, size - sample_size)}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for position in sorted(positions):
            stream.seek(position)
            chunk = stream.read(sample_size)
            digest.update(position.to_bytes(8, "little"))
            digest.update(len(chunk).to_bytes(8, "little"))
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_weight_directory(directory: Path) -> str:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise LatticeError(f"weight directory not found: {directory}")
    entries: list[dict[str, Any]] = []
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise LatticeError(f"no .safetensors shards found in {directory}")
    for path in shards:
        header = _safetensors_header(path)
        entries.append({
            "name": path.name,
            "size": path.stat().st_size,
            "header_sha256": sha256_bytes(header),
            "sampled_payload_sha256": _sample_payload(path, len(header)),
        })
    return sha256_bytes(canonical_json({"schema": 1, "files": entries}))


def _path_list(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(part.strip()).expanduser().resolve() for part in re.split(r"[;,]", value) if part.strip()]


def fingerprint_storage_topology(model: Path, env: dict[str, str]) -> dict[str, Any]:
    primary = model.expanduser().resolve()
    seen = {primary}

    def entries(key: str) -> list[dict[str, str]]:
        result = []
        for directory in _path_list(env.get(key)):
            if directory in seen:
                raise LatticeError(f"duplicate model directory in {key}: {directory}")
            seen.add(directory)
            result.append({"path": str(directory), "fingerprint": _fingerprint_weight_directory(directory)})
        return result

    topology: dict[str, Any] = {
        "schema_version": 1,
        "primary": {"path": str(primary), "fingerprint": fingerprint_model(primary)},
        "split_directories": entries("COLI_MODEL_DIRS"),
        "mirror_directories": entries("COLI_MODEL_MIRROR"),
        "disk_weights": env.get("COLI_DISK_WEIGHTS"),
    }
    topology["fingerprint"] = sha256_bytes(canonical_json(topology))
    return topology


def fingerprint_model(model: Path) -> str:
    model = model.expanduser().resolve()
    if not model.is_dir():
        raise LatticeError(f"model directory not found: {model}")
    entries: list[dict[str, Any]] = []
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json", "tiktoken.model"):
        path = model / name
        if path.is_file():
            entries.append({"name": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise LatticeError(f"no .safetensors shards found in {model}")
    for path in shards:
        header = _safetensors_header(path)
        entries.append({
            "name": path.name,
            "size": path.stat().st_size,
            "header_sha256": sha256_bytes(header),
            "sampled_payload_sha256": _sample_payload(path, len(header)),
        })
    return sha256_bytes(canonical_json({"schema": 1, "files": entries}))


def fingerprint_runtime(c_dir: Path, coli: Path, engine: Path) -> str:
    paths = [coli, engine]
    for name in ("resource_plan.py", "doctor.py", "autotune.py", "version.py"):
        path = c_dir / name
        if path.is_file():
            paths.append(path)
    entries = []
    for path in sorted({p.resolve() for p in paths}, key=lambda p: str(p)):
        entries.append({"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json({"schema": 1, "files": entries}))


def _qualification_key(key: str) -> bool:
    return key.startswith("COLI_") or key.startswith("OMP_") or key.startswith("GOMP_") or key in QUALIFICATION_SCALAR_KEYS


def clean_environment(
    source: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if source is None else source)
    for key in FORBIDDEN_AMBIENT_KEYS | SAFE_TUNABLE_KEYS | SERVING_ONLY_KEYS:
        env.pop(key, None)
    env.update({
        "COLI_POLICY": "quality",
        "COLI_COLOR": "0",
        "KVSAVE": "0",
        "AUTOPIN": "0",
        "REPIN": "0",
        "SEED": "104729",
    })
    if overrides:
        for key, value in overrides.items():
            text = str(value)
            if not _qualification_key(key) or key in SERVING_ONLY_KEYS:
                raise LatticeError(f"invalid qualification environment override: {key}")
            if not text or any(char in text for char in "\r\n\x00"):
                raise LatticeError(f"invalid qualification environment value: {key}")
            env[key] = text
    return env


def qualification_environment(env: dict[str, str]) -> dict[str, str]:
    return {key: str(value) for key, value in sorted(env.items()) if _qualification_key(key)}


@contextlib.contextmanager
def _temporary_environment(environment: dict[str, str]):
    original = dict(os.environ)
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def fingerprint_hardware(plan: dict[str, Any], storage_topology: dict[str, Any]) -> str:
    devices = plan.get("tiers", {}).get("vram", {}).get("devices") or []
    storage = []
    topology_entries = [storage_topology.get("primary", {})]
    topology_entries.extend(storage_topology.get("split_directories") or [])
    topology_entries.extend(storage_topology.get("mirror_directories") or [])
    for entry in topology_entries:
        path_value = entry.get("path")
        if not path_value:
            continue
        path = Path(path_value)
        stat = path.stat()
        usage = shutil.disk_usage(path)
        storage.append({
            "path": str(path.resolve()),
            "device": int(getattr(stat, "st_dev", 0)),
            "volume_total_bytes": int(usage.total),
        })
    payload = {
        "schema": 1,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": plan.get("cpu"),
        "gpus": [
            {
                "index": device.get("index"),
                "name": device.get("name"),
                "total_bytes": device.get("total_bytes"),
            }
            for device in devices
        ],
        "storage": storage,
    }
    return sha256_bytes(canonical_json(payload))


def create_context(
    repo_root: Path,
    model: Path,
    *,
    engine: Path | None = None,
    deep: bool = True,
    context_length: int = 4096,
    qualification_overrides: dict[str, str] | None = None,
) -> ColibriContext:
    repo_root = repo_root.expanduser().resolve()
    c_dir = repo_root / "c"
    coli = c_dir / "coli"
    if not coli.is_file():
        raise LatticeError(f"not a Colibri checkout (missing c/coli): {repo_root}")
    model = model.expanduser().resolve()
    if isinstance(context_length, bool) or not isinstance(context_length, int) or not 128 <= context_length <= 262144:
        raise LatticeError("qualification context must be between 128 and 262144 tokens")
    resolved_engine, family = resolve_engine(c_dir, model, engine)
    if family != "colibri":
        raise LatticeError(
            "Lattice v0.1 qualification currently supports the GLM/colibri "
            "deterministic replay contract only; Inkling, Kimi and OLMoE "
            "require engine-specific calibration/replay adapters"
        )
    controlled = clean_environment(dict(os.environ), qualification_overrides)
    controlled.update({
        "SNAP": str(model),
        "COLI_MODEL": str(model),
        "COLI_POLICY": "quality",
        "CTX": str(context_length),
    })
    sys.path.insert(0, str(c_dir))
    with _temporary_environment(controlled):
        resource_plan = _load_module(c_dir / "resource_plan.py", "lattice_colibri_resource_plan")
        doctor_module = _load_module(c_dir / "doctor.py", "lattice_colibri_doctor")
        plan = resource_plan.build_plan(str(model), context=context_length, policy="quality")
        report = doctor_module.run_doctor(
            str(model), 0, context_length, None, 0,
            engine_path=str(resolved_engine),
            deep=deep,
            mirror_dir=controlled.get("COLI_MODEL_MIRROR"),
        )
        controlled = resource_plan.environment_for_plan(plan, env=controlled, cuda_enabled=True)
    controlled.update({
        "SNAP": str(model),
        "COLI_MODEL": str(model),
        "COLI_POLICY": "quality",
        "CTX": str(context_length),
    })
    status = str(report.get("status") or "").lower()
    if status not in {"ok", "pass", "warning", "warn"}:
        failures = [
            check.get("summary", check.get("id", "unknown failure"))
            for check in report.get("checks", [])
            if check.get("status") == "fail"
        ]
        detail = "; ".join(map(str, failures[:5])) or f"doctor status={status or 'unknown'}"
        raise LatticeError(f"Colibri deep doctor failed: {detail}")
    topology = fingerprint_storage_topology(model, controlled)
    runtime_fingerprint = fingerprint_runtime(c_dir, coli, resolved_engine)
    hardware_fingerprint = fingerprint_hardware(plan, topology)
    controlled_snapshot = qualification_environment(controlled)
    execution_fingerprint = sha256_bytes(canonical_json({
        "schema": 1,
        "context": context_length,
        "environment": controlled_snapshot,
        "hardware_fingerprint": hardware_fingerprint,
        "model_fingerprint": topology["fingerprint"],
        "runtime_fingerprint": runtime_fingerprint,
    }))
    return ColibriContext(
        repo_root=repo_root,
        c_dir=c_dir,
        coli=coli.resolve(),
        engine=resolved_engine,
        model=model,
        model_family=family,
        runtime_fingerprint=runtime_fingerprint,
        model_fingerprint=topology["fingerprint"],
        plan=plan,
        doctor=report,
        base_environment={str(k): str(v) for k, v in controlled.items()},
        qualification_context=context_length,
        qualification_environment=controlled_snapshot,
        execution_fingerprint=execution_fingerprint,
        hardware_fingerprint=hardware_fingerprint,
        storage_topology=topology,
    )


def parse_calibration(output: str) -> dict[str, list[int]]:
    prompt_match = PROMPT_RE.search(output)
    token_match = TOKENS_RE.search(output)
    if not prompt_match or not token_match:
        raise LatticeError("engine did not emit token trace; rebuild Colibri with current instrumentation")
    prompt_ids = [int(value) for value in prompt_match.group(1).split()]
    continuation = [int(value) for value in token_match.group(1).split()]
    if len(prompt_ids) < 2 or not continuation:
        raise LatticeError("calibration produced an empty token trace")
    return {"prompt_ids": prompt_ids, "full_ids": prompt_ids + continuation}


def parse_replay_metrics(output: str) -> dict[str, float | None]:
    speed = SPEED_RE.search(output)
    if not speed:
        raise LatticeError("engine did not emit REPLAY throughput")
    hit = HIT_RE.search(output)
    latency = LATENCY_RE.search(output)
    return {
        "tok_s": float(speed.group(1)),
        "hit_pct": float(hit.group(1)) if hit else None,
        "p50_ms": float(latency.group(1)) if latency else None,
        "p99_ms": float(latency.group(2)) if latency else None,
    }


def calibrate_case(
    context: ColibriContext,
    *,
    prompt: str,
    tokens: int,
    ctx: int,
    timeout: int,
) -> tuple[dict[str, list[int]], ProcessResult]:
    env = dict(context.base_environment)
    env.update({
        "TOKENS": "1", "PROF": "1", "NGEN": str(tokens), "CTX": str(ctx),
        "COLI_ENGINE": str(context.engine),
    })
    command = [
        sys.executable,
        str(context.coli),
        "run",
        "--model", str(context.model),
        "--ctx", str(ctx),
        "--ngen", str(tokens),
        "--temp", "0",
        prompt,
    ]
    result = run_bounded(command, env=env, timeout=timeout, cwd=context.repo_root)
    output = f"{result.stdout}\n{result.stderr}"
    if result.timed_out:
        raise LatticeError("calibration timed out")
    if result.returncode:
        raise LatticeError(f"calibration failed with exit code {result.returncode}: {output[-2000:]}")
    if result.output_truncated:
        raise LatticeError("calibration output exceeded the evidence limit")
    return parse_calibration(output), result


def run_replay(
    context: ColibriContext,
    *,
    replay_path: Path,
    candidate: dict[str, str],
    ctx: int,
    timeout: int,
) -> tuple[dict[str, float | None], ProcessResult]:
    unknown = set(candidate) - SAFE_TUNABLE_KEYS
    if unknown:
        raise LatticeError(f"candidate contains unsupported or quality-affecting keys: {', '.join(sorted(unknown))}")
    env = dict(context.base_environment)
    env.update(candidate)
    env.update({
        "REF": str(replay_path),
        "REF_FORCE": "1",
        "REPLAY": "1",
        "PROF": "1",
        "CTX": str(ctx),
    })
    env.pop("PROMPT", None); env.pop("TOKENS", None)
    cap = context.plan.get("tiers", {}).get("ram", {}).get("cache_slots_per_layer", 0)
    command = [str(context.engine), str(int(cap or 0))]
    result = run_bounded(command, env=env, timeout=timeout, cwd=context.c_dir)
    output = f"{result.stdout}\n{result.stderr}"
    if result.timed_out:
        raise LatticeError("replay timed out")
    if result.returncode:
        raise LatticeError(f"replay failed with exit code {result.returncode}: {output[-2000:]}")
    if result.output_truncated:
        raise LatticeError("replay output exceeded the evidence limit")
    return parse_replay_metrics(output), result


def hardware_summary(context: ColibriContext) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": context.plan.get("cpu"),
        "vram": context.plan.get("tiers", {}).get("vram"),
        "ram": context.plan.get("tiers", {}).get("ram"),
        "disk": context.plan.get("tiers", {}).get("disk"),
        "expected_bottleneck": context.plan.get("expected_bottleneck"),
    }
