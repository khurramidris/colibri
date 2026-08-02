from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from .common import (
    LatticeError, canonical_json, load_json, sha256_bytes, sha256_file,
    strict_json_loads,
)
from .process import ProcessResult, run_bounded
from .oracle import ORACLE_POLICY, ORACLE_SCHEMA, validate_oracle

PROMPT_RE = re.compile(r"\[PROMPT_TOKENS\]\s+(\d+):\s*([0-9 ]+)")
TOKENS_RE = re.compile(r"\[TOKENS\]\s+(\d+)\s+generated:\s*([0-9 ]+)")
SPEED_RE = re.compile(
    r"REPLAY decode:\s+(\d+)\s+tokens\s+in\s+([0-9.]+)s\s*\|\s*([0-9.]+)\s+tok/s"
)
HIT_RE = re.compile(r"expert hit\s+([0-9.]+)%")
LATENCY_RE = re.compile(r"latency p50\s+([0-9.]+)\s*ms.*?p99\s+([0-9.]+)\s*ms")
ORACLE_WRITTEN_RE = re.compile(
    r"^REPLAY_ORACLE_WRITTEN v2 steps=(\d+) topk=(\d+) "
    r"measurement=separate_replay_pass transport=private_file$",
    re.MULTILINE,
)
MAX_ORACLE_BYTES = 4 * 1024 * 1024

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
    "DISK_SPLIT", "DRAFT", "PIN_GB", "PIPE", "PIPE_WORKERS", "DIRECT",
    "URING", "PREFETCH", "PILOT", "PILOT_REAL", "PILOT_K", "SEED",
    "KVSAVE", "AUTOPIN", "REPIN", "SNAP", "OMP_NUM_THREADS",
    "OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES", "GOMP_SPINCOUNT",
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
})

TOPOLOGY_KEYS = frozenset({"COLI_MODEL_DIRS", "COLI_MODEL_MIRROR", "COLI_DISK_WEIGHTS"})

# Explicitly reviewed Colibri variables permitted in qualification identity.
# New upstream knobs fail closed until their semantics are reviewed here.
QUALIFICATION_COLI_KEYS = frozenset({
    "COLI_POLICY", "COLI_COLOR", "COLI_MODEL", "COLI_GPU", "COLI_GPUS",
    "COLI_CUDA", "COLI_METAL", "COLI_VULKAN", "COLI_NUMA",
    "COLI_NO_OMP_TUNE", "COLI_CUDA_PIPE", "COLI_CUDA_ASYNC", "COLI_MMAP",
    "COLI_SSD_FAST_GBS", "COLI_NO_FUSED_PAIR", "COLI_RAM_OVERCOMMIT",
})

# Only hardware and storage selectors may be inherited from the operator shell.
AMBIENT_QUALIFICATION_KEYS = TOPOLOGY_KEYS | frozenset({
    "COLI_GPU", "COLI_GPUS", "COLI_CUDA", "COLI_METAL", "COLI_VULKAN",
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
})

# Keep only operating-system variables required to start local subprocesses.
# Secrets, Python import injection, dynamic-loader injection and unrelated
# numerical-library tuning variables are deliberately absent.
SYSTEM_ENV_KEYS = frozenset({
    "PATH", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT",
    "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "HOMEDRIVE",
    "HOMEPATH", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    "LANG", "LC_ALL", "LANGUAGE", "TZ",
})
ATTESTED_SYSTEM_KEYS = frozenset({"PATH", "SystemRoot", "WINDIR"})


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
    plan_fingerprint: str = ""
    replay_cap: int = 0


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LatticeError(f"cannot import Colibri support module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module
    except BaseException:
        raise
    finally:
        sys.modules.pop(name, None)


def _purge_new_support_modules(c_dir: Path, before: set[str]) -> None:
    root = c_dir.resolve()
    for name in set(sys.modules) - before:
        module = sys.modules.get(name)
        location = getattr(module, "__file__", None)
        if not location:
            continue
        try:
            Path(location).resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        sys.modules.pop(name, None)


def detect_family(config: dict[str, Any]) -> str:
    if not isinstance(config, dict):
        raise LatticeError("model config must be an object")
    model_type = str(config.get("model_type") or "").lower()
    architectures = " ".join(map(str, config.get("architectures") or [])).lower()
    text = f"{model_type} {architectures}"
    if "inkling" in text:
        return "inkling"
    if "kimi" in text:
        return "kimi_k3"
    if "olmoe" in text or "olmo_moe" in text or "olmoe" in text.replace("-", ""):
        return "olmoe"
    if "glm" in text:
        return "colibri"
    raise LatticeError(
        f"unsupported or unrecognized model family: model_type={model_type or '<missing>'!r}"
    )


def resolve_engine(c_dir: Path, model: Path, explicit: Path | None = None) -> tuple[Path, str]:
    config_path = model / "config.json"
    try:
        config = strict_json_loads(
            config_path.read_text(encoding="utf-8"), label=f"model config {config_path}"
        )
    except FileNotFoundError as error:
        raise LatticeError(f"missing model config: {config_path}") from error
    except UnicodeDecodeError as error:
        raise LatticeError(f"model config is not UTF-8: {config_path}") from error
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
            parsed = strict_json_loads(header.decode("utf-8"), label=f"safetensors header {path}")
        except UnicodeDecodeError as error:
            raise LatticeError(f"safetensors header is not UTF-8: {path}") from error
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
    c_dir = c_dir.expanduser().resolve()
    paths = {coli.expanduser().resolve(), engine.expanduser().resolve()}
    paths.update(path.resolve() for path in c_dir.glob("*.py") if path.is_file())
    entries = []
    for path in sorted(paths, key=lambda item: str(item)):
        if not path.is_file() or path.is_symlink():
            raise LatticeError(f"runtime component is not a regular file: {path}")
        try:
            name = str(path.relative_to(c_dir))
        except ValueError:
            name = str(path)
        entries.append({"name": name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json({"schema": 2, "files": entries}))


def _qualification_key(key: str) -> bool:
    if key in SERVING_ONLY_KEYS:
        return False
    if key in FORBIDDEN_AMBIENT_KEYS and key not in {"DRAFT", "PIN_GB", "AUTOPIN", "REPIN"}:
        return False
    return key in (
        SAFE_TUNABLE_KEYS
        | QUALIFICATION_SCALAR_KEYS
        | TOPOLOGY_KEYS
        | QUALIFICATION_COLI_KEYS
        | ATTESTED_SYSTEM_KEYS
    )


def _validate_qualification_value(key: str, value: str) -> None:
    if not value or any(char in value for char in "\r\n\x00"):
        raise LatticeError(f"invalid qualification environment value: {key}")
    fixed = {
        "COLI_POLICY": "quality",
        "DRAFT": "0",
        "KVSAVE": "0",
        "AUTOPIN": "0",
        "REPIN": "0",
    }
    if key in fixed and value != fixed[key]:
        raise LatticeError(
            f"qualification environment {key} must remain {fixed[key]!r}; got {value!r}"
        )
    if key == "PIN_GB" and value != "all":
        try:
            if float(value) <= 0:
                raise ValueError
        except ValueError as error:
            raise LatticeError("qualification environment PIN_GB must be 'all' or positive") from error


def clean_environment(
    source: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    original = dict(os.environ if source is None else source)
    env = {
        key: str(original[key])
        for key in SYSTEM_ENV_KEYS
        if key in original and str(original[key])
    }
    env.setdefault("PATH", os.defpath)
    inherited = {
        key: str(original[key])
        for key in AMBIENT_QUALIFICATION_KEYS
        if key in original
    }
    env.update(inherited)
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
            if key in ATTESTED_SYSTEM_KEYS:
                current = str(original.get(key) or env.get(key) or "")
                if text != current:
                    raise LatticeError(
                        f"recorded system environment changed for {key}; reinitialize"
                    )
                env[key] = text
                continue
            if not _qualification_key(key):
                raise LatticeError(f"invalid qualification environment override: {key}")
            _validate_qualification_value(key, text)
            env[key] = text
    return env


def qualification_environment(env: dict[str, str]) -> dict[str, str]:
    unreviewed = sorted(
        key
        for key in env
        if (key.startswith("COLI_") or key.startswith("OMP_") or key.startswith("GOMP_"))
        and not _qualification_key(key)
    )
    if unreviewed:
        raise LatticeError(
            "unreviewed Colibri qualification keys: " + ", ".join(unreviewed)
        )
    selected = {key: str(value) for key, value in sorted(env.items()) if _qualification_key(key)}
    for key, value in selected.items():
        _validate_qualification_value(key, value)
    return selected


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
    frozen_plan: dict[str, Any] | None = None,
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
            "Lattice numerical qualification currently supports the GLM/colibri "
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
    original_sys_path = list(sys.path)
    modules_before = set(sys.modules)
    try:
        sys.path.insert(0, str(c_dir))
        with _temporary_environment(controlled):
            resource_plan = _load_module(c_dir / "resource_plan.py", "lattice_colibri_resource_plan")
            doctor_module = _load_module(c_dir / "doctor.py", "lattice_colibri_doctor")
            computed_plan = resource_plan.build_plan(str(model), context=context_length, policy="quality")
            if frozen_plan is not None and not isinstance(frozen_plan, dict):
                raise LatticeError("frozen execution plan must be an object")
            plan = frozen_plan if frozen_plan is not None else computed_plan
            report = doctor_module.run_doctor(
                str(model), 0, context_length, None, 0,
                engine_path=str(resolved_engine),
                deep=deep,
                mirror_dir=controlled.get("COLI_MODEL_MIRROR"),
            )
            controlled = resource_plan.environment_for_plan(plan, env=controlled, cuda_enabled=True)
    finally:
        sys.path[:] = original_sys_path
        _purge_new_support_modules(c_dir, modules_before)
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
    plan_fingerprint = sha256_bytes(canonical_json(plan))
    replay_cap = int(plan.get("tiers", {}).get("ram", {}).get("cache_slots_per_layer", 0) or 0)
    if replay_cap < 0:
        raise LatticeError("execution plan has a negative replay cache cap")
    execution_fingerprint = sha256_bytes(canonical_json({
        "schema": 2,
        "context": context_length,
        "plan_fingerprint": plan_fingerprint,
        "native_replay_arguments": [str(replay_cap)],
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
        plan_fingerprint=plan_fingerprint,
        replay_cap=replay_cap,
    )


def _trace_ids(match: re.Match[str], label: str) -> list[int]:
    declared = int(match.group(1))
    values = [int(value) for value in match.group(2).split()]
    if declared != len(values):
        raise LatticeError(
            f"{label} declared {declared} token IDs but emitted {len(values)}"
        )
    if any(value < 0 for value in values):
        raise LatticeError(f"{label} contains a negative token ID")
    return values


def parse_calibration(output: str) -> dict[str, list[int]]:
    prompt_matches = list(PROMPT_RE.finditer(output))
    token_matches = list(TOKENS_RE.finditer(output))
    if len(prompt_matches) != 1 or len(token_matches) != 1:
        raise LatticeError("engine must emit exactly one prompt and continuation token trace")
    prompt_ids = _trace_ids(prompt_matches[0], "prompt trace")
    continuation = _trace_ids(token_matches[0], "continuation trace")
    if len(prompt_ids) < 2 or not continuation:
        raise LatticeError("calibration produced an empty token trace")
    return {"prompt_ids": prompt_ids, "full_ids": prompt_ids + continuation}


def read_replay_oracle_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LatticeError("engine did not create a readable regular replay oracle artifact") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise LatticeError("engine did not create a regular replay oracle artifact")
        if before.st_size > MAX_ORACLE_BYTES:
            raise LatticeError("replay numerical oracle artifact exceeded the evidence limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(MAX_ORACLE_BYTES + 1)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
            raise LatticeError("replay numerical oracle artifact changed while being read")
    finally:
        os.close(descriptor)
    if len(data) > MAX_ORACLE_BYTES:
        raise LatticeError("replay numerical oracle artifact exceeded the evidence limit")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LatticeError("replay numerical oracle artifact is not UTF-8") from error


def parse_replay_oracle(
    artifact: str, *, expected_forced: list[int] | None = None
) -> dict[str, Any]:
    steps: list[list[Any]] = []
    summary: list[str] | None = None
    for line_number, line in enumerate(artifact.splitlines(), start=1):
        parts = line.split("\t")
        if not parts or not parts[0]:
            continue
        if parts[0] == "STEP":
            if len(parts) != 16 or parts[1] != "v2":
                raise LatticeError(f"malformed replay oracle step at line {line_number}")
            try:
                identities = [int(value) for value in parts[2:6]]
                numerics = [float(value) for value in parts[6:15]]
                topk_ids = [int(value) for value in parts[15].split(",") if value]
            except ValueError as error:
                raise LatticeError(f"invalid replay oracle number at line {line_number}") from error
            steps.append([*identities, *numerics, topk_ids])
        elif parts[0] == "SUMMARY":
            if summary is not None or len(parts) != 6 or parts[1] != "v2":
                raise LatticeError("invalid replay numerical oracle summary")
            summary = parts
        else:
            raise LatticeError(f"unknown replay oracle record at line {line_number}")
    if not steps or summary is None:
        raise LatticeError("engine did not emit a complete replay numerical oracle artifact")
    try:
        summary_steps, topk = int(summary[2]), int(summary[3])
    except ValueError as error:
        raise LatticeError("invalid replay numerical oracle summary number") from error
    if (summary_steps != len(steps) or topk != ORACLE_POLICY["topk"]
            or summary[4] != ORACLE_POLICY["measurement"]
            or summary[5] != ORACLE_POLICY["transport"]):
        raise LatticeError("replay numerical oracle summary does not match policy")
    return validate_oracle({
        "schema": ORACLE_SCHEMA,
        "policy": dict(ORACLE_POLICY),
        "steps": steps,
    }, expected_forced=expected_forced)


def parse_replay_metrics(
    output: str, oracle_artifact: str, *, expected_forced: list[int] | None = None
) -> dict[str, Any]:
    speeds = list(SPEED_RE.finditer(output))
    markers = list(ORACLE_WRITTEN_RE.finditer(output))
    if len(speeds) != 1:
        raise LatticeError("engine must emit exactly one REPLAY throughput record")
    if len(markers) != 1:
        raise LatticeError("engine must emit exactly one replay oracle publication marker")
    oracle = parse_replay_oracle(oracle_artifact, expected_forced=expected_forced)
    speed, marker = speeds[0], markers[0]
    steps = int(speed.group(1))
    seconds = float(speed.group(2))
    tok_s = float(speed.group(3))
    if steps != len(oracle["steps"]):
        raise LatticeError("REPLAY throughput step count does not match numerical oracle")
    if (not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(tok_s) or tok_s <= 0
            or not math.isclose(tok_s, steps / seconds, rel_tol=0.02, abs_tol=0.02)):
        raise LatticeError("REPLAY throughput telemetry is invalid or internally inconsistent")
    if int(marker.group(1)) != len(oracle["steps"]) or int(marker.group(2)) != ORACLE_POLICY["topk"]:
        raise LatticeError("replay oracle publication marker does not match artifact")
    hits = list(HIT_RE.finditer(output))
    if len(hits) != 1:
        raise LatticeError("engine must emit exactly one expert-hit telemetry record")
    hit_pct = float(hits[0].group(1))
    if not math.isfinite(hit_pct) or not 0 <= hit_pct <= 100:
        raise LatticeError("expert-hit telemetry is outside 0..100")
    latencies = list(LATENCY_RE.finditer(output))
    if len(latencies) > 1:
        raise LatticeError("engine emitted duplicate latency telemetry")
    p50_ms = p99_ms = None
    if latencies:
        p50_ms = float(latencies[0].group(1))
        p99_ms = float(latencies[0].group(2))
        if (not math.isfinite(p50_ms) or not math.isfinite(p99_ms)
                or p50_ms < 0 or p99_ms < p50_ms):
            raise LatticeError("latency telemetry is invalid")
    return {
        "tok_s": tok_s,
        "decode_seconds": seconds,
        "decode_tokens": steps,
        "hit_pct": hit_pct,
        "p50_ms": p50_ms,
        "p99_ms": p99_ms,
        "oracle": oracle,
    }


def calibrate_case(
    context: ColibriContext,
    *,
    prompt: str,
    tokens: int,
    ctx: int,
    timeout: int,
) -> tuple[dict[str, list[int]], ProcessResult]:
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32768:
        raise LatticeError("calibration prompt must be non-empty and at most 32768 characters")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1 <= tokens <= 2048:
        raise LatticeError("calibration tokens must be between 1 and 2048")
    if isinstance(ctx, bool) or not isinstance(ctx, int) or not 128 <= ctx <= context.qualification_context:
        raise LatticeError("calibration context is outside the recorded qualification context")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise LatticeError("calibration timeout must be a positive integer")
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
) -> tuple[dict[str, Any], ProcessResult]:
    if not isinstance(candidate, dict):
        raise LatticeError("candidate environment must be an object")
    unknown = set(candidate) - SAFE_TUNABLE_KEYS
    if unknown:
        raise LatticeError(f"candidate contains unsupported or quality-affecting keys: {', '.join(sorted(unknown))}")
    for key, value in candidate.items():
        if not isinstance(value, str):
            raise LatticeError(f"candidate environment value for {key} must be a string")
        _validate_qualification_value(key, value)
    if isinstance(ctx, bool) or not isinstance(ctx, int) or not 128 <= ctx <= context.qualification_context:
        raise LatticeError("replay context is outside the recorded qualification context")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise LatticeError("replay timeout must be a positive integer")
    replay = load_json(replay_path, max_bytes=16 * 1024 * 1024)
    if not isinstance(replay, dict):
        raise LatticeError("replay payload must be an object")
    prompt_ids, full_ids = replay.get("prompt_ids"), replay.get("full_ids")
    if (not isinstance(prompt_ids, list) or not isinstance(full_ids, list)
            or not prompt_ids or len(full_ids) <= len(prompt_ids)
            or full_ids[:len(prompt_ids)] != prompt_ids
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in full_ids)):
        raise LatticeError("replay payload contains an invalid token path")
    if len(full_ids) > ctx:
        raise LatticeError("replay token path exceeds the declared context")
    expected_forced = full_ids[len(prompt_ids):]
    env = dict(context.base_environment)
    env.update(candidate)
    env.update({
        "REF": str(replay_path.resolve()),
        "REF_FORCE": "1",
        "REPLAY": "1",
        "REPLAY_ORACLE": "1",
        "REPLAY_ORACLE_TOPK": str(ORACLE_POLICY["topk"]),
        "PROF": "1",
        "CTX": str(ctx),
    })
    env.pop("PROMPT", None)
    env.pop("TOKENS", None)
    command = [str(context.engine), str(context.replay_cap)]
    with tempfile.TemporaryDirectory(prefix="lattice-oracle-") as directory:
        artifact_path = Path(directory) / "oracle.tsv"
        env["REPLAY_ORACLE_OUT"] = str(artifact_path)
        result = run_bounded(command, env=env, timeout=timeout, cwd=context.c_dir)
        output = f"{result.stdout}\n{result.stderr}"
        if result.timed_out:
            raise LatticeError("replay timed out")
        if result.returncode:
            raise LatticeError(f"replay failed with exit code {result.returncode}: {output[-2000:]}")
        if result.output_truncated:
            raise LatticeError("replay output exceeded the evidence limit")
        oracle_artifact = read_replay_oracle_file(artifact_path)
    return parse_replay_metrics(
        output, oracle_artifact, expected_forced=expected_forced
    ), result


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
