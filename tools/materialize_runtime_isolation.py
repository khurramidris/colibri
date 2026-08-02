#!/usr/bin/env python3
"""Materialize the reviewed Colibri runtime-isolation hardening."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8", newline="\n")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one replacement, found {count}")
    return text.replace(old, new)


def replace_between(text: str, start: str, end: str, replacement: str, label: str) -> str:
    first = text.find(start)
    if first < 0:
        raise SystemExit(f"{label}: start marker not found")
    second = text.find(end, first + len(start))
    if second < 0:
        raise SystemExit(f"{label}: end marker not found")
    return text[:first] + replacement + text[second:]


path = "lattice/colibri.py"
text = read(path)
text = replace_once(text, "import shutil\nimport sys\n", "import shutil\nimport stat\nimport sys\n", "stat import")
text = replace_once(
    text,
    "from .common import LatticeError, canonical_json, sha256_bytes, sha256_file\n",
    "from .common import (\n"
    "    LatticeError, canonical_json, load_json, sha256_bytes, sha256_file,\n"
    "    strict_json_loads,\n"
    ")\n",
    "common imports",
)
text = replace_once(
    text,
    'PROMPT_RE = re.compile(r"\\[PROMPT_TOKENS\\]\\s+\\d+:\\s*([0-9 ]+)")\n'
    'TOKENS_RE = re.compile(r"\\[TOKENS\\]\\s+\\d+\\s+generated:\\s*([0-9 ]+)")\n'
    'SPEED_RE = re.compile(r"REPLAY decode:\\s+\\d+\\s+tokens.*?\\|\\s*([0-9.]+)\\s+tok/s")\n',
    'PROMPT_RE = re.compile(r"\\[PROMPT_TOKENS\\]\\s+(\\d+):\\s*([0-9 ]+)")\n'
    'TOKENS_RE = re.compile(r"\\[TOKENS\\]\\s+(\\d+)\\s+generated:\\s*([0-9 ]+)")\n'
    'SPEED_RE = re.compile(\n'
    '    r"REPLAY decode:\\s+(\\d+)\\s+tokens\\s+in\\s+([0-9.]+)s\\s*\\|\\s*([0-9.]+)\\s+tok/s"\n'
    ')\n',
    "telemetry regexes",
)
text = replace_once(
    text,
    'AMBIENT_QUALIFICATION_KEYS = TOPOLOGY_KEYS | frozenset({\n'
    '    "COLI_GPU", "COLI_GPUS", "COLI_CUDA", "COLI_METAL", "COLI_VULKAN",\n'
    '    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",\n'
    '})\n',
    'AMBIENT_QUALIFICATION_KEYS = TOPOLOGY_KEYS | frozenset({\n'
    '    "COLI_GPU", "COLI_GPUS", "COLI_CUDA", "COLI_METAL", "COLI_VULKAN",\n'
    '    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",\n'
    '})\n\n'
    '# Keep only operating-system variables required to start local subprocesses.\n'
    '# Secrets, Python import injection, dynamic-loader injection and unrelated\n'
    '# numerical-library tuning variables are deliberately absent.\n'
    'SYSTEM_ENV_KEYS = frozenset({\n'
    '    "PATH", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT",\n'
    '    "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "HOMEDRIVE",\n'
    '    "HOMEPATH", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",\n'
    '    "LANG", "LC_ALL", "LANGUAGE", "TZ",\n'
    '})\n'
    'ATTESTED_SYSTEM_KEYS = frozenset({"PATH", "SystemRoot", "WINDIR"})\n',
    "system environment constants",
)
text = replace_between(
    text,
    "def _load_module(path: Path, name: str) -> ModuleType:\n",
    "def detect_family(config: dict[str, Any]) -> str:\n",
    '''def _load_module(path: Path, name: str) -> ModuleType:
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


''',
    "module loader",
)
text = replace_between(
    text,
    "def detect_family(config: dict[str, Any]) -> str:\n",
    "def resolve_engine(c_dir: Path, model: Path, explicit: Path | None = None) -> tuple[Path, str]:\n",
    '''def detect_family(config: dict[str, Any]) -> str:
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


''',
    "family detection",
)
text = replace_once(
    text,
    '        config = json.loads(config_path.read_text(encoding="utf-8"))\n'
    '    except FileNotFoundError as error:\n'
    '        raise LatticeError(f"missing model config: {config_path}") from error\n'
    '    except json.JSONDecodeError as error:\n'
    '        raise LatticeError(f"invalid model config: {error}") from error\n',
    '        config = strict_json_loads(\n'
    '            config_path.read_text(encoding="utf-8"), label=f"model config {config_path}"\n'
    '        )\n'
    '    except FileNotFoundError as error:\n'
    '        raise LatticeError(f"missing model config: {config_path}") from error\n'
    '    except UnicodeDecodeError as error:\n'
    '        raise LatticeError(f"model config is not UTF-8: {config_path}") from error\n',
    "strict model config",
)
text = replace_once(
    text,
    '            parsed = json.loads(header)\n'
    '        except json.JSONDecodeError as error:\n'
    '            raise LatticeError(f"invalid safetensors JSON header: {path}: {error}") from error\n',
    '            parsed = strict_json_loads(header.decode("utf-8"), label=f"safetensors header {path}")\n'
    '        except UnicodeDecodeError as error:\n'
    '            raise LatticeError(f"safetensors header is not UTF-8: {path}") from error\n',
    "strict safetensors header",
)
text = replace_between(
    text,
    "def fingerprint_runtime(c_dir: Path, coli: Path, engine: Path) -> str:\n",
    "def _qualification_key(key: str) -> bool:\n",
    '''def fingerprint_runtime(c_dir: Path, coli: Path, engine: Path) -> str:
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


''',
    "runtime fingerprint",
)
text = replace_once(
    text,
    '        SAFE_TUNABLE_KEYS\n        | QUALIFICATION_SCALAR_KEYS\n        | TOPOLOGY_KEYS\n        | QUALIFICATION_COLI_KEYS\n',
    '        SAFE_TUNABLE_KEYS\n        | QUALIFICATION_SCALAR_KEYS\n        | TOPOLOGY_KEYS\n        | QUALIFICATION_COLI_KEYS\n        | ATTESTED_SYSTEM_KEYS\n',
    "attested system keys",
)
text = replace_between(
    text,
    "def clean_environment(\n",
    "def qualification_environment(env: dict[str, str]) -> dict[str, str]:\n",
    '''def clean_environment(
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
            if key in ATTESTED_SYSTEM_KEYS or not _qualification_key(key):
                raise LatticeError(f"invalid qualification environment override: {key}")
            _validate_qualification_value(key, text)
            env[key] = text
    return env


''',
    "environment sanitation",
)
old_context = '''    sys.path.insert(0, str(c_dir))
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
'''
new_context = '''    original_sys_path = list(sys.path)
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
'''
text = replace_once(text, old_context, new_context, "support module isolation")
text = replace_between(
    text,
    "def parse_calibration(output: str) -> dict[str, list[int]]:\n",
    "def read_replay_oracle_file(path: Path) -> str:\n",
    '''def _trace_ids(match: re.Match[str], label: str) -> list[int]:
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


''',
    "calibration parser",
)
text = replace_between(
    text,
    "def read_replay_oracle_file(path: Path) -> str:\n",
    "def parse_replay_oracle(artifact: str) -> dict[str, Any]:\n",
    '''def read_replay_oracle_file(path: Path) -> str:
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


''',
    "oracle descriptor read",
)
text = replace_once(
    text,
    "def parse_replay_oracle(artifact: str) -> dict[str, Any]:\n",
    "def parse_replay_oracle(\n    artifact: str, *, expected_forced: list[int] | None = None\n) -> dict[str, Any]:\n",
    "oracle expected tokens signature",
)
text = replace_once(
    text,
    '        "steps": steps,\n    })\n\n\ndef parse_replay_metrics(output: str, oracle_artifact: str) -> dict[str, Any]:\n',
    '        "steps": steps,\n    }, expected_forced=expected_forced)\n\n\ndef parse_replay_metrics(\n'
    '    output: str, oracle_artifact: str, *, expected_forced: list[int] | None = None\n'
    ') -> dict[str, Any]:\n',
    "oracle expected tokens call",
)
text = replace_between(
    text,
    "def parse_replay_metrics(\n",
    "def calibrate_case(\n",
    '''def parse_replay_metrics(
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


''',
    "replay metric validation",
)
text = replace_between(
    text,
    "def calibrate_case(\n",
    "def run_replay(\n",
    '''def calibrate_case(
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


''',
    "calibration execution",
)
text = replace_between(
    text,
    "def run_replay(\n",
    "def hardware_summary(context: ColibriContext) -> dict[str, Any]:\n",
    '''def run_replay(
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


''',
    "replay execution",
)
write(path, text)

# Upgrade tests to the strict runtime and telemetry contract.
test_path = "lattice/tests/test_colibri.py"
test = read(test_path)
test = replace_once(
    test,
    '    fingerprint_model,\n    fingerprint_storage_topology,\n',
    '    fingerprint_model,\n    fingerprint_runtime,\n    fingerprint_storage_topology,\n',
    "test runtime import",
)
test = replace_once(
    test,
    '        self.assertEqual(detect_family({"model_type": "glm_moe"}), "colibri")\n',
    '        self.assertEqual(detect_family({"model_type": "glm_moe"}), "colibri")\n'
    '        with self.assertRaisesRegex(LatticeError, "unsupported or unrecognized"):\n'
    '            detect_family({"model_type": "mystery_model"})\n',
    "unknown family test",
)
test = replace_once(
    test,
    '        env = clean_environment({\n'
    '            "COLI_TEMP": "0.8",\n'
    '            "IDOT": "0",\n'
    '            "COLI_METAL": "1",\n'
    '            "COLI_UNREVIEWED_APPROX": "1",\n'
    '        })\n',
    '        env = clean_environment({\n'
    '            "PATH": "/safe/bin",\n'
    '            "COLI_TEMP": "0.8",\n'
    '            "IDOT": "0",\n'
    '            "COLI_METAL": "1",\n'
    '            "COLI_UNREVIEWED_APPROX": "1",\n'
    '            "LD_PRELOAD": "/tmp/inject.so",\n'
    '            "PYTHONPATH": "/tmp/import-inject",\n'
    '            "OPENAI_API_KEY": "secret",\n'
    '            "KMP_AFFINITY": "scatter",\n'
    '        })\n',
    "sanitized environment fixture",
)
test = replace_once(
    test,
    '        self.assertNotIn("COLI_UNREVIEWED_APPROX", env)\n'
    '        self.assertEqual(qualification_environment(env)["COLI_METAL"], "1")\n',
    '        self.assertNotIn("COLI_UNREVIEWED_APPROX", env)\n'
    '        self.assertNotIn("LD_PRELOAD", env)\n'
    '        self.assertNotIn("PYTHONPATH", env)\n'
    '        self.assertNotIn("OPENAI_API_KEY", env)\n'
    '        self.assertNotIn("KMP_AFFINITY", env)\n'
    '        snapshot = qualification_environment(env)\n'
    '        self.assertEqual(snapshot["COLI_METAL"], "1")\n'
    '        self.assertEqual(snapshot["PATH"], "/safe/bin")\n',
    "sanitized environment assertions",
)
test = replace_once(
    test,
    '        replay = parse_calibration("[PROMPT_TOKENS] 2: 1 2\\n[TOKENS] 3 generated: 3 4 5")\n'
    '        self.assertEqual(replay["full_ids"], [1, 2, 3, 4, 5])\n',
    '        replay = parse_calibration("[PROMPT_TOKENS] 2: 1 2\\n[TOKENS] 3 generated: 3 4 5")\n'
    '        self.assertEqual(replay["full_ids"], [1, 2, 3, 4, 5])\n'
    '        with self.assertRaisesRegex(LatticeError, "declared 4 token IDs"):\n'
    '            parse_calibration("[PROMPT_TOKENS] 2: 1 2\\n[TOKENS] 4 generated: 3 4 5")\n',
    "declared count test",
)
test = replace_once(
    test,
    '            "REPLAY decode: 16 tokens | 2.50 tok/s\\nexpert hit 70.5%\\nlatency p50 10.2 ms p99 18.4 ms"\n',
    '            "REPLAY decode: 1 tokens in 0.400s | 2.50 tok/s\\nexpert hit 70.5%\\nlatency p50 10.2 ms p99 18.4 ms"\n',
    "consistent speed fixture",
)
test = replace_once(
    test,
    '        metrics = parse_replay_metrics(output, artifact)\n',
    '        metrics = parse_replay_metrics(output, artifact, expected_forced=[3])\n',
    "bound metric parser test",
)
test = replace_once(
    test,
    '        self.assertEqual(metrics["oracle"]["steps"][0][1], 2)\n',
    '        self.assertEqual(metrics["oracle"]["steps"][0][1], 2)\n'
    '        self.assertEqual(metrics["decode_tokens"], 1)\n'
    '        with self.assertRaisesRegex(LatticeError, "internally inconsistent"):\n'
    '            parse_replay_metrics(output.replace("2.50 tok/s", "9.00 tok/s"), artifact)\n',
    "metric validation assertions",
)
insert_marker = '    def test_long_oracle_uses_separate_bounded_artifact(self):\n'
runtime_test = '''    def test_runtime_fingerprint_covers_all_support_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coli = root / "coli"
            engine = root / "colibri"
            support = root / "future_support.py"
            coli.write_text("launcher", encoding="utf-8")
            engine.write_text("engine", encoding="utf-8")
            support.write_text("VALUE = 1\n", encoding="utf-8")
            before = fingerprint_runtime(root, coli, engine)
            support.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(before, fingerprint_runtime(root, coli, engine))

'''
test = replace_once(test, insert_marker, runtime_test + insert_marker, "runtime fingerprint test")
write(test_path, test)

print("runtime isolation materialized")
