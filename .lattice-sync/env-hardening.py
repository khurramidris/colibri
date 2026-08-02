from pathlib import Path

colibri = Path("lattice/colibri.py")
text = colibri.read_text(encoding="utf-8")
old_constants = '''QUALIFICATION_SCALAR_KEYS = frozenset({
    "RAM_GB", "CTX", "CUDA_EXPERT_GB", "CAP", "CAP_RAISE", "MLOCK",
    "COLI_MMAP", "COLI_SSD_FAST_GBS", "COLI_NO_FUSED_PAIR", "DISK_SPLIT",
    "COLI_RAM_OVERCOMMIT", "DRAFT", "MTP", "IDOT", "ABSORB", "I4S", "SPEC_PIN",
    "PIPE", "PIPE_WORKERS", "DIRECT", "URING", "PREFETCH", "PILOT",
    "PILOT_REAL", "PILOT_K", "SEED", "KVSAVE", "AUTOPIN", "REPIN",
    "SNAP", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES",
    "GOMP_SPINCOUNT", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
})

TOPOLOGY_KEYS = frozenset({"COLI_MODEL_DIRS", "COLI_MODEL_MIRROR", "COLI_DISK_WEIGHTS"})
'''
new_constants = '''QUALIFICATION_SCALAR_KEYS = frozenset({
    "RAM_GB", "CTX", "CUDA_EXPERT_GB", "CAP", "CAP_RAISE", "MLOCK",
    "DISK_SPLIT", "PIPE", "PIPE_WORKERS", "DIRECT", "URING", "PREFETCH",
    "PILOT", "PILOT_REAL", "PILOT_K", "SEED", "KVSAVE", "AUTOPIN",
    "REPIN", "SNAP", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "OMP_PROC_BIND",
    "OMP_PLACES", "GOMP_SPINCOUNT", "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
})

TOPOLOGY_KEYS = frozenset({"COLI_MODEL_DIRS", "COLI_MODEL_MIRROR", "COLI_DISK_WEIGHTS"})

# Every Colibri variable allowed into a qualification identity is reviewed here.
# Unknown COLI_* variables are stripped from the ambient process and rejected in
# persisted overrides so new upstream knobs cannot silently invalidate the
# quality-preserving claim.
QUALIFICATION_COLI_KEYS = frozenset({
    "COLI_POLICY", "COLI_COLOR", "COLI_MODEL", "COLI_GPU", "COLI_GPUS",
    "COLI_CUDA", "COLI_METAL", "COLI_VULKAN", "COLI_NUMA",
    "COLI_NO_OMP_TUNE", "COLI_CUDA_PIPE", "COLI_CUDA_ASYNC", "COLI_MMAP",
    "COLI_SSD_FAST_GBS", "COLI_NO_FUSED_PAIR", "COLI_RAM_OVERCOMMIT",
})

# These are the only hardware/topology selections inherited from the operator's
# shell. All other execution knobs come from the generated Colibri plan or an
# explicit Lattice candidate.
AMBIENT_QUALIFICATION_KEYS = TOPOLOGY_KEYS | frozenset({
    "COLI_GPU", "COLI_GPUS", "COLI_CUDA", "COLI_METAL", "COLI_VULKAN",
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
})
'''
if old_constants not in text:
    raise SystemExit("qualification constant block did not match")
text = text.replace(old_constants, new_constants)

old_functions = '''def _qualification_key(key: str) -> bool:
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
            if not text or any(char in text for char in "\\r\\n\\x00"):
                raise LatticeError(f"invalid qualification environment value: {key}")
            env[key] = text
    return env


def qualification_environment(env: dict[str, str]) -> dict[str, str]:
    return {key: str(value) for key, value in sorted(env.items()) if _qualification_key(key)}
'''
new_functions = '''def _qualification_key(key: str) -> bool:
    if key in FORBIDDEN_AMBIENT_KEYS or key in SERVING_ONLY_KEYS:
        return False
    return key in (
        SAFE_TUNABLE_KEYS
        | QUALIFICATION_SCALAR_KEYS
        | TOPOLOGY_KEYS
        | QUALIFICATION_COLI_KEYS
    )


def clean_environment(
    source: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if source is None else source)
    inherited = {
        key: str(env[key])
        for key in AMBIENT_QUALIFICATION_KEYS
        if key in env
    }
    for key in list(env):
        if (
            key.startswith("COLI_")
            or key.startswith("OMP_")
            or key.startswith("GOMP_")
            or key in QUALIFICATION_SCALAR_KEYS
            or key in SAFE_TUNABLE_KEYS
            or key in SERVING_ONLY_KEYS
            or key in FORBIDDEN_AMBIENT_KEYS
        ):
            env.pop(key, None)
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
            if not _qualification_key(key):
                raise LatticeError(f"invalid qualification environment override: {key}")
            if not text or any(char in text for char in "\\r\\n\\x00"):
                raise LatticeError(f"invalid qualification environment value: {key}")
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
    return {key: str(value) for key, value in sorted(env.items()) if _qualification_key(key)}
'''
if old_functions not in text:
    raise SystemExit("qualification environment functions did not match")
colibri.write_text(text.replace(old_functions, new_functions), encoding="utf-8")

test_path = Path("lattice/tests/test_colibri.py")
test_text = test_path.read_text(encoding="utf-8")
old_test = '''    def test_quality_environment_is_stripped_but_backend_is_attested(self):
        env = clean_environment({"COLI_TEMP": "0.8", "IDOT": "0", "COLI_METAL": "1"})
        self.assertNotIn("COLI_TEMP", env)
        self.assertNotIn("IDOT", env)
        self.assertEqual(qualification_environment(env)["COLI_METAL"], "1")
'''
new_test = '''    def test_quality_environment_is_stripped_but_backend_is_attested(self):
        env = clean_environment({
            "COLI_TEMP": "0.8",
            "IDOT": "0",
            "COLI_METAL": "1",
            "COLI_UNREVIEWED_EXPERIMENT": "1",
        })
        self.assertNotIn("COLI_TEMP", env)
        self.assertNotIn("IDOT", env)
        self.assertNotIn("COLI_UNREVIEWED_EXPERIMENT", env)
        self.assertEqual(qualification_environment(env)["COLI_METAL"], "1")

    def test_forbidden_or_unknown_persisted_override_is_rejected(self):
        with self.assertRaisesRegex(LatticeError, "invalid qualification environment override"):
            clean_environment({}, {"DRAFT": "0"})
        with self.assertRaisesRegex(LatticeError, "invalid qualification environment override"):
            clean_environment({}, {"COLI_UNREVIEWED_EXPERIMENT": "1"})
        env = clean_environment({}, {"COLI_METAL": "1"})
        self.assertEqual(env["COLI_METAL"], "1")
'''
if old_test not in test_text:
    raise SystemExit("qualification environment test block did not match")
test_path.write_text(test_text.replace(old_test, new_test), encoding="utf-8")
