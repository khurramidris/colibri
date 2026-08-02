from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


# 1. Fail closed on unreviewed Colibri environment variables and constrain
# values that can affect model semantics.
replace_once(
    "lattice/colibri.py",
    '''QUALIFICATION_SCALAR_KEYS = frozenset({
    "RAM_GB", "CTX", "CUDA_EXPERT_GB", "CAP", "CAP_RAISE", "MLOCK",
    "COLI_MMAP", "COLI_SSD_FAST_GBS", "COLI_NO_FUSED_PAIR", "DISK_SPLIT",
    "COLI_RAM_OVERCOMMIT", "DRAFT", "MTP", "IDOT", "ABSORB", "I4S", "SPEC_PIN",
    "PIPE", "PIPE_WORKERS", "DIRECT", "URING", "PREFETCH", "PILOT",
    "PILOT_REAL", "PILOT_K", "SEED", "KVSAVE", "AUTOPIN", "REPIN",
    "SNAP", "OMP_NUM_THREADS", "OMP_WAIT_POLICY", "OMP_PROC_BIND", "OMP_PLACES",
    "GOMP_SPINCOUNT", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
})

TOPOLOGY_KEYS = frozenset({"COLI_MODEL_DIRS", "COLI_MODEL_MIRROR", "COLI_DISK_WEIGHTS"})
''',
    '''QUALIFICATION_SCALAR_KEYS = frozenset({
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
''',
)

replace_once(
    "lattice/colibri.py",
    '''def _qualification_key(key: str) -> bool:
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
''',
    '''def _qualification_key(key: str) -> bool:
    if key in FORBIDDEN_AMBIENT_KEYS or key in SERVING_ONLY_KEYS:
        return False
    return key in (
        SAFE_TUNABLE_KEYS
        | QUALIFICATION_SCALAR_KEYS
        | TOPOLOGY_KEYS
        | QUALIFICATION_COLI_KEYS
    )


def _validate_qualification_value(key: str, value: str) -> None:
    if not value or any(char in value for char in "\\r\\n\\x00"):
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
    env = dict(original)
    inherited = {
        key: str(original[key])
        for key in AMBIENT_QUALIFICATION_KEYS
        if key in original
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
            or key in TOPOLOGY_KEYS
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
''',
)

# 2. Application-level tamper evidence. This is intentionally described as a
# hash chain, not as cryptographic immutability or remote attestation.
Path("lattice/evidence.py").write_text('''from __future__ import annotations

from typing import Any

from .common import LatticeError, canonical_json, sha256_bytes

DIGEST_FIELD = "record_sha256"


def record_digest(record: dict[str, Any]) -> str:
    if not isinstance(record, dict):
        raise LatticeError("evidence record must be an object")
    payload = {key: value for key, value in record.items() if key != DIGEST_FIELD}
    return sha256_bytes(canonical_json(payload))


def seal_record(record: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(record)
    sealed[DIGEST_FIELD] = record_digest(sealed)
    return sealed


def verify_record_digest(record: dict[str, Any], label: str = "record") -> str:
    expected = record.get(DIGEST_FIELD)
    if not isinstance(expected, str) or len(expected) != 64:
        raise LatticeError(f"{label} has no valid record digest")
    actual = record_digest(record)
    if actual != expected:
        raise LatticeError(f"{label} digest mismatch")
    return expected


def session_evidence_root(session: dict[str, Any], runs: list[dict[str, Any]]) -> str:
    run_entries = []
    for run in runs:
        digest = verify_record_digest(run, f"run {run.get('id', '<unknown>')}")
        run_entries.append({"id": run.get("id"), "sha256": digest})
    replays = session.get("replays") or {}
    replay_entries = [
        {"case_id": case_id, "sha256": record.get("sha256")}
        for case_id, record in sorted(replays.items())
    ]
    session_core = {
        key: value
        for key, value in session.items()
        if key != "evidence_root_sha256"
    }
    payload = {
        "schema_version": 1,
        "session": session_core,
        "runs": sorted(run_entries, key=lambda item: (str(item["id"]), item["sha256"])),
        "replays": replay_entries,
    }
    return sha256_bytes(canonical_json(payload))
''', encoding="utf-8")

# 3. Seal runs automatically, verify them whenever loaded, and prevent a
# completed session from being rewritten through the application API.
replace_once(
    "lattice/workspace.py",
    'from .common import LatticeError, atomic_write_json, load_json, validate_id\n',
    'from .common import LatticeError, atomic_write_json, load_json, validate_id\nfrom .evidence import seal_record, verify_record_digest\n',
)
replace_once(
    "lattice/workspace.py",
    '''    def write_run(self, run: dict[str, Any]) -> Path:
        run_id = validate_id(run.get("id"), "run id")
        path = self.runs_dir / f"{run_id}.json"
        atomic_write_json(path, run, exclusive=True)
        return path

    def write_session(self, session: dict[str, Any], *, immutable: bool = False) -> Path:
        session_id = validate_id(session.get("id"), "session id")
        path = self.sessions_dir / f"{session_id}.json"
        atomic_write_json(path, session, exclusive=immutable)
        return path
''',
    '''    def write_run(self, run: dict[str, Any]) -> Path:
        run_id = validate_id(run.get("id"), "run id")
        path = self.runs_dir / f"{run_id}.json"
        sealed = seal_record(run)
        atomic_write_json(path, sealed, exclusive=True)
        run.clear()
        run.update(sealed)
        return path

    def write_session(self, session: dict[str, Any], *, immutable: bool = False) -> Path:
        session_id = validate_id(session.get("id"), "session id")
        path = self.sessions_dir / f"{session_id}.json"
        if path.exists():
            existing = load_json(path)
            if isinstance(existing, dict) and existing.get("status") == "completed" and existing != session:
                raise LatticeError(f"refusing to rewrite completed session: {session_id}")
        atomic_write_json(path, session, exclusive=immutable)
        return path
''',
)
replace_once(
    "lattice/workspace.py",
    '''            if not isinstance(data, dict) or data.get("schema_version") != 1:
                raise LatticeError(f"invalid run record: {path}")
            if session_id is None or data.get("session_id") == session_id:
                runs.append(data)
''',
    '''            if not isinstance(data, dict) or data.get("schema_version") != 1:
                raise LatticeError(f"invalid run record: {path}")
            verify_record_digest(data, f"run record {path.name}")
            if session_id is None or data.get("session_id") == session_id:
                runs.append(data)
''',
)

# 4. Finalize each completed experiment with a root over the exact session,
# replay hashes, and sealed run records.
replace_once(
    "lattice/experiment.py",
    'from .common import LatticeError, atomic_write_json, canonical_json, sha256_bytes, short_id, utc_now\n',
    'from .common import LatticeError, atomic_write_json, canonical_json, sha256_bytes, short_id, utc_now\nfrom .evidence import session_evidence_root\n',
)
replace_once(
    "lattice/experiment.py",
    '''    session["status"] = "running"
    session["completed_at"] = None
    workspace.write_session(session)
''',
    '''    session["status"] = "running"
    session["completed_at"] = None
    session.pop("evidence_root_sha256", None)
    workspace.write_session(session)
''',
)
replace_once(
    "lattice/experiment.py",
    '''        session["status"] = "completed"
        session["completed_at"] = utc_now()
        workspace.write_session(session)
        return session
    except BaseException:
        session["status"] = "interrupted"
        session["completed_at"] = utc_now()
        workspace.write_session(session)
        raise
''',
    '''        session["status"] = "completed"
        session["completed_at"] = utc_now()
        session["evidence_root_sha256"] = session_evidence_root(
            session, workspace.list_runs(session["id"])
        )
        workspace.write_session(session)
        return session
    except BaseException:
        session["status"] = "interrupted"
        session["completed_at"] = utc_now()
        session.pop("evidence_root_sha256", None)
        workspace.write_session(session)
        raise
''',
)
# There are two outer interruption handlers with the same body.
experiment = Path("lattice/experiment.py")
text = experiment.read_text(encoding="utf-8")
old = '''        except BaseException:
            session["status"] = "interrupted"
            session["completed_at"] = utc_now()
            workspace.write_session(session)
            raise
'''
new = '''        except BaseException:
            session["status"] = "interrupted"
            session["completed_at"] = utc_now()
            session.pop("evidence_root_sha256", None)
            workspace.write_session(session)
            raise
'''
if text.count(old) != 2:
    raise SystemExit(f"lattice/experiment.py: expected two outer interruption handlers, found {text.count(old)}")
experiment.write_text(text.replace(old, new), encoding="utf-8")

# 5. Verify every record digest and the completed session root before scoring.
replace_once(
    "lattice/integrity.py",
    'from .common import LatticeError, canonical_json, ensure_under, load_json, sha256_bytes, validate_id\n',
    'from .common import LatticeError, canonical_json, ensure_under, load_json, sha256_bytes, validate_id\nfrom .evidence import session_evidence_root, verify_record_digest\n',
)
replace_once(
    "lattice/integrity.py",
    '''    actual_ids = {run.get("id") for run in runs}
    if set(run_ids) != actual_ids:
''',
    '''    for run in runs:
        verify_record_digest(run, f"run {run.get('id', '<unknown>')}")
    actual_ids = {run.get("id") for run in runs}
    if set(run_ids) != actual_ids:
''',
)
replace_once(
    "lattice/integrity.py",
    '''    if require_complete and actual_tasks != expected_tasks:
        missing = expected_tasks - actual_tasks
        raise LatticeError(f"session task matrix is incomplete ({len(missing)} missing)")
    return candidates
''',
    '''    if require_complete and actual_tasks != expected_tasks:
        missing = expected_tasks - actual_tasks
        raise LatticeError(f"session task matrix is incomplete ({len(missing)} missing)")
    if status == "completed":
        expected_root = session.get("evidence_root_sha256")
        if not isinstance(expected_root, str) or len(expected_root) != 64:
            raise LatticeError("completed session has no valid evidence root")
        actual_root = session_evidence_root(session, runs)
        if actual_root != expected_root:
            raise LatticeError("completed session evidence root mismatch")
    return candidates
''',
)

# 6. Bind the promoted profile ID and verification to the exact evidence root.
replace_once(
    "lattice/recommend.py",
    '''        "suite_fingerprint": suite.fingerprint,
        "scores": [score.as_dict() for score in scores],
''',
    '''        "suite_fingerprint": suite.fingerprint,
        "evidence_root_sha256": session["evidence_root_sha256"],
        "scores": [score.as_dict() for score in scores],
''',
)
replace_once(
    "lattice/recommend.py",
    '''    profile_seed = {
        "session_id": session_id,
        "winner": evaluation["winner"]["id"],
        "policy": seed_policy,
    }
''',
    '''    profile_seed = {
        "session_id": session_id,
        "winner": evaluation["winner"]["id"],
        "policy": seed_policy,
        "evidence_root_sha256": evaluation["evidence_root_sha256"],
    }
''',
)
replace_once(
    "lattice/verify.py",
    '''        expected_id = short_id("profile", {
            "session_id": profile["session_id"],
            "winner": evaluation["winner"]["id"],
            "policy": evaluation["selection_policy"],
        })
''',
    '''        expected_id = short_id("profile", {
            "session_id": profile["session_id"],
            "winner": evaluation["winner"]["id"],
            "policy": evaluation["selection_policy"],
            "evidence_root_sha256": evaluation["evidence_root_sha256"],
        })
''',
)
replace_once(
    "lattice/verify.py",
    '''            "profile_suite_fingerprint": profile.get("suite_fingerprint") == project.get("suite_fingerprint"),
            "profile_recomputed": canonical_json({k: profile.get(k) for k in evaluation}) == canonical_json(evaluation),
''',
    '''            "profile_suite_fingerprint": profile.get("suite_fingerprint") == project.get("suite_fingerprint"),
            "profile_evidence_root": profile.get("evidence_root_sha256") == evaluation.get("evidence_root_sha256"),
            "profile_recomputed": canonical_json({k: profile.get(k) for k in evaluation}) == canonical_json(evaluation),
''',
)

# 7. Tests for the concrete gaps found by the audit.
replace_once(
    "lattice/tests/test_colibri.py",
    '''    def test_quality_environment_is_stripped_but_backend_is_attested(self):
        env = clean_environment({"COLI_TEMP": "0.8", "IDOT": "0", "COLI_METAL": "1"})
        self.assertNotIn("COLI_TEMP", env)
        self.assertNotIn("IDOT", env)
        self.assertEqual(qualification_environment(env)["COLI_METAL"], "1")
''',
    '''    def test_quality_environment_is_stripped_but_backend_is_attested(self):
        env = clean_environment({
            "COLI_TEMP": "0.8",
            "IDOT": "0",
            "COLI_METAL": "1",
            "COLI_UNREVIEWED_APPROX": "1",
        })
        self.assertNotIn("COLI_TEMP", env)
        self.assertNotIn("IDOT", env)
        self.assertNotIn("COLI_UNREVIEWED_APPROX", env)
        self.assertEqual(qualification_environment(env)["COLI_METAL"], "1")

    def test_unknown_or_semantic_override_is_rejected(self):
        with self.assertRaisesRegex(LatticeError, "invalid qualification environment override"):
            clean_environment({}, {"COLI_UNREVIEWED_APPROX": "1"})
        with self.assertRaisesRegex(LatticeError, "DRAFT must remain"):
            clean_environment({}, {"DRAFT": "3"})
        env = clean_environment({}, {"DRAFT": "0", "COLI_CUDA": "1"})
        self.assertEqual(env["DRAFT"], "0")
''',
)

# Recommend fixtures now finalize the same evidence root as production.
replace_once(
    "lattice/tests/test_recommend.py",
    'from lattice.common import atomic_write_json, canonical_json, sha256_bytes\n',
    'from lattice.common import atomic_write_json, canonical_json, sha256_bytes\nfrom lattice.evidence import session_evidence_root\n',
)
replace_once(
    "lattice/tests/test_recommend.py",
    '''    def _run(self, candidate: str, repeat: int, tok_s: float, replay_hash: str, suffix: str = "") -> dict:
''',
    '''    def _finalize(self, ws: Workspace, session: dict) -> None:
        session["evidence_root_sha256"] = session_evidence_root(
            session, ws.list_runs(session["id"])
        )
        ws.write_session(session)

    def _run(self, candidate: str, repeat: int, tok_s: float, replay_hash: str, suffix: str = "") -> dict:
''',
)
text = Path("lattice/tests/test_recommend.py").read_text(encoding="utf-8")
if text.count('            ws.write_session(session)\n') != 2:
    raise SystemExit("test_recommend.py: expected two session finalizations")
Path("lattice/tests/test_recommend.py").write_text(
    text.replace('            ws.write_session(session)\n', '            self._finalize(ws, session)\n'),
    encoding="utf-8",
)

Path("lattice/tests/test_evidence.py").write_text('''from __future__ import annotations

import unittest

from lattice.common import LatticeError
from lattice.evidence import seal_record, session_evidence_root, verify_record_digest


class EvidenceTests(unittest.TestCase):
    def test_record_digest_detects_edit(self):
        record = seal_record({"schema_version": 1, "id": "run-a", "metrics": {"tok_s": 1.0}})
        verify_record_digest(record, "fixture")
        record["metrics"]["tok_s"] = 9.0
        with self.assertRaisesRegex(LatticeError, "digest mismatch"):
            verify_record_digest(record, "fixture")

    def test_session_root_binds_runs_and_replays(self):
        run = seal_record({"schema_version": 1, "id": "run-a", "status": "success"})
        session = {
            "schema_version": 1,
            "id": "session-a",
            "status": "completed",
            "run_ids": ["run-a"],
            "replays": {"case": {"sha256": "a" * 64}},
        }
        first = session_evidence_root(session, [run])
        changed = dict(session)
        changed["replays"] = {"case": {"sha256": "b" * 64}}
        self.assertNotEqual(first, session_evidence_root(changed, [run]))


if __name__ == "__main__":
    unittest.main()
''', encoding="utf-8")

# Honest language: local hash-linked evidence is tamper-evident, not immutable.
for path in ("LATTICE.md", "docs/lattice-qualification.md"):
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    text = text.replace("immutable run", "tamper-evident run")
    text = text.replace("immutable evidence", "tamper-evident evidence")
    target.write_text(text, encoding="utf-8")
