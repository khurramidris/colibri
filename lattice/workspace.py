from __future__ import annotations

import os
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import LatticeError, atomic_write_json, load_json, validate_id
from .evidence import seal_record, verify_record_digest


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def project_path(self) -> Path:
        return self.root / "project.json"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def replays_dir(self) -> Path:
        return self.root / "replays"

    @property
    def sessions_dir(self) -> Path:
        return self.root / "sessions"

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def current_profile_path(self) -> Path:
        return self.root / "current-profile.json"

    def initialize(self, project: dict[str, Any], *, force: bool = False) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.project_path.exists():
            if not force:
                raise LatticeError(f"workspace already initialized: {self.root}")
            evidence = []
            for directory in (self.runs_dir, self.replays_dir, self.sessions_dir, self.profiles_dir):
                if directory.exists():
                    evidence.extend(path for path in directory.iterdir() if path.is_file())
            if evidence or self.current_profile_path.exists():
                raise LatticeError("refusing to replace a workspace that already contains evidence; use a new directory")
        for directory in (self.runs_dir, self.replays_dir, self.sessions_dir, self.profiles_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.project_path, project)

    def load_project(self) -> dict[str, Any]:
        data = load_json(self.project_path)
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise LatticeError("invalid or unsupported project.json")
        required = {
            "repo_root", "model_path", "engine_path", "model_family",
            "model_fingerprint", "runtime_fingerprint", "hardware_fingerprint",
            "execution_fingerprint", "plan_fingerprint", "replay_cap",
            "qualification_context",
            "qualification_environment", "storage_topology", "suite",
            "suite_fingerprint", "plan", "doctor",
        }
        missing = sorted(required - set(data))
        if missing:
            raise LatticeError("project.json is missing required fields: " + ", ".join(missing))
        if (isinstance(data["qualification_context"], bool)
                or not isinstance(data["qualification_context"], int)
                or not 128 <= data["qualification_context"] <= 262144):
            raise LatticeError("project.json has an invalid qualification_context")
        if not isinstance(data.get("plan_fingerprint"), str) or len(data["plan_fingerprint"]) != 64:
            raise LatticeError("project.json has an invalid plan_fingerprint")
        if (isinstance(data.get("replay_cap"), bool) or not isinstance(data.get("replay_cap"), int)
                or data["replay_cap"] < 0):
            raise LatticeError("project.json has an invalid replay_cap")
        if not isinstance(data["qualification_environment"], dict):
            raise LatticeError("project.json has an invalid qualification_environment")
        if not isinstance(data["storage_topology"], dict):
            raise LatticeError("project.json has an invalid storage_topology")
        return data

    def write_run(self, run: dict[str, Any]) -> Path:
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

    def load_session(self, session_id: str) -> dict[str, Any]:
        session_id = validate_id(session_id, "session id")
        data = load_json(self.sessions_dir / f"{session_id}.json")
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise LatticeError(f"invalid session: {session_id}")
        return data

    def list_runs(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id is not None:
            session_id = validate_id(session_id, "session id")
        runs: list[dict[str, Any]] = []
        for path in sorted(self.runs_dir.glob("*.json")):
            data = load_json(path)
            if not isinstance(data, dict) or data.get("schema_version") != 1:
                raise LatticeError(f"invalid run record: {path}")
            verify_record_digest(data, f"run record {path.name}")
            if session_id is None or data.get("session_id") == session_id:
                runs.append(data)
        return runs

    def write_profile(self, profile: dict[str, Any]) -> Path:
        profile_id = validate_id(profile.get("id"), "profile id")
        path = self.profiles_dir / f"{profile_id}.json"
        sealed = seal_record(profile)
        atomic_write_json(path, sealed, exclusive=True)
        atomic_write_json(
            self.current_profile_path,
            {"schema_version": 1, "profile_id": profile_id, "profile_sha256": sealed["record_sha256"]},
        )
        profile.clear()
        profile.update(sealed)
        return path

    def load_profile(self, profile_id: str | None = None) -> dict[str, Any]:
        pointer = None
        if profile_id is None:
            pointer = load_json(self.current_profile_path)
            if not isinstance(pointer, dict) or pointer.get("schema_version") != 1:
                raise LatticeError("invalid current profile pointer")
            profile_id = pointer.get("profile_id")
        profile_id = validate_id(profile_id, "profile id")
        data = load_json(self.profiles_dir / f"{profile_id}.json")
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise LatticeError(f"invalid profile: {profile_id}")
        digest = verify_record_digest(data, f"profile {profile_id}")
        if pointer is not None and pointer.get("profile_sha256") != digest:
            raise LatticeError("current profile pointer digest does not match profile")
        return data

    def acquire_lock(self, name: str = "operation"):
        name = validate_id(name, "lock name")
        return WorkspaceLock(self.root / f".{name}.lock")


class WorkspaceLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None
        self.token = uuid.uuid4().hex
        self.hostname = socket.gethostname()

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            # os.kill(pid, 0) is not a portable no-op on Windows: CPython's
            # Windows implementation maps ordinary signals to TerminateProcess.
            # Query the process handle without sending any signal instead.
            try:
                import ctypes
                from ctypes import wintypes

                process_query_limited_information = 0x1000
                still_active = 259
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel32.OpenProcess.restype = wintypes.HANDLE
                kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
                kernel32.GetExitCodeProcess.restype = wintypes.BOOL
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.CloseHandle.restype = wintypes.BOOL
                handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
                if not handle:
                    # Access denied means a process exists but is not queryable;
                    # any other failure is conservatively treated as not alive.
                    return ctypes.get_last_error() == 5
                try:
                    exit_code = wintypes.DWORD()
                    if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                        return True
                    return exit_code.value == still_active
                finally:
                    kernel32.CloseHandle(handle)
            except (AttributeError, OSError, ValueError):
                # Failure to establish death is not proof of staleness.
                return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True
        return True

    def _remove_provably_stale_local_lock(self) -> bool:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return False
        fields = {}
        for line in text.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                fields[key] = value
        try:
            pid = int(fields["pid"])
        except (KeyError, ValueError):
            return False
        if fields.get("hostname") != self.hostname or self._pid_is_alive(pid):
            return False
        try:
            self.path.unlink()
            return True
        except OSError:
            return False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                break
            except FileExistsError as error:
                if attempt == 0 and self._remove_provably_stale_local_lock():
                    continue
                raise LatticeError(
                    f"workspace is busy (lock exists: {self.path}); "
                    "remove it only after confirming the recorded process is not running"
                ) from error
        if self.fd is None:
            raise LatticeError(f"could not acquire workspace lock: {self.path}")
        payload = (
            f"pid={os.getpid()}\n"
            f"hostname={self.hostname}\n"
            f"created_unix={time.time():.6f}\n"
            f"token={self.token}\n"
        ).encode("utf-8")
        os.write(self.fd, payload)
        os.fsync(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            text = self.path.read_text(encoding="utf-8")
            if f"token={self.token}\n" in text:
                self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
