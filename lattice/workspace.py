from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import LatticeError, atomic_write_json, load_json


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
            "execution_fingerprint", "qualification_context",
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
        if not isinstance(data["qualification_environment"], dict):
            raise LatticeError("project.json has an invalid qualification_environment")
        if not isinstance(data["storage_topology"], dict):
            raise LatticeError("project.json has an invalid storage_topology")
        return data

    def write_run(self, run: dict[str, Any]) -> Path:
        run_id = run["id"]
        path = self.runs_dir / f"{run_id}.json"
        atomic_write_json(path, run, exclusive=True)
        return path

    def write_session(self, session: dict[str, Any], *, immutable: bool = False) -> Path:
        path = self.sessions_dir / f"{session['id']}.json"
        atomic_write_json(path, session, exclusive=immutable)
        return path

    def load_session(self, session_id: str) -> dict[str, Any]:
        data = load_json(self.sessions_dir / f"{session_id}.json")
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise LatticeError(f"invalid session: {session_id}")
        return data

    def list_runs(self, session_id: str | None = None) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        for path in sorted(self.runs_dir.glob("*.json")):
            data = load_json(path)
            if not isinstance(data, dict) or data.get("schema_version") != 1:
                raise LatticeError(f"invalid run record: {path}")
            if session_id is None or data.get("session_id") == session_id:
                runs.append(data)
        return runs

    def write_profile(self, profile: dict[str, Any]) -> Path:
        path = self.profiles_dir / f"{profile['id']}.json"
        atomic_write_json(path, profile, exclusive=True)
        atomic_write_json(self.current_profile_path, {"schema_version": 1, "profile_id": profile["id"]})
        return path

    def load_profile(self, profile_id: str | None = None) -> dict[str, Any]:
        if profile_id is None:
            pointer = load_json(self.current_profile_path)
            if not isinstance(pointer, dict) or pointer.get("schema_version") != 1:
                raise LatticeError("invalid current profile pointer")
            profile_id = pointer.get("profile_id")
        data = load_json(self.profiles_dir / f"{profile_id}.json")
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise LatticeError(f"invalid profile: {profile_id}")
        return data

    def acquire_lock(self, name: str = "operation"):
        return WorkspaceLock(self.root / f".{name}.lock")


class WorkspaceLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise LatticeError(f"workspace is busy (lock exists: {self.path})") from error
        os.write(self.fd, f"pid={os.getpid()}\n".encode())
        os.fsync(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
