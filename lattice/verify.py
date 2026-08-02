from __future__ import annotations

from pathlib import Path
from typing import Any

from .colibri import create_context
from .common import LatticeError, canonical_json, short_id
from .recommend import evaluate_session
from .suite import parse_suite
from .workspace import Workspace


def verify_workspace(workspace: Workspace, *, deep: bool = False) -> dict[str, Any]:
    project = workspace.load_project()
    context = create_context(
        Path(project["repo_root"]),
        Path(project["model_path"]),
        engine=Path(project["engine_path"]),
        deep=deep,
        context_length=int(project["qualification_context"]),
        qualification_overrides=project.get("qualification_environment"),
    )
    checks = {
        "model_fingerprint": context.model_fingerprint == project.get("model_fingerprint"),
        "runtime_fingerprint": context.runtime_fingerprint == project.get("runtime_fingerprint"),
        "hardware_fingerprint": context.hardware_fingerprint == project.get("hardware_fingerprint"),
        "execution_fingerprint": context.execution_fingerprint == project.get("execution_fingerprint"),
        "qualification_context": context.qualification_context == project.get("qualification_context"),
        "suite_fingerprint": parse_suite(project["suite"]).fingerprint == project.get("suite_fingerprint"),
    }
    profile = None
    if workspace.current_profile_path.exists():
        profile = workspace.load_profile()
        policy = profile.get("selection_policy")
        if not isinstance(policy, dict):
            raise LatticeError("profile selection policy is missing")
        evaluation = evaluate_session(workspace, profile["session_id"], **policy)
        expected_id = short_id("profile", {
            "session_id": profile["session_id"],
            "winner": evaluation["winner"]["id"],
            "policy": evaluation["selection_policy"],
        })
        checks.update({
            "profile_id": profile.get("id") == expected_id,
            "profile_model_fingerprint": profile.get("model_fingerprint") == context.model_fingerprint,
            "profile_runtime_fingerprint": profile.get("runtime_fingerprint") == context.runtime_fingerprint,
            "profile_hardware_fingerprint": profile.get("hardware_fingerprint") == context.hardware_fingerprint,
            "profile_execution_fingerprint": profile.get("execution_fingerprint") == context.execution_fingerprint,
            "profile_qualification_context": profile.get("qualification_context") == context.qualification_context,
            "profile_suite_fingerprint": profile.get("suite_fingerprint") == project.get("suite_fingerprint"),
            "profile_recomputed": canonical_json({k: profile.get(k) for k in evaluation}) == canonical_json(evaluation),
        })
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise LatticeError("workspace verification failed: " + ", ".join(failed))
    return {
        "status": "ok",
        "checks": checks,
        "model_family": context.model_family,
        "current_profile": None if profile is None else profile["id"],
    }
