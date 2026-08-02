from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .colibri import create_context
from .common import LatticeError
from .verify import verify_workspace
from .workspace import Workspace


PROFILE_OVERRIDE_FLAGS = frozenset({
    "--model", "--ram", "--ctx", "--gpu", "--vram", "--policy",
    "--repin", "--cap", "--topp", "--topk", "--temp", "--auto-tier",
    "--attach",
})
ALLOWED_LAUNCH_COMMANDS = frozenset({"run", "chat", "serve", "web", "info"})


def validate_launch_args(coli_args: list[str]) -> None:
    """Reject CLI options that would invalidate the promoted profile.

    Serving controls such as host, port, API key, queue depth and generation
    length remain available. Placement, model, policy and sampling controls are
    bound to qualification evidence and cannot be replaced after verification.
    """
    for argument in coli_args:
        flag = argument.split("=", 1)[0]
        if flag in PROFILE_OVERRIDE_FLAGS:
            raise LatticeError(
                f"launch forbids {flag} because it overrides the verified profile; "
                "create and qualify a new workspace instead"
            )


def prepare_launch_args(coli_args: list[str]) -> list[str]:
    prepared = list(coli_args or ["serve"])
    command = prepared[0]
    if command not in ALLOWED_LAUNCH_COMMANDS:
        raise LatticeError(
            f"launch supports only {', '.join(sorted(ALLOWED_LAUNCH_COMMANDS))}; "
            f"received {command!r}"
        )
    validate_launch_args(prepared)
    if command == "chat" and "--no-attach" not in prepared:
        # Colibri chat can auto-attach to an already-running server. Force a
        # private local engine so the verified profile cannot be bypassed.
        prepared.insert(1, "--no-attach")
    return prepared


def deployment_environment(workspace: Workspace, *, adaptive: bool = False) -> tuple[dict[str, str], dict[str, Any]]:
    """Return the exact promoted environment after re-verifying its evidence."""
    verify_workspace(workspace, deep=False)
    project = workspace.load_project()
    profile = workspace.load_profile()
    context = create_context(
        Path(project["repo_root"]),
        Path(project["model_path"]),
        engine=Path(project["engine_path"]),
        deep=False,
        context_length=int(project["qualification_context"]),
        qualification_overrides=project.get("qualification_environment"),
        frozen_plan=project.get("plan"),
    )
    env = dict(context.base_environment)
    env.update({str(key): str(value) for key, value in profile["winner"]["environment"].items()})
    env.update({
        "COLI_MODEL": str(context.model),
        "COLI_ENGINE": str(context.engine),
        "COLI_POLICY": "quality",
        "LATTICE_PROFILE_ID": profile["id"],
    })
    if adaptive:
        # Adaptive state is useful in production but was deliberately frozen
        # during qualification. The opt-in is explicit so operators know this
        # deployment is no longer byte-for-byte the measured environment.
        for key in ("KVSAVE", "AUTOPIN", "REPIN"):
            env.pop(key, None)
    return env, profile


def launch(workspace: Workspace, coli_args: list[str], *, adaptive: bool = False) -> int:
    coli_args = prepare_launch_args(coli_args)
    project = workspace.load_project()
    coli = Path(project["repo_root"]) / "c" / "coli"
    if not coli.is_file():
        raise LatticeError(f"Colibri launcher is missing: {coli}")
    with workspace.acquire_lock("operation"):
        env, _profile = deployment_environment(workspace, adaptive=adaptive)
        # Saved `coli tune` profiles are a second, independent source of
        # execution overrides. Disable them so the launched process uses only
        # the environment that Lattice just recomputed and verified.
        command = [sys.executable, str(coli), "--no-tune-profile", *coli_args]
        return subprocess.call(command, env=env, cwd=str(Path(project["repo_root"])))
