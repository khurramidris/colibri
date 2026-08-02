from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .colibri import create_context
from .common import LatticeError
from .verify import verify_workspace
from .workspace import Workspace


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
        for key in ("KVSAVE", "AUTOPIN", "REPIN"):
            env.pop(key, None)
    return env, profile


def launch(workspace: Workspace, coli_args: list[str], *, adaptive: bool = False) -> int:
    if not coli_args:
        coli_args = ["serve"]
    if "--model" in coli_args or any(arg.startswith("--model=") for arg in coli_args):
        raise LatticeError("launch forbids --model overrides because the profile is bound to one model fingerprint")
    project = workspace.load_project()
    coli = Path(project["repo_root"]) / "c" / "coli"
    if not coli.is_file():
        raise LatticeError(f"Colibri launcher is missing: {coli}")
    env, _profile = deployment_environment(workspace, adaptive=adaptive)
    command = [sys.executable, str(coli), *coli_args]
    return subprocess.call(command, env=env, cwd=str(Path(project["repo_root"])))
