from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .colibri import create_context
from .common import LatticeError
from .verify import verify_workspace
from .workspace import Workspace

ALLOWED_LAUNCH_COMMANDS = frozenset({"run", "chat", "serve", "web", "info"})
VALUE_OPTIONS = {
    "run": frozenset({"--ngen"}),
    "chat": frozenset({"--ngen", "--api-key"}),
    "serve": frozenset({
        "--ngen", "--host", "--port", "--model-id", "--api-key",
        "--cors-origin", "--max-queue", "--queue-timeout", "--kv-slots",
        "--allowed-host",
    }),
    "web": frozenset({
        "--ngen", "--host", "--port", "--model-id", "--api-key",
        "--cors-origin", "--max-queue", "--queue-timeout", "--kv-slots",
        "--allowed-host",
    }),
    "info": frozenset(),
}
SWITCH_OPTIONS = {
    "run": frozenset(),
    "chat": frozenset({"--no-attach"}),
    "serve": frozenset(),
    "web": frozenset({"--no-browser"}),
    "info": frozenset(),
}


def ensure_profile_deployable(profile: dict[str, Any]) -> None:
    if profile.get("deployable") is not True:
        blocker = profile.get("deployment_blocker")
        detail = blocker if isinstance(blocker, str) and blocker else "profile is not approved for deployment"
        raise LatticeError("deployment is blocked by the evidence profile: " + detail)


def validate_launch_args(coli_args: list[str]) -> None:
    """Accept only reviewed, subcommand-specific operational controls."""
    if not coli_args:
        raise LatticeError("launch command is missing")
    command = coli_args[0]
    if command not in ALLOWED_LAUNCH_COMMANDS:
        raise LatticeError(
            f"launch supports only {', '.join(sorted(ALLOWED_LAUNCH_COMMANDS))}; "
            f"received {command!r}"
        )
    values = VALUE_OPTIONS[command]
    switches = SWITCH_OPTIONS[command]
    positionals: list[str] = []
    index = 1
    while index < len(coli_args):
        argument = coli_args[index]
        if not isinstance(argument, str) or not argument or "\x00" in argument:
            raise LatticeError("launch arguments must be non-empty strings without NUL bytes")
        if argument == "--":
            raise LatticeError("nested '--' separators are not permitted in launch arguments")
        if argument.startswith("-"):
            flag, separator, inline_value = argument.partition("=")
            if flag in switches:
                if separator:
                    raise LatticeError(f"launch switch {flag} does not accept a value")
                index += 1
                continue
            if flag not in values:
                raise LatticeError(
                    f"launch flag {flag!r} is not in the reviewed allowlist for {command}"
                )
            if separator:
                if not inline_value:
                    raise LatticeError(f"launch option {flag} requires a non-empty value")
                index += 1
                continue
            if index + 1 >= len(coli_args):
                raise LatticeError(f"launch option {flag} requires a value")
            value = coli_args[index + 1]
            if not isinstance(value, str) or not value or value.startswith("-") or "\x00" in value:
                raise LatticeError(f"launch option {flag} requires a non-empty value")
            index += 2
            continue
        positionals.append(argument)
        index += 1
    if command == "run":
        if not positionals:
            raise LatticeError("launch run requires a prompt")
    elif positionals:
        raise LatticeError(f"launch {command} does not accept positional arguments")


def prepare_launch_args(coli_args: list[str]) -> list[str]:
    prepared = list(coli_args or ["serve"])
    validate_launch_args(prepared)
    if prepared[0] == "chat" and "--no-attach" not in prepared:
        prepared.insert(1, "--no-attach")
    return prepared


def deployment_environment(workspace: Workspace, *, adaptive: bool = False) -> tuple[dict[str, str], dict[str, Any]]:
    """Return a verified environment only for a deployable evidence profile."""
    verify_workspace(workspace, deep=False)
    project = workspace.load_project()
    profile = workspace.load_profile()
    ensure_profile_deployable(profile)
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
        for key in ("KVSAVE", "AUTOPIN", "REPIN"):
            env.pop(key, None)
        env["LATTICE_ADAPTIVE_OUTSIDE_EVIDENCE"] = "1"
    return env, profile


def launch(workspace: Workspace, coli_args: list[str], *, adaptive: bool = False) -> int:
    coli_args = prepare_launch_args(coli_args)
    project = workspace.load_project()
    coli = Path(project["repo_root"]) / "c" / "coli"
    if not coli.is_file():
        raise LatticeError(f"Colibri launcher is missing: {coli}")
    with workspace.acquire_lock("operation"):
        env, _profile = deployment_environment(workspace, adaptive=adaptive)
        command = [sys.executable, str(coli), "--no-tune-profile", *coli_args]
        try:
            return subprocess.call(command, env=env, cwd=str(Path(project["repo_root"])))
        except OSError as error:
            raise LatticeError(f"cannot launch Colibri: {error}") from error
