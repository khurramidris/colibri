from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .colibri import create_context, hardware_summary
from .common import LatticeError
from .deploy import deployment_environment, launch
from .experiment import create_project, resume_experiment, run_experiment
from .recommend import recommend
from .report import write_report
from .suite import load_suite, parse_suite
from .verify import verify_workspace
from .workspace import Workspace


def _workspace(value: str) -> Workspace:
    return Workspace(Path(value).expanduser().resolve())


def cmd_init(args: argparse.Namespace) -> int:
    suite = load_suite(Path(args.suite))
    qualification_context = max(case.context for case in suite.cases)
    context = create_context(
        Path(args.repo), Path(args.model),
        engine=Path(args.engine) if args.engine else None,
        deep=args.deep,
        context_length=qualification_context,
    )
    workspace = _workspace(args.workspace)
    project = create_project(context, suite)
    workspace.initialize(project, force=args.force)
    print(json.dumps({
        "status": "initialized",
        "workspace": str(workspace.root),
        "model_family": context.model_family,
        "model_fingerprint": context.model_fingerprint,
        "runtime_fingerprint": context.runtime_fingerprint,
        "hardware_fingerprint": context.hardware_fingerprint,
        "execution_fingerprint": context.execution_fingerprint,
        "qualification_context": context.qualification_context,
        "suite_fingerprint": suite.fingerprint,
        "hardware": hardware_summary(context),
    }, indent=2))
    return 0


def _context_from_project(workspace: Workspace, deep: bool = False):
    project = workspace.load_project()
    return create_context(
        Path(project["repo_root"]),
        Path(project["model_path"]),
        engine=Path(project["engine_path"]),
        deep=deep,
        context_length=int(project["qualification_context"]),
        qualification_overrides=project.get("qualification_environment"),
    )


def _assert_project_identity(workspace: Workspace, context) -> None:
    project = workspace.load_project()
    mismatches = []
    if project.get("model_fingerprint") != context.model_fingerprint:
        mismatches.append("model")
    if project.get("runtime_fingerprint") != context.runtime_fingerprint:
        mismatches.append("runtime")
    if project.get("hardware_fingerprint") != context.hardware_fingerprint:
        mismatches.append("hardware")
    if project.get("execution_fingerprint") != context.execution_fingerprint:
        mismatches.append("execution environment")
    if project.get("qualification_context") != context.qualification_context:
        mismatches.append("qualification context")
    if mismatches:
        raise LatticeError("project identity changed; reinitialize before qualification: " + ", ".join(mismatches))


def cmd_qualify(args: argparse.Namespace) -> int:
    workspace = _workspace(args.workspace)
    project = workspace.load_project()
    suite = parse_suite(project["suite"])
    context = _context_from_project(workspace, deep=False)
    _assert_project_identity(workspace, context)
    progress = lambda message: print(f"[lattice] {message}", flush=True)
    if args.resume:
        session = resume_experiment(
            workspace, context, suite, args.resume,
            timeout=args.timeout, retry_failed=args.retry_failed, progress=progress,
        )
    else:
        session = run_experiment(
            workspace, context, suite, repeats=args.repeats,
            timeout=args.timeout, progress=progress,
        )
    print(json.dumps({"status": session["status"], "session_id": session["id"], "runs": len(session["run_ids"])}, indent=2))
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    workspace = _workspace(args.workspace)
    profile, scores = recommend(
        workspace,
        args.session,
        min_runs=args.min_runs,
        min_gain=args.min_gain,
        max_regression=args.max_regression,
        confidence=args.confidence,
        require_confidence=args.require_confidence,
        hourly_cost_usd=args.hourly_cost,
    )
    print(json.dumps({
        "profile_id": profile["id"],
        "winner": profile["winner"]["id"],
        "baseline_retained": profile["baseline_retained"],
        "scores": [score.as_dict() for score in scores],
    }, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    workspace = _workspace(args.workspace)
    output = Path(args.output).expanduser().resolve() if args.output else workspace.reports_dir / "qualification-report.md"
    path = write_report(workspace, output, args.profile)
    print(path)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify_workspace(_workspace(args.workspace), deep=args.deep)
    print(json.dumps(result, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    workspace = _workspace(args.workspace)
    project = workspace.load_project()
    sessions = []
    for path in sorted(workspace.sessions_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        sessions.append({
            "id": data.get("id"),
            "status": data.get("status"),
            "started_at": data.get("started_at"),
            "completed_at": data.get("completed_at"),
            "runs": len(data.get("run_ids") or []),
        })
    profile = None
    if workspace.current_profile_path.exists():
        current = workspace.load_profile()
        profile = {"id": current["id"], "winner": current["winner"]["id"], "created_at": current["created_at"]}
    print(json.dumps({
        "workspace": str(workspace.root),
        "model_family": project.get("model_family"),
        "suite": project.get("suite", {}).get("name"),
        "sessions": sessions,
        "current_profile": profile,
    }, indent=2))
    return 0



def cmd_env(args: argparse.Namespace) -> int:
    workspace = _workspace(args.workspace)
    env, profile = deployment_environment(workspace, adaptive=args.adaptive)
    selected = {key: env[key] for key in sorted(env) if key.startswith(("COLI_", "CUDA_", "OMP_", "PIPE", "DIRECT", "URING", "PILOT", "RAM_GB", "CTX", "KVSAVE", "AUTOPIN", "REPIN", "LATTICE_"))}
    if args.format == "json":
        print(json.dumps({"profile_id": profile["id"], "environment": selected}, indent=2))
    else:
        for key, value in selected.items():
            escaped = value.replace("'", "'\"'\"'")
            print(f"export {key}='{escaped}'")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    workspace = _workspace(args.workspace)
    command = list(args.coli_args)
    if command and command[0] == "--":
        command = command[1:]
    return launch(workspace, command, adaptive=args.adaptive)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lattice",
        description="Evidence-driven qualification and deployment profiles for Colibri inference.",
    )
    parser.add_argument("--version", action="version", version=f"lattice {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="deep-check and fingerprint a Colibri model/runtime/workload")
    init.add_argument("--repo", default=".", help="Colibri checkout root")
    init.add_argument("--model", required=True, help="converted Colibri model directory")
    init.add_argument("--engine", help="explicit engine binary; otherwise resolved from model config")
    init.add_argument("--suite", required=True, help="workload suite JSON")
    init.add_argument("--workspace", default=".lattice", help="evidence workspace")
    init.add_argument("--deep", action=argparse.BooleanOptionalAction, default=True, help="run Colibri deep doctor")
    init.add_argument("--force", action="store_true", help="replace project.json in an existing workspace")
    init.set_defaults(func=cmd_init)

    qualify = sub.add_parser("qualify", help="run a controlled multi-workload candidate experiment")
    qualify.add_argument("--workspace", default=".lattice")
    qualify.add_argument("--repeats", type=int, default=2)
    qualify.add_argument("--timeout", type=int, default=900, help="seconds per calibration or replay")
    qualify.add_argument("--resume", metavar="SESSION_ID", help="resume an interrupted qualification session")
    qualify.add_argument("--retry-failed", action=argparse.BooleanOptionalAction, default=True,
                         help="when resuming, retry failed tasks as well as missing tasks")
    qualify.set_defaults(func=cmd_qualify)

    rec = sub.add_parser("recommend", help="apply promotion gates and write an immutable profile")
    rec.add_argument("--workspace", default=".lattice")
    rec.add_argument("--session", required=True)
    rec.add_argument("--min-runs", type=int, default=2)
    rec.add_argument("--min-gain", type=float, default=0.03)
    rec.add_argument("--max-regression", type=float, default=0.05)
    rec.add_argument("--confidence", type=float, default=0.90)
    rec.add_argument("--require-confidence", action="store_true")
    rec.add_argument("--hourly-cost", type=float, help="hardware cost in USD/hour for cost-per-million estimate")
    rec.set_defaults(func=cmd_recommend)

    report = sub.add_parser("report", help="generate an investor/customer-readable qualification report")
    report.add_argument("--workspace", default=".lattice")
    report.add_argument("--profile")
    report.add_argument("--output")
    report.set_defaults(func=cmd_report)

    verify = sub.add_parser("verify", help="recompute runtime/model/workload identity before deployment")
    verify.add_argument("--workspace", default=".lattice")
    verify.add_argument("--deep", action="store_true")
    verify.set_defaults(func=cmd_verify)

    status = sub.add_parser("status", help="show sessions and the promoted profile")
    status.add_argument("--workspace", default=".lattice")
    status.set_defaults(func=cmd_status)

    env = sub.add_parser("env", help="print the verified promoted deployment environment")
    env.add_argument("--workspace", default=".lattice")
    env.add_argument("--adaptive", action="store_true", help="allow runtime adaptive state not covered by the qualification")
    env.add_argument("--format", choices=("json", "shell"), default="json")
    env.set_defaults(func=cmd_env)

    launch_parser = sub.add_parser("launch", help="verify and run a Colibri command under the promoted profile")
    launch_parser.add_argument("--workspace", default=".lattice")
    launch_parser.add_argument("--adaptive", action="store_true", help="allow runtime adaptive state not covered by the qualification")
    launch_parser.add_argument("coli_args", nargs=argparse.REMAINDER, help="arguments after --, e.g. -- serve --port 8000")
    launch_parser.set_defaults(func=cmd_launch)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except LatticeError as error:
        print(f"lattice: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("lattice: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
