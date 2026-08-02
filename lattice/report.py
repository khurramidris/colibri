from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_json, utc_now
from .suite import parse_suite
from .workspace import Workspace


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(workspace: Workspace, profile_id: str | None = None) -> str:
    project = workspace.load_project()
    profile = workspace.load_profile(profile_id)
    session = workspace.load_session(profile["session_id"])
    suite = parse_suite(project["suite"])
    winner = profile["winner"]
    winner_score = profile.get("winner_score")
    plan = project.get("plan") or {}
    policy = profile["selection_policy"]
    lines = [
        f"# Lattice qualification report — {suite.name}",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Executive result",
        "",
    ]
    if profile.get("baseline_retained"):
        lines.append("**Baseline retained.** No tested candidate cleared every performance, regression and evidence gate.")
    else:
        speedup = winner_score.get("weighted_speedup") if winner_score else None
        ci = winner_score.get("confidence_interval") if winner_score else None
        lines.append(
            f"**Promoted `{winner['id']}`** with a workload-weighted speedup of "
            f"**{_fmt((speedup - 1.0) * 100 if speedup else None)}%** over the generated Colibri plan."
        )
        if ci:
            lines.append(f"Bootstrap confidence interval: **{(ci[0]-1)*100:.2f}% to {(ci[1]-1)*100:.2f}%**.")
        cost = winner_score.get("cost_per_million_usd") if winner_score else None
        if cost is not None:
            lines.append(f"Estimated decode-only hardware cost at the supplied hourly rate: **${cost:.2f} per million generated tokens**.")
            lines.append("This estimate excludes prompt prefill, idle capacity, batching effects, queueing and service overhead.")
    lines.extend([
        "",
        "## Qualified system",
        "",
        f"- Model family: `{project.get('model_family')}`",
        f"- Model fingerprint: `{project.get('model_fingerprint')}`",
        f"- Runtime fingerprint: `{project.get('runtime_fingerprint')}`",
        f"- Hardware fingerprint: `{project.get('hardware_fingerprint')}`",
        f"- Execution fingerprint: `{project.get('execution_fingerprint')}`",
        f"- Qualification context: `{project.get('qualification_context')}` tokens",
        f"- Expected bottleneck: `{plan.get('expected_bottleneck', 'unknown')}`",
        f"- Session: `{session['id']}`",
        f"- Repeats per workload/candidate: `{session['repeats']}`",
        "",
        "## Promotion policy",
        "",
        f"- Minimum successful paired runs: {policy['min_runs']}",
        f"- Minimum weighted gain: {policy['min_gain']:.2%}",
        f"- Maximum allowed per-workload regression: {policy['max_regression']:.2%}",
        f"- Bootstrap confidence: {policy['confidence']:.0%}",
        f"- Confidence lower bound required: {policy['require_confidence']}",
        "",
        "## Workload suite",
        "",
        "| Case | Weight | Context | Generated tokens |",
        "|---|---:|---:|---:|",
    ])
    for case in suite.cases:
        lines.append(f"| `{case.id}` | {case.weight:.2f} | {case.context} | {case.tokens} |")
    lines.extend([
        "",
        "## Candidate evidence",
        "",
        "| Candidate | Eligible | Weighted speedup | Worst case | Confidence interval | Effective tok/s | Reason |",
        "|---|:---:|---:|---:|---:|---:|---|",
    ])
    for score in profile.get("scores", []):
        ci = score.get("confidence_interval")
        ci_text = "—" if not ci else f"{(ci[0]-1)*100:.2f}%…{(ci[1]-1)*100:.2f}%"
        weighted = score.get("weighted_speedup")
        worst = score.get("worst_case_regression")
        lines.append(
            f"| `{score['candidate_id']}` | {'yes' if score['eligible'] else 'no'} | "
            f"{_fmt((weighted-1)*100 if weighted else None)}% | {_fmt(worst*100 if worst is not None else None)}% | "
            f"{ci_text} | {_fmt(score.get('effective_tok_s'))} | {score['reason']} |"
        )
    lines.extend([
        "",
        "## Promoted environment",
        "",
        "```text",
    ])
    env = winner.get("environment") or {}
    if env:
        lines.extend(f"{key}={value}" for key, value in sorted(env.items()))
    else:
        lines.append("# baseline: no additional environment overrides")
    lines.extend([
        "```",
        "",
        "## Interpretation and boundaries",
        "",
        "This report qualifies one exact combination of model bytes, Colibri runtime bytes, hardware plan and workload suite. "
        "It does not establish universal performance for other models, machines or workloads. A changed fingerprint requires requalification.",
        "",
        "The promoted candidate is limited to execution and placement controls. Lattice does not change quantization, expert selection, "
        "sampling policy, model weights or router semantics during this qualification.",
        "",
        "Failed candidates remain part of the evidence rather than being hidden from the report.",
        "",
    ])
    return "\n".join(lines)


def write_report(workspace: Workspace, output: Path, profile_id: str | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = render_report(workspace, profile_id)
    output.write_text(text, encoding="utf-8", newline="\n")
    return output
