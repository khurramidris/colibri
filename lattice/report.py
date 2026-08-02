from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import utc_now
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
    oracle = profile.get("oracle_policy") or session.get("oracle_policy") or {}
    lines = [
        f"# Lattice qualification report — {suite.name}",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Executive result",
        "",
    ]
    if profile.get("baseline_retained"):
        lines.append("**Baseline retained.** No tested candidate cleared every numerical-consistency, performance, regression and evidence gate.")
    else:
        speedup = winner_score.get("weighted_speedup") if winner_score else None
        ci = winner_score.get("confidence_interval") if winner_score else None
        lines.append(
            f"**Promoted `{winner['id']}`** with a measured workload-weighted replay speedup of "
            f"**{_fmt((speedup - 1.0) * 100 if speedup else None)}%** over the generated Colibri plan."
        )
        if ci:
            lines.append(f"Bootstrap confidence interval for replay throughput: **{(ci[0]-1)*100:.2f}% to {(ci[1]-1)*100:.2f}%**.")
        cost = winner_score.get("cost_per_million_usd") if winner_score else None
        if cost is not None:
            lines.append(f"Estimated decode-only hardware cost at the supplied hourly rate: **${cost:.2f} per million generated tokens**.")
            lines.append("This estimate excludes prompt prefill, idle capacity, batching effects, queueing and service overhead.")
    lines.extend([
        "",
        "## Assurance level",
        "",
        "**Numerical replay consistency.** Every measured candidate was forced over the same recorded prompt and continuation token IDs and compared with a versioned numerical sketch of every replay-step logit vector.",
        "",
        "The gate requires exact forced-token, top-1, top-2 and top-k identity; rejects non-finite logits; and compares selected logits, moments and four deterministic full-vector projections within the recorded tolerances.",
        "",
        "This is stronger than token replay alone, but it is not complete logit equality, formal numerical equivalence, semantic equivalence or downstream model-quality validation. The current tolerances still require calibration on real supported CPU/GPU deployments.",
        "",
        "## Numerical-oracle policy",
        "",
        f"- Schema: `{oracle.get('schema', 'missing')}`",
        f"- Exact top-k identity size: `{oracle.get('topk', 'missing')}`",
        f"- Absolute tolerance: `{oracle.get('absolute_tolerance', 'missing')}`",
        f"- Relative tolerance: `{oracle.get('relative_tolerance', 'missing')}`",
        "",
        "## Qualified system fingerprints",
        "",
        f"- Model family: `{project.get('model_family')}`",
        f"- Sampled model/topology fingerprint: `{project.get('model_fingerprint')}`",
        f"- Runtime fingerprint: `{project.get('runtime_fingerprint')}`",
        f"- Hardware-plan fingerprint: `{project.get('hardware_fingerprint')}`",
        f"- Execution-environment fingerprint: `{project.get('execution_fingerprint')}`",
        f"- Session evidence root: `{profile.get('evidence_root_sha256', 'missing')}`",
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
        "This report applies to the recorded sampled model/topology fingerprint, runtime fingerprint, hardware-plan fingerprint, controlled execution environment and workload suite. It does not establish universal performance for other models, machines or workloads. A changed fingerprint or oracle policy requires requalification.",
        "",
        "The promoted candidate is limited to reviewed execution and placement controls. Lattice does not intentionally change quantization, expert selection, sampling policy, model weights or router semantics during this qualification.",
        "",
        "Run records carry individual SHA-256 digests and the completed session carries an evidence root over the session definition, replay hashes and run digests. This detects ordinary local edits but is not remote attestation, a digital signature or write-once storage.",
        "",
        "Failed candidates and numerical-oracle mismatches remain part of the retained evidence rather than being hidden from the report.",
        "",
    ])
    return "\n".join(lines)


def write_report(workspace: Workspace, output: Path, profile_id: str | None = None) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = render_report(workspace, profile_id)
    output.write_text(text, encoding="utf-8", newline="\n")
    return output
