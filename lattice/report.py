from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_text, utc_now
from .suite import parse_suite
from .verify import verify_workspace
from .workspace import Workspace


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(workspace: Workspace, profile_id: str | None = None) -> str:
    """Render only after recomputing workspace and profile evidence."""
    verification = verify_workspace(workspace, deep=False)
    project = workspace.load_project()
    profile = workspace.load_profile(profile_id)
    session = workspace.load_session(profile["session_id"])
    runs = workspace.list_runs(session["id"])
    suite = parse_suite(project["suite"])
    winner = profile["winner"]
    winner_score = profile.get("winner_score")
    plan = project.get("plan") or {}
    policy = profile["selection_policy"]
    oracle = profile.get("oracle_policy") or session.get("oracle_policy") or {}
    stats = profile.get("statistics_policy") or {}
    assurance = profile.get("assurance_level", "missing")
    deployable = profile.get("deployable") is True
    blocker = profile.get("deployment_blocker") or "none recorded"
    failed_runs = [run for run in runs if run.get("status") == "failed"]

    lines = [
        f"# Lattice screening evidence report — {suite.name}",
        "",
        f"Generated: {utc_now()}",
        "",
        "> **This is an evidence and screening report, not a production-readiness certificate.**",
        "",
        "## Executive result",
        "",
        f"- Evidence verification: **{verification['status'].upper()}**",
        f"- Assurance level: `{assurance}`",
        f"- Deployable from this profile: **{'yes' if deployable else 'no'}**",
        f"- Deployment blocker: {_cell(blocker)}",
    ]
    if profile.get("baseline_retained"):
        lines.append(
            "- Screening selection: **baseline retained**; no tested candidate cleared every recorded gate."
        )
    else:
        speedup = winner_score.get("weighted_speedup") if winner_score else None
        ci = winner_score.get("confidence_interval") if winner_score else None
        lines.append(
            f"- Screening selection: `{winner['id']}` with a workload-weighted paired replay "
            f"point estimate of **{_fmt((speedup - 1.0) * 100 if speedup else None)}%** over baseline."
        )
        if ci:
            lines.append(
                "- Multiple-comparison-adjusted stratified bootstrap screening interval: "
                f"**{(ci[0]-1)*100:.2f}% to {(ci[1]-1)*100:.2f}%**."
            )
        else:
            lines.append("- Screening interval: **not computed; evidence is underpowered or exploratory**.")
        cost = winner_score.get("cost_per_million_usd") if winner_score else None
        if cost is not None:
            lines.append(
                f"- Decode-only arithmetic estimate at the supplied hourly rate: "
                f"**${cost:.2f} per million generated tokens**."
            )
            lines.append(
                "- Cost exclusion: prompt prefill, idle capacity, batching, queueing, networking, "
                "service overhead and reliability reserve."
            )
    lines.extend([
        "",
        "## Assurance boundary",
        "",
        "**Numerical replay consistency.** Measured candidates were forced over the same recorded prompt and continuation token IDs and compared with a versioned numerical sketch of each replay-step logit vector.",
        "",
        "The gate requires exact forced-token, top-1, top-2 and ordered top-k identity; rejects non-finite logits; and compares selected logits, moments and deterministic full-vector projections within recorded tolerances.",
        "",
        "This is not complete logit equality, semantic equivalence, downstream model-quality validation, serving reliability validation or proof that the selected configuration is globally optimal. Numerical tolerances and interval coverage have not been calibrated on real supported backends.",
        "",
        "## Numerical-oracle policy",
        "",
        f"- Schema: `{oracle.get('schema', 'missing')}`",
        f"- Exact top-k identity size: `{oracle.get('topk', 'missing')}`",
        f"- Measurement method: `{oracle.get('measurement', 'missing')}`",
        f"- Transport: `{oracle.get('transport', 'missing')}`",
        f"- Representation: `{oracle.get('representation', 'missing')}`",
        f"- Absolute tolerance: `{oracle.get('absolute_tolerance', 'missing')}`",
        f"- Relative tolerance: `{oracle.get('relative_tolerance', 'missing')}`",
        f"- Tolerance calibration: `{oracle.get('tolerance_calibration', 'missing')}`",
        "",
        "## Statistical methodology",
        "",
        f"- Schema: `{stats.get('schema', 'missing')}`",
        f"- Point estimator: `{stats.get('point_estimator', 'missing')}`",
        f"- Bootstrap: `{stats.get('bootstrap', 'missing')}`",
        f"- Bootstrap samples: `{stats.get('bootstrap_samples', 'missing')}`",
        f"- Multiple-comparison method: `{stats.get('multiple_comparisons', 'missing')}`",
        f"- Candidate comparisons: `{stats.get('candidate_comparisons', 'missing')}`",
        f"- Requested family-wise confidence: `{stats.get('familywise_confidence', 'missing')}`",
        f"- Per-candidate interval confidence: `{stats.get('per_candidate_confidence', 'missing')}`",
        f"- Minimum confidence runs per workload: `{stats.get('minimum_confidence_runs_per_workload', 'missing')}`",
        f"- Coverage calibration: `{stats.get('coverage_calibration', 'missing')}`",
        "",
        "Paired throughput ratios are calculated within repeat identity for each workload. The point estimate is the workload-weighted geometric mean of per-workload median paired ratios. Stratified bootstrap resampling occurs independently within each workload, preserving declared workload weights. Bonferroni adjustment controls the requested family-wise confidence across non-baseline candidates.",
        "",
        "This remains screening methodology. It does not include formal power analysis, controlled thermal or power state, robust outlier modelling, independent-machine replication or empirically calibrated interval coverage.",
        "",
        "## Recorded identities",
        "",
        f"- Model family: `{project.get('model_family')}`",
        f"- Sampled model/topology fingerprint: `{project.get('model_fingerprint')}`",
        f"- Runtime fingerprint: `{project.get('runtime_fingerprint')}`",
        f"- Hardware fingerprint: `{project.get('hardware_fingerprint')}`",
        f"- Frozen plan fingerprint: `{project.get('plan_fingerprint')}`",
        f"- Execution fingerprint: `{project.get('execution_fingerprint')}`",
        f"- Native replay cache argument: `{project.get('replay_cap')}`",
        f"- Session evidence root: `{profile.get('evidence_root_sha256', 'missing')}`",
        f"- Profile record digest: `{profile.get('record_sha256', 'missing')}`",
        f"- Qualification context: `{project.get('qualification_context')}` tokens",
        f"- Expected bottleneck: `{plan.get('expected_bottleneck', 'unknown')}`",
        f"- Session: `{session['id']}`",
        f"- Repeats per workload/candidate: `{session['repeats']}`",
        "",
        "## Screening policy",
        "",
        f"- Minimum successful paired runs: {policy['min_runs']}",
        f"- Minimum weighted gain: {policy['min_gain']:.2%}",
        f"- Maximum allowed per-workload regression: {policy['max_regression']:.2%}",
        f"- Requested family-wise confidence: {policy['confidence']:.0%}",
        f"- Confidence lower bound required: {policy['require_confidence']}",
        "",
        "## Workload suite",
        "",
        "| Case | Weight | Context | Requested generated tokens |",
        "|---|---:|---:|---:|",
    ])
    for case in suite.cases:
        lines.append(f"| `{case.id}` | {case.weight:.2f} | {case.context} | {case.tokens} |")
    lines.extend([
        "",
        "## Candidate evidence",
        "",
        "| Candidate | Cleared screening gates | Weighted paired speedup | Worst case | Adjusted interval | Effective tok/s | Reason |",
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
            f"{ci_text} | {_fmt(score.get('effective_tok_s'))} | {_cell(score['reason'])} |"
        )
    lines.extend([
        "",
        "## Selected experimental overrides",
        "",
        "These values describe the screening selection. They are **not authorized for launch** while the profile is non-deployable.",
        "",
        "```text",
    ])
    environment = winner.get("environment") or {}
    if environment:
        lines.extend(f"{key}={value}" for key, value in sorted(environment.items()))
    else:
        lines.append("# baseline: no additional environment overrides")
    lines.extend(["```", "", "## Failed-attempt evidence", ""])
    if not failed_runs:
        lines.append("No failed attempts were retained in this session.")
    else:
        lines.extend([
            "| Run | Candidate | Workload | Repeat | Attempt | Return code | Timed out | Error |",
            "|---|---|---|---:|---:|---:|:---:|---|",
        ])
        for run in failed_runs:
            lines.append(
                f"| `{run['id']}` | `{run['candidate_id']}` | `{run['case_id']}` | "
                f"{run['repeat']} | {run.get('attempt', 'missing')} | {_fmt(run.get('returncode'))} | "
                f"{'yes' if run.get('timed_out') else 'no'} | {_cell(run.get('error'))} |"
            )
    lines.extend([
        "",
        "## Integrity and threat model",
        "",
        "Run records carry individual SHA-256 digests, profiles are sealed, and completed sessions carry an evidence root. The controller uses application-level write-once publication for retained evidence. These controls detect ordinary local edits through the verification path.",
        "",
        "They are not filesystem immutability, remote attestation, a digital signature, an external timestamp or protection against a privileged administrator who can rewrite all related files coherently.",
        "",
        "Any change to the recorded model, runtime, machine, storage topology, frozen plan, environment, workload, numerical policy or statistical policy requires a new experiment.",
        "",
    ])
    return "\n".join(lines)


def write_report(workspace: Workspace, output: Path, profile_id: str | None = None) -> Path:
    output = output.expanduser().resolve()
    text = render_report(workspace, profile_id)
    atomic_write_text(output, text, exclusive=True)
    return output
