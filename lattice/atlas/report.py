from __future__ import annotations

from typing import Any


def render_report(evidence: dict[str, Any]) -> str:
    manifest = evidence["manifest"]
    decision = evidence["decision"]
    cache = decision["cache_prefetch"]
    best = cache["best_practical_cache"]
    oracle = cache["best_oracle_cache"]
    predictor = cache["best_prefetch_predictor"]
    reduction = cache["best_transfer_reduction"]
    ternary = decision["ternary"]
    lines = [
        "# Gemma Expert Atlas — Phase 0 Evidence Report",
        "",
        "## Identity",
        "",
        f"- Model: `{manifest['model']}`",
        f"- Pinned model revision: `{manifest['model_revision']}`",
        f"- Checkpoint SHA-256: `{manifest['checkpoint_sha256']}`",
        f"- Runtime: `{manifest['runtime']}` at `{manifest['runtime_revision']}`",
        f"- Workload SHA-256: `{manifest['workload_sha256']}`",
        f"- Architecture: {manifest['n_layers']} layers, {manifest['n_experts']} experts/layer, top-{manifest['top_k']}",
        f"- Expert bytes used in the cost model: {manifest['expert_bytes']:,}",
        f"- Pre-registered target cache: {manifest['target_cache_capacity']} experts/layer = {manifest['target_cache_capacity'] * manifest['n_layers'] * manifest['expert_bytes']:,} bytes",
        f"- Trace SHA-256: `{evidence['inputs']['trace_sha256']}`",
        f"- Manifest SHA-256: `{evidence['inputs']['manifest_sha256']}`",
        "",
        "## Held-out routing result",
        "",
        f"- Requests: {evidence['metrics']['requests']} total; split assignment is retained in evidence.",
        f"- Adjacent-token expert Jaccard median: {evidence['metrics']['adjacent_token_jaccard']['median']!r}",
        f"- Best practical cache: `{best['policy']}` at {best['capacity']} experts/layer, hit rate **{best['hit_rate']:.2%}**.",
        f"- Belady oracle at {oracle['capacity']} experts/layer: **{oracle['hit_rate']:.2%}**.",
        f"- Best modelled exposed-transfer reduction: **{reduction['reduction']:.2%}** on `{reduction.get('profile', 'n/a')}`.",
        f"- Best non-oracle prefetch predictor: `{predictor['predictor']}`, recall **{predictor['recall']:.2%}**, overfetch **{predictor['overfetch_fraction']:.2%}**.",
        "",
        "## Decision",
        "",
        f"**Cache/prefetch: {cache['decision'].upper()}**",
        "",
    ]
    for name, passed in cache["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines += ["", "## Ternary expert sensitivity", ""]
    if ternary["status"] == "not_measured":
        lines.append("No ternary probe evidence was supplied. Ternary experts remain an untested hypothesis.")
    else:
        lines.append(f"Safe expert fraction under configured local-output thresholds: **{ternary['safe_fraction']:.2%}**.")
        lines.append(f"Gate: **{'PASS' if ternary['pass'] else 'FAIL'}**")
    lines += [
        "",
        "## Scientific limitations",
        "",
        "- This report measures routing locality and simulates cache/prefetch policies; it is not an end-to-end speed benchmark.",
        "- Transfer time is a disclosed cost model. Actual overlap, launch overhead, allocator behaviour, PCIe contention, and kernel execution require hardware A/B runs.",
        "- Belady is an unattainable oracle and is used only to measure remaining policy headroom.",
        "- Static policies are trained only on the retained training requests; reported evaluation is held out by request identity.",
        "- Router traces must come from the unmodified native router. A predictor may stage weights but must never replace router semantics.",
        "- Local expert-output similarity is not proof of model-level quality after ternarization. Full perplexity and downstream evaluations remain mandatory.",
        "",
        f"Overall Phase 0 status: **{decision['overall'].upper()}**",
        "",
    ]
    return "\n".join(lines)
