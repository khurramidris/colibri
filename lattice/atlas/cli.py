from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

from .cache import evaluate_caches
from .decision import evaluate_decision
from .io import (
    atomic_write_json,
    atomic_write_text,
    load_manifest,
    load_routes,
    canonical_json,
    sha256_bytes,
    sha256_file,
    write_jsonl,
)
from .metrics import route_metrics
from .model import AtlasError, AtlasManifest
from .prefetch import evaluate_prefetch
from .report import render_report
from .split import stable_request_split
from .synthetic import generate_routes
from .ternary import load_ternary_probes, summarize_ternary
from .transfer import evaluate_transfer
from .verify import verify_output
from . import __version__


def cmd_analyze(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    trace_path = Path(args.trace).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    identities = {
        "model_revision": manifest.model_revision,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "runtime_revision": manifest.runtime_revision,
        "workload_sha256": manifest.workload_sha256,
    }
    for name, value in identities.items():
        if value.upper().startswith("REPLACE_") or value.lower() in {"main", "master", "latest", "head"}:
            raise AtlasError(f"{name} is not pinned: {value!r}")
    records = load_routes(trace_path, manifest)
    train, test, assignment = stable_request_split(records, manifest.train_fraction)
    metrics = route_metrics(records, manifest)
    cache_rows = evaluate_caches(train, test, manifest)
    prefetch_rows = evaluate_prefetch(train, test, manifest)
    transfer_rows = evaluate_transfer(cache_rows, manifest.transfer_profiles)
    ternary_summary = None
    ternary_sha = None
    if args.ternary:
        ternary_path = Path(args.ternary).expanduser().resolve()
        ternary_sha = sha256_file(ternary_path)
        ternary_summary = summarize_ternary(load_ternary_probes(ternary_path))
    decision = evaluate_decision(
        cache_rows, transfer_rows, prefetch_rows, ternary_summary, manifest.gates,
        target_capacity=manifest.target_cache_capacity,
    )
    evidence_body = {
        "schema": "lattice.atlas.evidence.v1",
        "tool": {
            "name": "gemma-expert-atlas",
            "version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "manifest": manifest.as_dict(),
        "inputs": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "trace_path": str(trace_path),
            "trace_sha256": sha256_file(trace_path),
            "ternary_path": str(Path(args.ternary).expanduser().resolve()) if args.ternary else None,
            "ternary_sha256": ternary_sha,
        },
        "split": assignment,
        "metrics": metrics,
        "cache": cache_rows,
        "prefetch": prefetch_rows,
        "transfer": transfer_rows,
        "ternary": ternary_summary or {"status": "not_measured"},
        "decision": decision,
    }
    evidence = dict(evidence_body)
    evidence["record_sha256"] = sha256_bytes(canonical_json(evidence_body))
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "atlas-evidence.json"
    report_path = output_dir / "atlas-report.md"
    receipt_path = output_dir / "atlas-receipt.json"
    atomic_write_json(evidence_path, evidence, exclusive=not args.force)
    atomic_write_text(report_path, render_report(evidence), exclusive=not args.force)
    receipt = {
        "schema": "lattice.atlas.receipt.v1",
        "record_sha256": evidence["record_sha256"],
        "evidence_sha256": sha256_file(evidence_path),
        "report_sha256": sha256_file(report_path),
    }
    atomic_write_json(receipt_path, receipt, exclusive=not args.force)
    print(json.dumps({
        "status": decision["overall"],
        "cache_prefetch": decision["cache_prefetch"]["decision"],
        "evidence": str(output_dir / "atlas-evidence.json"),
        "report": str(report_path),
        "receipt": str(receipt_path),
        "record_sha256": evidence["record_sha256"],
    }, indent=2))
    return 0 if decision["cache_prefetch"]["decision"] == "go" else 3


def cmd_synthetic(args: argparse.Namespace) -> int:
    manifest = AtlasManifest.from_dict(json.loads(Path(args.manifest).read_text("utf-8")))
    records = generate_routes(
        seed=args.seed,
        requests=args.requests,
        tokens=args.tokens,
        n_layers=manifest.n_layers,
        n_experts=manifest.n_experts,
        top_k=manifest.top_k,
        hot_experts=args.hot_experts,
        hot_probability=args.hot_probability,
        include_prefill=not args.no_prefill,
    )
    output = Path(args.output).expanduser().resolve()
    write_jsonl(output, (record.as_dict() for record in records), exclusive=not args.force)
    print(output)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify_output(
        Path(args.output_dir).expanduser().resolve(),
        require_inputs=args.require_inputs,
    )
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lattice.atlas",
        description="Held-out MoE routing locality, cache, prefetch and ternary feasibility analysis.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="produce write-once Phase 0 evidence from a canonical route trace")
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--trace", required=True)
    analyze.add_argument("--ternary", help="optional lattice.ternary.probe.v1 JSONL")
    analyze.add_argument("--output-dir", default="atlas-output")
    analyze.add_argument("--force", action="store_true", help="replace existing outputs; exploratory use only")
    analyze.set_defaults(func=cmd_analyze)

    synthetic = sub.add_parser("synthetic", help="generate a deterministic trace for falsification and CI")
    synthetic.add_argument("--manifest", required=True)
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--requests", type=int, default=12)
    synthetic.add_argument("--tokens", type=int, default=24)
    synthetic.add_argument("--hot-experts", type=int, default=24)
    synthetic.add_argument("--hot-probability", type=float, default=0.85)
    synthetic.add_argument("--no-prefill", action="store_true")
    synthetic.add_argument("--force", action="store_true")
    synthetic.set_defaults(func=cmd_synthetic)

    verify = sub.add_parser("verify", help="verify evidence, receipt, report, and available source inputs")
    verify.add_argument("--output-dir", default="atlas-output")
    verify.add_argument("--require-inputs", action="store_true")
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except AtlasError as error:
        print(f"atlas: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
