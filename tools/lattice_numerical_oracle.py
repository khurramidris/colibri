from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


# Engine integration: emit one compact numerical sketch per forced-token step.
replace_once(
    "c/colibri.c",
    '#include "route_trace.h"                           /* ROUTE_TRACE + .coli_usage, engine-agnostic (#700) */\n',
    '#include "route_trace.h"                           /* ROUTE_TRACE + .coli_usage, engine-agnostic (#700) */\n#include "replay_oracle.h"                         /* opt-in numerical sketch for fixed-token replay */\n',
)
replace_once(
    "c/colibri.c",
    '''/* Fixed-token decode benchmark: prefill all but the prompt's last token, then
 * replay the oracle sequence one token at a time. CPU and CUDA therefore see
 * identical hidden-state inputs even if their argmax predictions differ. */
static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires a non-empty prompt and continuation\\n"); return; }
    kv_alloc(m,nfull+2);
    float *logit=step(m,full,np-1,0); free(logit);
    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0; m->hit_pin=m->hit_ecache=0; m->hit_vk=0;
    profile_reset(m);
    ProfBase pb; prof_base(m,&pb);
    for(int r=0;r<MIR_REPS;r++){ atomic_store(&g_mir_bytes[r],0); atomic_store(&g_mir_nread[r],0); }
    double t0=now_s(); int steps=0;
    for(int i=np-1;i<nfull-1;i++){
        double tf0=g_prof?now_s():0;
        logit=step(m,full+i,1,i); free(logit); steps++;
        if(g_prof){ prof_lat(now_s()-tf0); m->n_fw++; m->n_emit++; }
    }
    double dt=now_s()-t0, tot=m->hits+m->miss;
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
    if(g_prof) prof_report(m,&pb,dt,steps,stdout);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
}
''',
    '''/* Fixed-token decode benchmark: prefill all but the prompt's last token, then
 * replay the oracle sequence one token at a time. CPU and CUDA therefore see
 * identical hidden-state inputs even if their argmax predictions differ.
 *
 * REPLAY_ORACLE=1 adds a compact numerical sketch before each logit vector is
 * freed. It does not claim full-logit equality: the host compares exact top-k
 * identity plus top logits, moments and four full-vector projections using a
 * versioned tolerance policy. */
static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires a non-empty prompt and continuation\\n"); return; }
    int oracle_on=getenv("REPLAY_ORACLE")?atoi(getenv("REPLAY_ORACLE")):0;
    int oracle_topk=getenv("REPLAY_ORACLE_TOPK")?atoi(getenv("REPLAY_ORACLE_TOPK")):8;
    if(oracle_topk<2) oracle_topk=2;
    if(oracle_topk>COLI_REPLAY_ORACLE_TOPK_MAX) oracle_topk=COLI_REPLAY_ORACLE_TOPK_MAX;
    kv_alloc(m,nfull+2);
    float *logit=step(m,full,np-1,0); free(logit);
    m->hits=m->miss=m->ereq=m->gpu_expert_calls=0; m->hit_pin=m->hit_ecache=0; m->hit_vk=0;
    profile_reset(m);
    ProfBase pb; prof_base(m,&pb);
    for(int r=0;r<MIR_REPS;r++){ atomic_store(&g_mir_bytes[r],0); atomic_store(&g_mir_nread[r],0); }
    double t0=now_s(); int steps=0;
    for(int i=np-1;i<nfull-1;i++){
        double tf0=g_prof?now_s():0;
        logit=step(m,full+i,1,i);
        if(oracle_on){
            ColiReplayOracleStep o;
            if(!coli_replay_oracle_step(logit,m->c.vocab,full[i+1],oracle_topk,&o)){
                fprintf(stderr,"REPLAY_ORACLE failed at step %d (invalid/non-finite forced logit)\\n",steps);
                free(logit); exit(2);
            }
            printf("REPLAY_ORACLE_STEP v1 step=%d forced=%d top1=%d top2=%d "
                   "top1_logit=%.9g forced_logit=%.9g margin=%.9g mean=%.12g rms=%.12g "
                   "p0=%.12g p1=%.12g p2=%.12g p3=%.12g topk_ids=%016llx nonfinite=%d\\n",
                   steps,o.forced,o.top1,o.top2,(double)o.top1_logit,(double)o.forced_logit,
                   (double)o.margin,o.mean,o.rms,o.projection[0],o.projection[1],
                   o.projection[2],o.projection[3],(unsigned long long)o.topk_ids_hash,o.nonfinite);
        }
        free(logit); steps++;
        if(g_prof){ prof_lat(now_s()-tf0); m->n_fw++; m->n_emit++; }
    }
    if(oracle_on) printf("REPLAY_ORACLE_SUMMARY v1 steps=%d topk=%d\\n",steps,oracle_topk);
    double dt=now_s()-t0, tot=m->hits+m->miss;
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
    if(g_prof) prof_report(m,&pb,dt,steps,stdout);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif
}
''',
)

# C unit-test gate.
replace_once(
    "c/Makefile",
    '''tests/test_route_trace$(EXE): tests/test_route_trace.c route_trace.h
\t$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
''',
    '''tests/test_route_trace$(EXE): tests/test_route_trace.c route_trace.h
\t$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_replay_oracle$(EXE): tests/test_replay_oracle.c replay_oracle.h
\t$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
''',
)

# Python parser and engine invocation.
replace_once("lattice/colibri.py", "import json\n", "import json\nimport math\n")
replace_once(
    "lattice/colibri.py",
    'from .process import ProcessResult, run_bounded\n',
    'from .process import ProcessResult, run_bounded\nfrom .oracle import ORACLE_POLICY, ORACLE_SCHEMA, validate_oracle\n',
)
replace_once(
    "lattice/colibri.py",
    'LATENCY_RE = re.compile(r"latency p50\\s+([0-9.]+)\\s*ms.*?p99\\s+([0-9.]+)\\s*ms")\n',
    'LATENCY_RE = re.compile(r"latency p50\\s+([0-9.]+)\\s*ms.*?p99\\s+([0-9.]+)\\s*ms")\nORACLE_STEP_PREFIX = "REPLAY_ORACLE_STEP "\nORACLE_SUMMARY_PREFIX = "REPLAY_ORACLE_SUMMARY "\n',
)
replace_once(
    "lattice/colibri.py",
    '''def parse_replay_metrics(output: str) -> dict[str, float | None]:
    speed = SPEED_RE.search(output)
    if not speed:
        raise LatticeError("engine did not emit REPLAY throughput")
    hit = HIT_RE.search(output)
    latency = LATENCY_RE.search(output)
    return {
        "tok_s": float(speed.group(1)),
        "hit_pct": float(hit.group(1)) if hit else None,
        "p50_ms": float(latency.group(1)) if latency else None,
        "p99_ms": float(latency.group(2)) if latency else None,
    }
''',
    '''def _oracle_fields(line: str, prefix: str) -> tuple[str, dict[str, str]]:
    parts = line.strip().split()
    prefix_parts = prefix.strip().split()
    if parts[:len(prefix_parts)] != prefix_parts or len(parts) <= len(prefix_parts):
        raise LatticeError("malformed replay numerical oracle line")
    version = parts[len(prefix_parts)]
    fields: dict[str, str] = {}
    for part in parts[len(prefix_parts) + 1:]:
        if "=" not in part:
            raise LatticeError("malformed replay numerical oracle field")
        key, value = part.split("=", 1)
        if not key or key in fields:
            raise LatticeError("duplicate replay numerical oracle field")
        fields[key] = value
    return version, fields


def parse_replay_oracle(output: str) -> dict[str, Any]:
    raw_steps = [line for line in output.splitlines() if line.startswith(ORACLE_STEP_PREFIX)]
    summaries = [line for line in output.splitlines() if line.startswith(ORACLE_SUMMARY_PREFIX)]
    if not raw_steps or len(summaries) != 1:
        raise LatticeError("engine did not emit a complete replay numerical oracle")
    steps: list[dict[str, Any]] = []
    integer_fields = ("step", "forced", "top1", "top2", "nonfinite")
    float_mapping = {
        "top1_logit": "top1_logit", "forced_logit": "forced_logit", "margin": "margin",
        "mean": "mean", "rms": "rms", "p0": "projection_0", "p1": "projection_1",
        "p2": "projection_2", "p3": "projection_3",
    }
    for expected_step, line in enumerate(raw_steps):
        version, fields = _oracle_fields(line, ORACLE_STEP_PREFIX)
        if version != "v1":
            raise LatticeError(f"unsupported replay numerical oracle version: {version}")
        required = set(integer_fields) | set(float_mapping) | {"topk_ids"}
        if set(fields) != required:
            raise LatticeError("replay numerical oracle step has an unexpected field set")
        try:
            step = {field: int(fields[field]) for field in integer_fields}
            step.update({target: float(fields[source]) for source, target in float_mapping.items()})
        except ValueError as error:
            raise LatticeError("replay numerical oracle step has an invalid number") from error
        step["topk_ids_hash"] = fields["topk_ids"].lower()
        if step["step"] != expected_step or any(not math.isfinite(step[target]) for target in float_mapping.values()):
            raise LatticeError("replay numerical oracle step is not finite or sequential")
        steps.append(step)
    version, summary = _oracle_fields(summaries[0], ORACLE_SUMMARY_PREFIX)
    if version != "v1" or set(summary) != {"steps", "topk"}:
        raise LatticeError("invalid replay numerical oracle summary")
    try:
        summary_steps, topk = int(summary["steps"]), int(summary["topk"])
    except ValueError as error:
        raise LatticeError("invalid replay numerical oracle summary number") from error
    if summary_steps != len(steps) or topk != ORACLE_POLICY["topk"]:
        raise LatticeError("replay numerical oracle summary does not match policy")
    return validate_oracle({
        "schema": ORACLE_SCHEMA,
        "policy": dict(ORACLE_POLICY),
        "steps": steps,
    })


def parse_replay_metrics(output: str) -> dict[str, Any]:
    speed = SPEED_RE.search(output)
    if not speed:
        raise LatticeError("engine did not emit REPLAY throughput")
    hit = HIT_RE.search(output)
    latency = LATENCY_RE.search(output)
    return {
        "tok_s": float(speed.group(1)),
        "hit_pct": float(hit.group(1)) if hit else None,
        "p50_ms": float(latency.group(1)) if latency else None,
        "p99_ms": float(latency.group(2)) if latency else None,
        "oracle": parse_replay_oracle(output),
    }
''',
)
replace_once(
    "lattice/colibri.py",
    ') -> tuple[dict[str, float | None], ProcessResult]:\n',
    ') -> tuple[dict[str, Any], ProcessResult]:\n',
)
replace_once(
    "lattice/colibri.py",
    '''        "REPLAY": "1",
        "PROF": "1",
        "CTX": str(ctx),
''',
    '''        "REPLAY": "1",
        "REPLAY_ORACLE": "1",
        "REPLAY_ORACLE_TOPK": str(ORACLE_POLICY["topk"]),
        "PROF": "1",
        "CTX": str(ctx),
''',
)

# Session and evidence contracts.
replace_once(
    "lattice/experiment.py",
    'from .integrity import validate_session_evidence\n',
    'from .integrity import validate_session_evidence\nfrom .oracle import ORACLE_POLICY\n',
)
replace_once(
    "lattice/experiment.py",
    '''            "qualification_context": context.qualification_context,
            "repeats": repeats,
''',
    '''            "qualification_context": context.qualification_context,
            "oracle_policy": dict(ORACLE_POLICY),
            "repeats": repeats,
''',
)
replace_once(
    "lattice/integrity.py",
    'from .colibri import SAFE_TUNABLE_KEYS\n',
    'from .colibri import SAFE_TUNABLE_KEYS\nfrom .oracle import ORACLE_POLICY, validate_oracle\n',
)
replace_once(
    "lattice/integrity.py",
    '''    if session.get("qualification_context") != project.get("qualification_context"):
        raise LatticeError("session qualification context does not match project")
''',
    '''    if session.get("qualification_context") != project.get("qualification_context"):
        raise LatticeError("session qualification context does not match project")
    if session.get("oracle_policy") != ORACLE_POLICY:
        raise LatticeError("session replay numerical oracle policy does not match Lattice")
''',
)
replace_once(
    "lattice/integrity.py",
    '''            tok_s = metrics.get("tok_s") if isinstance(metrics, dict) else None
            if (not isinstance(tok_s, (int, float)) or isinstance(tok_s, bool) or tok_s <= 0
                    or run.get("returncode") != 0 or run.get("output_truncated") is not False):
                raise LatticeError(f"successful run is incomplete: {run.get('id')}")
''',
    '''            tok_s = metrics.get("tok_s") if isinstance(metrics, dict) else None
            if (not isinstance(tok_s, (int, float)) or isinstance(tok_s, bool) or tok_s <= 0
                    or run.get("returncode") != 0 or run.get("output_truncated") is not False):
                raise LatticeError(f"successful run is incomplete: {run.get('id')}")
            validate_oracle(metrics.get("oracle"))
''',
)

# Replace recommendation logic with numerical-gated selection.
Path("lattice/recommend.py").write_text('''from __future__ import annotations

import math
from typing import Any

from .common import LatticeError, short_id, utc_now
from .integrity import validate_session_evidence
from .oracle import ORACLE_POLICY, compare_oracles, validate_oracle
from .stats import CandidateScore, score_candidate
from .suite import WorkloadSuite, parse_suite
from .workspace import Workspace


def _successful_samples(runs: list[dict[str, Any]], candidate_id: str) -> dict[str, list[float]]:
    values: dict[str, list[tuple[int, float]]] = {}
    seen: set[tuple[str, int]] = set()
    for run in runs:
        if run.get("candidate_id") != candidate_id or run.get("status") != "success":
            continue
        case_id = run.get("case_id")
        repeat = run.get("repeat")
        metrics = run.get("metrics") or {}
        tok_s = metrics.get("tok_s")
        if not isinstance(case_id, str) or not isinstance(repeat, int) or not isinstance(tok_s, (int, float)) or tok_s <= 0:
            continue
        key = (case_id, repeat)
        if key in seen:
            raise LatticeError(f"duplicate successful evidence for {candidate_id}/{case_id}/repeat-{repeat}")
        seen.add(key)
        values.setdefault(case_id, []).append((repeat, float(tok_s)))
    return {case_id: [value for _, value in sorted(samples)] for case_id, samples in values.items()}


def _successful_run_map(runs: list[dict[str, Any]], candidate_id: str) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        if run.get("candidate_id") != candidate_id or run.get("status") != "success":
            continue
        key = (run["case_id"], run["repeat"])
        if key in result:
            raise LatticeError(f"duplicate successful evidence for {candidate_id}/{key[0]}/repeat-{key[1]}")
        validate_oracle((run.get("metrics") or {}).get("oracle"))
        result[key] = run
    return result


def _baseline_oracle_stability(baseline_runs: dict[tuple[str, int], dict[str, Any]]) -> None:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for (case_id, _repeat), run in sorted(baseline_runs.items()):
        by_case.setdefault(case_id, []).append(run)
    for case_id, runs in by_case.items():
        reference = (runs[0]["metrics"] or {})["oracle"]
        for run in runs[1:]:
            mismatch = compare_oracles(reference, (run["metrics"] or {})["oracle"])
            if mismatch:
                raise LatticeError(f"baseline numerical oracle is unstable for {case_id}: {mismatch}")


def _candidate_oracle_mismatch(
    baseline_runs: dict[tuple[str, int], dict[str, Any]],
    candidate_runs: dict[tuple[str, int], dict[str, Any]],
) -> str | None:
    for key, run in sorted(candidate_runs.items()):
        baseline = baseline_runs.get(key)
        if baseline is None:
            continue
        mismatch = compare_oracles((baseline["metrics"] or {})["oracle"], (run["metrics"] or {})["oracle"])
        if mismatch:
            return f"numerical oracle mismatch for {key[0]}/repeat-{key[1]}: {mismatch}"
    return None


def _ineligible_oracle_score(candidate_id: str, reason: str) -> CandidateScore:
    return CandidateScore(candidate_id, False, reason, None, None, None, None, None, None, {})


def evaluate_session(
    workspace: Workspace,
    session_id: str,
    *,
    min_runs: int = 2,
    min_gain: float = 0.03,
    max_regression: float = 0.05,
    confidence: float = 0.90,
    require_confidence: bool = False,
    hourly_cost_usd: float | None = None,
) -> dict[str, Any]:
    if not 1 <= min_runs <= 20:
        raise LatticeError("min_runs must be between 1 and 20")
    if not 0 <= min_gain <= 1:
        raise LatticeError("min_gain must be between 0 and 1")
    if not 0 <= max_regression <= 1:
        raise LatticeError("max_regression must be between 0 and 1")
    if not 0.5 < confidence < 1:
        raise LatticeError("confidence must be between 0.5 and 1")
    if hourly_cost_usd is not None and (not math.isfinite(hourly_cost_usd) or hourly_cost_usd < 0):
        raise LatticeError("hourly_cost_usd must be finite and non-negative")
    project = workspace.load_project()
    suite: WorkloadSuite = parse_suite(project["suite"])
    session = workspace.load_session(session_id)
    if session.get("repeats", 0) < min_runs:
        raise LatticeError("promotion requires more runs than the session contains")
    runs = workspace.list_runs(session_id)
    candidate_defs = validate_session_evidence(workspace, project, suite, session, runs)
    baseline = _successful_samples(runs, "baseline")
    baseline_runs = _successful_run_map(runs, "baseline")
    _baseline_oracle_stability(baseline_runs)
    weights = {case.id: case.weight for case in suite.cases}
    hourly = suite.hourly_cost_usd if hourly_cost_usd is None else hourly_cost_usd
    scores: list[CandidateScore] = []
    for candidate_id in candidate_defs:
        if candidate_id == "baseline":
            continue
        candidate_runs = _successful_run_map(runs, candidate_id)
        mismatch = _candidate_oracle_mismatch(baseline_runs, candidate_runs)
        if mismatch:
            scores.append(_ineligible_oracle_score(candidate_id, mismatch))
            continue
        scores.append(score_candidate(
            candidate_id,
            weights,
            baseline,
            _successful_samples(runs, candidate_id),
            min_runs=min_runs,
            min_gain=min_gain,
            max_regression=max_regression,
            confidence=confidence,
            require_confidence=require_confidence,
            hourly_cost_usd=hourly,
        ))
    eligible = [score for score in scores if score.eligible and score.weighted_speedup is not None]
    winner_score = max(eligible, key=lambda score: score.weighted_speedup or 0.0) if eligible else None
    winner_id = winner_score.candidate_id if winner_score else "baseline"
    policy = {
        "min_runs": min_runs,
        "min_gain": min_gain,
        "max_regression": max_regression,
        "confidence": confidence,
        "require_confidence": require_confidence,
        "hourly_cost_usd": hourly,
    }
    return {
        "session_id": session_id,
        "winner": candidate_defs[winner_id],
        "winner_score": None if winner_score is None else winner_score.as_dict(),
        "baseline_retained": winner_score is None,
        "selection_policy": policy,
        "oracle_policy": dict(ORACLE_POLICY),
        "model_fingerprint": project["model_fingerprint"],
        "runtime_fingerprint": project["runtime_fingerprint"],
        "hardware_fingerprint": project["hardware_fingerprint"],
        "execution_fingerprint": project["execution_fingerprint"],
        "qualification_context": project["qualification_context"],
        "suite_fingerprint": suite.fingerprint,
        "evidence_root_sha256": session["evidence_root_sha256"],
        "scores": [score.as_dict() for score in scores],
    }


def recommend(workspace: Workspace, session_id: str, **policy: Any) -> tuple[dict[str, Any], list[CandidateScore]]:
    evaluation = evaluate_session(workspace, session_id, **policy)
    seed_policy = evaluation["selection_policy"]
    profile_seed = {
        "session_id": session_id,
        "winner": evaluation["winner"]["id"],
        "policy": seed_policy,
        "oracle_policy": evaluation["oracle_policy"],
        "evidence_root_sha256": evaluation["evidence_root_sha256"],
    }
    profile = {
        "schema_version": 1,
        "id": short_id("profile", profile_seed),
        "created_at": utc_now(),
        **evaluation,
    }
    workspace.write_profile(profile)
    scores = [CandidateScore(
        candidate_id=item["candidate_id"],
        eligible=item["eligible"],
        reason=item["reason"],
        weighted_speedup=item["weighted_speedup"],
        effective_tok_s=item["effective_tok_s"],
        cost_per_million_usd=item["cost_per_million_usd"],
        worst_case_regression=item["worst_case_regression"],
        ci_low=None if item["confidence_interval"] is None else item["confidence_interval"][0],
        ci_high=None if item["confidence_interval"] is None else item["confidence_interval"][1],
        per_case=item["per_case"],
    ) for item in evaluation["scores"]]
    return profile, scores
''', encoding="utf-8")

replace_once(
    "lattice/verify.py",
    '''            "policy": evaluation["selection_policy"],
            "evidence_root_sha256": evaluation["evidence_root_sha256"],
''',
    '''            "policy": evaluation["selection_policy"],
            "oracle_policy": evaluation["oracle_policy"],
            "evidence_root_sha256": evaluation["evidence_root_sha256"],
''',
)
replace_once(
    "lattice/verify.py",
    '''            "profile_evidence_root": profile.get("evidence_root_sha256") == evaluation.get("evidence_root_sha256"),
            "profile_recomputed": canonical_json({k: profile.get(k) for k in evaluation}) == canonical_json(evaluation),
''',
    '''            "profile_evidence_root": profile.get("evidence_root_sha256") == evaluation.get("evidence_root_sha256"),
            "profile_oracle_policy": profile.get("oracle_policy") == evaluation.get("oracle_policy"),
            "profile_recomputed": canonical_json({k: profile.get(k) for k in evaluation}) == canonical_json(evaluation),
''',
)

# Test fixtures now emit and retain numerical sketches.
replace_once(
    "lattice/tests/test_colibri.py",
    '''        metrics = parse_replay_metrics(
            "REPLAY decode: 16 tokens | 2.50 tok/s\\nexpert hit 70.5%\\nlatency p50 10.2 ms p99 18.4 ms"
        )
        self.assertEqual(metrics["tok_s"], 2.5)
        self.assertEqual(metrics["hit_pct"], 70.5)
''',
    '''        oracle_line = (
            "REPLAY_ORACLE_STEP v1 step=0 forced=3 top1=2 top2=4 "
            "top1_logit=3 forced_logit=1.25 margin=0.5 mean=0.8 rms=1.9 "
            "p0=1 p1=-2 p2=3 p3=-4 topk_ids=0123456789abcdef nonfinite=0"
        )
        metrics = parse_replay_metrics(
            oracle_line + "\\nREPLAY_ORACLE_SUMMARY v1 steps=1 topk=8\\n"
            "REPLAY decode: 16 tokens | 2.50 tok/s\\nexpert hit 70.5%\\nlatency p50 10.2 ms p99 18.4 ms"
        )
        self.assertEqual(metrics["tok_s"], 2.5)
        self.assertEqual(metrics["hit_pct"], 70.5)
        self.assertEqual(metrics["oracle"]["steps"][0]["top1"], 2)
''',
)

# Recommend test helper and mismatch coverage.
recommend_test = Path("lattice/tests/test_recommend.py")
text = recommend_test.read_text(encoding="utf-8")
text = text.replace(
    'from lattice.evidence import session_evidence_root\n',
    'from lattice.evidence import session_evidence_root\nfrom lattice.oracle import ORACLE_POLICY, ORACLE_SCHEMA\n',
)
text = text.replace(
    '''            "qualification_context": 4096,
            "repeats": repeats,
''',
    '''            "qualification_context": 4096,
            "oracle_policy": dict(ORACLE_POLICY),
            "repeats": repeats,
''',
)
text = text.replace(
    '''    def _run(self, candidate: str, repeat: int, tok_s: float, replay_hash: str, suffix: str = "") -> dict:
        environment = {} if candidate == "baseline" else {"PIPE": "1"}
        return {
''',
    '''    def _oracle(self, *, top1: int = 2, forced_logit: float = 1.25) -> dict:
        return {
            "schema": ORACLE_SCHEMA,
            "policy": dict(ORACLE_POLICY),
            "steps": [{
                "step": 0, "forced": 3, "top1": top1, "top2": 4, "nonfinite": 0,
                "top1_logit": 3.0, "forced_logit": forced_logit, "margin": 0.5,
                "mean": 0.8, "rms": 1.9,
                "projection_0": 1.0, "projection_1": -2.0,
                "projection_2": 3.0, "projection_3": -4.0,
                "topk_ids_hash": "0123456789abcdef",
            }],
        }

    def _run(self, candidate: str, repeat: int, tok_s: float, replay_hash: str, suffix: str = "", *, top1: int = 2) -> dict:
        environment = {} if candidate == "baseline" else {"PIPE": "1"}
        return {
''',
)
text = text.replace('''            "metrics": {"tok_s": tok_s},
''', '''            "metrics": {"tok_s": tok_s, "oracle": self._oracle(top1=top1)},
''')
insert = '''
    def test_numerical_oracle_mismatch_retains_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            ws, replay_hash = self._workspace(Path(directory))
            session = self._session(ws, replay_hash, repeats=2)
            for repeat in range(2):
                baseline = self._run("baseline", repeat, 1.0, replay_hash)
                fast = self._run("fast", repeat, 2.0, replay_hash, top1=9)
                for run in (baseline, fast):
                    ws.write_run(run)
                    session["run_ids"].append(run["id"])
            self._finalize(ws, session)
            profile, scores = recommend(ws, "session-a", min_runs=2)
            self.assertTrue(profile["baseline_retained"])
            self.assertEqual(profile["winner"]["id"], "baseline")
            self.assertIn("numerical oracle mismatch", scores[0].reason)
'''
needle = '\n    def test_duplicate_success_is_rejected(self):\n'
if needle not in text:
    raise SystemExit("test_recommend.py: insertion point missing")
text = text.replace(needle, insert + needle)
recommend_test.write_text(text, encoding="utf-8")

# Resume mock includes a valid oracle and session policy is generated by production.
replace_once(
    "lattice/tests/test_resume.py",
    'from lattice.process import ProcessResult\n',
    'from lattice.process import ProcessResult\nfrom lattice.oracle import ORACLE_POLICY, ORACLE_SCHEMA\n',
)
replace_once(
    "lattice/tests/test_resume.py",
    '''            def replay(_context, *, replay_path, candidate, ctx, timeout):
                return {"tok_s": 1.0, "hit_pct": 50.0, "p50_ms": 1.0, "p99_ms": 2.0}, result("replay")
''',
    '''            def replay(_context, *, replay_path, candidate, ctx, timeout):
                oracle = {
                    "schema": ORACLE_SCHEMA,
                    "policy": dict(ORACLE_POLICY),
                    "steps": [{
                        "step": 0, "forced": 3, "top1": 2, "top2": 4, "nonfinite": 0,
                        "top1_logit": 3.0, "forced_logit": 1.25, "margin": 0.5,
                        "mean": 0.8, "rms": 1.9,
                        "projection_0": 1.0, "projection_1": -2.0,
                        "projection_2": 3.0, "projection_3": -4.0,
                        "topk_ids_hash": "0123456789abcdef",
                    }],
                }
                return {"tok_s": 1.0, "hit_pct": 50.0, "p50_ms": 1.0, "p99_ms": 2.0, "oracle": oracle}, result("replay")
''',
)

# Protocol fixture emits eight deterministic oracle steps.
replace_once(
    "lattice/tests/test_integration.py",
    '''print(f'REPLAY decode: 8 tokens | {speed:.2f} tok/s')
print('expert hit 70.0%')
''',
    '''for step, forced in enumerate(range(4, 12)):
    print(
        f'REPLAY_ORACLE_STEP v1 step={step} forced={forced} top1=2 top2=4 '
        'top1_logit=3 forced_logit=1.25 margin=0.5 mean=0.8 rms=1.9 '
        'p0=1 p1=-2 p2=3 p3=-4 topk_ids=0123456789abcdef nonfinite=0'
    )
print('REPLAY_ORACLE_SUMMARY v1 steps=8 topk=8')
print(f'REPLAY decode: 8 tokens | {speed:.2f} tok/s')
print('expert hit 70.0%')
''',
)

# Version and documentation surface.
replace_once('lattice/__init__.py', '__version__ = "0.2.0-alpha.1"\n', '__version__ = "0.3.0-alpha.1"\n')
