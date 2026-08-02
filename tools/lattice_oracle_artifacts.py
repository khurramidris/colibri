from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


Path("c/replay_oracle.h").write_text(r'''#ifndef COLI_REPLAY_ORACLE_H
#define COLI_REPLAY_ORACLE_H

#include <math.h>
#include <stdint.h>
#include <stddef.h>

#define COLI_REPLAY_ORACLE_SCHEMA "coli-replay-oracle/2"
#define COLI_REPLAY_ORACLE_TOPK_MAX 16
#define COLI_REPLAY_ORACLE_PROJECTIONS 4

typedef struct {
    int forced;
    int top1;
    int top2;
    int nonfinite;
    int topk;
    int topk_ids[COLI_REPLAY_ORACLE_TOPK_MAX];
    float forced_logit;
    float top1_logit;
    float margin;
    double mean;
    double rms;
    double projection[COLI_REPLAY_ORACLE_PROJECTIONS];
} ColiReplayOracleStep;

static inline uint64_t coli_replay_oracle_mix64(uint64_t x){
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

static inline int coli_replay_oracle_better(float value, int id, float other, int other_id){
    return value > other || (value == other && id < other_id);
}

/* Build a compact numerical sketch of one replay step.
 *
 * This is not a proof of full-logit equality. It records exact ordered top-k
 * token IDs, selected logits, distribution moments, deterministic full-vector
 * projections and non-finite count. The O(vocab * topk) work is opt-in and is
 * executed in a separate replay pass outside the published decode timing. */
static inline int coli_replay_oracle_step(const float *logits, int vocab, int forced,
                                           int requested_topk,
                                           ColiReplayOracleStep *out){
    if(!logits || !out || vocab < 2 || forced < 0 || forced >= vocab) return 0;
    int k=requested_topk;
    if(k<2) k=2;
    if(k>COLI_REPLAY_ORACLE_TOPK_MAX) k=COLI_REPLAY_ORACLE_TOPK_MAX;
    if(k>vocab) k=vocab;
    int ids[COLI_REPLAY_ORACLE_TOPK_MAX];
    float values[COLI_REPLAY_ORACLE_TOPK_MAX];
    for(int j=0;j<k;j++){ ids[j]=-1; values[j]=-INFINITY; }
    double sum=0.0, sumsq=0.0, proj[COLI_REPLAY_ORACLE_PROJECTIONS]={0,0,0,0};
    int finite_count=0, nonfinite=0;
    for(int i=0;i<vocab;i++){
        float value=logits[i];
        if(!isfinite(value)){ nonfinite++; continue; }
        finite_count++;
        double dv=(double)value;
        sum+=dv; sumsq+=dv*dv;
        for(int p=0;p<COLI_REPLAY_ORACLE_PROJECTIONS;p++){
            uint64_t key=coli_replay_oracle_mix64((uint64_t)(i+1) ^
                (UINT64_C(0x9e3779b97f4a7c15)*(uint64_t)(p+1)));
            proj[p] += (key & 1) ? dv : -dv;
        }
        int pos=k;
        for(int j=0;j<k;j++){
            if(ids[j]<0 || coli_replay_oracle_better(value,i,values[j],ids[j])){
                pos=j; break;
            }
        }
        if(pos<k){
            for(int j=k-1;j>pos;j--){ values[j]=values[j-1]; ids[j]=ids[j-1]; }
            values[pos]=value; ids[pos]=i;
        }
    }
    if(finite_count < 2 || ids[0] < 0 || ids[1] < 0 || !isfinite(logits[forced])) return 0;
    out->forced=forced;
    out->top1=ids[0];
    out->top2=ids[1];
    out->nonfinite=nonfinite;
    out->topk=k;
    for(int j=0;j<k;j++) out->topk_ids[j]=ids[j];
    for(int j=k;j<COLI_REPLAY_ORACLE_TOPK_MAX;j++) out->topk_ids[j]=-1;
    out->forced_logit=logits[forced];
    out->top1_logit=values[0];
    out->margin=values[0]-values[1];
    out->mean=sum/(double)finite_count;
    out->rms=sqrt(sumsq/(double)finite_count);
    double scale=sqrt((double)finite_count);
    for(int p=0;p<COLI_REPLAY_ORACLE_PROJECTIONS;p++) out->projection[p]=proj[p]/scale;
    return 1;
}

#endif
''', encoding="utf-8")

Path("c/tests/test_replay_oracle.c").write_text(r'''#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../replay_oracle.h"

static int close_enough(double a, double b, double tol){
    return fabs(a-b) <= tol;
}

int main(void){
    const float logits[] = {-2.0f, 0.5f, 3.0f, 1.25f, 2.5f, -0.25f};
    ColiReplayOracleStep a, b, changed;
    memset(&a,0,sizeof(a)); memset(&b,0,sizeof(b)); memset(&changed,0,sizeof(changed));
    assert(coli_replay_oracle_step(logits,6,3,4,&a));
    assert(coli_replay_oracle_step(logits,6,3,4,&b));
    assert(a.forced==3);
    assert(a.top1==2);
    assert(a.top2==4);
    assert(a.topk==4);
    assert(a.topk_ids[0]==2 && a.topk_ids[1]==4 && a.topk_ids[2]==3 && a.topk_ids[3]==1);
    assert(!memcmp(a.topk_ids,b.topk_ids,sizeof(a.topk_ids)));
    assert(a.nonfinite==0);
    assert(close_enough(a.forced_logit,1.25,1e-7));
    assert(close_enough(a.top1_logit,3.0,1e-7));
    assert(close_enough(a.margin,0.5,1e-7));
    for(int p=0;p<COLI_REPLAY_ORACLE_PROJECTIONS;p++)
        assert(a.projection[p]==b.projection[p]);

    float mutated[6]; memcpy(mutated,logits,sizeof(mutated)); mutated[0]+=0.75f;
    assert(coli_replay_oracle_step(mutated,6,3,4,&changed));
    int projection_changed=0;
    for(int p=0;p<COLI_REPLAY_ORACLE_PROJECTIONS;p++)
        if(changed.projection[p]!=a.projection[p]) projection_changed=1;
    assert(projection_changed);
    assert(!memcmp(changed.topk_ids,a.topk_ids,sizeof(a.topk_ids)));

    float nonfinite[6]; memcpy(nonfinite,logits,sizeof(nonfinite)); nonfinite[0]=NAN;
    assert(coli_replay_oracle_step(nonfinite,6,3,4,&changed));
    assert(changed.nonfinite==1);

    assert(!coli_replay_oracle_step(logits,1,0,4,&changed));
    assert(!coli_replay_oracle_step(logits,6,9,4,&changed));
    puts("replay oracle: ok");
    return 0;
}
''', encoding="utf-8")

old_replay = r'''/* Fixed-token decode benchmark: prefill all but the prompt's last token, then
 * replay the oracle sequence one token at a time. CPU and CUDA therefore see
 * identical hidden-state inputs even if their argmax predictions differ.
 *
 * Numerical validation deliberately runs in a SECOND replay pass. Computing
 * top-k identities and full-vector projections inside the timed loop would
 * contaminate the throughput that the oracle is meant to gate. */
static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires a non-empty prompt and continuation\n"); return; }
    int oracle_on=getenv("REPLAY_ORACLE")?atoi(getenv("REPLAY_ORACLE")):0;
    int oracle_topk=getenv("REPLAY_ORACLE_TOPK")?atoi(getenv("REPLAY_ORACLE_TOPK")):8;
    if(oracle_topk<2) oracle_topk=2;
    if(oracle_topk>COLI_REPLAY_ORACLE_TOPK_MAX) oracle_topk=COLI_REPLAY_ORACLE_TOPK_MAX;

    /* Phase 1: uninstrumented performance replay. */
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
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
    if(g_prof) prof_report(m,&pb,dt,steps,stdout);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif

    if(!oracle_on) return;

    /* Phase 2: reset KV state and replay again for the numerical sketch. This
     * phase is outside the published decode time and cannot alter its ranking. */
    kv_alloc(m,nfull+2);
    logit=step(m,full,np-1,0); free(logit);
    int oracle_steps=0;
    for(int i=np-1;i<nfull-1;i++){
        logit=step(m,full+i,1,i);
        ColiReplayOracleStep o;
        if(!coli_replay_oracle_step(logit,m->c.vocab,full[i+1],oracle_topk,&o)){
            fprintf(stderr,"REPLAY_ORACLE failed at step %d (invalid/non-finite forced logit)\n",oracle_steps);
            free(logit); exit(2);
        }
        printf("REPLAY_ORACLE_STEP v1 step=%d forced=%d top1=%d top2=%d "
               "top1_logit=%.9g forced_logit=%.9g margin=%.9g mean=%.12g rms=%.12g "
               "p0=%.12g p1=%.12g p2=%.12g p3=%.12g topk_ids=%016llx nonfinite=%d\n",
               oracle_steps,o.forced,o.top1,o.top2,(double)o.top1_logit,(double)o.forced_logit,
               (double)o.margin,o.mean,o.rms,o.projection[0],o.projection[1],
               o.projection[2],o.projection[3],(unsigned long long)o.topk_ids_hash,o.nonfinite);
        free(logit); oracle_steps++;
    }
    if(oracle_steps!=steps){
        fprintf(stderr,"REPLAY_ORACLE step count differs from measured replay\n"); exit(2);
    }
    printf("REPLAY_ORACLE_SUMMARY v1 steps=%d topk=%d measurement=separate_replay_pass\n",
           oracle_steps,oracle_topk);
}
'''
new_replay = r'''/* Fixed-token decode benchmark: prefill all but the prompt's last token, then
 * replay the oracle sequence one token at a time. CPU and CUDA therefore see
 * identical hidden-state inputs even if their argmax predictions differ.
 * Numerical validation is a separate replay pass and writes to a controller-
 * supplied private artifact path so long continuations cannot overflow stdout. */
static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires a non-empty prompt and continuation\n"); return; }
    int oracle_on=getenv("REPLAY_ORACLE")?atoi(getenv("REPLAY_ORACLE")):0;
    int oracle_topk=getenv("REPLAY_ORACLE_TOPK")?atoi(getenv("REPLAY_ORACLE_TOPK")):8;
    const char *oracle_path=getenv("REPLAY_ORACLE_OUT");
    if(oracle_topk<2) oracle_topk=2;
    if(oracle_topk>COLI_REPLAY_ORACLE_TOPK_MAX) oracle_topk=COLI_REPLAY_ORACLE_TOPK_MAX;
    if(oracle_on && (!oracle_path || !*oracle_path)){
        fprintf(stderr,"REPLAY_ORACLE_OUT is required when REPLAY_ORACLE=1\n"); exit(2);
    }

    /* Phase 1: uninstrumented performance replay. */
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
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
    if(g_prof) prof_report(m,&pb,dt,steps,stdout);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\n",
        m->gpu_expert_count,m->gpu_expert_bytes/1e9,(unsigned long long)m->gpu_expert_calls);
    if(g_cuda_enabled) cuda_stats_print();
#endif

    if(!oracle_on) return;
    FILE *oracle_file=fopen(oracle_path,"wb");
    if(!oracle_file){ perror("REPLAY_ORACLE_OUT"); exit(2); }

    /* Phase 2: reset KV state and replay again for the numerical sketch. */
    kv_alloc(m,nfull+2);
    logit=step(m,full,np-1,0); free(logit);
    int oracle_steps=0;
    for(int i=np-1;i<nfull-1;i++){
        logit=step(m,full+i,1,i);
        ColiReplayOracleStep o;
        if(!coli_replay_oracle_step(logit,m->c.vocab,full[i+1],oracle_topk,&o)){
            fprintf(stderr,"REPLAY_ORACLE failed at step %d (invalid/non-finite forced logit)\n",oracle_steps);
            free(logit); fclose(oracle_file); exit(2);
        }
        fprintf(oracle_file,"STEP\tv2\t%d\t%d\t%d\t%d\t%.9g\t%.9g\t%.9g\t%.12g\t%.12g"
                            "\t%.12g\t%.12g\t%.12g\t%.12g\t",
                o.forced,o.top1,o.top2,o.nonfinite,(double)o.top1_logit,(double)o.forced_logit,
                (double)o.margin,o.mean,o.rms,o.projection[0],o.projection[1],
                o.projection[2],o.projection[3]);
        for(int j=0;j<o.topk;j++) fprintf(oracle_file,j?",%d":"%d",o.topk_ids[j]);
        fputc('\n',oracle_file);
        free(logit); oracle_steps++;
    }
    if(oracle_steps!=steps){
        fprintf(stderr,"REPLAY_ORACLE step count differs from measured replay\n");
        fclose(oracle_file); exit(2);
    }
    fprintf(oracle_file,"SUMMARY\tv2\t%d\t%d\tseparate_replay_pass\tprivate_file\n",
            oracle_steps,oracle_topk);
    if(fflush(oracle_file)!=0 || ferror(oracle_file) || fclose(oracle_file)!=0){
        fprintf(stderr,"failed to publish replay oracle artifact\n"); exit(2);
    }
    printf("REPLAY_ORACLE_WRITTEN v2 steps=%d topk=%d measurement=separate_replay_pass transport=private_file\n",
           oracle_steps,oracle_topk);
}
'''
replace_once("c/colibri.c", old_replay, new_replay)

Path("lattice/oracle.py").write_text('''from __future__ import annotations

import math
from typing import Any

from .common import LatticeError

ORACLE_SCHEMA = "coli-replay-oracle/2"
ORACLE_POLICY = {
    "schema": ORACLE_SCHEMA,
    "topk": 8,
    "measurement": "separate_replay_pass",
    "transport": "private_file",
    "representation": "compact_rows",
    "absolute_tolerance": 0.005,
    "relative_tolerance": 0.0005,
}

FORCED, TOP1, TOP2, NONFINITE = 0, 1, 2, 3
TOP1_LOGIT, FORCED_LOGIT, MARGIN, MEAN, RMS = 4, 5, 6, 7, 8
PROJECTION_0, PROJECTION_1, PROJECTION_2, PROJECTION_3 = 9, 10, 11, 12
TOPK_IDS = 13
ROW_LENGTH = 14
FLOAT_INDEXES = tuple(range(TOP1_LOGIT, PROJECTION_3 + 1))


def validate_oracle(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LatticeError("replay numerical oracle is missing")
    if value.get("schema") != ORACLE_SCHEMA:
        raise LatticeError("unsupported replay numerical oracle schema")
    if value.get("policy") != ORACLE_POLICY:
        raise LatticeError("replay numerical oracle policy mismatch")
    steps = value.get("steps")
    if not isinstance(steps, list) or not steps:
        raise LatticeError("replay numerical oracle has no steps")
    for index, row in enumerate(steps):
        if not isinstance(row, list) or len(row) != ROW_LENGTH:
            raise LatticeError(f"replay numerical oracle row {index} is invalid")
        for field in (FORCED, TOP1, TOP2, NONFINITE):
            item = row[field]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise LatticeError(f"replay numerical oracle row {index} has invalid identity")
        for field in FLOAT_INDEXES:
            item = row[field]
            if (isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))):
                raise LatticeError(f"replay numerical oracle row {index} has invalid numeric sketch")
        topk = row[TOPK_IDS]
        if (not isinstance(topk, list) or len(topk) != ORACLE_POLICY["topk"]
                or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in topk)
                or len(set(topk)) != len(topk)):
            raise LatticeError(f"replay numerical oracle row {index} has invalid top-k identity")
        if row[TOP1] != topk[0] or row[TOP2] != topk[1]:
            raise LatticeError(f"replay numerical oracle row {index} top-k order is inconsistent")
    return value


def _close(reference: float, candidate: float, *, absolute: float, relative: float) -> bool:
    return math.isclose(reference, candidate, abs_tol=absolute, rel_tol=relative)


def compare_oracles(reference: Any, candidate: Any) -> str | None:
    """Compare compact numerical sketches under the versioned policy."""
    reference = validate_oracle(reference)
    candidate = validate_oracle(candidate)
    if len(reference["steps"]) != len(candidate["steps"]):
        return "step count differs"
    absolute = float(ORACLE_POLICY["absolute_tolerance"])
    relative = float(ORACLE_POLICY["relative_tolerance"])
    for index, (base, trial) in enumerate(zip(reference["steps"], candidate["steps"])):
        if base[NONFINITE] or trial[NONFINITE]:
            return f"step {index} contains non-finite logits"
        for field, label in ((FORCED, "forced"), (TOP1, "top1"), (TOP2, "top2"), (TOPK_IDS, "topk_ids")):
            if base[field] != trial[field]:
                return f"step {index} {label} differs"
        for field in FLOAT_INDEXES:
            if not _close(float(base[field]), float(trial[field]), absolute=absolute, relative=relative):
                return f"step {index} numeric field {field} differs beyond tolerance"
    return None
''', encoding="utf-8")

replace_once("lattice/colibri.py", "import sys\n", "import sys\nimport tempfile\n")
replace_once(
    "lattice/colibri.py",
    '''ORACLE_STEP_PREFIX = "REPLAY_ORACLE_STEP "
ORACLE_SUMMARY_PREFIX = "REPLAY_ORACLE_SUMMARY "
''',
    '''ORACLE_WRITTEN_RE = re.compile(
    r"^REPLAY_ORACLE_WRITTEN v2 steps=(\\d+) topk=(\\d+) "
    r"measurement=separate_replay_pass transport=private_file$",
    re.MULTILINE,
)
MAX_ORACLE_BYTES = 4 * 1024 * 1024
''',
)
old_parser = '''def _oracle_fields(line: str, prefix: str) -> tuple[str, dict[str, str]]:
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
    if version != "v1" or set(summary) != {"steps", "topk", "measurement"}:
        raise LatticeError("invalid replay numerical oracle summary")
    if summary["measurement"] != ORACLE_POLICY["measurement"]:
        raise LatticeError("replay numerical oracle measurement method mismatch")
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
'''
new_parser = '''def read_replay_oracle_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise LatticeError("engine did not create a regular replay oracle artifact")
    if path.stat().st_size > MAX_ORACLE_BYTES:
        raise LatticeError("replay numerical oracle artifact exceeded the evidence limit")
    with path.open("rb") as stream:
        data = stream.read(MAX_ORACLE_BYTES + 1)
    if len(data) > MAX_ORACLE_BYTES:
        raise LatticeError("replay numerical oracle artifact exceeded the evidence limit")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LatticeError("replay numerical oracle artifact is not UTF-8") from error


def parse_replay_oracle(artifact: str) -> dict[str, Any]:
    steps: list[list[Any]] = []
    summary: list[str] | None = None
    for line_number, line in enumerate(artifact.splitlines(), start=1):
        parts = line.split("\\t")
        if not parts or not parts[0]:
            continue
        if parts[0] == "STEP":
            if len(parts) != 16 or parts[1] != "v2":
                raise LatticeError(f"malformed replay oracle step at line {line_number}")
            try:
                identities = [int(value) for value in parts[2:6]]
                numerics = [float(value) for value in parts[6:15]]
                topk_ids = [int(value) for value in parts[15].split(",") if value]
            except ValueError as error:
                raise LatticeError(f"invalid replay oracle number at line {line_number}") from error
            steps.append([*identities, *numerics, topk_ids])
        elif parts[0] == "SUMMARY":
            if summary is not None or len(parts) != 6 or parts[1] != "v2":
                raise LatticeError("invalid replay numerical oracle summary")
            summary = parts
        else:
            raise LatticeError(f"unknown replay oracle record at line {line_number}")
    if not steps or summary is None:
        raise LatticeError("engine did not emit a complete replay numerical oracle artifact")
    try:
        summary_steps, topk = int(summary[2]), int(summary[3])
    except ValueError as error:
        raise LatticeError("invalid replay numerical oracle summary number") from error
    if (summary_steps != len(steps) or topk != ORACLE_POLICY["topk"]
            or summary[4] != ORACLE_POLICY["measurement"]
            or summary[5] != ORACLE_POLICY["transport"]):
        raise LatticeError("replay numerical oracle summary does not match policy")
    return validate_oracle({
        "schema": ORACLE_SCHEMA,
        "policy": dict(ORACLE_POLICY),
        "steps": steps,
    })


def parse_replay_metrics(output: str, oracle_artifact: str) -> dict[str, Any]:
    speed = SPEED_RE.search(output)
    marker = ORACLE_WRITTEN_RE.search(output)
    if not speed:
        raise LatticeError("engine did not emit REPLAY throughput")
    if not marker:
        raise LatticeError("engine did not confirm replay oracle publication")
    oracle = parse_replay_oracle(oracle_artifact)
    if int(marker.group(1)) != len(oracle["steps"]) or int(marker.group(2)) != ORACLE_POLICY["topk"]:
        raise LatticeError("replay oracle publication marker does not match artifact")
    hit = HIT_RE.search(output)
    latency = LATENCY_RE.search(output)
    return {
        "tok_s": float(speed.group(1)),
        "hit_pct": float(hit.group(1)) if hit else None,
        "p50_ms": float(latency.group(1)) if latency else None,
        "p99_ms": float(latency.group(2)) if latency else None,
        "oracle": oracle,
    }
'''
replace_once("lattice/colibri.py", old_parser, new_parser)
old_run = '''    env = dict(context.base_environment)
    env.update(candidate)
    env.update({
        "REF": str(replay_path),
        "REF_FORCE": "1",
        "REPLAY": "1",
        "REPLAY_ORACLE": "1",
        "REPLAY_ORACLE_TOPK": str(ORACLE_POLICY["topk"]),
        "PROF": "1",
        "CTX": str(ctx),
    })
    env.pop("PROMPT", None); env.pop("TOKENS", None)
    cap = context.plan.get("tiers", {}).get("ram", {}).get("cache_slots_per_layer", 0)
    command = [str(context.engine), str(int(cap or 0))]
    result = run_bounded(command, env=env, timeout=timeout, cwd=context.c_dir)
    output = f"{result.stdout}\\n{result.stderr}"
    if result.timed_out:
        raise LatticeError("replay timed out")
    if result.returncode:
        raise LatticeError(f"replay failed with exit code {result.returncode}: {output[-2000:]}")
    if result.output_truncated:
        raise LatticeError("replay output exceeded the evidence limit")
    return parse_replay_metrics(output), result
'''
new_run = '''    env = dict(context.base_environment)
    env.update(candidate)
    env.update({
        "REF": str(replay_path),
        "REF_FORCE": "1",
        "REPLAY": "1",
        "REPLAY_ORACLE": "1",
        "REPLAY_ORACLE_TOPK": str(ORACLE_POLICY["topk"]),
        "PROF": "1",
        "CTX": str(ctx),
    })
    env.pop("PROMPT", None); env.pop("TOKENS", None)
    cap = context.plan.get("tiers", {}).get("ram", {}).get("cache_slots_per_layer", 0)
    command = [str(context.engine), str(int(cap or 0))]
    with tempfile.TemporaryDirectory(prefix="lattice-oracle-") as directory:
        artifact_path = Path(directory) / "oracle.tsv"
        env["REPLAY_ORACLE_OUT"] = str(artifact_path)
        result = run_bounded(command, env=env, timeout=timeout, cwd=context.c_dir)
        output = f"{result.stdout}\\n{result.stderr}"
        if result.timed_out:
            raise LatticeError("replay timed out")
        if result.returncode:
            raise LatticeError(f"replay failed with exit code {result.returncode}: {output[-2000:]}")
        if result.output_truncated:
            raise LatticeError("replay output exceeded the evidence limit")
        oracle_artifact = read_replay_oracle_file(artifact_path)
    return parse_replay_metrics(output, oracle_artifact), result
'''
replace_once("lattice/colibri.py", old_run, new_run)

Path("lattice/tests/test_oracle.py").write_text('''from __future__ import annotations

import copy
import unittest

from lattice.common import LatticeError
from lattice.oracle import ORACLE_POLICY, ORACLE_SCHEMA, compare_oracles, validate_oracle


def row(*, top1: int = 2, forced_logit: float = 1.25, topk: list[int] | None = None) -> list:
    return [
        3, top1, 4, 0,
        3.0, forced_logit, 0.5, 0.8, 1.9,
        1.0, -2.0, 3.0, -4.0,
        topk or [top1, 4, 3, 1, 5, 6, 7, 8],
    ]


def fixture() -> dict:
    return {"schema": ORACLE_SCHEMA, "policy": dict(ORACLE_POLICY), "steps": [row()]}


class OracleTests(unittest.TestCase):
    def test_identical_oracles_match(self):
        value = fixture()
        self.assertIsNone(compare_oracles(value, copy.deepcopy(value)))

    def test_small_numeric_drift_is_tolerated(self):
        base = fixture(); trial = copy.deepcopy(base)
        trial["steps"][0][11] += 0.001
        self.assertIsNone(compare_oracles(base, trial))

    def test_exact_topk_or_large_numeric_drift_is_rejected(self):
        base = fixture(); trial = copy.deepcopy(base)
        trial["steps"][0][13][3] = 9
        self.assertIn("topk_ids differs", compare_oracles(base, trial) or "")
        trial = copy.deepcopy(base)
        trial["steps"][0][5] += 0.1
        self.assertIn("beyond tolerance", compare_oracles(base, trial) or "")

    def test_nonfinite_or_malformed_oracle_is_rejected(self):
        value = fixture(); value["steps"][0][3] = 1
        self.assertIn("non-finite", compare_oracles(value, value) or "")
        value = fixture(); value["steps"][0][13] = [2, 4]
        with self.assertRaisesRegex(LatticeError, "top-k identity"):
            validate_oracle(value)


if __name__ == "__main__":
    unittest.main()
''', encoding="utf-8")

# Compact fixtures used by recommendation and recovery tests.
for path in ("lattice/tests/test_recommend.py", "lattice/tests/test_resume.py"):
    target=Path(path)
    text=target.read_text(encoding="utf-8")
    old='''            "steps": [{
                "step": 0, "forced": 3, "top1": top1, "top2": 4, "nonfinite": 0,
                "top1_logit": 3.0, "forced_logit": forced_logit, "margin": 0.5,
                "mean": 0.8, "rms": 1.9,
                "projection_0": 1.0, "projection_1": -2.0,
                "projection_2": 3.0, "projection_3": -4.0,
                "topk_ids_hash": "0123456789abcdef",
            }],
'''
    if path.endswith("test_resume.py"):
        old='''                    "steps": [{
                        "step": 0, "forced": 3, "top1": 2, "top2": 4, "nonfinite": 0,
                        "top1_logit": 3.0, "forced_logit": 1.25, "margin": 0.5,
                        "mean": 0.8, "rms": 1.9,
                        "projection_0": 1.0, "projection_1": -2.0,
                        "projection_2": 3.0, "projection_3": -4.0,
                        "topk_ids_hash": "0123456789abcdef",
                    }],
'''
        new='''                    "steps": [[
                        3, 2, 4, 0, 3.0, 1.25, 0.5, 0.8, 1.9,
                        1.0, -2.0, 3.0, -4.0, [2, 4, 3, 1, 5, 6, 7, 8],
                    ]],
'''
    else:
        new='''            "steps": [[
                3, top1, 4, 0, 3.0, forced_logit, 0.5, 0.8, 1.9,
                1.0, -2.0, 3.0, -4.0, [top1, 4, 3, 1, 5, 6, 7, 8],
            ]],
'''
    if text.count(old)!=1:
        raise SystemExit(f"{path}: oracle fixture replacement count {text.count(old)}")
    target.write_text(text.replace(old,new),encoding="utf-8")

# Parser and large-artifact tests.
test=Path("lattice/tests/test_colibri.py")
text=test.read_text(encoding="utf-8")
text=text.replace('''    parse_calibration,
    parse_replay_metrics,
)''','''    parse_calibration,
    parse_replay_metrics,
    parse_replay_oracle,
    read_replay_oracle_file,
)''')
old='''        oracle_line = (
            "REPLAY_ORACLE_STEP v1 step=0 forced=3 top1=2 top2=4 "
            "top1_logit=3 forced_logit=1.25 margin=0.5 mean=0.8 rms=1.9 "
            "p0=1 p1=-2 p2=3 p3=-4 topk_ids=0123456789abcdef nonfinite=0"
        )
        metrics = parse_replay_metrics(
            oracle_line + "\\nREPLAY_ORACLE_SUMMARY v1 steps=1 topk=8 measurement=separate_replay_pass\\n"
            "REPLAY decode: 16 tokens | 2.50 tok/s\\nexpert hit 70.5%\\nlatency p50 10.2 ms p99 18.4 ms"
        )
        self.assertEqual(metrics["tok_s"], 2.5)
        self.assertEqual(metrics["hit_pct"], 70.5)
        self.assertEqual(metrics["oracle"]["steps"][0]["top1"], 2)
'''
new='''        artifact = (
            "STEP\\tv2\\t3\\t2\\t4\\t0\\t3\\t1.25\\t0.5\\t0.8\\t1.9"
            "\\t1\\t-2\\t3\\t-4\\t2,4,3,1,5,6,7,8\\n"
            "SUMMARY\\tv2\\t1\\t8\\tseparate_replay_pass\\tprivate_file\\n"
        )
        output = (
            "REPLAY_ORACLE_WRITTEN v2 steps=1 topk=8 measurement=separate_replay_pass transport=private_file\\n"
            "REPLAY decode: 16 tokens | 2.50 tok/s\\nexpert hit 70.5%\\nlatency p50 10.2 ms p99 18.4 ms"
        )
        metrics = parse_replay_metrics(output, artifact)
        self.assertEqual(metrics["tok_s"], 2.5)
        self.assertEqual(metrics["hit_pct"], 70.5)
        self.assertEqual(metrics["oracle"]["steps"][0][1], 2)

    def test_long_oracle_uses_separate_bounded_artifact(self):
        step = (
            "STEP\\tv2\\t3\\t2\\t4\\t0\\t3\\t1.25\\t0.5\\t0.8\\t1.9"
            "\\t1\\t-2\\t3\\t-4\\t2,4,3,1,5,6,7,8\\n"
        )
        artifact = step * 2048 + "SUMMARY\\tv2\\t2048\\t8\\tseparate_replay_pass\\tprivate_file\\n"
        self.assertGreater(len(artifact.encode()), 65536)
        parsed = parse_replay_oracle(artifact)
        self.assertEqual(len(parsed["steps"]), 2048)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.tsv"
            path.write_text(artifact, encoding="utf-8")
            self.assertEqual(read_replay_oracle_file(path), artifact)
'''
if text.count(old)!=1:
    raise SystemExit("test_colibri parser block not found")
test.write_text(text.replace(old,new),encoding="utf-8")

# Fake integration engine writes the detailed artifact and only a marker to stdout.
test=Path("lattice/tests/test_integration.py")
text=test.read_text(encoding="utf-8")
old='''for step, forced in enumerate(range(4, 12)):
    print(
        f'REPLAY_ORACLE_STEP v1 step={step} forced={forced} top1=2 top2=4 '
        'top1_logit=3 forced_logit=1.25 margin=0.5 mean=0.8 rms=1.9 '
        'p0=1 p1=-2 p2=3 p3=-4 topk_ids=0123456789abcdef nonfinite=0'
    )
print('REPLAY_ORACLE_SUMMARY v1 steps=8 topk=8 measurement=separate_replay_pass')
print(f'REPLAY decode: 8 tokens | {speed:.2f} tok/s')
'''
new='''oracle_path=os.environ.get('REPLAY_ORACLE_OUT')
if not oracle_path:
    print('missing REPLAY_ORACLE_OUT', file=sys.stderr)
    raise SystemExit(2)
with open(oracle_path, 'w', encoding='utf-8') as oracle:
    for forced in range(4, 12):
        oracle.write(
            f'STEP\\tv2\\t{forced}\\t2\\t4\\t0\\t3\\t1.25\\t0.5\\t0.8\\t1.9'
            '\\t1\\t-2\\t3\\t-4\\t2,4,3,1,5,6,7,8\\n'
        )
    oracle.write('SUMMARY\\tv2\\t8\\t8\\tseparate_replay_pass\\tprivate_file\\n')
print('REPLAY_ORACLE_WRITTEN v2 steps=8 topk=8 measurement=separate_replay_pass transport=private_file')
print(f'REPLAY decode: 8 tokens | {speed:.2f} tok/s')
'''
if text.count(old)!=1:
    raise SystemExit("integration oracle output block not found")
text=text.replace(old,new).replace('"coli-replay-oracle/1"','"coli-replay-oracle/2"')
test.write_text(text,encoding="utf-8")

replace_once('lattice/__init__.py', '__version__ = "0.3.0-alpha.1"\n', '__version__ = "0.4.0-alpha.1"\n')
