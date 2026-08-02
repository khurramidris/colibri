from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "c/colibri.c",
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
    '''/* Fixed-token decode benchmark: prefill all but the prompt's last token, then
 * replay the oracle sequence one token at a time. CPU and CUDA therefore see
 * identical hidden-state inputs even if their argmax predictions differ.
 *
 * Numerical validation deliberately runs in a SECOND replay pass. Computing
 * top-k identities and full-vector projections inside the timed loop would
 * contaminate the throughput that the oracle is meant to gate. */
static void run_replay(Model *m, const int *full, int nfull, int np){
    if(np<2||nfull<=np){ fprintf(stderr,"REPLAY requires a non-empty prompt and continuation\\n"); return; }
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
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    profile_print(m,dt);
    if(g_prof) prof_report(m,&pb,dt,steps,stdout);
#ifdef COLI_CUDA
    if(m->gpu_expert_count) printf("CUDA expert tier: %d resident experts (%.2f GB) | %llu calls served from VRAM\\n",
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
            fprintf(stderr,"REPLAY_ORACLE failed at step %d (invalid/non-finite forced logit)\\n",oracle_steps);
            free(logit); exit(2);
        }
        printf("REPLAY_ORACLE_STEP v1 step=%d forced=%d top1=%d top2=%d "
               "top1_logit=%.9g forced_logit=%.9g margin=%.9g mean=%.12g rms=%.12g "
               "p0=%.12g p1=%.12g p2=%.12g p3=%.12g topk_ids=%016llx nonfinite=%d\\n",
               oracle_steps,o.forced,o.top1,o.top2,(double)o.top1_logit,(double)o.forced_logit,
               (double)o.margin,o.mean,o.rms,o.projection[0],o.projection[1],
               o.projection[2],o.projection[3],(unsigned long long)o.topk_ids_hash,o.nonfinite);
        free(logit); oracle_steps++;
    }
    if(oracle_steps!=steps){
        fprintf(stderr,"REPLAY_ORACLE step count differs from measured replay\\n"); exit(2);
    }
    printf("REPLAY_ORACLE_SUMMARY v1 steps=%d topk=%d measurement=separate_replay_pass\\n",
           oracle_steps,oracle_topk);
}
''',
)

replace_once(
    "lattice/oracle.py",
    '''    "topk": 8,
    "absolute_tolerance": 0.005,
''',
    '''    "topk": 8,
    "measurement": "separate_replay_pass",
    "absolute_tolerance": 0.005,
''',
)
replace_once(
    "lattice/colibri.py",
    '''    if version != "v1" or set(summary) != {"steps", "topk"}:
        raise LatticeError("invalid replay numerical oracle summary")
''',
    '''    if version != "v1" or set(summary) != {"steps", "topk", "measurement"}:
        raise LatticeError("invalid replay numerical oracle summary")
    if summary["measurement"] != ORACLE_POLICY["measurement"]:
        raise LatticeError("replay numerical oracle measurement method mismatch")
''',
)

for path in ("lattice/tests/test_colibri.py", "lattice/tests/test_integration.py"):
    target=Path(path)
    text=target.read_text(encoding="utf-8")
    old="REPLAY_ORACLE_SUMMARY v1 steps="
    text=text.replace(old, old)
    text=text.replace(" topk=8\\n", " topk=8 measurement=separate_replay_pass\\n")
    text=text.replace(" topk=8')", " topk=8 measurement=separate_replay_pass')")
    target.write_text(text,encoding="utf-8")

replace_once(
    "lattice/report.py",
    '''        f"- Exact top-k identity size: `{oracle.get('topk', 'missing')}`",
        f"- Absolute tolerance: `{oracle.get('absolute_tolerance', 'missing')}`",
''',
    '''        f"- Exact top-k identity size: `{oracle.get('topk', 'missing')}`",
        f"- Measurement method: `{oracle.get('measurement', 'missing')}`",
        f"- Absolute tolerance: `{oracle.get('absolute_tolerance', 'missing')}`",
''',
)
replace_once(
    "LATTICE.md",
    '''Before each replay-step logit vector is freed, Colibri records a compact numerical sketch. Lattice rejects candidates whose exact token identities differ or whose numeric sketch exceeds the versioned tolerance.
''',
    '''Colibri first performs an uninstrumented timed replay. It then resets KV state and performs a separate numerical-validation replay, so sketch computation and output cannot contaminate the measured throughput. Lattice rejects candidates whose exact token identities differ or whose numeric sketch exceeds the versioned tolerance.
''',
)
