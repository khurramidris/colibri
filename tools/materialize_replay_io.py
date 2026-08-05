#!/usr/bin/env python3
"""Materialize replay I/O accounting into committed C/Makefile source.

This script is intentionally exact-anchor based. It refuses to write when the
reviewed source shape has drifted, preventing a partial or misplaced patch.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C_PATH = ROOT / "c" / "colibri.c"
MAKE_PATH = ROOT / "c" / "Makefile"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one anchor, found {count}")
    return text.replace(old, new, 1)


source = C_PATH.read_text(encoding="utf-8")
source = replace_once(
    source,
    '#include "replay_oracle.h"                         /* opt-in numerical sketch for fixed-token replay */\n',
    '#include "replay_oracle.h"                         /* opt-in numerical sketch for fixed-token replay */\n'
    '#include "replay_io.h"                             /* exact-window Linux process I/O accounting */\n',
    "replay_io include",
)

old_replay = '''    double t0=now_s(); int steps=0;
    for(int i=np-1;i<nfull-1;i++){
        logit=step(m,full+i,1,i); free(logit); steps++;
    }
    double dt=now_s()-t0, tot=m->hits+m->miss;
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    g_prof=saved_prof;
'''
new_replay = '''    /* Bracket the exact profiler-free decode window. g_prof_io is a lightweight
     * engine demand counter and remains active while stage profiling is disabled. */
    ReplayIoSnapshot io_before={0}, io_after={0};
    int io_before_ok=replay_io_snapshot(&io_before);
    int64_t logical_before=atomic_load_explicit(&g_prof_io,memory_order_relaxed);
    double t0=now_s(); int steps=0;
    for(int i=np-1;i<nfull-1;i++){
        logit=step(m,full+i,1,i); free(logit); steps++;
    }
    double dt=now_s()-t0, tot=m->hits+m->miss;
    int64_t logical_bytes=atomic_load_explicit(&g_prof_io,memory_order_relaxed)-logical_before;
    if(logical_bytes<0) logical_bytes=0;
    int io_after_ok=replay_io_snapshot(&io_after);
    ReplayIoDelta io_delta={0};
    int io_ok=io_before_ok && io_after_ok && replay_io_delta(&io_before,&io_after,&io_delta);
    printf("REPLAY decode: %d tokens in %.3fs | %.2f tok/s | expert hit %.1f%%\\n",
        steps,dt,steps/dt,tot?100.0*m->hits/tot:0.0);
    if(io_ok){
        double ratio=logical_bytes>0?(double)io_delta.read_bytes/(double)logical_bytes:0.0;
        double logical_per_token=steps>0?(double)logical_bytes/(double)steps:0.0;
        double physical_per_token=steps>0?(double)io_delta.read_bytes/(double)steps:0.0;
        printf("REPLAY_IO v1 available=1 scope=teacher_forced_decode "
               "logical_expert_bytes=%lld process_read_bytes=%llu process_rchar=%llu "
               "process_read_syscalls=%llu physical_to_logical=%.6f "
               "logical_bytes_per_token=%.3f physical_bytes_per_token=%.3f\\n",
               (long long)logical_bytes,
               (unsigned long long)io_delta.read_bytes,
               (unsigned long long)io_delta.rchar,
               (unsigned long long)io_delta.read_syscalls,
               ratio,logical_per_token,physical_per_token);
    } else {
        printf("REPLAY_IO v1 available=0 scope=teacher_forced_decode "
               "logical_expert_bytes=%lld reason=proc_self_io_unavailable\\n",
               (long long)logical_bytes);
    }
    g_prof=saved_prof;
'''
source = replace_once(source, old_replay, new_replay, "run_replay window")
C_PATH.write_text(source, encoding="utf-8")

makefile = MAKE_PATH.read_text(encoding="utf-8")
makefile = replace_once(
    makefile,
    "colibri$(EXE): colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h quant.h sample.h kv_persist.h telemetry.h route_trace.h $(CUDA_OBJ) $(METAL_OBJ) $(VK_OBJ) $(VK_SPV) .build-config\n",
    "colibri$(EXE): colibri.c st.h uring.h json.h tok.h tok_unicode.h compat.h grammar.h quant.h sample.h kv_persist.h telemetry.h route_trace.h replay_oracle.h replay_io.h $(CUDA_OBJ) $(METAL_OBJ) $(VK_OBJ) $(VK_SPV) .build-config\n",
    "colibri dependency",
)
makefile = replace_once(
    makefile,
    "tests/test_replay_oracle$(EXE): tests/test_replay_oracle.c replay_oracle.h\n\t$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)\n",
    "tests/test_replay_oracle$(EXE): tests/test_replay_oracle.c replay_oracle.h\n\t$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)\n\n"
    "tests/test_replay_io$(EXE): tests/test_replay_io.c replay_io.h\n"
    "\t$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)\n",
    "replay I/O test rule",
)
MAKE_PATH.write_text(makefile, encoding="utf-8")
print("materialized replay I/O accounting")
