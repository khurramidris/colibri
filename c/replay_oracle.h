#ifndef COLI_REPLAY_ORACLE_H
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
