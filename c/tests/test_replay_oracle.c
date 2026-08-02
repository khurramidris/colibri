#include <assert.h>
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
    assert(a.nonfinite==0);
    assert(close_enough(a.forced_logit,1.25,1e-7));
    assert(close_enough(a.top1_logit,3.0,1e-7));
    assert(close_enough(a.margin,0.5,1e-7));
    assert(a.topk_ids_hash==b.topk_ids_hash);
    for(int p=0;p<COLI_REPLAY_ORACLE_PROJECTIONS;p++)
        assert(a.projection[p]==b.projection[p]);

    float mutated[6]; memcpy(mutated,logits,sizeof(mutated)); mutated[0]+=0.75f;
    assert(coli_replay_oracle_step(mutated,6,3,4,&changed));
    int projection_changed=0;
    for(int p=0;p<COLI_REPLAY_ORACLE_PROJECTIONS;p++)
        if(changed.projection[p]!=a.projection[p]) projection_changed=1;
    assert(projection_changed);
    assert(changed.topk_ids_hash==a.topk_ids_hash); /* tail change keeps top-k identity */

    float nonfinite[6]; memcpy(nonfinite,logits,sizeof(nonfinite)); nonfinite[0]=NAN;
    assert(coli_replay_oracle_step(nonfinite,6,3,4,&changed));
    assert(changed.nonfinite==1);

    assert(!coli_replay_oracle_step(logits,1,0,4,&changed));
    assert(!coli_replay_oracle_step(logits,6,9,4,&changed));
    puts("replay oracle: ok");
    return 0;
}
