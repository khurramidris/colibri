/* Real K3 layer lifecycle proof.
 *
 * This research executable includes Colibri's production Kimi K3 engine in the
 * same translation unit, then exercises its real Layer/W/KDA/AttnRes/dense-MLP
 * code with a memory-dial lifecycle:
 *
 *   resident prefix -> keep weights
 *   streamed suffix -> reload before the layer, execute, release immediately
 *
 * The current gate deliberately reloads from the ordinary Colibri repacked
 * safetensors container. Gate 1 separately proves the one-read, double-buffered
 * packed-trunk reader. Combining the two is the next gate.
 */
#define _GNU_SOURCE
#define main cerno_k3_original_main
#include "../../c/kimi_k3.c"
#undef main

static uint64_t g_stream_layer_loads = 0;
static uint64_t g_stream_layer_releases = 0;

static void stream_w_release(W *w){
    if(!w) return;
#ifdef COLI_VULKAN
    if(w->vk){
        /* This executable forces K3_VK=0 before model_init. Refuse rather than
         * silently leaking or freeing a backend object through the wrong API. */
        fprintf(stderr,"stream lifecycle: unexpected Vulkan-resident tensor\n");
        exit(1);
    }
#endif
    free(w->f); free(w->q8); free(w->q4); free(w->s);
    memset(w,0,sizeof(*w));
}

static void stream_layer_release(Layer *l){
    if(!l) return;
    int kda=l->kda, sparse=l->sparse;
    free(l->in_ln); free(l->post_ln); free(l->attn_sw); free(l->mlp_sw);
    if(kda){
        stream_w_release(&l->a.q); stream_w_release(&l->a.k);
        stream_w_release(&l->a.v); stream_w_release(&l->a.o);
        stream_w_release(&l->a.g);
        free(l->a.conv_q); free(l->a.conv_k); free(l->a.conv_v);
        free(l->a.fa); free(l->a.fb); free(l->a.bp);
        free(l->a.dt); free(l->a.A); free(l->a.onw);
    } else {
        stream_w_release(&l->m.qa); stream_w_release(&l->m.qb);
        stream_w_release(&l->m.kva); stream_w_release(&l->m.kvb);
        stream_w_release(&l->m.o); stream_w_release(&l->m.g);
        free(l->m.qa_ln); free(l->m.kva_ln);
    }
    if(sparse){
        free(l->moe.router); free(l->moe.rbias); free(l->moe.lat_norm);
        stream_w_release(&l->moe.lat_down); stream_w_release(&l->moe.lat_up);
        stream_w_release(&l->moe.sh_gate); stream_w_release(&l->moe.sh_up);
        stream_w_release(&l->moe.sh_down);
    } else {
        stream_w_release(&l->d_gate); stream_w_release(&l->d_up);
        stream_w_release(&l->d_down);
    }
    memset(l,0,sizeof(*l));
    l->kda=kda; l->sparse=sparse;
    g_stream_layer_releases++;
}

static void stream_layer_load(Model *m, Layer *l, int i, int bits, int mbits){
    Cfg *c=&m->c;
    int kda=l->kda, sparse=l->sparse;
    memset(l,0,sizeof(*l));
    l->kda=kda; l->sparse=sparse;
    char nm[512];
    #define SNM(...) (snprintf(nm,sizeof(nm),__VA_ARGS__),nm)

    l->in_ln=f32_load(m,SNM("model.layers.%d.input_layernorm.weight",i),c->hidden);
    l->post_ln=f32_load(m,SNM("model.layers.%d.post_attention_layernorm.weight",i),c->hidden);
    {
        float *rn=f32_load(m,SNM("model.layers.%d.self_attention_res_norm.weight",i),c->hidden);
        float *rp=f32_load(m,SNM("model.layers.%d.self_attention_res_proj.weight",i),c->hidden);
        l->attn_sw=falloc(c->hidden);
        for(int d=0;d<c->hidden;d++) l->attn_sw[d]=rn[d]*rp[d];
        free(rn); free(rp);
        rn=f32_load(m,SNM("model.layers.%d.mlp_res_norm.weight",i),c->hidden);
        rp=f32_load(m,SNM("model.layers.%d.mlp_res_proj.weight",i),c->hidden);
        l->mlp_sw=falloc(c->hidden);
        for(int d=0;d<c->hidden;d++) l->mlp_sw[d]=rn[d]*rp[d];
        free(rn); free(rp);
    }

    if(l->kda){
        Kda *a=&l->a; int P=c->kda_proj;
        w_load(m,&a->q,SNM("model.layers.%d.self_attn.q_proj.weight",i),P,c->hidden,bits);
        w_load(m,&a->k,SNM("model.layers.%d.self_attn.k_proj.weight",i),P,c->hidden,bits);
        w_load(m,&a->v,SNM("model.layers.%d.self_attn.v_proj.weight",i),P,c->hidden,bits);
        w_load(m,&a->g,SNM("model.layers.%d.self_attn.g_proj.weight",i),P,c->hidden,bits);
        w_load(m,&a->o,SNM("model.layers.%d.self_attn.o_proj.weight",i),c->hidden,P,bits);
        a->conv_q=f32_load(m,SNM("model.layers.%d.self_attn.q_conv1d.weight",i),(int64_t)P*c->conv_k);
        a->conv_k=f32_load(m,SNM("model.layers.%d.self_attn.k_conv1d.weight",i),(int64_t)P*c->conv_k);
        a->conv_v=f32_load(m,SNM("model.layers.%d.self_attn.v_conv1d.weight",i),(int64_t)P*c->conv_k);
        a->fa=f32_load(m,SNM("model.layers.%d.self_attn.f_a_proj.weight",i),(int64_t)c->kda_hd*c->hidden);
        a->fb=f32_load(m,SNM("model.layers.%d.self_attn.f_b_proj.weight",i),(int64_t)P*c->kda_hd);
        a->bp=f32_load(m,SNM("model.layers.%d.self_attn.b_proj.weight",i),(int64_t)c->kda_heads*c->hidden);
        a->dt=f32_load(m,SNM("model.layers.%d.self_attn.dt_bias",i),P);
        a->onw=f32_load(m,SNM("model.layers.%d.self_attn.o_norm.weight",i),c->kda_hd);
        {
            char an[512];
            snprintf(an,sizeof(an),"%smodel.layers.%d.self_attn.A_log",m->pfx,i);
            st_tensor *t=st_find(&m->S,an);
            if(!t) st_die_missing(&m->S,an);
            if(t->numel<c->kda_heads){
                fprintf(stderr,"%s: %lld < heads\n",an,(long long)t->numel);
                exit(1);
            }
            float *al=falloc(t->numel);
            st_read_f32(&m->S,an,al,0);
            a->A=falloc(c->kda_heads);
            for(int h=0;h<c->kda_heads;h++) a->A[h]=expf(al[h]);
            free(al);
        }
    } else {
        Mla *a=&l->m;
        w_load(m,&a->qa,SNM("model.layers.%d.self_attn.q_a_proj.weight",i),c->q_lora,c->hidden,mbits);
        w_load(m,&a->qb,SNM("model.layers.%d.self_attn.q_b_proj.weight",i),c->n_heads*c->qk_head,c->q_lora,mbits);
        w_load(m,&a->kva,SNM("model.layers.%d.self_attn.kv_a_proj_with_mqa.weight",i),c->kv_lora+c->qk_rope,c->hidden,mbits);
        w_load(m,&a->kvb,SNM("model.layers.%d.self_attn.kv_b_proj.weight",i),c->n_heads*(c->qk_nope+c->v_head),c->kv_lora,mbits);
        w_load(m,&a->o,SNM("model.layers.%d.self_attn.o_proj.weight",i),c->hidden,c->n_heads*c->v_head,mbits);
        w_load(m,&a->g,SNM("model.layers.%d.self_attn.g_proj.weight",i),c->n_heads*c->v_head,c->hidden,mbits);
        a->qa_ln=f32_load(m,SNM("model.layers.%d.self_attn.q_a_layernorm.weight",i),c->q_lora);
        a->kva_ln=f32_load(m,SNM("model.layers.%d.self_attn.kv_a_layernorm.weight",i),c->kv_lora);
    }

    if(l->sparse){
        Moe *o=&l->moe;
        o->router=f32_load(m,SNM("model.layers.%d.block_sparse_moe.gate.weight",i),(int64_t)c->n_experts*c->hidden);
        o->rbias=f32_load(m,SNM("model.layers.%d.block_sparse_moe.gate.e_score_correction_bias",i),c->n_experts);
        o->lat_norm=f32_load(m,SNM("model.layers.%d.block_sparse_moe.routed_expert_norm.weight",i),c->latent);
        w_load(m,&o->lat_down,SNM("model.layers.%d.block_sparse_moe.routed_expert_down_proj.weight",i),c->latent,c->hidden,bits);
        w_load(m,&o->lat_up,SNM("model.layers.%d.block_sparse_moe.routed_expert_up_proj.weight",i),c->hidden,c->latent,bits);
        int shi=c->moe_inter*c->n_shared;
        w_load(m,&o->sh_gate,SNM("model.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight",i),shi,c->hidden,bits);
        w_load(m,&o->sh_up,SNM("model.layers.%d.block_sparse_moe.shared_experts.up_proj.weight",i),shi,c->hidden,bits);
        w_load(m,&o->sh_down,SNM("model.layers.%d.block_sparse_moe.shared_experts.down_proj.weight",i),c->hidden,shi,bits);
    } else {
        w_load(m,&l->d_gate,SNM("model.layers.%d.mlp.gate_proj.weight",i),c->dense_inter,c->hidden,bits);
        w_load(m,&l->d_up,SNM("model.layers.%d.mlp.up_proj.weight",i),c->dense_inter,c->hidden,bits);
        w_load(m,&l->d_down,SNM("model.layers.%d.mlp.down_proj.weight",i),c->hidden,c->dense_inter,bits);
    }
    #undef SNM
    g_stream_layer_loads++;
}

static float *stream_step_chunk(
    Model *m,
    const int *ids,
    int pos0,
    int C,
    int resident_layers,
    int bits,
    int mbits
){
    Cfg *c=&m->c; int D=c->hidden;
    int nbmax=(c->n_layers+c->res_bs-1)/c->res_bs;
    float *hidden=falloc((int64_t)C*D), *bres=falloc((int64_t)C*nbmax*D);
    float *prefix=falloc((int64_t)C*D), *nrm=falloc((int64_t)C*D);
    float *att=falloc((int64_t)C*D), *mix=falloc(D), *mlp=falloc((int64_t)C*D);
    int nb=0;

    for(int t=0;t<C;t++){
        if(g_x0){
            if(pos0+t>=g_x0_n){
                fprintf(stderr,"K3_X0: pos %d beyond %d injected rows\n",pos0+t,g_x0_n);
                exit(1);
            }
            memcpy(hidden+(int64_t)t*D,g_x0+(int64_t)(pos0+t)*D,D*sizeof(float));
        } else {
            char nm[512];
            snprintf(nm,sizeof(nm),"%smodel.embed_tokens.weight",m->pfx);
            st_read_slice_f32(&m->S,nm,(int64_t)ids[t]*D,D,hidden+(int64_t)t*D,0);
        }
    }

    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        int transient=i>=resident_layers;
        if(transient) stream_layer_load(m,l,i,bits,mbits);

        int snap=(i%c->res_bs==0);
        for(int t=0;t<C;t++){
            float *h=hidden+(int64_t)t*D, *p=prefix+(int64_t)t*D;
            memcpy(p,h,D*sizeof(float));
            if(nb>0) res_mix(h,p,bres+(int64_t)t*nbmax*D,nb,D,l->attn_sw,c->eps);
            if(snap) memcpy(bres+(int64_t)t*nbmax*D+(int64_t)nb*D,p,D*sizeof(float));
            rmsnorm_(nrm+(int64_t)t*D,h,l->in_ln,D,c->eps);
        }
        int have_prefix=!snap;
        if(snap) nb++;

        if(l->kda) kda_forward(m,l,i,nrm,C,att);
        else       mla_forward(m,l,i,nrm,pos0,C,att);

        for(int t=0;t<C;t++){
            float *p=prefix+(int64_t)t*D, *a=att+(int64_t)t*D;
            if(have_prefix){ for(int d=0;d<D;d++) p[d]+=a[d]; }
            else           { memcpy(p,a,D*sizeof(float)); }
            res_mix(mix,p,bres+(int64_t)t*nbmax*D,nb,D,l->mlp_sw,c->eps);
            rmsnorm_(nrm+(int64_t)t*D,mix,l->post_ln,D,c->eps);
        }

        if(l->sparse) moe_forward(m,l,i,nrm,C,mlp);
        else          dense_forward(m,l,nrm,C,mlp);

        for(int t=0;t<C;t++){
            float *p=prefix+(int64_t)t*D;
            for(int d=0;d<D;d++) p[d]+=mlp[(int64_t)t*D+d];
            memcpy(hidden+(int64_t)t*D,p,D*sizeof(float));
            if(m->trace) fwrite(hidden+(int64_t)t*D,sizeof(float),D,m->trace);
        }
        if(transient) stream_layer_release(l);
    }

    float *logits=NULL;
    if(m->has_head){
        for(int t=0;t<C;t++){
            if(!g_lfp && t<C-1) continue;
            res_mix(mix,hidden+(int64_t)t*D,bres+(int64_t)t*nbmax*D,nb,D,m->out_sw,c->eps);
            rmsnorm_(mix,mix,m->final_norm,D,c->eps);
            if(m->trace) fwrite(mix,sizeof(float),D,m->trace);
            float *lo=falloc(c->vocab);
            w_matmul(lo,mix,&m->lm_head,1);
            if(g_lfp) fwrite(lo,sizeof(float),(size_t)c->vocab,g_lfp);
            if(t==C-1) logits=lo; else free(lo);
        }
    }
    kv_prefix_record(&m->kvp,ids,pos0,C);
    free(hidden); free(bres); free(prefix); free(nrm);
    free(att); free(mix); free(mlp);
    return logits;
}

static int load_x0_file(Model *m, const char *path){
    FILE *f=fopen(path,"rb");
    if(!f){ perror(path); return -1; }
    if(fseek(f,0,SEEK_END)){ fclose(f); return -1; }
    long bytes=ftell(f);
    if(bytes<0 || bytes%(m->c.hidden*(long)sizeof(float))){
        fprintf(stderr,"%s: input size is not a whole number of hidden rows\n",path);
        fclose(f); return -1;
    }
    rewind(f);
    g_x0_n=(int)(bytes/(m->c.hidden*(long)sizeof(float)));
    g_x0=falloc((int64_t)g_x0_n*m->c.hidden);
    if(fread(g_x0,sizeof(float),(size_t)g_x0_n*m->c.hidden,f)!=(size_t)g_x0_n*m->c.hidden){
        fprintf(stderr,"%s: short input read\n",path);
        fclose(f); return -1;
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv){
    if(argc!=6){
        fprintf(stderr,
            "usage: %s SNAPSHOT X0_F32 TRACE_OUT RESIDENT_LAYERS N_LAYERS\n",
            argv[0]);
        return 2;
    }
    const char *snap=argv[1], *x0_path=argv[2], *trace_path=argv[3];
    int resident_layers=atoi(argv[4]), n_layers=atoi(argv[5]);
    if(n_layers<1 || resident_layers<0 || resident_layers>n_layers){
        fprintf(stderr,"invalid layer counts\n"); return 2;
    }

    setenv("K3_VK","0",1);
    setenv("AUTOPIN","0",1);
    setenv("K3_EXPERT_GB","0.001",1);

    Model m;
    model_init(&m,snap,n_layers);
    if(resident_layers>m.c.n_layers) resident_layers=m.c.n_layers;
    if(load_x0_file(&m,x0_path)!=0) return 1;
    kv_alloc(&m,g_x0_n+1);

    /* Functional lifecycle gate: original model_init validates and loads the
     * snapshot once; then the suffix is evicted before execution. A later gate
     * will skip those startup allocations entirely using the packed trunk. */
    for(int i=resident_layers;i<m.c.n_layers;i++) stream_layer_release(&m.L[i]);

    m.trace=fopen(trace_path,"wb");
    if(!m.trace){ perror(trace_path); return 1; }
    int id=0;
    for(int pos=0;pos<g_x0_n;pos++){
        float *lo=stream_step_chunk(
            &m,&id,pos,1,resident_layers,
            getenv("K3_BITS")?atoi(getenv("K3_BITS")):4,
            getenv("K3_MLA_BITS")?atoi(getenv("K3_MLA_BITS")):8
        );
        free(lo);
    }
    fclose(m.trace); m.trace=NULL;

    uint64_t expected=(uint64_t)(m.c.n_layers-resident_layers)*(uint64_t)g_x0_n;
    if(g_stream_layer_loads!=expected){
        fprintf(stderr,"layer reload count %llu != expected %llu\n",
            (unsigned long long)g_stream_layer_loads,
            (unsigned long long)expected);
        return 1;
    }
    fprintf(stderr,
        "[K3-STREAM] resident %d/%d | transient loads %llu | releases %llu | RSS %.3f GB\n",
        resident_layers,m.c.n_layers,
        (unsigned long long)g_stream_layer_loads,
        (unsigned long long)g_stream_layer_releases,
        rss_gb());
    return 0;
}
