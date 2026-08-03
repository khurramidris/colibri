/*
 * olmoe_bats.c — exact OLMoE inference with BATS shadow instrumentation.
 *
 * The stock engine is included unchanged. When BATS_SHADOW_OUT is unset this
 * binary delegates to the stock main. When set, it runs an arithmetic-identical
 * copy of the OLMoE step/MoE path and records predicted marginal expert-transfer
 * cost beside measured expert_get time. Shadow mode never changes routing,
 * expert admission, eviction, arithmetic, sampling, or emitted tokens.
 */
#define main olmoe_stock_main
#include "olmoe.c"
#undef main

#include "bats.h"
#include "route_trace.h"

static FILE *g_bats_fp;
static bats_hw_profile g_bats_hw;
static uint64_t g_bats_expert_bytes;
static uint64_t g_bats_call;

static double bats_env_double(const char *name, double fallback, double min_value, double max_value) {
    const char *text = getenv(name);
    if (!text || !*text) return fallback;
    char *end = NULL;
    double value = strtod(text, &end);
    if (!end || *end || !isfinite(value) || value < min_value || value > max_value) {
        fprintf(stderr, "[BATS] ignoring invalid %s=%s\n", name, text);
        return fallback;
    }
    return value;
}

static void bats_shadow_init(Model *m, const char *path) {
    memset(&g_bats_hw, 0, sizeof(g_bats_hw));
    g_bats_hw.bandwidth_gbps[BATS_TIER_VRAM] =
        bats_env_double("BATS_VRAM_GBPS", 300.0, 0.001, 100000.0);
    g_bats_hw.bandwidth_gbps[BATS_TIER_PINNED] =
        bats_env_double("BATS_PINNED_GBPS", 24.0, 0.001, 100000.0);
    g_bats_hw.bandwidth_gbps[BATS_TIER_RAM] =
        bats_env_double("BATS_RAM_GBPS", 12.0, 0.001, 100000.0);
    g_bats_hw.bandwidth_gbps[BATS_TIER_NVME] =
        bats_env_double("BATS_NVME_GBPS", 4.0, 0.001, 100000.0);
    g_bats_hw.fixed_us[BATS_TIER_NVME] =
        bats_env_double("BATS_NVME_FIXED_US", 80.0, 0.0, 1e9);
    g_bats_hw.queue_us[BATS_TIER_NVME] =
        bats_env_double("BATS_NVME_QUEUE_US", 0.0, 0.0, 1e9);
    g_bats_hw.overlap[BATS_TIER_NVME] =
        bats_env_double("BATS_NVME_OVERLAP", 0.0, 0.0, 1.0);

    const uint64_t ng = (uint64_t)m->c.inter * (uint64_t)m->c.hidden;
    const uint64_t nd = (uint64_t)m->c.hidden * (uint64_t)m->c.inter;
    const uint64_t scales = (uint64_t)(2 * m->c.inter + m->c.hidden) * sizeof(float);
    g_bats_expert_bytes = ng + ng + nd + scales;

    g_bats_fp = fopen(path, "w");
    if (!g_bats_fp) {
        fprintf(stderr, "[BATS] cannot open %s\n", path);
        exit(1);
    }
    setvbuf(g_bats_fp, NULL, _IOLBF, 0);
    fprintf(stderr,
            "[BATS] exact shadow mode -> %s | expert=%llu bytes | nvme=%.3f GB/s "
            "fixed=%.1f us queue=%.1f us overlap=%.2f\n",
            path, (unsigned long long)g_bats_expert_bytes,
            g_bats_hw.bandwidth_gbps[BATS_TIER_NVME],
            g_bats_hw.fixed_us[BATS_TIER_NVME],
            g_bats_hw.queue_us[BATS_TIER_NVME],
            g_bats_hw.overlap[BATS_TIER_NVME]);
}

static int bats_cache_contains(Model *m, int layer, int eid) {
    int found = 0;
    LCache *lc = &m->cache[layer];
    pthread_mutex_lock(&g_pilot_mx);
    for (int i = 0; i < lc->n; ++i) {
        if (lc->slots[i].eid == eid) {
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&g_pilot_mx);
    return found;
}

static void bats_log_row(Model *m, int layer, int row,
                         const int *idx, const float *val, int k,
                         uint64_t marginal_bytes, double predicted_us,
                         int resident_before, uint64_t misses,
                         double expert_get_us, double miss_get_us) {
    if (!g_bats_fp) return;
    double gate_mass = 0.0;
    for (int i = 0; i < k; ++i) gate_mass += val[i];
    fprintf(g_bats_fp,
            "{\"call\":%llu,\"token_base\":%d,\"layer\":%d,\"row\":%d,"
            "\"gate_mass\":%.9g,\"resident_before\":%d,\"misses\":%llu,"
            "\"marginal_bytes\":%llu,\"predicted_transfer_us\":%.9g,"
            "\"expert_get_us\":%.9g,\"miss_get_us\":%.9g,\"experts\":[",
            (unsigned long long)g_bats_call, m->token_count, layer, row,
            gate_mass, resident_before, (unsigned long long)misses,
            (unsigned long long)marginal_bytes, predicted_us,
            expert_get_us, miss_get_us);
    for (int i = 0; i < k; ++i) {
        if (i) fputc(',', g_bats_fp);
        fprintf(g_bats_fp, "{\"id\":%d,\"gate\":%.9g}", idx[i], val[i]);
    }
    fputs("]}\n", g_bats_fp);
}

/* Arithmetic-identical copy of olmoe.c::moe with read-only BATS telemetry. */
static void moe_bats(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    float *logits = falloc((int64_t)S * E);
    matmul(logits, x, l->gate, S, D, E);
    memset(out, 0, (int64_t)S * D * sizeof(float));
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    uint8_t *union_mask = calloc((size_t)E, 1);
    uint8_t *new_mask = calloc((size_t)E, 1);
    bats_expert_state *states = calloc((size_t)E, sizeof(*states));
    if (!union_mask || !new_mask || !states) {
        fprintf(stderr, "[BATS] OOM allocating shadow state\n");
        exit(1);
    }

    for (int s = 0; s < S; ++s) {
        float *pr = logits + (int64_t)s * E;
        if (m->momentum_logits && m->pilot_smooth > 0.f) {
            float *ema = m->momentum_logits + (int64_t)layer * E;
            int is_zero = 1;
            for (int e = 0; e < E; ++e) {
                if (ema[e] != 0.f) {
                    is_zero = 0;
                    break;
                }
            }
            if (is_zero) {
                for (int e = 0; e < E; ++e) ema[e] = pr[e];
            } else {
                for (int e = 0; e < E; ++e)
                    ema[e] = (1.f - m->pilot_smooth) * pr[e] + m->pilot_smooth * ema[e];
            }
        }

        softmax_row(pr, E);
        int idx[64];
        float val[64];
        for (int kk = 0; kk < K; ++kk) {
            int best = -1;
            float bv = -1e30f;
            for (int e = 0; e < E; ++e) {
                int taken = 0;
                for (int j = 0; j < kk; ++j)
                    if (idx[j] == e) {
                        taken = 1;
                        break;
                    }
                if (!taken && pr[e] > bv) {
                    bv = pr[e];
                    best = e;
                }
            }
            idx[kk] = best;
            val[kk] = bv;
        }
        if (c->norm_topk) {
            float sm = 0;
            for (int kk = 0; kk < K; ++kk) sm += val[kk];
            for (int kk = 0; kk < K; ++kk) val[kk] /= sm;
        }
        if (!m->hot_pinned && m->freq) {
            uint32_t *freq_l = m->freq + (int64_t)layer * E;
            for (int kk = 0; kk < K; ++kk)
                if (idx[kk] >= 0) freq_l[idx[kk]]++;
        }

        int resident_before = 0;
        for (int kk = 0; kk < K; ++kk) {
            int eid = idx[kk];
            int resident = bats_cache_contains(m, layer, eid);
            states[eid].bytes = g_bats_expert_bytes;
            states[eid].tier = resident ? BATS_TIER_EXEC : BATS_TIER_NVME;
            states[eid].resident_in_exec = (uint8_t)resident;
            resident_before += resident;
        }
        bats_candidate candidate = {
            .id = s,
            .expected_accepted_tokens = 1.0,
            .verify_compute_us = 0.0,
            .kv_us = 0.0,
            .expert_ids = idx,
            .expert_count = (size_t)K,
        };
        uint64_t marginal_bytes = 0;
        double predicted_us = bats_candidate_marginal_cost(
            &g_bats_hw, states, (size_t)E, &candidate,
            union_mask, new_mask, &marginal_bytes);

        const float *xs = x + (int64_t)s * D;
        double expert_get_us = 0.0;
        double miss_get_us = 0.0;
        uint64_t misses_before = m->miss;
        for (int kk = 0; kk < K; ++kk) {
            Slot *expert;
            uint64_t miss_before = m->miss;
            double get_start = now_s();
            expert_get(m, layer, idx[kk], &expert);
            double get_us = (now_s() - get_start) * 1e6;
            expert_get_us += get_us;
            if (m->miss != miss_before) miss_get_us += get_us;

            matmul_q(g, xs, expert->g, expert->gs, D, I);
            matmul_q(u, xs, expert->u, expert->us, D, I);
            for (int i = 0; i < I; ++i) {
                float gv = g[i];
                g[i] = (gv / (1.f + expf(-gv))) * u[i];
            }
            matmul_q(hh, g, expert->d, expert->ds, I, D);
            float w = val[kk];
            float *os = out + (int64_t)s * D;
            for (int d = 0; d < D; ++d) os[d] += w * hh[d];
        }

        rt_route(layer, s, idx, val, K);
        bats_log_row(m, layer, s, idx, val, K, marginal_bytes, predicted_us,
                     resident_before, m->miss - misses_before,
                     expert_get_us, miss_get_us);
        for (int kk = 0; kk < K; ++kk)
            if (idx[kk] >= 0) union_mask[idx[kk]] = 1;
    }

    rt_trace_end();
    ++g_bats_call;
    free(union_mask);
    free(new_mask);
    free(states);
    free(logits);
    free(g);
    free(u);
    free(hh);
}

/* Arithmetic-identical copy of olmoe.c::step, calling moe_bats. */
static float *step_bats(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c;
    int D = c->hidden;
    if (g_pilot && m->token_count > 0) {
        pthread_mutex_lock(&g_pilot_mx);
        memset(m->is_queued, 0, (size_t)c->n_layers * c->n_experts);
        pthread_mutex_unlock(&g_pilot_mx);
    }
    float *x = falloc((int64_t)S * D);
    for (int s = 0; s < S; ++s)
        memcpy(x + (int64_t)s * D, m->embed + (int64_t)ids[s] * D, D * sizeof(float));
    float *nrm = falloc((int64_t)S * D), *tmp = falloc((int64_t)S * D);
    for (int i = 0; i < c->n_layers; ++i) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; ++s)
            rmsnorm_row(nrm + (int64_t)s * D, x + (int64_t)s * D, l->in_ln, D, c->eps);
        attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S * D; ++j) x[j] += tmp[j];
        if (g_pilot >= 1 && S <= 8 && i + 1 < c->n_layers)
            pilot_prefetch(m, i + 1, x, S);
        for (int s = 0; s < S; ++s)
            rmsnorm_row(nrm + (int64_t)s * D, x + (int64_t)s * D, l->post_ln, D, c->eps);
        moe_bats(m, l, i, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S * D; ++j) x[j] += tmp[j];
        if (g_pilot >= 2 && S <= 8 && i + 2 < c->n_layers)
            pilot_prefetch(m, i + 2, x, S);
        if (g_pilot >= 3 && S <= 8 && i + 3 < c->n_layers)
            pilot_prefetch(m, i + 3, x, S);
    }
    m->token_count += S;
    m->freq_token_count += S;
    if (!m->hot_pinned && m->hot_n > 0 && m->freq_token_count >= m->warmup_tokens)
        pin_hot_experts(m);
    m->kv_len = pos_base + S;
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S - 1) * D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    matmul(logit, last, m->lm_head, 1, D, c->vocab);
    free(x);
    free(nrm);
    free(tmp);
    free(last);
    return logit;
}

static void generate_bats(Model *m, const int *prompt, int np, int n_new, int *out) {
    Cfg *c = &m->c;
    m->max_t = np + n_new;
    m->K = calloc(c->n_layers, sizeof(float *));
    m->V = calloc(c->n_layers, sizeof(float *));
    for (int i = 0; i < c->n_layers; ++i) {
        m->K[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
    }
    for (int i = 0; i < np; ++i) out[i] = prompt[i];
    float *logit = step_bats(m, prompt, np, 0);
    int len = np;
    for (int s = 0; s < n_new; ++s) {
        int best = 0;
        float bv = logit[0];
        for (int i = 1; i < c->vocab; ++i)
            if (logit[i] > bv) {
                bv = logit[i];
                best = i;
            }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step_bats(m, &one, 1, len - 1);
    }
}

static int tf_nll_bats(Model *m, const int *full, int nfull, int np, double *nll_out) {
    Cfg *c = &m->c;
    m->max_t = nfull;
    m->K = calloc(c->n_layers, sizeof(float *));
    m->V = calloc(c->n_layers, sizeof(float *));
    for (int i = 0; i < c->n_layers; ++i) {
        m->K[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * m->max_t * c->head_dim);
    }
    double nll = 0;
    int scored = 0;
    float *logit = step_bats(m, full, np, 0);
    for (int i = np; i < nfull; ++i) {
        float mx = logit[0];
        for (int v = 1; v < c->vocab; ++v)
            if (logit[v] > mx) mx = logit[v];
        double Z = 0;
        for (int v = 0; v < c->vocab; ++v) Z += exp((double)logit[v] - mx);
        nll += -((double)logit[full[i]] - mx - log(Z));
        ++scored;
        free(logit);
        logit = NULL;
        if (i == nfull - 1) break;
        logit = step_bats(m, &full[i], 1, i);
    }
    if (logit) free(logit);
    *nll_out = nll / scored;
    return scored;
}

static void bats_save_hot_pins(Model *m, const char *snap) {
    if (!m->hot_pinned) return;
    char pinpath[512];
    snprintf(pinpath, sizeof(pinpath), "%s/hot_pinned.bin", snap);
    FILE *check = fopen(pinpath, "rb");
    if (check) {
        fclose(check);
        return;
    }
    FILE *save = fopen(pinpath, "wb");
    if (!save) return;
    size_t expected = (size_t)m->c.n_layers * m->c.n_experts;
    fwrite(m->is_pinned, 1, expected, save);
    fclose(save);
    printf("[HOT] Saved persistent pinning to %s\n", pinpath);
}

int main(int argc, char **argv) {
    const char *shadow_path = getenv("BATS_SHADOW_OUT");
    if (!shadow_path || !*shadow_path) return olmoe_stock_main(argc, argv);
    if (getenv("CHAT")) {
        fprintf(stderr, "[BATS] CHAT shadow path is not implemented yet; running stock chat\n");
        return olmoe_stock_main(argc, argv);
    }

    coli_omp_tune_threads("olmoe-bats");
    const char *snap = getenv("SNAP");
    if (!snap) {
        fprintf(stderr, "set SNAP=<snapshot directory>\n");
        return 1;
    }
    g_pilot = getenv("PILOT") ? atoi(getenv("PILOT")) : 0;
    g_wide = getenv("WIDE") ? atoi(getenv("WIDE")) : 1;
    g_pilot_evict_guard =
        getenv("PILOT_EVICT_GUARD") ? atoi(getenv("PILOT_EVICT_GUARD")) : 1;
    g_expert_drop = getenv("EXPERT_DROP") ? atoi(getenv("EXPERT_DROP")) : 0;
    if (g_wide < 1) g_wide = 1;
    if (g_wide > 4) g_wide = 4;

    int hot_n = getenv("HOT") ? atoi(getenv("HOT")) : 0;
    int cap = argc > 1 ? atoi(argv[1]) : 16;
    int bits = argc > 2 ? atoi(argv[2]) : 8;
    if (bits < 2 || bits > 8) {
        fprintf(stderr, "quant_bits must be 2..8 (got %d)\n", bits);
        return 1;
    }
    const char *refpath = argc > 3 ? argv[3] : "ref.json";
    float smooth = getenv("SMOOTH") ? (float)atof(getenv("SMOOTH")) : 0.3f;
    float conf = getenv("CONF_LIMIT") ? (float)atof(getenv("CONF_LIMIT")) : 0.92f;
    printf("== OLMoE BATS exact shadow | cache=%d/layer bits=%d pilot=%d wide=%d "
           "guard=%d hot=%d smooth=%.2f conf=%.2f ==\n",
           cap, bits, g_pilot, g_wide, g_pilot_evict_guard, hot_n, smooth, conf);

    FILE *f = fopen(refpath, "rb");
    if (!f) {
        perror(refpath);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fclose(f);
        return 1;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "%s: short read\n", refpath);
        fclose(f);
        free(buf);
        return 1;
    }
    buf[n] = 0;
    fclose(f);
    char *arena = NULL;
    jval *ref = json_parse(buf, &arena);
    int np, nfull;
    int *prompt = read_int_array(ref, "prompt_ids", &np);
    int *full = read_int_array(ref, "full_ids", &nfull);
    int n_new = nfull - np;

    Model m;
    model_init(&m, snap, cap, bits);
    rt_init("olmoe", m.c.n_layers, m.c.n_experts);
    bats_shadow_init(&m, shadow_path);
    printf("resident weights loaded in %.1fs | RSS after load: %.2f GB\n",
           m.dense_load_s, rss_gb());

    if (getenv("PPL") && atoi(getenv("PPL")) == 1) {
        double nll;
        double start = now_s();
        int scored = tf_nll_bats(&m, full, nfull, np, &nll);
        double elapsed = now_s() - start;
        double total = m.hits + m.miss;
        printf("TF-NLL: %.4f nats/token over %d tokens  |  ppl = %.2f\n",
               nll, scored, exp(nll));
        printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n",
               total ? 100.0 * m.hits / total : 0.0,
               (unsigned long long)m.hits, (unsigned long long)m.miss);
        printf("Speed: %.2f tok/s (%.1fs for %d tokens) | PEAK RSS: %.2f GB\n",
               scored / elapsed, elapsed, scored, rss_gb());
        fclose(g_bats_fp);
        free(prompt);
        free(full);
        free(buf);
        free(arena);
        return 0;
    }

    int *out = malloc((size_t)(np + n_new) * sizeof(int));
    double start = now_s();
    generate_bats(&m, prompt, np, n_new, out);
    double elapsed = now_s() - start;

    int match = 0;
    printf("\nReference: ");
    for (int i = np; i < nfull; ++i) printf("%d ", full[i]);
    printf("\nC engine : ");
    for (int i = np; i < nfull; ++i) {
        printf("%d ", out[i]);
        if (out[i] == full[i]) ++match;
    }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    double total = m.hits + m.miss;
    printf("\nPEAK RSS: %.2f GB\n", rss_gb());
    printf("Expert cache hit rate: %.1f%%  (hit=%llu miss=%llu)\n",
           total ? 100.0 * m.hits / total : 0.0,
           (unsigned long long)m.hits, (unsigned long long)m.miss);
    bats_save_hot_pins(&m, snap);
    printf("Speed: %.2f tok/s (%.1fs for %d tokens)\n", n_new / elapsed, elapsed, n_new);

    fclose(g_bats_fp);
    free(out);
    free(prompt);
    free(full);
    free(buf);
    free(arena);
    return match == n_new ? 0 : 2;
}
