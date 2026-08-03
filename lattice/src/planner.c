#include "lattice/planner.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void lt_set_error(char *dst, size_t cap, const char *msg) {
    if (!dst || cap == 0) return;
    snprintf(dst, cap, "%s", msg ? msg : "unknown error");
}

static double lt_clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static uint64_t lt_effective_budget(uint64_t raw, double reserve_fraction) {
    const double keep = 1.0 - lt_clamp(reserve_fraction, 0.0, 0.90);
    const long double value = (long double)raw * keep;
    if (value <= 0.0L) return 0;
    if (value >= (long double)UINT64_MAX) return UINT64_MAX;
    return (uint64_t)value;
}

static double lt_rank_weight(uint32_t rank_zero_based, double skew) {
    if (skew <= 0.0) return 1.0;
    return pow((double)rank_zero_based + 1.0, -skew);
}

static double lt_weight_denominator(const lt_tensor_class_t *c) {
    double sum = 0.0;
    uint32_t i;
    for (i = 0; i < c->count; ++i) sum += lt_rank_weight(i, c->skew);
    return sum > 0.0 ? sum : 1.0;
}

static double lt_mass_range(const lt_tensor_class_t *c,
                            double denominator,
                            uint32_t begin,
                            uint32_t count) {
    double sum = 0.0;
    uint32_t i;
    const uint32_t end = begin + count > c->count ? c->count : begin + count;
    for (i = begin; i < end; ++i) sum += lt_rank_weight(i, c->skew);
    return lt_clamp(sum / denominator, 0.0, 1.0);
}

static int lt_u64_mul(uint64_t a, uint32_t b, uint64_t *out) {
    if (b != 0 && a > UINT64_MAX / b) return -1;
    *out = a * (uint64_t)b;
    return 0;
}

const char *lt_role_name(lt_role_t role) {
    switch (role) {
        case LT_ROLE_DENSE: return "dense";
        case LT_ROLE_ATTENTION: return "attention";
        case LT_ROLE_SHARED_EXPERT: return "shared_expert";
        case LT_ROLE_ROUTED_EXPERT: return "routed_expert";
        case LT_ROLE_EMBEDDING: return "embedding";
        case LT_ROLE_KV_CACHE: return "kv_cache";
        case LT_ROLE_RECURRENT_STATE: return "recurrent_state";
        case LT_ROLE_NORM: return "norm";
        case LT_ROLE_OTHER: return "other";
        default: return "unknown";
    }
}

int lt_role_parse(const char *text, lt_role_t *out) {
    lt_role_t role;
    if (!text || !out) return -1;
    for (role = LT_ROLE_DENSE; role <= LT_ROLE_OTHER; role = (lt_role_t)(role + 1)) {
        if (strcmp(text, lt_role_name(role)) == 0) {
            *out = role;
            return 0;
        }
    }
    return -1;
}

const char *lt_tier_name(lt_tier_t tier) {
    switch (tier) {
        case LT_TIER_NVME: return "nvme";
        case LT_TIER_RAM: return "ram";
        case LT_TIER_VRAM: return "vram";
        default: return "unknown";
    }
}

static double lt_item_touch_mass(const lt_tensor_class_t *c,
                                 double denominator,
                                 uint32_t rank) {
    if (rank >= c->count || c->touches_per_token <= 0.0) return 0.0;
    return c->touches_per_token * lt_rank_weight(rank, c->skew) / denominator;
}

static double lt_nvme_ms_for_bytes(const lt_hardware_t *hw, uint64_t bytes) {
    if (bytes == 0) return 0.0;
    if (hw->nvme_read_gbps <= 0.0) return INFINITY;
    return ((double)bytes / (hw->nvme_read_gbps * 1.0e9)) * 1000.0;
}

static int lt_allocate_one_vram(const lt_hardware_t *hw,
                                const lt_tensor_class_t *classes,
                                const double *den,
                                lt_plan_t *plan,
                                uint64_t budget) {
    size_t best = (size_t)-1;
    double best_score = -1.0;
    size_t i;
    for (i = 0; i < plan->nclasses; ++i) {
        const lt_tensor_class_t *c = &classes[i];
        lt_class_plan_t *p = &plan->classes[i];
        uint32_t rank = p->vram_count + p->ram_count;
        double touches, movement, compute_gain, score;
        if (!(c->flags & LT_CLASS_GPU_CAPABLE)) continue;
        if (rank >= c->count || c->bytes_each == 0) continue;
        if (c->bytes_each > budget - plan->vram_used_bytes) continue;
        touches = lt_item_touch_mass(c, den[i], rank);
        movement = lt_nvme_ms_for_bytes(hw, c->bytes_each);
        compute_gain = c->cpu_ms_per_touch - c->gpu_ms_per_touch;
        if (compute_gain < 0.0) compute_gain = 0.0;
        score = touches * (movement + compute_gain) / (double)c->bytes_each;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    if (best == (size_t)-1) return 0;
    plan->classes[best].vram_count++;
    plan->vram_used_bytes += classes[best].bytes_each;
    return 1;
}

static int lt_allocate_one_ram(const lt_hardware_t *hw,
                               const lt_tensor_class_t *classes,
                               const double *den,
                               lt_plan_t *plan,
                               uint64_t budget) {
    size_t best = (size_t)-1;
    double best_score = -1.0;
    size_t i;
    for (i = 0; i < plan->nclasses; ++i) {
        const lt_tensor_class_t *c = &classes[i];
        lt_class_plan_t *p = &plan->classes[i];
        uint32_t rank = p->vram_count + p->ram_count;
        double touches, movement, score;
        if (rank >= c->count || c->bytes_each == 0) continue;
        if (c->bytes_each > budget - plan->ram_used_bytes) continue;
        touches = lt_item_touch_mass(c, den[i], rank);
        movement = lt_nvme_ms_for_bytes(hw, c->bytes_each);
        score = touches * movement / (double)c->bytes_each;
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    if (best == (size_t)-1) return 0;
    plan->classes[best].ram_count++;
    plan->ram_used_bytes += classes[best].bytes_each;
    return 1;
}

static int lt_place_fixed(const lt_tensor_class_t *c,
                          lt_class_plan_t *p,
                          uint32_t want,
                          int to_vram,
                          uint64_t *used,
                          uint64_t budget,
                          char *error,
                          size_t error_cap) {
    uint32_t already = p->vram_count + p->ram_count;
    uint32_t available = already < c->count ? c->count - already : 0;
    uint64_t add_bytes;
    if (want > available) want = available;
    if (lt_u64_mul(c->bytes_each, want, &add_bytes) != 0 || add_bytes > budget - *used) {
        char msg[LT_ERROR_MAX];
        snprintf(msg, sizeof(msg),
                 "%s cannot satisfy %s placement: need %.3f GiB, have %.3f GiB",
                 c->name,
                 to_vram ? "VRAM" : "RAM",
                 (double)add_bytes / (1024.0 * 1024.0 * 1024.0),
                 (double)(budget - *used) / (1024.0 * 1024.0 * 1024.0));
        lt_set_error(error, error_cap, msg);
        return -1;
    }
    if (to_vram) p->vram_count += want;
    else p->ram_count += want;
    *used += add_bytes;
    return 0;
}

int lt_plan_build(const lt_hardware_t *hw,
                  const lt_tensor_class_t *classes,
                  size_t nclasses,
                  lt_plan_t *out) {
    uint64_t ram_budget, vram_budget;
    double *den = NULL;
    size_t i;
    char validation[LT_ERROR_MAX];

    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    if (!hw || !classes || nclasses == 0) {
        lt_set_error(out->error, sizeof(out->error), "planner requires hardware and at least one tensor class");
        return -1;
    }
    if (!isfinite(hw->nvme_read_gbps) || hw->nvme_read_gbps <= 0.0 ||
        !isfinite(hw->overlap_efficiency) ||
        hw->overlap_efficiency < 0.0 || hw->overlap_efficiency > 1.0 ||
        !isfinite(hw->reserve_fraction) || hw->reserve_fraction < 0.0 || hw->reserve_fraction >= 1.0) {
        lt_set_error(out->error, sizeof(out->error), "invalid hardware profile");
        return -1;
    }

    out->classes = (lt_class_plan_t *)calloc(nclasses, sizeof(*out->classes));
    den = (double *)calloc(nclasses, sizeof(*den));
    if (!out->classes || !den) {
        free(out->classes);
        out->classes = NULL;
        lt_set_error(out->error, sizeof(out->error), "out of memory");
        return -1;
    }
    out->nclasses = nclasses;
    ram_budget = lt_effective_budget(hw->ram_budget_bytes, hw->reserve_fraction);
    vram_budget = lt_effective_budget(hw->vram_budget_bytes, hw->reserve_fraction);

    for (i = 0; i < nclasses; ++i) {
        const lt_tensor_class_t *c = &classes[i];
        if (c->name[0] == '\0' || c->bytes_each == 0 || c->count == 0 ||
            !isfinite(c->touches_per_token) || c->touches_per_token < 0.0 ||
            !isfinite(c->skew) || c->skew < 0.0 ||
            c->min_ram_count > c->count || c->min_vram_count > c->count ||
            c->min_ram_count + c->min_vram_count > c->count) {
            snprintf(out->error, sizeof(out->error), "invalid tensor class at index %zu", i);
            free(den);
            return -1;
        }
        if ((c->flags & LT_CLASS_VRAM_REQUIRED) && !(c->flags & LT_CLASS_GPU_CAPABLE)) {
            snprintf(out->error, sizeof(out->error), "%s requires VRAM but is not GPU-capable", c->name);
            free(den);
            return -1;
        }
        den[i] = lt_weight_denominator(c);
    }

    /* Hard VRAM requirements are admitted first. */
    for (i = 0; i < nclasses; ++i) {
        const lt_tensor_class_t *c = &classes[i];
        uint32_t want = (c->flags & LT_CLASS_VRAM_REQUIRED) ? c->count : c->min_vram_count;
        if (lt_place_fixed(c, &out->classes[i], want, 1,
                           &out->vram_used_bytes, vram_budget,
                           out->error, sizeof(out->error)) != 0) {
            free(den);
            return -1;
        }
    }

    /* Use remaining VRAM for the largest marginal reduction in movement plus
     * measured compute time. Each next item is the next hottest rank. */
    while (out->vram_used_bytes < vram_budget &&
           lt_allocate_one_vram(hw, classes, den, out, vram_budget)) {
        /* deliberate empty body */
    }

    /* Non-streamable and explicit RAM requirements are exact feasibility
     * constraints, not preferences. VRAM-resident items do not consume RAM. */
    for (i = 0; i < nclasses; ++i) {
        const lt_tensor_class_t *c = &classes[i];
        lt_class_plan_t *p = &out->classes[i];
        uint32_t remaining = c->count - p->vram_count - p->ram_count;
        uint32_t want = c->min_ram_count;
        if (!(c->flags & LT_CLASS_STREAMABLE)) want = remaining;
        if (want > remaining) want = remaining;
        if (lt_place_fixed(c, p, want, 0,
                           &out->ram_used_bytes, ram_budget,
                           out->error, sizeof(out->error)) != 0) {
            free(den);
            return -1;
        }
    }

    while (out->ram_used_bytes < ram_budget &&
           lt_allocate_one_ram(hw, classes, den, out, ram_budget)) {
        /* deliberate empty body */
    }

    for (i = 0; i < nclasses; ++i) {
        const lt_tensor_class_t *c = &classes[i];
        lt_class_plan_t *p = &out->classes[i];
        uint32_t placed = p->vram_count + p->ram_count;
        double tv, tr, tn;
        uint64_t bytes;
        p->nvme_count = c->count - placed;
        p->vram_bytes = c->bytes_each * (uint64_t)p->vram_count;
        p->ram_bytes = c->bytes_each * (uint64_t)p->ram_count;
        p->nvme_bytes = c->bytes_each * (uint64_t)p->nvme_count;
        p->hot_mass_vram = lt_mass_range(c, den[i], 0, p->vram_count);
        p->hot_mass_ram = lt_mass_range(c, den[i], p->vram_count, p->ram_count);
        p->hot_mass_nvme = lt_clamp(1.0 - p->hot_mass_vram - p->hot_mass_ram, 0.0, 1.0);
        p->ram_score_per_byte = c->touches_per_token *
            lt_nvme_ms_for_bytes(hw, c->bytes_each) / (double)c->bytes_each;
        p->vram_score_per_byte = p->ram_score_per_byte;
        if (c->cpu_ms_per_touch > c->gpu_ms_per_touch) {
            p->vram_score_per_byte += c->touches_per_token *
                (c->cpu_ms_per_touch - c->gpu_ms_per_touch) / (double)c->bytes_each;
        }

        out->nvme_resident_bytes += p->nvme_bytes;
        tv = c->touches_per_token * p->hot_mass_vram;
        tr = c->touches_per_token * p->hot_mass_ram;
        tn = c->touches_per_token * p->hot_mass_nvme;
        out->nvme_bytes_per_token += tn * (double)c->bytes_each;
        out->gpu_compute_ms_per_token += tv * c->gpu_ms_per_touch;
        out->cpu_compute_ms_per_token += (tr + tn) * c->cpu_ms_per_touch;

        /* Overflow guard for malformed manifests, although class validation
         * and ordinary model sizes keep this path far below UINT64_MAX. */
        if (lt_u64_mul(c->bytes_each, c->count, &bytes) != 0) {
            snprintf(out->error, sizeof(out->error), "%s byte size overflows uint64", c->name);
            free(den);
            return -1;
        }
        (void)bytes;
    }

    out->movement_ms_per_token = lt_nvme_ms_for_bytes(hw, (uint64_t)out->nvme_bytes_per_token);
    out->compute_ms_per_token = out->cpu_compute_ms_per_token + out->gpu_compute_ms_per_token;
    out->predicted_ms_per_token = out->movement_ms_per_token + out->compute_ms_per_token -
        hw->overlap_efficiency * fmin(out->movement_ms_per_token, out->compute_ms_per_token);
    out->feasible = 1;

    if (lt_plan_validate(hw, classes, out, validation, sizeof(validation)) != 0) {
        out->feasible = 0;
        lt_set_error(out->error, sizeof(out->error), validation);
        free(den);
        return -1;
    }
    free(den);
    return 0;
}

int lt_plan_validate(const lt_hardware_t *hw,
                     const lt_tensor_class_t *classes,
                     const lt_plan_t *plan,
                     char *error,
                     size_t error_cap) {
    uint64_t ram_budget, vram_budget;
    uint64_t ram = 0, vram = 0;
    size_t i;
    if (!hw || !classes || !plan || !plan->classes) {
        lt_set_error(error, error_cap, "cannot validate an empty plan");
        return -1;
    }
    ram_budget = lt_effective_budget(hw->ram_budget_bytes, hw->reserve_fraction);
    vram_budget = lt_effective_budget(hw->vram_budget_bytes, hw->reserve_fraction);
    for (i = 0; i < plan->nclasses; ++i) {
        const lt_tensor_class_t *c = &classes[i];
        const lt_class_plan_t *p = &plan->classes[i];
        if (p->vram_count + p->ram_count + p->nvme_count != c->count) {
            snprintf(error, error_cap, "%s placement count mismatch", c->name);
            return -1;
        }
        if (p->vram_count && !(c->flags & LT_CLASS_GPU_CAPABLE)) {
            snprintf(error, error_cap, "%s placed in VRAM without GPU capability", c->name);
            return -1;
        }
        if (!(c->flags & LT_CLASS_STREAMABLE) && p->nvme_count != 0) {
            snprintf(error, error_cap, "%s is non-streamable but remains on NVMe", c->name);
            return -1;
        }
        if (p->ram_count < c->min_ram_count || p->vram_count < c->min_vram_count) {
            snprintf(error, error_cap, "%s minimum placement not met", c->name);
            return -1;
        }
        if ((c->flags & LT_CLASS_RAM_REQUIRED) && p->ram_count + p->vram_count != c->count) {
            snprintf(error, error_cap, "%s requires fast memory", c->name);
            return -1;
        }
        if ((c->flags & LT_CLASS_VRAM_REQUIRED) && p->vram_count != c->count) {
            snprintf(error, error_cap, "%s requires VRAM", c->name);
            return -1;
        }
        ram += p->ram_bytes;
        vram += p->vram_bytes;
    }
    if (ram > ram_budget || vram > vram_budget) {
        lt_set_error(error, error_cap, "plan exceeds effective memory budget");
        return -1;
    }
    lt_set_error(error, error_cap, "");
    return 0;
}

void lt_plan_free(lt_plan_t *plan) {
    if (!plan) return;
    free(plan->classes);
    memset(plan, 0, sizeof(*plan));
}

static void lt_print_gib(FILE *fp, uint64_t bytes) {
    fprintf(fp, "%.3f", (double)bytes / (1024.0 * 1024.0 * 1024.0));
}

void lt_plan_print(FILE *fp,
                   const lt_hardware_t *hw,
                   const lt_tensor_class_t *classes,
                   const lt_plan_t *plan) {
    size_t i;
    if (!fp || !hw || !classes || !plan) return;
    fprintf(fp, "Lattice exact placement plan\n");
    fprintf(fp, "  RAM  used "); lt_print_gib(fp, plan->ram_used_bytes);
    fprintf(fp, " GiB / effective "); lt_print_gib(fp, lt_effective_budget(hw->ram_budget_bytes, hw->reserve_fraction));
    fprintf(fp, " GiB\n  VRAM used "); lt_print_gib(fp, plan->vram_used_bytes);
    fprintf(fp, " GiB / effective "); lt_print_gib(fp, lt_effective_budget(hw->vram_budget_bytes, hw->reserve_fraction));
    fprintf(fp, " GiB\n");
    fprintf(fp, "  predicted: %.3f GiB NVMe/token, %.3f ms movement, %.3f ms compute, %.3f ms/token\n",
            plan->nvme_bytes_per_token / (1024.0 * 1024.0 * 1024.0),
            plan->movement_ms_per_token,
            plan->compute_ms_per_token,
            plan->predicted_ms_per_token);
    fprintf(fp, "\n%-28s %-16s %10s %10s %10s %10s\n",
            "class", "role", "VRAM", "RAM", "NVMe", "NVMe hit%");
    for (i = 0; i < plan->nclasses; ++i) {
        const lt_class_plan_t *p = &plan->classes[i];
        fprintf(fp, "%-28s %-16s %5u/%-4u %5u/%-4u %5u/%-4u %9.2f\n",
                classes[i].name,
                lt_role_name(classes[i].role),
                p->vram_count, classes[i].count,
                p->ram_count, classes[i].count,
                p->nvme_count, classes[i].count,
                100.0 * p->hot_mass_nvme);
    }
}

static void lt_json_string(FILE *fp, const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    fputc('"', fp);
    while (*p) {
        switch (*p) {
            case '"': fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc(*p, fp);
        }
        ++p;
    }
    fputc('"', fp);
}

int lt_plan_write_json(FILE *fp,
                       const lt_hardware_t *hw,
                       const lt_tensor_class_t *classes,
                       const lt_plan_t *plan) {
    size_t i;
    if (!fp || !hw || !classes || !plan) return -1;
    fprintf(fp, "{\n");
    fprintf(fp, "  \"schema\": \"lattice.plan.v1\",\n");
    fprintf(fp, "  \"exact\": true,\n");
    fprintf(fp, "  \"feasible\": %s,\n", plan->feasible ? "true" : "false");
    fprintf(fp, "  \"hardware\": {\"ram_budget_bytes\": %llu, \"vram_budget_bytes\": %llu, \"nvme_read_gbps\": %.9g, \"ram_read_gbps\": %.9g, \"host_to_device_gbps\": %.9g, \"overlap_efficiency\": %.9g, \"reserve_fraction\": %.9g},\n",
            (unsigned long long)hw->ram_budget_bytes,
            (unsigned long long)hw->vram_budget_bytes,
            hw->nvme_read_gbps, hw->ram_read_gbps, hw->host_to_device_gbps,
            hw->overlap_efficiency, hw->reserve_fraction);
    fprintf(fp, "  \"summary\": {\"ram_used_bytes\": %llu, \"vram_used_bytes\": %llu, \"nvme_resident_bytes\": %llu, \"nvme_bytes_per_token\": %.17g, \"movement_ms_per_token\": %.17g, \"compute_ms_per_token\": %.17g, \"predicted_ms_per_token\": %.17g},\n",
            (unsigned long long)plan->ram_used_bytes,
            (unsigned long long)plan->vram_used_bytes,
            (unsigned long long)plan->nvme_resident_bytes,
            plan->nvme_bytes_per_token,
            plan->movement_ms_per_token,
            plan->compute_ms_per_token,
            plan->predicted_ms_per_token);
    fprintf(fp, "  \"classes\": [\n");
    for (i = 0; i < plan->nclasses; ++i) {
        const lt_class_plan_t *p = &plan->classes[i];
        fprintf(fp, "    {\"name\": "); lt_json_string(fp, classes[i].name);
        fprintf(fp, ", \"role\": "); lt_json_string(fp, lt_role_name(classes[i].role));
        fprintf(fp, ", \"bytes_each\": %llu, \"count\": %u, \"vram_count\": %u, \"ram_count\": %u, \"nvme_count\": %u, \"hot_mass_vram\": %.17g, \"hot_mass_ram\": %.17g, \"hot_mass_nvme\": %.17g}%s\n",
                (unsigned long long)classes[i].bytes_each,
                classes[i].count,
                p->vram_count, p->ram_count, p->nvme_count,
                p->hot_mass_vram, p->hot_mass_ram, p->hot_mass_nvme,
                i + 1 == plan->nclasses ? "" : ",");
    }
    fprintf(fp, "  ]\n}\n");
    return ferror(fp) ? -1 : 0;
}

static char *lt_trim(char *s) {
    char *end;
    while (*s && isspace((unsigned char)*s)) ++s;
    end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) --end;
    *end = '\0';
    return s;
}

static int lt_split_tabs(char *line, char **fields, int max_fields) {
    int n = 0;
    char *p = line;
    while (n < max_fields) {
        char *tab;
        fields[n++] = p;
        tab = strchr(p, '\t');
        if (!tab) break;
        *tab = '\0';
        p = tab + 1;
    }
    return n;
}

static int lt_parse_u64(const char *s, uint64_t *out) {
    char *end = NULL;
    unsigned long long v;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno || !end || *lt_trim(end) != '\0') return -1;
    *out = (uint64_t)v;
    return 0;
}

static int lt_parse_u32(const char *s, uint32_t *out) {
    uint64_t v;
    if (lt_parse_u64(s, &v) != 0 || v > UINT32_MAX) return -1;
    *out = (uint32_t)v;
    return 0;
}

static int lt_parse_double(const char *s, double *out) {
    char *end = NULL;
    double v;
    errno = 0;
    v = strtod(s, &end);
    if (errno || !end || *lt_trim(end) != '\0' || !isfinite(v)) return -1;
    *out = v;
    return 0;
}

static int lt_parse_flags(char *s, unsigned *out) {
    unsigned flags = 0;
    char *p = s;
    while (p && *p) {
        char *comma = strchr(p, ',');
        char *token;
        if (comma) *comma = '\0';
        token = lt_trim(p);
        if (*token == '\0' || strcmp(token, "none") == 0) {
            /* no-op */
        } else if (strcmp(token, "streamable") == 0) flags |= LT_CLASS_STREAMABLE;
        else if (strcmp(token, "gpu") == 0) flags |= LT_CLASS_GPU_CAPABLE;
        else if (strcmp(token, "ram_required") == 0) flags |= LT_CLASS_RAM_REQUIRED;
        else if (strcmp(token, "vram_required") == 0) flags |= LT_CLASS_VRAM_REQUIRED;
        else return -1;
        p = comma ? comma + 1 : NULL;
    }
    *out = flags;
    return 0;
}

int lt_manifest_load_tsv(const char *path,
                         lt_tensor_class_t **out_classes,
                         size_t *out_nclasses,
                         char *error,
                         size_t error_cap) {
    FILE *fp;
    lt_tensor_class_t *items = NULL;
    size_t n = 0, cap = 0;
    char line[8192];
    unsigned long line_no = 0;
    if (!path || !out_classes || !out_nclasses) {
        lt_set_error(error, error_cap, "manifest arguments are required");
        return -1;
    }
    *out_classes = NULL;
    *out_nclasses = 0;
    fp = fopen(path, "rb");
    if (!fp) {
        char msg[LT_ERROR_MAX];
        snprintf(msg, sizeof(msg), "cannot open manifest %s: %s", path, strerror(errno));
        lt_set_error(error, error_cap, msg);
        return -1;
    }
    while (fgets(line, sizeof(line), fp)) {
        char *fields[16];
        char *row;
        int nf;
        lt_tensor_class_t item;
        ++line_no;
        if (!strchr(line, '\n') && !feof(fp)) {
            snprintf(error, error_cap, "manifest line %lu is too long", line_no);
            free(items); fclose(fp); return -1;
        }
        row = lt_trim(line);
        if (*row == '\0' || *row == '#') continue;
        nf = lt_split_tabs(row, fields, 16);
        if (nf < 9 || nf > 11) {
            snprintf(error, error_cap, "manifest line %lu has %d fields; expected 9-11", line_no, nf);
            free(items); fclose(fp); return -1;
        }
        memset(&item, 0, sizeof(item));
        snprintf(item.name, sizeof(item.name), "%s", lt_trim(fields[0]));
        if (item.name[0] == '\0' || strlen(lt_trim(fields[0])) >= sizeof(item.name) ||
            lt_role_parse(lt_trim(fields[1]), &item.role) != 0 ||
            lt_parse_u64(lt_trim(fields[2]), &item.bytes_each) != 0 ||
            lt_parse_u32(lt_trim(fields[3]), &item.count) != 0 ||
            lt_parse_double(lt_trim(fields[4]), &item.touches_per_token) != 0 ||
            lt_parse_double(lt_trim(fields[5]), &item.skew) != 0 ||
            lt_parse_double(lt_trim(fields[6]), &item.cpu_ms_per_touch) != 0 ||
            lt_parse_double(lt_trim(fields[7]), &item.gpu_ms_per_touch) != 0 ||
            lt_parse_flags(fields[8], &item.flags) != 0 ||
            (nf >= 10 && lt_parse_u32(lt_trim(fields[9]), &item.min_ram_count) != 0) ||
            (nf >= 11 && lt_parse_u32(lt_trim(fields[10]), &item.min_vram_count) != 0)) {
            snprintf(error, error_cap, "manifest line %lu contains an invalid value", line_no);
            free(items); fclose(fp); return -1;
        }
        if (n == cap) {
            size_t next = cap ? cap * 2 : 16;
            lt_tensor_class_t *tmp = (lt_tensor_class_t *)realloc(items, next * sizeof(*items));
            if (!tmp) {
                lt_set_error(error, error_cap, "out of memory while reading manifest");
                free(items); fclose(fp); return -1;
            }
            items = tmp;
            cap = next;
        }
        items[n++] = item;
    }
    if (ferror(fp)) {
        snprintf(error, error_cap, "error reading manifest %s", path);
        free(items); fclose(fp); return -1;
    }
    fclose(fp);
    if (n == 0) {
        free(items);
        lt_set_error(error, error_cap, "manifest contains no tensor classes");
        return -1;
    }
    *out_classes = items;
    *out_nclasses = n;
    lt_set_error(error, error_cap, "");
    return 0;
}

void lt_manifest_free(lt_tensor_class_t *classes) {
    free(classes);
}
