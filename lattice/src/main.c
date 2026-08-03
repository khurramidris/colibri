#include "lattice/planner.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GIB (1024.0 * 1024.0 * 1024.0)

static void usage(FILE *fp, const char *argv0) {
    fprintf(fp,
        "usage: %s [--manifest FILE | --demo] [options]\n"
        "\n"
        "Exact model-wide placement planner for NVMe/RAM/VRAM.\n"
        "\n"
        "options:\n"
        "  --ram-gib N       physical RAM budget (default 16)\n"
        "  --vram-gib N      physical VRAM budget (default 8)\n"
        "  --nvme-gbps N     measured aggregate NVMe read bandwidth (default 7)\n"
        "  --ram-gbps N      measured RAM bandwidth, metadata only for v1 (default 60)\n"
        "  --h2d-gbps N      measured host-to-device bandwidth (default 24)\n"
        "  --overlap F       transfer/compute overlap efficiency 0..1 (default 0.65)\n"
        "  --reserve F       RAM/VRAM reserve fraction 0..0.9 (default 0.10)\n"
        "  --json FILE       write machine-readable lattice.plan.v1 JSON; '-' means stdout\n"
        "  --quiet           suppress human-readable table\n"
        "  --help            show this help\n"
        "\n"
        "Manifest columns are documented in lattice/include/lattice/planner.h.\n",
        argv0);
}

static int parse_double_arg(const char *name, const char *text, double *out) {
    char *end = NULL;
    double value;
    errno = 0;
    value = strtod(text, &end);
    if (errno || !end || *end || !isfinite(value)) {
        fprintf(stderr, "%s: invalid number: %s\n", name, text);
        return -1;
    }
    *out = value;
    return 0;
}

static int gib_to_bytes(const char *name, double gib, uint64_t *out) {
    long double value;
    if (!isfinite(gib) || gib < 0.0) {
        fprintf(stderr, "%s must be non-negative\n", name);
        return -1;
    }
    value = (long double)gib * (long double)GIB;
    if (value > (long double)UINT64_MAX) {
        fprintf(stderr, "%s is too large\n", name);
        return -1;
    }
    *out = (uint64_t)value;
    return 0;
}

static void set_class(lt_tensor_class_t *c,
                      const char *name,
                      lt_role_t role,
                      uint64_t bytes_each,
                      uint32_t count,
                      double touches,
                      double skew,
                      double cpu_ms,
                      double gpu_ms,
                      unsigned flags,
                      uint32_t min_ram,
                      uint32_t min_vram) {
    memset(c, 0, sizeof(*c));
    snprintf(c->name, sizeof(c->name), "%s", name);
    c->role = role;
    c->bytes_each = bytes_each;
    c->count = count;
    c->touches_per_token = touches;
    c->skew = skew;
    c->cpu_ms_per_touch = cpu_ms;
    c->gpu_ms_per_touch = gpu_ms;
    c->flags = flags;
    c->min_ram_count = min_ram;
    c->min_vram_count = min_vram;
}

static int make_demo(lt_tensor_class_t **out, size_t *n) {
    lt_tensor_class_t *classes = (lt_tensor_class_t *)calloc(8, sizeof(*classes));
    if (!classes) return -1;
    /* Synthetic 35B-class MoE. The numbers are deliberately transparent and
     * are not claims about a named checkpoint. They exercise every placement
     * rule under constrained RAM and VRAM. */
    set_class(&classes[0], "attention_blocks", LT_ROLE_ATTENTION,
              320ULL << 20, 40, 40.0, 0.0, 1.50, 0.35,
              LT_CLASS_STREAMABLE | LT_CLASS_GPU_CAPABLE, 0, 0);
    set_class(&classes[1], "shared_experts", LT_ROLE_SHARED_EXPERT,
              240ULL << 20, 40, 40.0, 0.0, 1.10, 0.28,
              LT_CLASS_STREAMABLE | LT_CLASS_GPU_CAPABLE, 0, 0);
    set_class(&classes[2], "routed_experts", LT_ROLE_ROUTED_EXPERT,
              26ULL << 20, 10240, 320.0, 0.85, 0.08, 0.022,
              LT_CLASS_STREAMABLE | LT_CLASS_GPU_CAPABLE, 0, 0);
    set_class(&classes[3], "router_and_norms", LT_ROLE_NORM,
              16ULL << 20, 40, 40.0, 0.0, 0.08, 0.03,
              LT_CLASS_RAM_REQUIRED | LT_CLASS_GPU_CAPABLE, 0, 0);
    set_class(&classes[4], "embeddings", LT_ROLE_EMBEDDING,
              1200ULL << 20, 1, 1.0, 0.0, 3.0, 0.8,
              LT_CLASS_STREAMABLE | LT_CLASS_GPU_CAPABLE, 0, 0);
    set_class(&classes[5], "lm_head", LT_ROLE_EMBEDDING,
              1200ULL << 20, 1, 1.0, 0.0, 3.0, 0.8,
              LT_CLASS_STREAMABLE | LT_CLASS_GPU_CAPABLE, 0, 0);
    set_class(&classes[6], "kv_cache", LT_ROLE_KV_CACHE,
              1024ULL << 20, 1, 1.0, 0.0, 0.1, 0.05,
              LT_CLASS_RAM_REQUIRED | LT_CLASS_GPU_CAPABLE, 0, 0);
    set_class(&classes[7], "recurrent_state", LT_ROLE_RECURRENT_STATE,
              768ULL << 20, 1, 1.0, 0.0, 0.2, 0.08,
              LT_CLASS_RAM_REQUIRED | LT_CLASS_GPU_CAPABLE, 0, 0);
    *out = classes;
    *n = 8;
    return 0;
}

int main(int argc, char **argv) {
    const char *manifest = NULL;
    const char *json_path = NULL;
    int demo = 0, quiet = 0;
    double ram_gib = 16.0, vram_gib = 8.0;
    lt_hardware_t hw;
    lt_tensor_class_t *classes = NULL;
    size_t nclasses = 0;
    lt_plan_t plan;
    char error[LT_ERROR_MAX];
    int i, rc = 1;

    memset(&hw, 0, sizeof(hw));
    hw.nvme_read_gbps = 7.0;
    hw.ram_read_gbps = 60.0;
    hw.host_to_device_gbps = 24.0;
    hw.overlap_efficiency = 0.65;
    hw.reserve_fraction = 0.10;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout, argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--demo") == 0) {
            demo = 1;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            quiet = 1;
        } else if (strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) {
            manifest = argv[++i];
        } else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else if (strcmp(argv[i], "--ram-gib") == 0 && i + 1 < argc) {
            if (parse_double_arg("--ram-gib", argv[++i], &ram_gib) != 0) return 2;
        } else if (strcmp(argv[i], "--vram-gib") == 0 && i + 1 < argc) {
            if (parse_double_arg("--vram-gib", argv[++i], &vram_gib) != 0) return 2;
        } else if (strcmp(argv[i], "--nvme-gbps") == 0 && i + 1 < argc) {
            if (parse_double_arg("--nvme-gbps", argv[++i], &hw.nvme_read_gbps) != 0) return 2;
        } else if (strcmp(argv[i], "--ram-gbps") == 0 && i + 1 < argc) {
            if (parse_double_arg("--ram-gbps", argv[++i], &hw.ram_read_gbps) != 0) return 2;
        } else if (strcmp(argv[i], "--h2d-gbps") == 0 && i + 1 < argc) {
            if (parse_double_arg("--h2d-gbps", argv[++i], &hw.host_to_device_gbps) != 0) return 2;
        } else if (strcmp(argv[i], "--overlap") == 0 && i + 1 < argc) {
            if (parse_double_arg("--overlap", argv[++i], &hw.overlap_efficiency) != 0) return 2;
        } else if (strcmp(argv[i], "--reserve") == 0 && i + 1 < argc) {
            if (parse_double_arg("--reserve", argv[++i], &hw.reserve_fraction) != 0) return 2;
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", argv[i]);
            usage(stderr, argv[0]);
            return 2;
        }
    }

    if (!!manifest + !!demo != 1) {
        fprintf(stderr, "choose exactly one of --manifest FILE or --demo\n");
        usage(stderr, argv[0]);
        return 2;
    }
    if (gib_to_bytes("--ram-gib", ram_gib, &hw.ram_budget_bytes) != 0 ||
        gib_to_bytes("--vram-gib", vram_gib, &hw.vram_budget_bytes) != 0)
        return 2;

    if (demo) {
        if (make_demo(&classes, &nclasses) != 0) {
            fprintf(stderr, "out of memory creating demo manifest\n");
            return 1;
        }
    } else if (lt_manifest_load_tsv(manifest, &classes, &nclasses,
                                    error, sizeof(error)) != 0) {
        fprintf(stderr, "manifest: %s\n", error);
        return 1;
    }

    memset(&plan, 0, sizeof(plan));
    if (lt_plan_build(&hw, classes, nclasses, &plan) != 0) {
        fprintf(stderr, "planning failed: %s\n", plan.error[0] ? plan.error : "unknown error");
        goto done;
    }
    if (!quiet) lt_plan_print(stdout, &hw, classes, &plan);
    if (json_path) {
        FILE *fp = stdout;
        if (strcmp(json_path, "-") != 0) {
            fp = fopen(json_path, "wb");
            if (!fp) {
                fprintf(stderr, "cannot open %s for writing: %s\n", json_path, strerror(errno));
                goto done;
            }
        }
        if (lt_plan_write_json(fp, &hw, classes, &plan) != 0) {
            fprintf(stderr, "failed to write plan JSON\n");
            if (fp != stdout) fclose(fp);
            goto done;
        }
        if (fp != stdout && fclose(fp) != 0) {
            fprintf(stderr, "failed to close %s\n", json_path);
            goto done;
        }
    }
    rc = 0;

done:
    lt_plan_free(&plan);
    lt_manifest_free(classes);
    return rc;
}
