#include "lattice/planner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GIB ((uint64_t)1024 * 1024 * 1024)
#define MIB ((uint64_t)1024 * 1024)

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static lt_hardware_t hardware(uint64_t ram, uint64_t vram) {
    lt_hardware_t hw;
    memset(&hw, 0, sizeof(hw));
    hw.ram_budget_bytes = ram;
    hw.vram_budget_bytes = vram;
    hw.nvme_read_gbps = 5.0;
    hw.ram_read_gbps = 50.0;
    hw.host_to_device_gbps = 20.0;
    hw.overlap_efficiency = 0.5;
    hw.reserve_fraction = 0.0;
    return hw;
}

static lt_tensor_class_t tensor(const char *name,
                                lt_role_t role,
                                uint64_t bytes,
                                uint32_t count,
                                double touches,
                                double skew,
                                unsigned flags) {
    lt_tensor_class_t c;
    memset(&c, 0, sizeof(c));
    snprintf(c.name, sizeof(c.name), "%s", name);
    c.role = role;
    c.bytes_each = bytes;
    c.count = count;
    c.touches_per_token = touches;
    c.skew = skew;
    c.cpu_ms_per_touch = 0.1;
    c.gpu_ms_per_touch = 0.02;
    c.flags = flags;
    return c;
}

static void test_budget_and_counts(void) {
    lt_tensor_class_t c[3];
    lt_hardware_t hw = hardware(4 * GIB, 2 * GIB);
    lt_plan_t plan;
    size_t i;
    memset(&plan, 0, sizeof(plan));
    c[0] = tensor("required", LT_ROLE_NORM, GIB, 1, 1.0, 0.0,
                  LT_CLASS_RAM_REQUIRED);
    c[1] = tensor("experts", LT_ROLE_ROUTED_EXPERT, 64 * MIB, 100, 8.0, 1.0,
                  LT_CLASS_STREAMABLE | LT_CLASS_GPU_CAPABLE);
    c[2] = tensor("dense", LT_ROLE_DENSE, 256 * MIB, 8, 8.0, 0.0,
                  LT_CLASS_STREAMABLE | LT_CLASS_GPU_CAPABLE);
    CHECK(lt_plan_build(&hw, c, 3, &plan) == 0);
    CHECK(plan.feasible == 1);
    CHECK(plan.ram_used_bytes <= hw.ram_budget_bytes);
    CHECK(plan.vram_used_bytes <= hw.vram_budget_bytes);
    for (i = 0; i < 3; ++i)
        CHECK(plan.classes[i].ram_count + plan.classes[i].vram_count +
              plan.classes[i].nvme_count == c[i].count);
    CHECK(plan.classes[0].nvme_count == 0);
    lt_plan_free(&plan);
}

static void test_more_ram_reduces_traffic(void) {
    lt_tensor_class_t c = tensor("experts", LT_ROLE_ROUTED_EXPERT,
                                 32 * MIB, 512, 8.0, 0.9,
                                 LT_CLASS_STREAMABLE);
    lt_hardware_t small = hardware(1 * GIB, 0);
    lt_hardware_t large = hardware(8 * GIB, 0);
    lt_plan_t a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    CHECK(lt_plan_build(&small, &c, 1, &a) == 0);
    CHECK(lt_plan_build(&large, &c, 1, &b) == 0);
    CHECK(b.nvme_bytes_per_token < a.nvme_bytes_per_token);
    CHECK(b.predicted_ms_per_token < a.predicted_ms_per_token);
    CHECK(b.classes[0].ram_count > a.classes[0].ram_count);
    lt_plan_free(&a);
    lt_plan_free(&b);
}

static void test_hot_class_wins_first_byte(void) {
    lt_tensor_class_t c[2];
    lt_hardware_t hw = hardware(64 * MIB, 0);
    lt_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    c[0] = tensor("hot", LT_ROLE_ROUTED_EXPERT, 64 * MIB, 8, 20.0, 0.0,
                  LT_CLASS_STREAMABLE);
    c[1] = tensor("cold", LT_ROLE_ROUTED_EXPERT, 64 * MIB, 8, 1.0, 0.0,
                  LT_CLASS_STREAMABLE);
    CHECK(lt_plan_build(&hw, c, 2, &plan) == 0);
    CHECK(plan.classes[0].ram_count == 1);
    CHECK(plan.classes[1].ram_count == 0);
    lt_plan_free(&plan);
}

static void test_infeasible_required_memory(void) {
    lt_tensor_class_t c = tensor("state", LT_ROLE_RECURRENT_STATE,
                                 2 * GIB, 1, 1.0, 0.0,
                                 LT_CLASS_RAM_REQUIRED);
    lt_hardware_t hw = hardware(1 * GIB, 0);
    lt_plan_t plan;
    memset(&plan, 0, sizeof(plan));
    CHECK(lt_plan_build(&hw, &c, 1, &plan) != 0);
    CHECK(plan.feasible == 0);
    CHECK(plan.error[0] != '\0');
    lt_plan_free(&plan);
}

static void test_manifest_parser(void) {
    char path[256];
    FILE *fp;
    lt_tensor_class_t *classes = NULL;
    size_t nclasses = 0;
    char error[LT_ERROR_MAX];
    snprintf(path, sizeof(path), "/tmp/lattice-manifest-%ld.tsv", (long)getpid());
    fp = fopen(path, "wb");
    CHECK(fp != NULL);
    if (!fp) return;
    fputs("# name role bytes count touches skew cpu gpu flags min_ram min_vram\n", fp);
    fputs("experts\trouted_expert\t33554432\t256\t8\t0.8\t0.1\t0.02\tstreamable,gpu\t0\t0\n", fp);
    fputs("state\trecurrent_state\t1048576\t1\t1\t0\t0.1\t0.1\tram_required\t0\t0\n", fp);
    CHECK(fclose(fp) == 0);
    CHECK(lt_manifest_load_tsv(path, &classes, &nclasses, error, sizeof(error)) == 0);
    CHECK(nclasses == 2);
    if (nclasses == 2) {
        CHECK(classes[0].role == LT_ROLE_ROUTED_EXPERT);
        CHECK(classes[0].flags == (LT_CLASS_STREAMABLE | LT_CLASS_GPU_CAPABLE));
        CHECK(classes[1].flags == LT_CLASS_RAM_REQUIRED);
    }
    lt_manifest_free(classes);
    unlink(path);
}

int main(void) {
    test_budget_and_counts();
    test_more_ram_reduces_traffic();
    test_hot_class_wins_first_byte();
    test_infeasible_required_memory();
    test_manifest_parser();
    if (failures) {
        fprintf(stderr, "%d planner test(s) failed\n", failures);
        return 1;
    }
    puts("planner tests passed");
    return 0;
}
