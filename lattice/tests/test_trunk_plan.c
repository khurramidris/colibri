#include "lattice/trunk_plan.h"

#include <stdio.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static void test_prefix_and_ring(void) {
    const uint64_t layers[] = {100, 80, 60};
    lt_trunk_plan_t plan;
    CHECK(lt_trunk_plan_build(layers, 3, 224, 16, 1, 16, &plan) == 0);
    CHECK(plan.feasible == 1);
    CHECK(plan.pinned_layers == 1);
    CHECK(plan.pinned_allocation_bytes == 128);
    CHECK(plan.ring_slot_bytes == 96);
    CHECK(plan.ring_allocation_bytes == 96);
    CHECK(plan.total_allocation_bytes == 224);
    CHECK(plan.streamed_bytes_per_token == 140);
}

static void test_all_pinned_needs_no_ring(void) {
    const uint64_t layers[] = {100, 80, 60};
    lt_trunk_plan_t plan;
    CHECK(lt_trunk_plan_build(layers, 3, 304, 16, 1, 16, &plan) == 0);
    CHECK(plan.pinned_layers == 3);
    CHECK(plan.pinned_allocation_bytes == 304);
    CHECK(plan.ring_slot_bytes == 0);
    CHECK(plan.ring_allocation_bytes == 0);
    CHECK(plan.streamed_bytes_per_token == 0);
}

static void test_floor_is_largest_layer_ring(void) {
    const uint64_t layers[] = {100, 80, 60};
    lt_trunk_plan_t plan;
    CHECK(lt_trunk_plan_build(layers, 3, 127, 16, 1, 16, &plan) != 0);
    CHECK(plan.feasible == 0);
    CHECK(plan.error[0] != '\0');
    CHECK(lt_trunk_plan_build(layers, 3, 128, 16, 1, 16, &plan) == 0);
    CHECK(plan.pinned_layers == 0);
    CHECK(plan.ring_slot_bytes == 128);
}

static void test_two_ring_slots_raise_floor(void) {
    const uint64_t layers[] = {100, 80, 60};
    lt_trunk_plan_t plan;
    CHECK(lt_trunk_plan_build(layers, 3, 255, 16, 2, 16, &plan) != 0);
    CHECK(lt_trunk_plan_build(layers, 3, 256, 16, 2, 16, &plan) == 0);
    CHECK(plan.pinned_layers == 0);
    CHECK(plan.ring_allocation_bytes == 256);
}

static void test_more_budget_is_monotonic(void) {
    const uint64_t layers[] = {100, 80, 60, 40};
    lt_trunk_plan_t small, large;
    CHECK(lt_trunk_plan_build(layers, 4, 224, 16, 1, 16, &small) == 0);
    CHECK(lt_trunk_plan_build(layers, 4, 400, 16, 1, 16, &large) == 0);
    CHECK(large.pinned_layers >= small.pinned_layers);
    CHECK(large.streamed_bytes_per_token <= small.streamed_bytes_per_token);
}

int main(void) {
    test_prefix_and_ring();
    test_all_pinned_needs_no_ring();
    test_floor_is_largest_layer_ring();
    test_two_ring_slots_raise_floor();
    test_more_budget_is_monotonic();
    if (failures) {
        fprintf(stderr, "%d trunk plan test(s) failed\n", failures);
        return 1;
    }
    puts("trunk plan tests passed");
    return 0;
}
