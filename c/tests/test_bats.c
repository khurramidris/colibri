#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "../bats.h"

static bats_hw_profile profile(void) {
    bats_hw_profile hw;
    memset(&hw, 0, sizeof(hw));
    hw.bandwidth_gbps[BATS_TIER_VRAM] = 300.0;
    hw.bandwidth_gbps[BATS_TIER_PINNED] = 24.0;
    hw.bandwidth_gbps[BATS_TIER_RAM] = 12.0;
    hw.bandwidth_gbps[BATS_TIER_NVME] = 4.0;
    hw.bandwidth_gbps[BATS_TIER_REMOTE] = 1.0;
    hw.fixed_us[BATS_TIER_NVME] = 80.0;
    hw.queue_us[BATS_TIER_NVME] = 20.0;
    return hw;
}

static void test_transfer_cost(void) {
    bats_hw_profile hw = profile();
    bats_expert_state e = { .bytes = 4 * 1000 * 1000, .tier = BATS_TIER_NVME };
    double us = bats_expert_transfer_us(&hw, &e);
    assert(fabs(us - 1100.0) < 1e-9);
    e.resident_in_exec = 1;
    assert(bats_expert_transfer_us(&hw, &e) == 0.0);
    e.resident_in_exec = 0; e.in_flight = 1; e.remaining_us = 17.5;
    assert(bats_expert_transfer_us(&hw, &e) == 17.5);
}

static void test_resident_path_beats_fewer_cold_experts(void) {
    bats_hw_profile hw = profile();
    bats_expert_state experts[8];
    memset(experts, 0, sizeof(experts));
    for (int i = 0; i < 6; ++i) {
        experts[i].bytes = 4 * 1000 * 1000;
        experts[i].tier = BATS_TIER_EXEC;
        experts[i].resident_in_exec = 1;
    }
    for (int i = 6; i < 8; ++i) {
        experts[i].bytes = 4 * 1000 * 1000;
        experts[i].tier = BATS_TIER_NVME;
    }

    const int resident_route[] = {0,1,2,3,4,5};
    const int cold_route[] = {6,7};
    bats_candidate c[] = {
        {.id=10, .expected_accepted_tokens=1.8, .verify_compute_us=50,
         .expert_ids=resident_route, .expert_count=6},
        {.id=20, .expected_accepted_tokens=2.0, .verify_compute_us=50,
         .expert_ids=cold_route, .expert_count=2}
    };
    bats_plan plan;
    bats_limits lim = {.max_candidates=1};
    assert(bats_plan_candidates(&hw, experts, 8, c, 2, lim, &plan) == 0);
    assert(plan.selected_count == 1);
    assert(plan.selected[0] == 0);
    assert(plan.marginal_bytes == 0);
}

static void test_union_reuse_and_budget(void) {
    bats_hw_profile hw = profile();
    bats_expert_state experts[4];
    memset(experts, 0, sizeof(experts));
    for (int i = 0; i < 4; ++i) {
        experts[i].bytes = 1000 * 1000;
        experts[i].tier = BATS_TIER_NVME;
    }
    const int a[] = {0,1};
    const int b[] = {1,2};
    const int croute[] = {3};
    bats_candidate c[] = {
        {.id=1, .expected_accepted_tokens=2.0, .expert_ids=a, .expert_count=2},
        {.id=2, .expected_accepted_tokens=1.9, .expert_ids=b, .expert_count=2},
        {.id=3, .expected_accepted_tokens=0.2, .expert_ids=croute, .expert_count=1}
    };
    bats_plan plan;
    bats_limits lim = {.max_candidates=3, .budget_us=1100.0, .minimum_gain=0.1};
    assert(bats_plan_candidates(&hw, experts, 4, c, 3, lim, &plan) == 0);
    assert(plan.selected_count == 2);
    assert(plan.selected[0] == 0);
    assert(plan.selected[1] == 1);
    assert(plan.marginal_bytes == 3ULL * 1000 * 1000);
    assert(fabs(plan.predicted_us - 1050.0) < 1e-9);
}

static void test_deterministic_tie_break(void) {
    bats_hw_profile hw = profile();
    bats_expert_state experts[2];
    memset(experts, 0, sizeof(experts));
    for (int i = 0; i < 2; ++i) {
        experts[i].bytes = 1000 * 1000;
        experts[i].tier = BATS_TIER_NVME;
    }
    const int e0[] = {0};
    const int e1[] = {1};
    bats_candidate c[] = {
        {.id=9, .expected_accepted_tokens=1.0, .expert_ids=e0, .expert_count=1},
        {.id=4, .expected_accepted_tokens=1.0, .expert_ids=e1, .expert_count=1}
    };
    bats_plan plan;
    bats_limits lim = {.max_candidates=1};
    assert(bats_plan_candidates(&hw, experts, 2, c, 2, lim, &plan) == 0);
    assert(plan.selected[0] == 1);
}

static void test_prefetch_admission(void) {
    bats_hw_profile hw = profile();
    bats_expert_state cold = {.bytes=4*1000*1000, .tier=BATS_TIER_NVME};
    assert(bats_admit_prefetch(&hw, &cold, 0.9, 0.5) == 1);
    assert(bats_admit_prefetch(&hw, &cold, 0.01, 0.5) == 0);
    cold.resident_in_exec = 1;
    assert(bats_admit_prefetch(&hw, &cold, 1.0, 0.0) == 0);
}

int main(void) {
    test_transfer_cost();
    test_resident_path_beats_fewer_cold_experts();
    test_union_reuse_and_budget();
    test_deterministic_tie_break();
    test_prefetch_admission();
    puts("test_bats: ok");
    return 0;
}
