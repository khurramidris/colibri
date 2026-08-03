#ifndef COLIBRI_BATS_H
#define COLIBRI_BATS_H

/*
 * BATS-MoE: byte-aware tiered scheduling primitives.
 *
 * This header is deliberately dependency-free and side-effect-free. It does
 * not alter router decisions or model arithmetic. Runtime integrations provide
 * a snapshot of expert placement and candidate routes; BATS only predicts the
 * marginal transfer cost and chooses a verification/prefetch set.
 */

#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef BATS_MAX_CANDIDATES
#define BATS_MAX_CANDIDATES 64
#endif

#ifndef BATS_MAX_EXPERTS
#define BATS_MAX_EXPERTS 4096
#endif

typedef enum {
    BATS_TIER_EXEC = 0,
    BATS_TIER_VRAM = 1,
    BATS_TIER_PINNED = 2,
    BATS_TIER_RAM = 3,
    BATS_TIER_NVME = 4,
    BATS_TIER_REMOTE = 5,
    BATS_TIER_COUNT = 6
} bats_tier;

typedef struct {
    double bandwidth_gbps[BATS_TIER_COUNT];
    double fixed_us[BATS_TIER_COUNT];
    double queue_us[BATS_TIER_COUNT];
    double overlap[BATS_TIER_COUNT];
} bats_hw_profile;

typedef struct {
    uint64_t bytes;
    uint64_t remaining_bytes;
    bats_tier tier;
    uint8_t resident_in_exec;
    uint8_t in_flight;
    double remaining_us;
} bats_expert_state;

typedef struct {
    int id;
    double expected_accepted_tokens;
    double verify_compute_us;
    double kv_us;
    const int *expert_ids;
    size_t expert_count;
} bats_candidate;

typedef struct {
    int selected[BATS_MAX_CANDIDATES];
    size_t selected_count;
    uint64_t marginal_bytes;
    double predicted_us;
    double expected_accepted_tokens;
    double objective;
} bats_plan;

typedef struct {
    size_t max_candidates;
    double budget_us;
    double minimum_gain;
} bats_limits;

static inline double bats_clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

static inline uint64_t bats_expert_marginal_bytes(const bats_expert_state *expert) {
    if (!expert || expert->resident_in_exec || expert->tier == BATS_TIER_EXEC)
        return 0;
    if (expert->in_flight && expert->remaining_bytes > 0)
        return expert->remaining_bytes < expert->bytes
            ? expert->remaining_bytes : expert->bytes;
    return expert->bytes;
}

static inline double bats_expert_transfer_us(const bats_hw_profile *hw,
                                              const bats_expert_state *expert) {
    if (!hw || !expert || expert->resident_in_exec || expert->tier == BATS_TIER_EXEC)
        return 0.0;
    if (expert->in_flight)
        return expert->remaining_us > 0.0 ? expert->remaining_us : 0.0;
    if (expert->tier < 0 || expert->tier >= BATS_TIER_COUNT)
        return DBL_MAX;

    const double gbps = hw->bandwidth_gbps[expert->tier];
    if (gbps <= 0.0) return DBL_MAX;

    /* 1 GB/s == 1000 bytes/us. Calibration supplies the effective rate. */
    const double transfer_us =
        (double)bats_expert_marginal_bytes(expert) / (gbps * 1000.0);
    const double exposed =
        transfer_us * (1.0 - bats_clamp01(hw->overlap[expert->tier]));
    return hw->fixed_us[expert->tier] + hw->queue_us[expert->tier] + exposed;
}

static inline int bats_valid_expert_id(int eid, size_t expert_count) {
    return eid >= 0 && (size_t)eid < expert_count;
}

/* Cost of one candidate relative to an existing expert union. `new_mask` is
 * overwritten and marks only newly required experts. */
static inline double bats_candidate_marginal_cost(
        const bats_hw_profile *hw,
        const bats_expert_state *experts,
        size_t expert_count,
        const bats_candidate *candidate,
        const uint8_t *union_mask,
        uint8_t *new_mask,
        uint64_t *new_bytes) {
    size_t i;
    double cost = 0.0;
    uint64_t bytes = 0;
    if (!hw || !experts || !candidate || !new_mask ||
        expert_count > BATS_MAX_EXPERTS)
        return DBL_MAX;

    memset(new_mask, 0, expert_count);
    for (i = 0; i < candidate->expert_count; ++i) {
        const int eid = candidate->expert_ids[i];
        if (!bats_valid_expert_id(eid, expert_count)) return DBL_MAX;
        if ((union_mask && union_mask[eid]) || new_mask[eid]) continue;
        new_mask[eid] = 1;
        if (!experts[eid].resident_in_exec &&
            experts[eid].tier != BATS_TIER_EXEC) {
            const double t = bats_expert_transfer_us(hw, &experts[eid]);
            if (t == DBL_MAX) return DBL_MAX;
            cost += t;
            bytes += bats_expert_marginal_bytes(&experts[eid]);
        }
    }
    if (new_bytes) *new_bytes = bytes;
    return cost + candidate->verify_compute_us + candidate->kv_us;
}

/* Greedy union-aware planner. Every iteration scores remaining candidates
 * against the union already selected. Ties are deterministic. */
static inline int bats_plan_candidates(
        const bats_hw_profile *hw,
        const bats_expert_state *experts,
        size_t expert_count,
        const bats_candidate *candidates,
        size_t candidate_count,
        bats_limits limits,
        bats_plan *out) {
    uint8_t union_mask[BATS_MAX_EXPERTS];
    uint8_t trial_mask[BATS_MAX_EXPERTS];
    uint8_t chosen[BATS_MAX_CANDIDATES];
    size_t step;

    if (!out || !hw || !experts || !candidates ||
        expert_count > BATS_MAX_EXPERTS ||
        candidate_count > BATS_MAX_CANDIDATES)
        return -1;

    memset(out, 0, sizeof(*out));
    memset(union_mask, 0, expert_count);
    memset(chosen, 0, candidate_count);
    if (limits.max_candidates == 0 || limits.max_candidates > candidate_count)
        limits.max_candidates = candidate_count;

    for (step = 0; step < limits.max_candidates; ++step) {
        int best = -1;
        double best_score = -DBL_MAX;
        double best_cost = 0.0;
        uint64_t best_bytes = 0;
        size_t i;

        for (i = 0; i < candidate_count; ++i) {
            uint64_t bytes = 0;
            double cost;
            double score;
            if (chosen[i]) continue;
            if (candidates[i].expected_accepted_tokens <= limits.minimum_gain)
                continue;

            cost = bats_candidate_marginal_cost(
                hw, experts, expert_count, &candidates[i],
                union_mask, trial_mask, &bytes);
            if (cost == DBL_MAX) continue;
            if (limits.budget_us > 0.0 &&
                out->predicted_us + cost > limits.budget_us)
                continue;

            score = candidates[i].expected_accepted_tokens / (cost + 1e-9);
            if (best < 0 || score > best_score + 1e-15 ||
                ((score >= best_score - 1e-15) &&
                 (candidates[i].id < candidates[best].id ||
                  (candidates[i].id == candidates[best].id &&
                   (int)i < best)))) {
                best = (int)i;
                best_score = score;
                best_cost = cost;
                best_bytes = bytes;
            }
        }

        if (best < 0) break;
        chosen[best] = 1;
        out->selected[out->selected_count++] = best;
        out->predicted_us += best_cost;
        out->marginal_bytes += best_bytes;
        out->expected_accepted_tokens +=
            candidates[best].expected_accepted_tokens;

        for (size_t j = 0; j < candidates[best].expert_count; ++j) {
            int eid = candidates[best].expert_ids[j];
            if (bats_valid_expert_id(eid, expert_count))
                union_mask[eid] = 1;
        }
    }

    out->objective =
        out->expected_accepted_tokens / (out->predicted_us + 1e-9);
    return 0;
}

/* Admission primitive for the current PILOT path. */
static inline int bats_admit_prefetch(const bats_hw_profile *hw,
                                      const bats_expert_state *expert,
                                      double benefit,
                                      double min_benefit_per_ms) {
    const double us = bats_expert_transfer_us(hw, expert);
    if (expert && expert->resident_in_exec) return 0;
    if (benefit <= 0.0 || us == DBL_MAX) return 0;
    if (us <= 0.0) return 1;
    return (benefit / (us / 1000.0)) >= min_benefit_per_ms;
}

#endif /* COLIBRI_BATS_H */
