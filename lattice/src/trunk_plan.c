#include "lattice/trunk_plan.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return -1;
    *out = a + b;
    return 0;
}

static int mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return -1;
    *out = a * b;
    return 0;
}

static int align_u64(uint64_t value, uint64_t alignment, uint64_t *out) {
    uint64_t add;
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return -1;
    add = alignment - 1;
    if (value > UINT64_MAX - add) return -1;
    *out = (value + add) & ~add;
    return 0;
}

static int layer_allocation(uint64_t layer_bytes,
                            uint64_t widen_bytes,
                            uint64_t alignment,
                            uint64_t *out) {
    uint64_t aligned, total;
    if (align_u64(layer_bytes, alignment, &aligned) != 0 ||
        add_u64(aligned, widen_bytes, &total) != 0 ||
        align_u64(total, alignment, out) != 0)
        return -1;
    return 0;
}

static void set_error(lt_trunk_plan_t *out, const char *message) {
    snprintf(out->error, sizeof(out->error), "%s", message ? message : "trunk plan error");
}

int lt_trunk_plan_build(const uint64_t *layer_bytes,
                        uint32_t layers,
                        uint64_t budget_bytes,
                        uint64_t alignment,
                        uint32_t ring_slots,
                        uint64_t widen_bytes_per_slot,
                        lt_trunk_plan_t *out) {
    uint64_t *suffix_max = NULL;
    uint64_t packed = 0;
    uint64_t prefix_allocation = 0;
    uint32_t best_pin = UINT32_MAX;
    uint64_t best_slot = 0, best_ring = 0, best_total = 0, best_pinned = 0;
    uint32_t layer, pin;

    if (!out) return -1;
    memset(out, 0, sizeof(*out));
    out->layers = layers;
    out->ring_slots = ring_slots;
    out->alignment = alignment;
    out->widen_bytes_per_slot = widen_bytes_per_slot;
    out->budget_bytes = budget_bytes;
    if (!layer_bytes || layers == 0 || ring_slots == 0 ||
        alignment == 0 || (alignment & (alignment - 1)) != 0) {
        set_error(out, "invalid trunk geometry");
        return -1;
    }

    suffix_max = (uint64_t *)calloc((size_t)layers + 1, sizeof(*suffix_max));
    if (!suffix_max) {
        set_error(out, "out of memory planning trunk");
        return -1;
    }
    for (layer = 0; layer < layers; ++layer) {
        uint64_t ignored;
        if (layer_bytes[layer] == 0 ||
            add_u64(packed, layer_bytes[layer], &packed) != 0 ||
            layer_allocation(layer_bytes[layer], widen_bytes_per_slot,
                             alignment, &ignored) != 0) {
            set_error(out, "layer size is zero or overflows allocation arithmetic");
            free(suffix_max);
            return -1;
        }
    }
    out->packed_bytes = packed;
    suffix_max[layers] = 0;
    for (layer = layers; layer > 0; --layer) {
        uint64_t current = layer_bytes[layer - 1];
        suffix_max[layer - 1] = current > suffix_max[layer] ? current : suffix_max[layer];
    }

    /* Enumerate every possible pinned prefix. For prefix p, the ring only has
     * to hold layers p..N-1. This solves the circular dependency exactly and
     * selects the deepest feasible prefix. */
    for (pin = 0; pin <= layers; ++pin) {
        uint64_t slot = 0, ring = 0, total;
        if (suffix_max[pin] != 0 &&
            layer_allocation(suffix_max[pin], widen_bytes_per_slot,
                             alignment, &slot) != 0) {
            set_error(out, "ring slot size overflow");
            free(suffix_max);
            return -1;
        }
        if (mul_u64(slot, ring_slots, &ring) != 0 ||
            add_u64(prefix_allocation, ring, &total) != 0) {
            set_error(out, "trunk allocation overflow");
            free(suffix_max);
            return -1;
        }
        if (total <= budget_bytes) {
            best_pin = pin;
            best_slot = slot;
            best_ring = ring;
            best_total = total;
            best_pinned = prefix_allocation;
        }
        if (pin < layers) {
            uint64_t need;
            if (layer_allocation(layer_bytes[pin], widen_bytes_per_slot,
                                 alignment, &need) != 0 ||
                add_u64(prefix_allocation, need, &prefix_allocation) != 0) {
                set_error(out, "pinned layer allocation overflow");
                free(suffix_max);
                return -1;
            }
        }
    }

    if (best_pin == UINT32_MAX) {
        set_error(out, "budget cannot hold one streaming ring for the largest unpinned layer");
        free(suffix_max);
        return -1;
    }

    out->streamed_bytes_per_token = 0;
    for (layer = best_pin; layer < layers; ++layer) {
        if (add_u64(out->streamed_bytes_per_token, layer_bytes[layer],
                    &out->streamed_bytes_per_token) != 0) {
            set_error(out, "streamed byte total overflow");
            free(suffix_max);
            return -1;
        }
    }
    out->pinned_layers = best_pin;
    out->pinned_allocation_bytes = best_pinned;
    out->ring_slot_bytes = best_slot;
    out->ring_allocation_bytes = best_ring;
    out->total_allocation_bytes = best_total;
    out->feasible = 1;
    out->error[0] = '\0';
    free(suffix_max);
    return 0;
}
