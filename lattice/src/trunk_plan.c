#include "lattice/trunk_plan.h"

#include <stdio.h>
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
    uint32_t npin = 0;
    uint32_t pass;
    uint64_t packed = 0;
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
    for (pass = 0; pass < layers; ++pass) {
        uint64_t ignored;
        if (layer_bytes[pass] == 0 ||
            add_u64(packed, layer_bytes[pass], &packed) != 0 ||
            layer_allocation(layer_bytes[pass], widen_bytes_per_slot,
                             alignment, &ignored) != 0) {
            set_error(out, "layer size is zero or overflows allocation arithmetic");
            return -1;
        }
    }
    out->packed_bytes = packed;

    for (pass = 0; pass <= layers + 1u; ++pass) {
        uint64_t largest = 0;
        uint64_t slot = 0;
        uint64_t ring_bytes = 0;
        uint64_t spent;
        uint32_t next_pin = 0;
        uint32_t layer;

        for (layer = npin; layer < layers; ++layer)
            if (layer_bytes[layer] > largest) largest = layer_bytes[layer];
        if (largest != 0 &&
            layer_allocation(largest, widen_bytes_per_slot, alignment, &slot) != 0) {
            set_error(out, "ring slot size overflow");
            return -1;
        }
        if (mul_u64(slot, ring_slots, &ring_bytes) != 0) {
            set_error(out, "ring allocation overflow");
            return -1;
        }
        if (ring_bytes > budget_bytes) {
            set_error(out, "budget cannot hold one streaming ring for the largest unpinned layer");
            return -1;
        }
        spent = ring_bytes;
        while (next_pin < layers) {
            uint64_t need;
            if (layer_allocation(layer_bytes[next_pin], widen_bytes_per_slot,
                                 alignment, &need) != 0) {
                set_error(out, "pinned layer allocation overflow");
                return -1;
            }
            if (need > budget_bytes - spent) break;
            spent += need;
            ++next_pin;
        }
        if (next_pin == npin) {
            uint64_t pinned_bytes = 0;
            uint64_t streamed = 0;
            for (layer = 0; layer < npin; ++layer) {
                uint64_t need;
                if (layer_allocation(layer_bytes[layer], widen_bytes_per_slot,
                                     alignment, &need) != 0 ||
                    add_u64(pinned_bytes, need, &pinned_bytes) != 0) {
                    set_error(out, "pinned byte total overflow");
                    return -1;
                }
            }
            for (layer = npin; layer < layers; ++layer) {
                if (add_u64(streamed, layer_bytes[layer], &streamed) != 0) {
                    set_error(out, "streamed byte total overflow");
                    return -1;
                }
            }
            out->pinned_layers = npin;
            out->pinned_allocation_bytes = pinned_bytes;
            out->ring_slot_bytes = slot;
            out->ring_allocation_bytes = ring_bytes;
            out->total_allocation_bytes = pinned_bytes + ring_bytes;
            out->streamed_bytes_per_token = streamed;
            out->feasible = 1;
            out->error[0] = '\0';
            return 0;
        }
        npin = next_pin;
    }
    set_error(out, "trunk fixed-point planner did not converge");
    return -1;
}
