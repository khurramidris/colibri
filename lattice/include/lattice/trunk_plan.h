#ifndef LATTICE_TRUNK_PLAN_H
#define LATTICE_TRUNK_PLAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t layers;
    uint32_t pinned_layers;
    uint32_t ring_slots;
    uint64_t alignment;
    uint64_t widen_bytes_per_slot;
    uint64_t budget_bytes;
    uint64_t packed_bytes;
    uint64_t pinned_allocation_bytes;
    uint64_t ring_slot_bytes;
    uint64_t ring_allocation_bytes;
    uint64_t total_allocation_bytes;
    uint64_t streamed_bytes_per_token;
    int feasible;
    char error[256];
} lt_trunk_plan_t;

/* Plan Fareed-style exact dense-trunk streaming: a resident prefix and a
 * uniform rotating ring for the remaining sequential layers. Ring size and
 * prefix depth are solved to a fixed point because pinning the largest early
 * layers may shrink the required ring slot and free more pinning capacity. */
int lt_trunk_plan_build(const uint64_t *layer_bytes,
                        uint32_t layers,
                        uint64_t budget_bytes,
                        uint64_t alignment,
                        uint32_t ring_slots,
                        uint64_t widen_bytes_per_slot,
                        lt_trunk_plan_t *out);

#ifdef __cplusplus
}
#endif

#endif /* LATTICE_TRUNK_PLAN_H */
