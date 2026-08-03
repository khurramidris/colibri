#ifndef LATTICE_BLOCK_LAYOUT_H
#define LATTICE_BLOCK_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LT_BLOCK_HEADER_BYTES 4096u
#define LT_BLOCK_VERSION 1u
#define LT_BLOCK_MAX_RANGES 4096u

typedef struct {
    uint32_t latent;
    uint32_t intermediate;
    uint32_t group_size;
    uint32_t block_channels;
    uint32_t experts;
    uint32_t blocks_per_expert;
    uint64_t w1_packed_bytes;
    uint64_t w1_scale_bytes;
    uint64_t w3_packed_bytes;
    uint64_t w3_scale_bytes;
    uint64_t w2_packed_bytes;
    uint64_t w2_scale_bytes;
    uint64_t record_bytes;
    uint64_t total_bytes;
} lt_block_layout_t;

typedef struct {
    uint64_t offset;
    uint64_t length;
    uint32_t first_block;
    uint32_t block_count;
} lt_block_range_t;

/* Kimi-style native MXFP4 assumptions: packed e2m1 nibbles and one u8 scale
 * per group_size input channels. block_channels must be group-aligned so a
 * selected block is independently decodable. */
int lt_block_layout_init(lt_block_layout_t *layout,
                         uint32_t latent,
                         uint32_t intermediate,
                         uint32_t group_size,
                         uint32_t block_channels,
                         uint32_t experts,
                         char *error,
                         size_t error_cap);

uint64_t lt_block_record_offset(const lt_block_layout_t *layout,
                                uint32_t expert,
                                uint32_t block);

/* Converts a block-selection bitmap for one expert into coalesced file ranges.
 * Adjacent selected records are one read. Returns number of ranges, or -1. */
int lt_block_ranges(const lt_block_layout_t *layout,
                    uint32_t expert,
                    const uint8_t *selected,
                    size_t selected_count,
                    lt_block_range_t *out_ranges,
                    size_t range_capacity,
                    char *error,
                    size_t error_cap);

/* 4096-byte little-endian header used by pack_expert_blocks.py. Only the first
 * 96 bytes are meaningful in v1; the rest are zero and reserved. */
int lt_block_header_encode(const lt_block_layout_t *layout,
                           uint32_t layer,
                           uint8_t out_header[LT_BLOCK_HEADER_BYTES]);
int lt_block_header_decode(const uint8_t header[LT_BLOCK_HEADER_BYTES],
                           lt_block_layout_t *layout,
                           uint32_t *out_layer,
                           char *error,
                           size_t error_cap);

#ifdef __cplusplus
}
#endif

#endif /* LATTICE_BLOCK_LAYOUT_H */
