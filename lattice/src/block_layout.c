#include "lattice/block_layout.h"

#include <stdio.h>
#include <string.h>

static const uint8_t LT_BLOCK_MAGIC[8] = {'L','T','B','L','K','1',0,0};

static void lt_block_error(char *error, size_t cap, const char *message) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", message ? message : "block layout error");
}

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

static void put_u32(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static void put_u64(uint8_t *p, uint64_t value) {
    unsigned i;
    for (i = 0; i < 8; ++i) p[i] = (uint8_t)(value >> (8 * i));
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t get_u64(const uint8_t *p) {
    uint64_t value = 0;
    unsigned i;
    for (i = 0; i < 8; ++i) value |= (uint64_t)p[i] << (8 * i);
    return value;
}

static uint64_t fnv1a64(const uint8_t *data, size_t size) {
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t i;
    for (i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

int lt_block_layout_init(lt_block_layout_t *layout,
                         uint32_t latent,
                         uint32_t intermediate,
                         uint32_t group_size,
                         uint32_t block_channels,
                         uint32_t experts,
                         char *error,
                         size_t error_cap) {
    uint64_t w1p, w1s, w2p, w2s, payload, records, total;
    uint32_t blocks;
    if (!layout) {
        lt_block_error(error, error_cap, "layout output is required");
        return -1;
    }
    memset(layout, 0, sizeof(*layout));
    if (latent == 0 || intermediate == 0 || group_size == 0 ||
        block_channels == 0 || experts == 0 ||
        (group_size & 1u) != 0 || latent % group_size != 0 ||
        block_channels % group_size != 0 || intermediate % block_channels != 0) {
        lt_block_error(error, error_cap,
                       "dimensions must be non-zero; latent and block width must be group-aligned; intermediate must divide into fixed blocks");
        return -1;
    }
    blocks = intermediate / block_channels;
    if (mul_u64(block_channels, latent / 2u, &w1p) != 0 ||
        mul_u64(block_channels, latent / group_size, &w1s) != 0 ||
        mul_u64(latent, block_channels / 2u, &w2p) != 0 ||
        mul_u64(latent, block_channels / group_size, &w2s) != 0) {
        lt_block_error(error, error_cap, "block component size overflow");
        return -1;
    }
    payload = 0;
    if (add_u64(payload, w1p, &payload) != 0 ||
        add_u64(payload, w1s, &payload) != 0 ||
        add_u64(payload, w1p, &payload) != 0 ||
        add_u64(payload, w1s, &payload) != 0 ||
        add_u64(payload, w2p, &payload) != 0 ||
        add_u64(payload, w2s, &payload) != 0 ||
        align_u64(payload, LT_BLOCK_HEADER_BYTES, &layout->record_bytes) != 0 ||
        mul_u64(experts, blocks, &records) != 0 ||
        mul_u64(records, layout->record_bytes, &total) != 0 ||
        add_u64(total, LT_BLOCK_HEADER_BYTES, &total) != 0) {
        lt_block_error(error, error_cap, "block file size overflow");
        memset(layout, 0, sizeof(*layout));
        return -1;
    }
    layout->latent = latent;
    layout->intermediate = intermediate;
    layout->group_size = group_size;
    layout->block_channels = block_channels;
    layout->experts = experts;
    layout->blocks_per_expert = blocks;
    layout->w1_packed_bytes = w1p;
    layout->w1_scale_bytes = w1s;
    layout->w3_packed_bytes = w1p;
    layout->w3_scale_bytes = w1s;
    layout->w2_packed_bytes = w2p;
    layout->w2_scale_bytes = w2s;
    layout->total_bytes = total;
    lt_block_error(error, error_cap, "");
    return 0;
}

uint64_t lt_block_record_offset(const lt_block_layout_t *layout,
                                uint32_t expert,
                                uint32_t block) {
    uint64_t index;
    if (!layout || expert >= layout->experts || block >= layout->blocks_per_expert) return UINT64_MAX;
    index = (uint64_t)expert * layout->blocks_per_expert + block;
    if (index > (UINT64_MAX - LT_BLOCK_HEADER_BYTES) / layout->record_bytes) return UINT64_MAX;
    return LT_BLOCK_HEADER_BYTES + index * layout->record_bytes;
}

int lt_block_ranges(const lt_block_layout_t *layout,
                    uint32_t expert,
                    const uint8_t *selected,
                    size_t selected_count,
                    lt_block_range_t *out_ranges,
                    size_t range_capacity,
                    char *error,
                    size_t error_cap) {
    size_t block = 0, ranges = 0;
    if (!layout || !selected || selected_count != layout->blocks_per_expert ||
        expert >= layout->experts || (!out_ranges && range_capacity != 0)) {
        lt_block_error(error, error_cap, "invalid block range arguments");
        return -1;
    }
    while (block < selected_count) {
        size_t first, end;
        uint64_t offset, length;
        while (block < selected_count && selected[block] == 0) ++block;
        if (block == selected_count) break;
        first = block;
        while (block < selected_count && selected[block] != 0) ++block;
        end = block;
        if (ranges >= range_capacity) {
            lt_block_error(error, error_cap, "block range output capacity is too small");
            return -1;
        }
        offset = lt_block_record_offset(layout, expert, (uint32_t)first);
        if (offset == UINT64_MAX ||
            mul_u64(end - first, layout->record_bytes, &length) != 0) {
            lt_block_error(error, error_cap, "block range overflow");
            return -1;
        }
        out_ranges[ranges].offset = offset;
        out_ranges[ranges].length = length;
        out_ranges[ranges].first_block = (uint32_t)first;
        out_ranges[ranges].block_count = (uint32_t)(end - first);
        ++ranges;
    }
    lt_block_error(error, error_cap, "");
    return (int)ranges;
}

int lt_block_header_encode(const lt_block_layout_t *layout,
                           uint32_t layer,
                           uint8_t out_header[LT_BLOCK_HEADER_BYTES]) {
    uint64_t checksum;
    if (!layout || !out_header || layout->record_bytes == 0 || layout->total_bytes < LT_BLOCK_HEADER_BYTES)
        return -1;
    memset(out_header, 0, LT_BLOCK_HEADER_BYTES);
    memcpy(out_header, LT_BLOCK_MAGIC, sizeof(LT_BLOCK_MAGIC));
    put_u32(out_header + 8, LT_BLOCK_VERSION);
    put_u32(out_header + 12, LT_BLOCK_HEADER_BYTES);
    put_u32(out_header + 16, layer);
    put_u32(out_header + 20, layout->latent);
    put_u32(out_header + 24, layout->intermediate);
    put_u32(out_header + 28, layout->group_size);
    put_u32(out_header + 32, layout->block_channels);
    put_u32(out_header + 36, layout->experts);
    put_u32(out_header + 40, layout->blocks_per_expert);
    put_u64(out_header + 48, layout->record_bytes);
    put_u64(out_header + 56, layout->total_bytes);
    checksum = fnv1a64(out_header, 64);
    put_u64(out_header + 64, checksum);
    return 0;
}

int lt_block_header_decode(const uint8_t header[LT_BLOCK_HEADER_BYTES],
                           lt_block_layout_t *layout,
                           uint32_t *out_layer,
                           char *error,
                           size_t error_cap) {
    lt_block_layout_t computed;
    uint32_t layer;
    uint64_t checksum;
    if (!header || !layout) {
        lt_block_error(error, error_cap, "header and layout output are required");
        return -1;
    }
    if (memcmp(header, LT_BLOCK_MAGIC, sizeof(LT_BLOCK_MAGIC)) != 0 ||
        get_u32(header + 8) != LT_BLOCK_VERSION ||
        get_u32(header + 12) != LT_BLOCK_HEADER_BYTES) {
        lt_block_error(error, error_cap, "block header magic or version mismatch");
        return -1;
    }
    checksum = fnv1a64(header, 64);
    if (get_u64(header + 64) != checksum) {
        lt_block_error(error, error_cap, "block header checksum mismatch");
        return -1;
    }
    layer = get_u32(header + 16);
    if (lt_block_layout_init(&computed,
                             get_u32(header + 20),
                             get_u32(header + 24),
                             get_u32(header + 28),
                             get_u32(header + 32),
                             get_u32(header + 36),
                             error, error_cap) != 0)
        return -1;
    if (computed.blocks_per_expert != get_u32(header + 40) ||
        computed.record_bytes != get_u64(header + 48) ||
        computed.total_bytes != get_u64(header + 56)) {
        lt_block_error(error, error_cap, "block header derived sizes do not match dimensions");
        return -1;
    }
    *layout = computed;
    if (out_layer) *out_layer = layer;
    lt_block_error(error, error_cap, "");
    return 0;
}
