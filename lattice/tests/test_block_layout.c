#include "lattice/block_layout.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static void test_geometry(void) {
    lt_block_layout_t layout;
    char error[128];
    CHECK(lt_block_layout_init(&layout, 64, 128, 32, 64, 2,
                               error, sizeof(error)) == 0);
    CHECK(layout.blocks_per_expert == 2);
    CHECK(layout.w1_packed_bytes == 2048);
    CHECK(layout.w1_scale_bytes == 128);
    CHECK(layout.w2_packed_bytes == 2048);
    CHECK(layout.w2_scale_bytes == 128);
    CHECK(layout.record_bytes == 8192);
    CHECK(layout.total_bytes == 4096 + 4 * 8192);
    CHECK(lt_block_record_offset(&layout, 0, 0) == 4096);
    CHECK(lt_block_record_offset(&layout, 1, 0) == 4096 + 2 * 8192);
    CHECK(lt_block_record_offset(&layout, 2, 0) == UINT64_MAX);
}

static void test_ranges(void) {
    lt_block_layout_t layout;
    lt_block_range_t ranges[4];
    uint8_t selected[] = {1, 1, 0, 1};
    char error[128];
    int n;
    CHECK(lt_block_layout_init(&layout, 64, 256, 32, 64, 3,
                               error, sizeof(error)) == 0);
    n = lt_block_ranges(&layout, 1, selected, 4, ranges, 4,
                        error, sizeof(error));
    CHECK(n == 2);
    CHECK(ranges[0].first_block == 0 && ranges[0].block_count == 2);
    CHECK(ranges[0].length == 2 * layout.record_bytes);
    CHECK(ranges[1].first_block == 3 && ranges[1].block_count == 1);
    CHECK(ranges[1].offset == lt_block_record_offset(&layout, 1, 3));
}

static void test_header_roundtrip(void) {
    lt_block_layout_t a, b;
    uint8_t header[LT_BLOCK_HEADER_BYTES];
    uint32_t layer = 0;
    char error[128];
    CHECK(lt_block_layout_init(&a, 3584, 3072, 32, 256, 896,
                               error, sizeof(error)) == 0);
    CHECK(lt_block_header_encode(&a, 17, header) == 0);
    CHECK(lt_block_header_decode(header, &b, &layer, error, sizeof(error)) == 0);
    CHECK(layer == 17);
    CHECK(memcmp(&a, &b, sizeof(a)) == 0);
    header[24] ^= 1;
    CHECK(lt_block_header_decode(header, &b, &layer, error, sizeof(error)) != 0);
}

static void test_rejects_misalignment(void) {
    lt_block_layout_t layout;
    char error[128];
    CHECK(lt_block_layout_init(&layout, 64, 130, 32, 64, 2,
                               error, sizeof(error)) != 0);
    CHECK(lt_block_layout_init(&layout, 65, 128, 32, 64, 2,
                               error, sizeof(error)) != 0);
    CHECK(lt_block_layout_init(&layout, 64, 128, 32, 48, 2,
                               error, sizeof(error)) != 0);
}

int main(void) {
    test_geometry();
    test_ranges();
    test_header_roundtrip();
    test_rejects_misalignment();
    if (failures) {
        fprintf(stderr, "%d block layout test(s) failed\n", failures);
        return 1;
    }
    puts("block layout tests passed");
    return 0;
}
