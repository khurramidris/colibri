#ifndef CERNO_TRUNK_STREAM_H
#define CERNO_TRUNK_STREAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cerno_trunk_reader cerno_trunk_reader;

typedef struct {
    uint64_t resident_budget_bytes;
    uint64_t resident_payload_bytes;
    uint64_t streamed_payload_bytes_per_pass;
    uint64_t streamed_io_bytes_per_pass;
    uint64_t startup_payload_bytes;
    uint64_t io_bytes_requested;
    uint64_t payload_bytes_consumed;
    uint64_t read_calls;
    uint64_t resident_hits;
    uint64_t streamed_hits;
    uint64_t modeled_peak_working_bytes;
    uint32_t resident_layers;
    uint32_t total_layers;
    int direct_io_active;
} cerno_trunk_stats;

typedef int (*cerno_trunk_callback)(
    uint32_t layer_id,
    const void *payload,
    size_t payload_bytes,
    void *context
);

/* direct_mode: 0 = buffered, 1 = require O_DIRECT, 2 = prefer O_DIRECT. */
int cerno_trunk_open(
    cerno_trunk_reader **out,
    const char *path,
    uint64_t resident_budget_bytes,
    int direct_mode
);

void cerno_trunk_close(cerno_trunk_reader *reader);

int cerno_trunk_for_each(
    cerno_trunk_reader *reader,
    cerno_trunk_callback callback,
    void *context
);

void cerno_trunk_get_stats(
    const cerno_trunk_reader *reader,
    cerno_trunk_stats *out
);

uint64_t cerno_trunk_prefix_payload_bytes(
    const cerno_trunk_reader *reader,
    uint32_t layer_count
);

const char *cerno_trunk_error(const cerno_trunk_reader *reader);

#ifdef __cplusplus
}
#endif

#endif
