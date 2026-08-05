#include "trunk_stream.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t width;
    double *state;
    double *scratch;
} execution_context;

static uint32_t rd32(const unsigned char *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static float rdf32(const unsigned char *p) {
    uint32_t bits = rd32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int apply_layer(
    uint32_t layer_id,
    const void *payload_raw,
    size_t payload_bytes,
    void *context_raw
) {
    (void)layer_id;
    execution_context *context = (execution_context *)context_raw;
    const unsigned char *payload = (const unsigned char *)payload_raw;
    if (payload_bytes < 12 || memcmp(payload, "QTM1", 4) != 0) {
        return -1;
    }
    const uint32_t rows = rd32(payload + 4);
    const uint32_t cols = rd32(payload + 8);
    if (rows != context->width || cols != context->width) {
        return -1;
    }
    const size_t scales_offset = 12;
    const size_t weights_offset = scales_offset + (size_t)rows * 4;
    if (weights_offset + (size_t)rows * cols != payload_bytes) {
        return -1;
    }
    const int8_t *weights = (const int8_t *)(payload + weights_offset);
    for (uint32_t row = 0; row < rows; ++row) {
        double acc = 0.0;
        for (uint32_t col = 0; col < cols; ++col) {
            acc += (double)weights[(size_t)row * cols + col] * context->state[col];
        }
        const float scale = rdf32(payload + scales_offset + (size_t)row * 4);
        context->scratch[row] = tanh(acc * (double)scale + 0.125 * context->state[row]);
    }
    memcpy(context->state, context->scratch, context->width * sizeof(double));
    return 0;
}

static int execute_once(
    cerno_trunk_reader *reader,
    size_t width,
    double *output
) {
    execution_context context = {0};
    context.width = width;
    context.state = calloc(width, sizeof(double));
    context.scratch = calloc(width, sizeof(double));
    if (!context.state || !context.scratch) {
        free(context.state);
        free(context.scratch);
        return -1;
    }
    for (size_t i = 0; i < width; ++i) {
        context.state[i] = (double)(i + 1) / (double)width;
    }
    const int rc = cerno_trunk_for_each(reader, apply_layer, &context);
    if (rc == 0) {
        memcpy(output, context.state, width * sizeof(double));
    }
    free(context.state);
    free(context.scratch);
    return rc;
}

static int verify_reader(
    const char *label,
    cerno_trunk_reader *reader,
    size_t width,
    const double *reference,
    unsigned passes
) {
    double *output = calloc(width, sizeof(double));
    if (!output) {
        return -1;
    }
    for (unsigned pass = 0; pass < passes; ++pass) {
        if (execute_once(reader, width, output) != 0) {
            fprintf(stderr, "%s pass %u failed: %s\n",
                    label, pass, cerno_trunk_error(reader));
            free(output);
            return -1;
        }
        if (memcmp(output, reference, width * sizeof(double)) != 0) {
            fprintf(stderr, "%s pass %u was not bit-identical\n", label, pass);
            free(output);
            return -1;
        }
    }
    free(output);

    cerno_trunk_stats stats;
    memset(&stats, 0, sizeof(stats));
    cerno_trunk_get_stats(reader, &stats);
    const uint64_t expected_payload =
        stats.streamed_payload_bytes_per_pass * (uint64_t)passes;
    const uint64_t expected_io =
        stats.streamed_io_bytes_per_pass * (uint64_t)passes;
    if (
        stats.payload_bytes_consumed != expected_payload ||
        stats.io_bytes_requested != expected_io
    ) {
        fprintf(
            stderr,
            "%s accounting mismatch: payload %llu/%llu io %llu/%llu\n",
            label,
            (unsigned long long)stats.payload_bytes_consumed,
            (unsigned long long)expected_payload,
            (unsigned long long)stats.io_bytes_requested,
            (unsigned long long)expected_io
        );
        return -1;
    }
    printf(
        "%s: direct=%d resident=%u/%u payload/pass=%llu io/pass=%llu "
        "reads=%llu peak=%llu bit-identical=%u/%u\n",
        label,
        stats.direct_io_active,
        stats.resident_layers,
        stats.total_layers,
        (unsigned long long)stats.streamed_payload_bytes_per_pass,
        (unsigned long long)stats.streamed_io_bytes_per_pass,
        (unsigned long long)stats.read_calls,
        (unsigned long long)stats.modeled_peak_working_bytes,
        passes,
        passes
    );
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s TRUNK_BIN\n", argv[0]);
        return 2;
    }
    const char *path = argv[1];
    const size_t width = 48;
    const unsigned passes = 5;

    cerno_trunk_reader *probe = NULL;
    if (cerno_trunk_open(&probe, path, 0, 0) != 0) {
        fprintf(stderr, "probe open failed\n");
        return 1;
    }
    const uint64_t partial_budget = cerno_trunk_prefix_payload_bytes(probe, 7);
    cerno_trunk_close(probe);

    cerno_trunk_reader *full = NULL;
    if (cerno_trunk_open(&full, path, UINT64_MAX, 0) != 0) {
        fprintf(stderr, "full open failed\n");
        return 1;
    }
    double *reference = calloc(width, sizeof(double));
    if (!reference) {
        cerno_trunk_close(full);
        return 1;
    }
    if (execute_once(full, width, reference) != 0) {
        fprintf(stderr, "full execution failed: %s\n", cerno_trunk_error(full));
        free(reference);
        cerno_trunk_close(full);
        return 1;
    }
    cerno_trunk_stats full_stats;
    memset(&full_stats, 0, sizeof(full_stats));
    cerno_trunk_get_stats(full, &full_stats);
    if (
        full_stats.resident_layers != full_stats.total_layers ||
        full_stats.io_bytes_requested != 0
    ) {
        fprintf(stderr, "full residency accounting failed\n");
        free(reference);
        cerno_trunk_close(full);
        return 1;
    }
    cerno_trunk_close(full);

    cerno_trunk_reader *buffered = NULL;
    if (cerno_trunk_open(&buffered, path, partial_budget, 0) != 0) {
        fprintf(stderr, "buffered open failed\n");
        free(reference);
        return 1;
    }
    if (verify_reader("buffered", buffered, width, reference, passes) != 0) {
        cerno_trunk_close(buffered);
        free(reference);
        return 1;
    }
    cerno_trunk_close(buffered);

    cerno_trunk_reader *direct = NULL;
    if (cerno_trunk_open(&direct, path, partial_budget, 2) != 0) {
        fprintf(stderr, "direct-preferred open failed\n");
        free(reference);
        return 1;
    }
    if (verify_reader("direct-preferred", direct, width, reference, passes) != 0) {
        cerno_trunk_close(direct);
        free(reference);
        return 1;
    }
    cerno_trunk_stats direct_stats;
    memset(&direct_stats, 0, sizeof(direct_stats));
    cerno_trunk_get_stats(direct, &direct_stats);
    if (
        direct_stats.direct_io_active &&
        direct_stats.streamed_io_bytes_per_pass <
            direct_stats.streamed_payload_bytes_per_pass
    ) {
        fprintf(stderr, "direct I/O accounting underflow\n");
        cerno_trunk_close(direct);
        free(reference);
        return 1;
    }
    cerno_trunk_close(direct);
    free(reference);

    puts("native trunk streaming tests: ok");
    return 0;
}
