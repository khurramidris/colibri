#define _POSIX_C_SOURCE 200809L
#include "lattice/runtime.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define BASE_ID UINT64_C(1000)
#define TENSOR_BYTES 64u
#define TENSOR_COUNT 8u

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

typedef struct {
    int fd;
    unsigned char destinations[TENSOR_COUNT][TENSOR_BYTES];
    unsigned char ready[TENSOR_COUNT];
    pthread_mutex_t mutex;
    int completions;
    int failed_completions;
} Adapter;

static int resolve(void *context,
                   const lt_transfer_request_t *request,
                   lt_io_job_t *out,
                   char *error,
                   size_t error_cap) {
    Adapter *adapter = (Adapter *)context;
    uint64_t index;
    if (request->tensor_id < BASE_ID ||
        request->tensor_id - BASE_ID >= TENSOR_COUNT) {
        snprintf(error, error_cap, "tensor id is outside fixture");
        return -1;
    }
    index = request->tensor_id - BASE_ID;
    out->fd = adapter->fd;
    out->offset = index * TENSOR_BYTES;
    out->destination = adapter->destinations[index];
    out->bytes = request->bytes;
    out->opaque = index;
    return 0;
}

static void complete(void *context,
                     const lt_io_job_t *job,
                     int status,
                     uint64_t bytes_read) {
    Adapter *adapter = (Adapter *)context;
    pthread_mutex_lock(&adapter->mutex);
    adapter->completions++;
    if (status == 0 && bytes_read == job->bytes)
        adapter->ready[job->opaque] = 1;
    else
        adapter->failed_completions++;
    pthread_mutex_unlock(&adapter->mutex);
}

static int make_fixture(char path[256], unsigned char expected[TENSOR_COUNT][TENSOR_BYTES]) {
    int fd;
    uint32_t tensor, byte;
    snprintf(path, 256, "/tmp/lattice-runtime-%ld.bin", (long)getpid());
    fd = open(path, O_CREAT | O_TRUNC | O_RDWR, 0600);
    if (fd < 0) return -1;
    for (tensor = 0; tensor < TENSOR_COUNT; ++tensor) {
        for (byte = 0; byte < TENSOR_BYTES; ++byte)
            expected[tensor][byte] = (unsigned char)((tensor * 31u + byte * 7u) & 0xffu);
        if (write(fd, expected[tensor], TENSOR_BYTES) != TENSOR_BYTES) {
            close(fd);
            unlink(path);
            return -1;
        }
    }
    return fd;
}

static lt_runtime_t *make_runtime(Adapter *adapter,
                                  uint64_t max_inflight,
                                  char *error,
                                  size_t error_cap) {
    lt_runtime_config_t config;
    lt_runtime_adapter_t callbacks;
    memset(&config, 0, sizeof(config));
    config.layers = 2;
    config.experts = 4;
    config.expert_tensor_id_base = BASE_ID;
    config.expert_bytes_each = TENSOR_BYTES;
    config.scheduler_capacity = 32;
    config.max_inflight_bytes = max_inflight;
    config.io_workers = 2;
    config.io_queue_capacity = 2;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.context = adapter;
    callbacks.resolve = resolve;
    callbacks.complete = complete;
    return lt_runtime_create(&config, &callbacks, error, error_cap);
}

static void test_prefetch_promotion_and_reads(void) {
    char path[256], error[256];
    unsigned char expected[TENSOR_COUNT][TENSOR_BYTES];
    uint32_t ids[] = {2, 1, 2};
    double weights[] = {0.4, 0.2, 0.4};
    Adapter adapter;
    lt_runtime_t *runtime;
    lt_runtime_stats_t stats;
    uint64_t expert2_id = BASE_ID + 4 + 2;
    uint64_t expert1_id = BASE_ID + 4 + 1;
    memset(&adapter, 0, sizeof(adapter));
    pthread_mutex_init(&adapter.mutex, NULL);
    adapter.fd = make_fixture(path, expected);
    CHECK(adapter.fd >= 0);
    runtime = make_runtime(&adapter, 128, error, sizeof(error));
    CHECK(runtime != NULL);
    CHECK(lt_runtime_observe_experts(runtime, 1, ids, weights, 3) == 0);
    CHECK(lt_runtime_prefetch_experts(runtime, 1, 2, 100,
                                      LT_PATH_PROACTIVE, 0.0,
                                      error, sizeof(error)) == 2);
    CHECK(lt_runtime_demand(runtime, expert2_id, TENSOR_BYTES, 1, 10, 2,
                            error, sizeof(error)) == 2);
    CHECK(lt_runtime_run_until_idle(runtime, error, sizeof(error)) == 0);
    CHECK(adapter.ready[expert2_id - BASE_ID] == 1);
    CHECK(adapter.ready[expert1_id - BASE_ID] == 1);
    CHECK(memcmp(adapter.destinations[expert2_id - BASE_ID],
                 expected[expert2_id - BASE_ID], TENSOR_BYTES) == 0);
    CHECK(memcmp(adapter.destinations[expert1_id - BASE_ID],
                 expected[expert1_id - BASE_ID], TENSOR_BYTES) == 0);
    stats = lt_runtime_stats(runtime);
    CHECK(stats.scheduler.submitted == 2);
    CHECK(stats.scheduler.deduplicated == 1);
    CHECK(stats.scheduler.promoted == 1);
    CHECK(stats.scheduler.completed == 2);
    CHECK(stats.scheduler.failed == 0);
    CHECK(stats.io.completed_jobs == 2);
    CHECK(stats.dispatched_jobs == 2);
    CHECK(adapter.completions == 2 && adapter.failed_completions == 0);
    lt_runtime_destroy(runtime);
    close(adapter.fd);
    unlink(path);
    pthread_mutex_destroy(&adapter.mutex);
}

static void test_resolve_failure_is_not_resident(void) {
    char path[256], error[256];
    unsigned char expected[TENSOR_COUNT][TENSOR_BYTES];
    Adapter adapter;
    lt_runtime_t *runtime;
    lt_runtime_stats_t stats;
    memset(&adapter, 0, sizeof(adapter));
    pthread_mutex_init(&adapter.mutex, NULL);
    adapter.fd = make_fixture(path, expected);
    runtime = make_runtime(&adapter, 128, error, sizeof(error));
    CHECK(runtime != NULL);
    CHECK(lt_runtime_demand(runtime, BASE_ID + 99, TENSOR_BYTES, 0, 1, 0,
                            error, sizeof(error)) == 1);
    CHECK(lt_runtime_run_until_idle(runtime, error, sizeof(error)) == -1);
    CHECK(strstr(error, "outside fixture") != NULL);
    stats = lt_runtime_stats(runtime);
    CHECK(stats.scheduler.failed == 1);
    CHECK(stats.resolve_failures == 1);
    CHECK(stats.io.submitted_jobs == 0);
    CHECK(adapter.completions == 0);
    lt_runtime_destroy(runtime);
    close(adapter.fd);
    unlink(path);
    pthread_mutex_destroy(&adapter.mutex);
}

static void test_inflight_cap_detects_impossible_request(void) {
    char path[256], error[256];
    unsigned char expected[TENSOR_COUNT][TENSOR_BYTES];
    Adapter adapter;
    lt_runtime_t *runtime;
    memset(&adapter, 0, sizeof(adapter));
    pthread_mutex_init(&adapter.mutex, NULL);
    adapter.fd = make_fixture(path, expected);
    runtime = make_runtime(&adapter, TENSOR_BYTES - 1, error, sizeof(error));
    CHECK(runtime != NULL);
    CHECK(lt_runtime_demand(runtime, BASE_ID, TENSOR_BYTES, 0, 1, 0,
                            error, sizeof(error)) == 1);
    CHECK(lt_runtime_run_until_idle(runtime, error, sizeof(error)) == -1);
    CHECK(strstr(error, "inflight-byte cap") != NULL);
    lt_runtime_destroy(runtime);
    close(adapter.fd);
    unlink(path);
    pthread_mutex_destroy(&adapter.mutex);
}

int main(void) {
    test_prefetch_promotion_and_reads();
    test_resolve_failure_is_not_resident();
    test_inflight_cap_detects_impossible_request();
    if (failures) {
        fprintf(stderr, "%d runtime test(s) failed\n", failures);
        return 1;
    }
    puts("runtime tests passed");
    return 0;
}
