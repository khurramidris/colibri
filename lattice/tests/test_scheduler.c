#include "lattice/scheduler.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static lt_transfer_request_t request(uint64_t id,
                                     uint64_t bytes,
                                     lt_transfer_path_t path,
                                     uint64_t deadline,
                                     double confidence) {
    lt_transfer_request_t r;
    memset(&r, 0, sizeof(r));
    r.tensor_id = id;
    r.bytes = bytes;
    r.path = path;
    r.deadline_ns = deadline;
    r.confidence = confidence;
    r.layer = 1;
    return r;
}

static void test_priority(void) {
    lt_scheduler_t *s = lt_scheduler_create(8, 0);
    lt_transfer_request_t out;
    char error[128];
    lt_transfer_request_t a = request(1, 100, LT_PATH_SPECULATIVE, 10, 0.9);
    lt_transfer_request_t b = request(2, 100, LT_PATH_PROACTIVE, 20, 0.7);
    lt_transfer_request_t c = request(3, 100, LT_PATH_DEMAND, 30, 1.0);
    CHECK(s != NULL);
    CHECK(lt_scheduler_submit(s, &a, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_submit(s, &b, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_submit(s, &c, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_pop(s, &out) == 1 && out.tensor_id == 3);
    CHECK(lt_scheduler_complete(s, out.tensor_id, error, sizeof(error)) == 0);
    CHECK(lt_scheduler_pop(s, &out) == 1 && out.tensor_id == 2);
    CHECK(lt_scheduler_complete(s, out.tensor_id, error, sizeof(error)) == 0);
    CHECK(lt_scheduler_pop(s, &out) == 1 && out.tensor_id == 1);
    CHECK(lt_scheduler_complete(s, out.tensor_id, error, sizeof(error)) == 0);
    CHECK(lt_scheduler_pop(s, &out) == 0);
    lt_scheduler_destroy(s);
}

static void test_deadline_and_confidence(void) {
    lt_scheduler_t *s = lt_scheduler_create(8, 0);
    lt_transfer_request_t out;
    char error[128];
    lt_transfer_request_t a = request(1, 100, LT_PATH_PROACTIVE, 100, 0.2);
    lt_transfer_request_t b = request(2, 100, LT_PATH_PROACTIVE, 50, 0.1);
    lt_transfer_request_t c = request(3, 100, LT_PATH_PROACTIVE, 50, 0.9);
    CHECK(lt_scheduler_submit(s, &a, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_submit(s, &b, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_submit(s, &c, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_pop(s, &out) == 1 && out.tensor_id == 3);
    lt_scheduler_destroy(s);
}

static void test_dedup_and_promotion(void) {
    lt_scheduler_t *s = lt_scheduler_create(4, 0);
    lt_transfer_request_t out;
    lt_scheduler_stats_t stats;
    char error[128];
    lt_transfer_request_t predicted = request(77, 4096, LT_PATH_SPECULATIVE, 500, 0.4);
    lt_transfer_request_t demand = request(77, 4096, LT_PATH_DEMAND, 100, 1.0);
    CHECK(lt_scheduler_submit(s, &predicted, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_submit(s, &demand, error, sizeof(error)) == 2);
    CHECK(lt_scheduler_queued(s) == 1);
    CHECK(lt_scheduler_pop(s, &out) == 1);
    CHECK(out.tensor_id == 77);
    CHECK(out.path == LT_PATH_DEMAND);
    CHECK(out.deadline_ns == 100);
    stats = lt_scheduler_stats(s);
    CHECK(stats.submitted == 1);
    CHECK(stats.deduplicated == 1);
    CHECK(stats.promoted == 1);
    lt_scheduler_destroy(s);
}

static void test_inflight_cap(void) {
    lt_scheduler_t *s = lt_scheduler_create(4, 100);
    lt_transfer_request_t out;
    char error[128];
    lt_transfer_request_t a = request(1, 60, LT_PATH_DEMAND, 1, 1.0);
    lt_transfer_request_t b = request(2, 60, LT_PATH_DEMAND, 2, 1.0);
    CHECK(lt_scheduler_submit(s, &a, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_submit(s, &b, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_pop(s, &out) == 1 && out.tensor_id == 1);
    CHECK(lt_scheduler_pop(s, &out) == 0);
    CHECK(lt_scheduler_complete(s, 1, error, sizeof(error)) == 0);
    CHECK(lt_scheduler_pop(s, &out) == 1 && out.tensor_id == 2);
    lt_scheduler_destroy(s);
}

static void test_cancel_speculation(void) {
    lt_scheduler_t *s = lt_scheduler_create(8, 0);
    lt_transfer_request_t a = request(1, 100, LT_PATH_SPECULATIVE, 0, 0.2);
    lt_transfer_request_t b = request(2, 100, LT_PATH_SPECULATIVE, 0, 0.8);
    lt_transfer_request_t c = request(3, 100, LT_PATH_PROACTIVE, 0, 0.1);
    lt_scheduler_stats_t stats;
    char error[128];
    CHECK(lt_scheduler_submit(s, &a, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_submit(s, &b, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_submit(s, &c, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_cancel_speculative(s, 0.5) == 1);
    CHECK(lt_scheduler_queued(s) == 2);
    stats = lt_scheduler_stats(s);
    CHECK(stats.cancelled == 1);
    CHECK(stats.queued_bytes == 200);
    lt_scheduler_compact(s);
    CHECK(lt_scheduler_queued(s) == 2);
    lt_scheduler_destroy(s);
}

static void test_invalid_duplicate_size(void) {
    lt_scheduler_t *s = lt_scheduler_create(4, 0);
    lt_transfer_request_t a = request(10, 100, LT_PATH_SPECULATIVE, 0, 0.5);
    lt_transfer_request_t b = request(10, 200, LT_PATH_DEMAND, 0, 1.0);
    char error[128];
    CHECK(lt_scheduler_submit(s, &a, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_submit(s, &b, error, sizeof(error)) == -1);
    CHECK(error[0] != '\0');
    lt_scheduler_destroy(s);
}

static void test_failure_can_retry(void) {
    lt_scheduler_t *s = lt_scheduler_create(2, 0);
    lt_transfer_request_t a = request(41, 100, LT_PATH_DEMAND, 0, 1.0);
    lt_transfer_request_t out;
    lt_scheduler_stats_t stats;
    char error[128];
    CHECK(lt_scheduler_submit(s, &a, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_pop(s, &out) == 1);
    CHECK(lt_scheduler_fail(s, 41, error, sizeof(error)) == 0);
    CHECK(lt_scheduler_submit(s, &a, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_pop(s, &out) == 1);
    CHECK(lt_scheduler_complete(s, 41, error, sizeof(error)) == 0);
    stats = lt_scheduler_stats(s);
    CHECK(stats.failed == 1 && stats.failed_bytes == 100);
    CHECK(stats.completed == 1 && stats.completed_bytes == 100);
    lt_scheduler_destroy(s);
}

static void test_completed_tensor_must_be_forgotten_after_eviction(void) {
    lt_scheduler_t *s = lt_scheduler_create(2, 0);
    lt_transfer_request_t a = request(51, 100, LT_PATH_DEMAND, 0, 1.0);
    lt_transfer_request_t out;
    lt_scheduler_stats_t stats;
    char error[128];
    CHECK(lt_scheduler_submit(s, &a, error, sizeof(error)) == 1);
    CHECK(lt_scheduler_pop(s, &out) == 1);
    CHECK(lt_scheduler_complete(s, 51, error, sizeof(error)) == 0);
    CHECK(lt_scheduler_submit(s, &a, error, sizeof(error)) == 0);
    CHECK(lt_scheduler_forget(s, 51) == 1);
    CHECK(lt_scheduler_submit(s, &a, error, sizeof(error)) == 1);
    stats = lt_scheduler_stats(s);
    CHECK(stats.forgotten == 1);
    lt_scheduler_destroy(s);
}

int main(void) {
    test_priority();
    test_deadline_and_confidence();
    test_dedup_and_promotion();
    test_inflight_cap();
    test_cancel_speculation();
    test_invalid_duplicate_size();
    test_failure_can_retry();
    test_completed_tensor_must_be_forgotten_after_eviction();
    if (failures) {
        fprintf(stderr, "%d scheduler test(s) failed\n", failures);
        return 1;
    }
    puts("scheduler tests passed");
    return 0;
}
