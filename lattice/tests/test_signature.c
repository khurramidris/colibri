#include "lattice/signature.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

static void test_ranking_and_decay(void) {
    lt_signature_t *s = lt_signature_create(2, 4);
    uint32_t ids[] = {2, 1, 2};
    double weights[] = {0.2, 0.3, 0.5};
    uint32_t out_ids[4];
    double scores[4];
    size_t n;
    CHECK(s != NULL);
    CHECK(lt_signature_observe(s, 0, ids, weights, 3) == 0);
    CHECK(lt_signature_layer_mass(s, 0) == 1.0);
    n = lt_signature_topk(s, 0, 4, out_ids, scores);
    CHECK(n == 2);
    CHECK(out_ids[0] == 2);
    CHECK(out_ids[1] == 1);
    CHECK(scores[0] > 0.69 && scores[0] < 0.71);
    CHECK(scores[1] > 0.29 && scores[1] < 0.31);
    CHECK(lt_signature_decay(s, 0.5) == 0);
    CHECK(lt_signature_layer_mass(s, 0) == 0.5);
    lt_signature_destroy(s);
}

static void test_schedule_and_promotion(void) {
    lt_signature_t *sig = lt_signature_create(3, 8);
    lt_scheduler_t *scheduler = lt_scheduler_create(16, 0);
    uint32_t ids[] = {5, 3, 5, 1};
    double weights[] = {0.4, 0.3, 0.2, 0.1};
    lt_transfer_request_t out;
    char error[128];
    int n;
    CHECK(sig != NULL && scheduler != NULL);
    CHECK(lt_signature_observe(sig, 1, ids, weights, 4) == 0);
    n = lt_signature_schedule(sig, scheduler, 1, 3, 1000, 4096, 50,
                              LT_PATH_PROACTIVE, 0.0, error, sizeof(error));
    CHECK(n == 3);
    CHECK(lt_scheduler_queued(scheduler) == 3);
    CHECK(lt_scheduler_pop(scheduler, &out) == 1);
    CHECK(out.tensor_id == 1000 + 8 + 5);
    CHECK(out.opaque == 5);
    CHECK(out.path == LT_PATH_PROACTIVE);
    lt_scheduler_destroy(scheduler);
    lt_signature_destroy(sig);
}

static void test_schedule_overflow_fails_closed(void) {
    lt_signature_t *sig = lt_signature_create(2, 4);
    lt_scheduler_t *scheduler = lt_scheduler_create(4, 0);
    uint32_t ids[] = {3};
    char error[128];
    CHECK(sig != NULL && scheduler != NULL);
    CHECK(lt_signature_observe(sig, 1, ids, NULL, 1) == 0);
    CHECK(lt_signature_schedule(sig, scheduler, 1, 1, UINT64_MAX - 1,
                                4096, 50, LT_PATH_PROACTIVE, 0.0,
                                error, sizeof(error)) == -1);
    CHECK(error[0] != '\0');
    CHECK(lt_scheduler_queued(scheduler) == 0);
    lt_scheduler_destroy(scheduler);
    lt_signature_destroy(sig);
}

static void test_persistence(void) {
    lt_signature_t *a = lt_signature_create(2, 4);
    lt_signature_t *b = lt_signature_create(2, 4);
    uint32_t ids[] = {0, 3};
    double weights[] = {0.25, 0.75};
    uint32_t out_ids[2];
    double scores[2];
    char path[256], error[128];
    snprintf(path, sizeof(path), "/tmp/lattice-signature-%ld.txt", (long)getpid());
    CHECK(a != NULL && b != NULL);
    CHECK(lt_signature_observe(a, 1, ids, weights, 2) == 0);
    CHECK(lt_signature_save(a, path, error, sizeof(error)) == 0);
    CHECK(lt_signature_load(b, path, error, sizeof(error)) == 0);
    CHECK(lt_signature_topk(b, 1, 2, out_ids, scores) == 2);
    CHECK(out_ids[0] == 3 && out_ids[1] == 0);
    unlink(path);
    lt_signature_destroy(a);
    lt_signature_destroy(b);
}

static void test_bad_observation_is_transactional(void) {
    lt_signature_t *s = lt_signature_create(1, 2);
    uint32_t ids[] = {0, 9};
    double weights[] = {1.0, 1.0};
    CHECK(s != NULL);
    CHECK(lt_signature_observe(s, 0, ids, weights, 2) != 0);
    CHECK(lt_signature_layer_mass(s, 0) == 0.0);
    lt_signature_destroy(s);
}

int main(void) {
    test_ranking_and_decay();
    test_schedule_and_promotion();
    test_schedule_overflow_fails_closed();
    test_persistence();
    test_bad_observation_is_transactional();
    if (failures) {
        fprintf(stderr, "%d signature test(s) failed\n", failures);
        return 1;
    }
    puts("signature tests passed");
    return 0;
}
