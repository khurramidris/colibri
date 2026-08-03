#ifndef LATTICE_SCHEDULER_H
#define LATTICE_SCHEDULER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LT_PATH_SPECULATIVE = 0,
    LT_PATH_PROACTIVE = 1,
    LT_PATH_DEMAND = 2
} lt_transfer_path_t;

typedef enum {
    LT_REQ_QUEUED = 0,
    LT_REQ_INFLIGHT = 1,
    LT_REQ_DONE = 2,
    LT_REQ_CANCELLED = 3,
    LT_REQ_FAILED = 4
} lt_request_state_t;

typedef struct {
    uint64_t tensor_id;
    uint64_t bytes;
    uint32_t layer;
    uint64_t deadline_ns;
    double confidence;
    lt_transfer_path_t path;
    uint64_t opaque;
} lt_transfer_request_t;

typedef struct {
    uint64_t submitted;
    uint64_t deduplicated;
    uint64_t promoted;
    uint64_t dispatched;
    uint64_t completed;
    uint64_t failed;
    uint64_t cancelled;
    uint64_t forgotten;
    uint64_t queued_bytes;
    uint64_t inflight_bytes;
    uint64_t completed_bytes;
    uint64_t failed_bytes;
} lt_scheduler_stats_t;

typedef struct lt_scheduler lt_scheduler_t;

/* max_inflight_bytes=0 means no byte cap. The scheduler is deterministic:
 * demand before proactive before speculative, then earliest deadline, then
 * highest confidence, then insertion order. Duplicate tensor IDs are merged;
 * later demand requests promote earlier predictions instead of causing a
 * second read. */
lt_scheduler_t *lt_scheduler_create(size_t capacity, uint64_t max_inflight_bytes);
void lt_scheduler_destroy(lt_scheduler_t *scheduler);

/* Returns 1 when inserted, 2 when merged/promoted, 0 when an identical
 * completed/inflight request already covers it, and -1 on validation/capacity
 * failure. error may be NULL. */
int lt_scheduler_submit(lt_scheduler_t *scheduler,
                        const lt_transfer_request_t *request,
                        char *error,
                        size_t error_cap);

/* Returns 1 and writes a request when work can be dispatched, 0 when no
 * dispatchable work exists, -1 on invalid arguments. */
int lt_scheduler_pop(lt_scheduler_t *scheduler,
                     lt_transfer_request_t *out_request);

/* Mark one in-flight tensor complete or failed. Failed records may be compacted
 * and resubmitted; they never masquerade as a resident tensor. */
int lt_scheduler_complete(lt_scheduler_t *scheduler,
                          uint64_t tensor_id,
                          char *error,
                          size_t error_cap);
int lt_scheduler_fail(lt_scheduler_t *scheduler,
                      uint64_t tensor_id,
                      char *error,
                      size_t error_cap);

/* Tell the scheduler that a previously completed tensor was evicted. Returns
 * 1 when forgotten, 0 when absent/already terminal, and -1 for queued/inflight
 * tensors or invalid arguments. The next submission may read it again. */
int lt_scheduler_forget(lt_scheduler_t *scheduler, uint64_t tensor_id);

/* Cancel queued speculative work below min_confidence. Returns count removed. */
size_t lt_scheduler_cancel_speculative(lt_scheduler_t *scheduler,
                                       double min_confidence);

/* Cancel any queued request for a tensor. In-flight reads are not aborted. */
int lt_scheduler_cancel(lt_scheduler_t *scheduler, uint64_t tensor_id);

/* Forget terminal records while preserving queued/in-flight work. */
void lt_scheduler_compact(lt_scheduler_t *scheduler);

size_t lt_scheduler_queued(const lt_scheduler_t *scheduler);
size_t lt_scheduler_inflight(const lt_scheduler_t *scheduler);
lt_scheduler_stats_t lt_scheduler_stats(const lt_scheduler_t *scheduler);
const char *lt_transfer_path_name(lt_transfer_path_t path);

#ifdef __cplusplus
}
#endif

#endif /* LATTICE_SCHEDULER_H */
