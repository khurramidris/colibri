#include "lattice/scheduler.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    lt_transfer_request_t request;
    lt_request_state_t state;
    uint64_t sequence;
} lt_scheduler_entry_t;

struct lt_scheduler {
    lt_scheduler_entry_t *entries;
    size_t capacity;
    size_t count;
    uint64_t next_sequence;
    uint64_t max_inflight_bytes;
    lt_scheduler_stats_t stats;
};

static void lt_scheduler_error(char *error, size_t cap, const char *message) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", message ? message : "scheduler error");
}

const char *lt_transfer_path_name(lt_transfer_path_t path) {
    switch (path) {
        case LT_PATH_SPECULATIVE: return "speculative";
        case LT_PATH_PROACTIVE: return "proactive";
        case LT_PATH_DEMAND: return "demand";
        default: return "unknown";
    }
}

static int lt_request_valid(const lt_transfer_request_t *request) {
    return request && request->tensor_id != 0 && request->bytes != 0 &&
           request->path >= LT_PATH_SPECULATIVE && request->path <= LT_PATH_DEMAND &&
           isfinite(request->confidence) && request->confidence >= 0.0 &&
           request->confidence <= 1.0;
}

static size_t lt_find_active(const lt_scheduler_t *scheduler, uint64_t tensor_id) {
    size_t i;
    for (i = 0; i < scheduler->count; ++i) {
        lt_request_state_t state = scheduler->entries[i].state;
        if (scheduler->entries[i].request.tensor_id == tensor_id &&
            state != LT_REQ_CANCELLED && state != LT_REQ_FAILED) return i;
    }
    return (size_t)-1;
}

static int lt_entry_better(const lt_scheduler_entry_t *a,
                           const lt_scheduler_entry_t *b) {
    if (a->request.path != b->request.path)
        return a->request.path > b->request.path;
    if (a->request.deadline_ns != b->request.deadline_ns) {
        /* deadline 0 means no deadline and sorts last. */
        if (a->request.deadline_ns == 0) return 0;
        if (b->request.deadline_ns == 0) return 1;
        return a->request.deadline_ns < b->request.deadline_ns;
    }
    if (a->request.confidence != b->request.confidence)
        return a->request.confidence > b->request.confidence;
    return a->sequence < b->sequence;
}

lt_scheduler_t *lt_scheduler_create(size_t capacity, uint64_t max_inflight_bytes) {
    lt_scheduler_t *scheduler;
    if (capacity == 0) return NULL;
    scheduler = (lt_scheduler_t *)calloc(1, sizeof(*scheduler));
    if (!scheduler) return NULL;
    scheduler->entries = (lt_scheduler_entry_t *)calloc(capacity, sizeof(*scheduler->entries));
    if (!scheduler->entries) {
        free(scheduler);
        return NULL;
    }
    scheduler->capacity = capacity;
    scheduler->max_inflight_bytes = max_inflight_bytes;
    scheduler->next_sequence = 1;
    return scheduler;
}

void lt_scheduler_destroy(lt_scheduler_t *scheduler) {
    if (!scheduler) return;
    free(scheduler->entries);
    free(scheduler);
}

int lt_scheduler_submit(lt_scheduler_t *scheduler,
                        const lt_transfer_request_t *request,
                        char *error,
                        size_t error_cap) {
    size_t found;
    lt_scheduler_entry_t *entry;
    if (!scheduler || !lt_request_valid(request)) {
        lt_scheduler_error(error, error_cap, "invalid transfer request");
        return -1;
    }
    found = lt_find_active(scheduler, request->tensor_id);
    if (found != (size_t)-1) {
        entry = &scheduler->entries[found];
        if (entry->request.bytes != request->bytes) {
            lt_scheduler_error(error, error_cap, "duplicate tensor ID has a different byte size");
            return -1;
        }
        scheduler->stats.deduplicated++;
        if (entry->state == LT_REQ_DONE || entry->state == LT_REQ_INFLIGHT) return 0;
        if (request->path > entry->request.path) {
            entry->request.path = request->path;
            scheduler->stats.promoted++;
        }
        if (request->deadline_ns &&
            (!entry->request.deadline_ns || request->deadline_ns < entry->request.deadline_ns))
            entry->request.deadline_ns = request->deadline_ns;
        if (request->confidence > entry->request.confidence)
            entry->request.confidence = request->confidence;
        if (request->layer < entry->request.layer)
            entry->request.layer = request->layer;
        if (request->opaque) entry->request.opaque = request->opaque;
        return 2;
    }
    if (scheduler->count == scheduler->capacity) {
        lt_scheduler_compact(scheduler);
        if (scheduler->count == scheduler->capacity) {
            lt_scheduler_error(error, error_cap, "transfer queue capacity exhausted");
            return -1;
        }
    }
    entry = &scheduler->entries[scheduler->count++];
    memset(entry, 0, sizeof(*entry));
    entry->request = *request;
    entry->state = LT_REQ_QUEUED;
    entry->sequence = scheduler->next_sequence++;
    scheduler->stats.submitted++;
    scheduler->stats.queued_bytes += request->bytes;
    lt_scheduler_error(error, error_cap, "");
    return 1;
}

int lt_scheduler_pop(lt_scheduler_t *scheduler,
                     lt_transfer_request_t *out_request) {
    size_t best = (size_t)-1;
    size_t i;
    if (!scheduler || !out_request) return -1;
    for (i = 0; i < scheduler->count; ++i) {
        lt_scheduler_entry_t *entry = &scheduler->entries[i];
        uint64_t cap = scheduler->max_inflight_bytes;
        if (entry->state != LT_REQ_QUEUED) continue;
        if (cap != 0 &&
            (scheduler->stats.inflight_bytes >= cap ||
             entry->request.bytes > cap - scheduler->stats.inflight_bytes)) continue;
        if (best == (size_t)-1 || lt_entry_better(entry, &scheduler->entries[best])) best = i;
    }
    if (best == (size_t)-1) return 0;
    scheduler->entries[best].state = LT_REQ_INFLIGHT;
    *out_request = scheduler->entries[best].request;
    scheduler->stats.dispatched++;
    scheduler->stats.queued_bytes -= out_request->bytes;
    scheduler->stats.inflight_bytes += out_request->bytes;
    return 1;
}

static int lt_scheduler_finish(lt_scheduler_t *scheduler,
                               uint64_t tensor_id,
                               int success,
                               char *error,
                               size_t error_cap) {
    size_t found;
    lt_scheduler_entry_t *entry;
    if (!scheduler || tensor_id == 0) {
        lt_scheduler_error(error, error_cap, "invalid transfer completion");
        return -1;
    }
    found = lt_find_active(scheduler, tensor_id);
    if (found == (size_t)-1) {
        lt_scheduler_error(error, error_cap, "completion references an unknown tensor");
        return -1;
    }
    entry = &scheduler->entries[found];
    if (entry->state != LT_REQ_INFLIGHT) {
        lt_scheduler_error(error, error_cap, "completion references a tensor that is not in flight");
        return -1;
    }
    if (scheduler->stats.inflight_bytes < entry->request.bytes) {
        lt_scheduler_error(error, error_cap, "scheduler inflight byte accounting underflow");
        return -1;
    }
    scheduler->stats.inflight_bytes -= entry->request.bytes;
    if (success) {
        entry->state = LT_REQ_DONE;
        scheduler->stats.completed_bytes += entry->request.bytes;
        scheduler->stats.completed++;
    } else {
        entry->state = LT_REQ_FAILED;
        scheduler->stats.failed_bytes += entry->request.bytes;
        scheduler->stats.failed++;
    }
    lt_scheduler_error(error, error_cap, "");
    return 0;
}

int lt_scheduler_complete(lt_scheduler_t *scheduler,
                          uint64_t tensor_id,
                          char *error,
                          size_t error_cap) {
    return lt_scheduler_finish(scheduler, tensor_id, 1, error, error_cap);
}

int lt_scheduler_fail(lt_scheduler_t *scheduler,
                      uint64_t tensor_id,
                      char *error,
                      size_t error_cap) {
    return lt_scheduler_finish(scheduler, tensor_id, 0, error, error_cap);
}

size_t lt_scheduler_cancel_speculative(lt_scheduler_t *scheduler,
                                       double min_confidence) {
    size_t removed = 0;
    size_t i;
    if (!scheduler || !isfinite(min_confidence)) return 0;
    if (min_confidence < 0.0) min_confidence = 0.0;
    if (min_confidence > 1.0) min_confidence = 1.0;
    for (i = 0; i < scheduler->count; ++i) {
        lt_scheduler_entry_t *entry = &scheduler->entries[i];
        if (entry->state == LT_REQ_QUEUED &&
            entry->request.path == LT_PATH_SPECULATIVE &&
            entry->request.confidence < min_confidence) {
            entry->state = LT_REQ_CANCELLED;
            scheduler->stats.queued_bytes -= entry->request.bytes;
            scheduler->stats.cancelled++;
            ++removed;
        }
    }
    return removed;
}

int lt_scheduler_cancel(lt_scheduler_t *scheduler, uint64_t tensor_id) {
    size_t found;
    lt_scheduler_entry_t *entry;
    if (!scheduler || tensor_id == 0) return -1;
    found = lt_find_active(scheduler, tensor_id);
    if (found == (size_t)-1) return 0;
    entry = &scheduler->entries[found];
    if (entry->state != LT_REQ_QUEUED) return 0;
    entry->state = LT_REQ_CANCELLED;
    scheduler->stats.queued_bytes -= entry->request.bytes;
    scheduler->stats.cancelled++;
    return 1;
}

void lt_scheduler_compact(lt_scheduler_t *scheduler) {
    size_t read_index, write_index = 0;
    if (!scheduler) return;
    for (read_index = 0; read_index < scheduler->count; ++read_index) {
        lt_request_state_t state = scheduler->entries[read_index].state;
        if (state == LT_REQ_DONE || state == LT_REQ_CANCELLED || state == LT_REQ_FAILED) continue;
        if (write_index != read_index) scheduler->entries[write_index] = scheduler->entries[read_index];
        ++write_index;
    }
    scheduler->count = write_index;
}

size_t lt_scheduler_queued(const lt_scheduler_t *scheduler) {
    size_t count = 0, i;
    if (!scheduler) return 0;
    for (i = 0; i < scheduler->count; ++i)
        if (scheduler->entries[i].state == LT_REQ_QUEUED) ++count;
    return count;
}

size_t lt_scheduler_inflight(const lt_scheduler_t *scheduler) {
    size_t count = 0, i;
    if (!scheduler) return 0;
    for (i = 0; i < scheduler->count; ++i)
        if (scheduler->entries[i].state == LT_REQ_INFLIGHT) ++count;
    return count;
}

lt_scheduler_stats_t lt_scheduler_stats(const lt_scheduler_t *scheduler) {
    lt_scheduler_stats_t empty;
    memset(&empty, 0, sizeof(empty));
    return scheduler ? scheduler->stats : empty;
}
