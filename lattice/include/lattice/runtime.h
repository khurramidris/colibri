#ifndef LATTICE_RUNTIME_H
#define LATTICE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "lattice/io_reader.h"
#include "lattice/scheduler.h"
#include "lattice/signature.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lt_runtime lt_runtime_t;

/* Resolve a logical tensor request into one concrete positional read and a
 * caller-owned destination. The adapter may reserve a cache slot here. It must
 * not mark the tensor resident until completion reports status==0. */
typedef int (*lt_runtime_resolve_fn)(void *adapter_context,
                                     const lt_transfer_request_t *request,
                                     lt_io_job_t *out_job,
                                     char *error,
                                     size_t error_cap);

/* Called exactly once for every successfully resolved job, including submit or
 * read failures. Adapters release reservations on failure and publish the cache
 * entry on success. May run concurrently on I/O worker threads. */
typedef void (*lt_runtime_complete_fn)(void *adapter_context,
                                       const lt_io_job_t *job,
                                       int status,
                                       uint64_t bytes_read);

typedef struct {
    void *context;
    lt_runtime_resolve_fn resolve;
    lt_runtime_complete_fn complete;
} lt_runtime_adapter_t;

typedef struct {
    uint32_t layers;
    uint32_t experts;
    uint64_t expert_tensor_id_base;
    uint64_t expert_bytes_each;
    size_t scheduler_capacity;
    uint64_t max_inflight_bytes;
    size_t io_workers;
    size_t io_queue_capacity;
} lt_runtime_config_t;

typedef struct {
    lt_scheduler_stats_t scheduler;
    lt_io_stats_t io;
    uint64_t resolve_failures;
    uint64_t submit_failures;
    uint64_t completion_failures;
    uint64_t dispatched_jobs;
} lt_runtime_stats_t;

lt_runtime_t *lt_runtime_create(const lt_runtime_config_t *config,
                                const lt_runtime_adapter_t *adapter,
                                char *error,
                                size_t error_cap);

/* Drains accepted I/O before releasing resources. */
void lt_runtime_destroy(lt_runtime_t *runtime);

/* Generic exact transfer submission. Returns scheduler submit semantics. */
int lt_runtime_submit(lt_runtime_t *runtime,
                      const lt_transfer_request_t *request,
                      char *error,
                      size_t error_cap);

/* Convenience helper for an authoritative model-router demand. */
int lt_runtime_demand(lt_runtime_t *runtime,
                      uint64_t tensor_id,
                      uint64_t bytes,
                      uint32_t layer,
                      uint64_t deadline_ns,
                      uint64_t opaque,
                      char *error,
                      size_t error_cap);

/* Record request-local prefill routes and schedule likely experts. */
int lt_runtime_observe_experts(lt_runtime_t *runtime,
                               uint32_t layer,
                               const uint32_t *expert_ids,
                               const double *weights,
                               size_t count);
int lt_runtime_prefetch_experts(lt_runtime_t *runtime,
                                uint32_t layer,
                                size_t k,
                                uint64_t deadline_ns,
                                lt_transfer_path_t path,
                                double min_confidence,
                                char *error,
                                size_t error_cap);

/* Resolve and submit up to max_jobs currently dispatchable requests. max_jobs=0
 * means no count limit. Returns submitted job count, or -1 if any request could
 * not be resolved/submitted (other independent requests are still attempted). */
int lt_runtime_dispatch(lt_runtime_t *runtime,
                        size_t max_jobs,
                        char *error,
                        size_t error_cap);

/* Repeatedly dispatch and wait until queued and in-flight work reach zero.
 * Detects an impossible inflight-byte cap instead of spinning forever. */
int lt_runtime_run_until_idle(lt_runtime_t *runtime,
                              char *error,
                              size_t error_cap);

size_t lt_runtime_cancel_speculative(lt_runtime_t *runtime,
                                     double min_confidence);
lt_runtime_stats_t lt_runtime_stats(lt_runtime_t *runtime);

#ifdef __cplusplus
}
#endif

#endif /* LATTICE_RUNTIME_H */
