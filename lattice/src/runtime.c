#define _POSIX_C_SOURCE 200809L
#include "lattice/runtime.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct lt_runtime {
    lt_runtime_config_t config;
    lt_runtime_adapter_t adapter;
    lt_scheduler_t *scheduler;
    lt_signature_t *signature;
    lt_io_reader_t *reader;
    pthread_mutex_t mutex;
    int mutex_initialized;
    uint64_t resolve_failures;
    uint64_t submit_failures;
    uint64_t completion_failures;
    uint64_t dispatched_jobs;
};

static void lt_runtime_error(char *error, size_t cap, const char *message) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", message ? message : "runtime error");
}

static void lt_runtime_io_complete(const lt_io_job_t *job,
                                   int status,
                                   uint64_t bytes_read,
                                   void *user) {
    lt_runtime_t *runtime = (lt_runtime_t *)user;
    char ignored[128];
    int finish_status;

    pthread_mutex_lock(&runtime->mutex);
    if (status == 0 && bytes_read == job->bytes)
        finish_status = lt_scheduler_complete(runtime->scheduler, job->tensor_id,
                                              ignored, sizeof(ignored));
    else
        finish_status = lt_scheduler_fail(runtime->scheduler, job->tensor_id,
                                          ignored, sizeof(ignored));
    if (status != 0 || bytes_read != job->bytes || finish_status != 0)
        runtime->completion_failures++;
    pthread_mutex_unlock(&runtime->mutex);

    if (runtime->adapter.complete)
        runtime->adapter.complete(runtime->adapter.context, job,
                                  (status == 0 && bytes_read != job->bytes) ? -EIO : status,
                                  bytes_read);
}

lt_runtime_t *lt_runtime_create(const lt_runtime_config_t *config,
                                const lt_runtime_adapter_t *adapter,
                                char *error,
                                size_t error_cap) {
    lt_runtime_t *runtime;
    if (!config || !adapter || !adapter->resolve ||
        config->layers == 0 || config->experts == 0 ||
        config->expert_bytes_each == 0 || config->scheduler_capacity == 0 ||
        config->io_workers == 0 || config->io_queue_capacity == 0) {
        lt_runtime_error(error, error_cap, "invalid runtime configuration or adapter");
        return NULL;
    }
    runtime = (lt_runtime_t *)calloc(1, sizeof(*runtime));
    if (!runtime) {
        lt_runtime_error(error, error_cap, "out of memory creating runtime");
        return NULL;
    }
    runtime->config = *config;
    runtime->adapter = *adapter;
    if (pthread_mutex_init(&runtime->mutex, NULL) != 0) {
        lt_runtime_error(error, error_cap, "cannot initialize runtime mutex");
        free(runtime);
        return NULL;
    }
    runtime->mutex_initialized = 1;
    runtime->scheduler = lt_scheduler_create(config->scheduler_capacity,
                                             config->max_inflight_bytes);
    runtime->signature = lt_signature_create(config->layers, config->experts);
    if (!runtime->scheduler || !runtime->signature) {
        lt_runtime_error(error, error_cap, "out of memory creating scheduler or signature");
        lt_runtime_destroy(runtime);
        return NULL;
    }
    runtime->reader = lt_io_reader_create(config->io_workers,
                                          config->io_queue_capacity,
                                          lt_runtime_io_complete,
                                          runtime);
    if (!runtime->reader) {
        lt_runtime_error(error, error_cap, "cannot create asynchronous I/O reader");
        lt_runtime_destroy(runtime);
        return NULL;
    }
    lt_runtime_error(error, error_cap, "");
    return runtime;
}

void lt_runtime_destroy(lt_runtime_t *runtime) {
    if (!runtime) return;
    if (runtime->reader) {
        lt_io_reader_destroy(runtime->reader);
        runtime->reader = NULL;
    }
    lt_signature_destroy(runtime->signature);
    lt_scheduler_destroy(runtime->scheduler);
    if (runtime->mutex_initialized) pthread_mutex_destroy(&runtime->mutex);
    free(runtime);
}

int lt_runtime_submit(lt_runtime_t *runtime,
                      const lt_transfer_request_t *request,
                      char *error,
                      size_t error_cap) {
    int result;
    if (!runtime || !request) {
        lt_runtime_error(error, error_cap, "runtime and request are required");
        return -1;
    }
    pthread_mutex_lock(&runtime->mutex);
    result = lt_scheduler_submit(runtime->scheduler, request, error, error_cap);
    pthread_mutex_unlock(&runtime->mutex);
    return result;
}

int lt_runtime_demand(lt_runtime_t *runtime,
                      uint64_t tensor_id,
                      uint64_t bytes,
                      uint32_t layer,
                      uint64_t deadline_ns,
                      uint64_t opaque,
                      char *error,
                      size_t error_cap) {
    lt_transfer_request_t request;
    memset(&request, 0, sizeof(request));
    request.tensor_id = tensor_id;
    request.bytes = bytes;
    request.layer = layer;
    request.deadline_ns = deadline_ns;
    request.confidence = 1.0;
    request.path = LT_PATH_DEMAND;
    request.opaque = opaque;
    return lt_runtime_submit(runtime, &request, error, error_cap);
}

int lt_runtime_observe_experts(lt_runtime_t *runtime,
                               uint32_t layer,
                               const uint32_t *expert_ids,
                               const double *weights,
                               size_t count) {
    int result;
    if (!runtime) return -1;
    pthread_mutex_lock(&runtime->mutex);
    result = lt_signature_observe(runtime->signature, layer, expert_ids, weights, count);
    pthread_mutex_unlock(&runtime->mutex);
    return result;
}

int lt_runtime_prefetch_experts(lt_runtime_t *runtime,
                                uint32_t layer,
                                size_t k,
                                uint64_t deadline_ns,
                                lt_transfer_path_t path,
                                double min_confidence,
                                char *error,
                                size_t error_cap) {
    int result;
    if (!runtime || path == LT_PATH_DEMAND) {
        lt_runtime_error(error, error_cap, "prefetch path must be proactive or speculative");
        return -1;
    }
    pthread_mutex_lock(&runtime->mutex);
    result = lt_signature_schedule(runtime->signature,
                                   runtime->scheduler,
                                   layer,
                                   k,
                                   runtime->config.expert_tensor_id_base,
                                   runtime->config.expert_bytes_each,
                                   deadline_ns,
                                   path,
                                   min_confidence,
                                   error,
                                   error_cap);
    pthread_mutex_unlock(&runtime->mutex);
    return result;
}

static void lt_runtime_mark_failed(lt_runtime_t *runtime,
                                   uint64_t tensor_id,
                                   int resolve_failure) {
    char ignored[128];
    pthread_mutex_lock(&runtime->mutex);
    (void)lt_scheduler_fail(runtime->scheduler, tensor_id, ignored, sizeof(ignored));
    if (resolve_failure) runtime->resolve_failures++;
    else runtime->submit_failures++;
    pthread_mutex_unlock(&runtime->mutex);
}

int lt_runtime_dispatch(lt_runtime_t *runtime,
                        size_t max_jobs,
                        char *error,
                        size_t error_cap) {
    size_t submitted = 0;
    int had_failure = 0;
    char first_error[256] = {0};
    if (!runtime) {
        lt_runtime_error(error, error_cap, "runtime is required");
        return -1;
    }
    while (max_jobs == 0 || submitted < max_jobs) {
        lt_transfer_request_t request;
        lt_io_job_t job;
        char local_error[256] = {0};
        int popped;

        pthread_mutex_lock(&runtime->mutex);
        popped = lt_scheduler_pop(runtime->scheduler, &request);
        pthread_mutex_unlock(&runtime->mutex);
        if (popped < 0) {
            lt_runtime_error(error, error_cap, "scheduler pop failed");
            return -1;
        }
        if (popped == 0) break;

        memset(&job, 0, sizeof(job));
        job.fd = -1;
        if (runtime->adapter.resolve(runtime->adapter.context,
                                     &request,
                                     &job,
                                     local_error,
                                     sizeof(local_error)) != 0 ||
            job.fd < 0 || !job.destination || job.bytes != request.bytes) {
            lt_runtime_mark_failed(runtime, request.tensor_id, 1);
            if (!had_failure) {
                snprintf(first_error, sizeof(first_error), "%s",
                         local_error[0] ? local_error : "adapter produced an invalid I/O job");
            }
            had_failure = 1;
            continue;
        }
        job.tensor_id = request.tensor_id;
        job.layer = request.layer;
        job.transfer_path = (uint32_t)request.path;
        job.confidence = request.confidence;
        job.request_opaque = request.opaque;

        if (lt_io_reader_submit_wait(runtime->reader, &job) != 1) {
            lt_runtime_mark_failed(runtime, request.tensor_id, 0);
            if (runtime->adapter.complete)
                runtime->adapter.complete(runtime->adapter.context, &job, -ECANCELED, 0);
            if (!had_failure)
                snprintf(first_error, sizeof(first_error), "asynchronous I/O submission failed");
            had_failure = 1;
            continue;
        }
        pthread_mutex_lock(&runtime->mutex);
        runtime->dispatched_jobs++;
        pthread_mutex_unlock(&runtime->mutex);
        ++submitted;
    }
    if (had_failure) {
        lt_runtime_error(error, error_cap, first_error);
        return -1;
    }
    lt_runtime_error(error, error_cap, "");
    return submitted > (size_t)INT32_MAX ? INT32_MAX : (int)submitted;
}

int lt_runtime_run_until_idle(lt_runtime_t *runtime,
                              char *error,
                              size_t error_cap) {
    int had_failure = 0;
    char first_error[256] = {0};
    if (!runtime) {
        lt_runtime_error(error, error_cap, "runtime is required");
        return -1;
    }
    for (;;) {
        int dispatched = lt_runtime_dispatch(runtime, 0, error, error_cap);
        size_t queued, inflight;
        if (dispatched < 0) {
            had_failure = 1;
            if (!first_error[0]) snprintf(first_error, sizeof(first_error), "%s", error && *error ? error : "dispatch failed");
        }
        if (lt_io_reader_wait_idle(runtime->reader) != 0) {
            had_failure = 1;
            if (!first_error[0]) snprintf(first_error, sizeof(first_error), "one or more asynchronous reads failed");
        }
        pthread_mutex_lock(&runtime->mutex);
        queued = lt_scheduler_queued(runtime->scheduler);
        inflight = lt_scheduler_inflight(runtime->scheduler);
        pthread_mutex_unlock(&runtime->mutex);
        if (queued == 0 && inflight == 0) break;
        if (dispatched == 0 && inflight == 0) {
            lt_runtime_error(error, error_cap,
                             "queued request cannot satisfy the configured inflight-byte cap");
            return -1;
        }
    }
    if (had_failure) {
        lt_runtime_error(error, error_cap, first_error);
        return -1;
    }
    lt_runtime_error(error, error_cap, "");
    return 0;
}

int lt_runtime_forget(lt_runtime_t *runtime, uint64_t tensor_id) {
    int result;
    if (!runtime) return -1;
    pthread_mutex_lock(&runtime->mutex);
    result = lt_scheduler_forget(runtime->scheduler, tensor_id);
    pthread_mutex_unlock(&runtime->mutex);
    return result;
}

size_t lt_runtime_cancel_speculative(lt_runtime_t *runtime,
                                     double min_confidence) {
    size_t result;
    if (!runtime) return 0;
    pthread_mutex_lock(&runtime->mutex);
    result = lt_scheduler_cancel_speculative(runtime->scheduler, min_confidence);
    pthread_mutex_unlock(&runtime->mutex);
    return result;
}

lt_runtime_stats_t lt_runtime_stats(lt_runtime_t *runtime) {
    lt_runtime_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    if (!runtime) return stats;
    pthread_mutex_lock(&runtime->mutex);
    stats.scheduler = lt_scheduler_stats(runtime->scheduler);
    stats.resolve_failures = runtime->resolve_failures;
    stats.submit_failures = runtime->submit_failures;
    stats.completion_failures = runtime->completion_failures;
    stats.dispatched_jobs = runtime->dispatched_jobs;
    pthread_mutex_unlock(&runtime->mutex);
    stats.io = lt_io_reader_stats(runtime->reader);
    return stats;
}
