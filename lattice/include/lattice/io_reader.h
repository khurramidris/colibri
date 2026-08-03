#ifndef LATTICE_IO_READER_H
#define LATTICE_IO_READER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lt_io_reader lt_io_reader_t;

typedef struct {
    uint64_t tensor_id;
    int fd;
    uint64_t offset;
    void *destination;
    uint64_t bytes;
    uint64_t opaque;
} lt_io_job_t;

typedef void (*lt_io_completion_fn)(const lt_io_job_t *job,
                                    int status,
                                    uint64_t bytes_read,
                                    void *user);

typedef struct {
    uint64_t submitted_jobs;
    uint64_t completed_jobs;
    uint64_t failed_jobs;
    uint64_t submitted_bytes;
    uint64_t completed_bytes;
    uint64_t inflight_jobs;
    uint64_t queued_jobs;
} lt_io_stats_t;

/* Portable POSIX worker pool for already-prioritized tensor reads. Scheduling
 * policy remains in scheduler.c; this component only executes concrete pread
 * jobs concurrently. User buffers remain owned by the caller and must stay
 * valid until completion. */
lt_io_reader_t *lt_io_reader_create(size_t workers,
                                    size_t queue_capacity,
                                    lt_io_completion_fn completion,
                                    void *completion_user);

/* Drains queued/in-flight jobs, joins workers and releases the reader. */
void lt_io_reader_destroy(lt_io_reader_t *reader);

/* Non-blocking submission: 1 accepted, 0 queue full, -1 invalid/stopped. */
int lt_io_reader_submit(lt_io_reader_t *reader, const lt_io_job_t *job);

/* Wait until every accepted job completes. Returns 0, or -1 if any job failed. */
int lt_io_reader_wait_idle(lt_io_reader_t *reader);

lt_io_stats_t lt_io_reader_stats(lt_io_reader_t *reader);

#ifdef __cplusplus
}
#endif

#endif /* LATTICE_IO_READER_H */
