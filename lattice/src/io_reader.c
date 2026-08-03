#define _POSIX_C_SOURCE 200809L
#include "lattice/io_reader.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

struct lt_io_reader {
    pthread_t *threads;
    size_t workers;
    lt_io_job_t *queue;
    size_t queue_capacity;
    size_t queue_head;
    size_t queue_tail;
    size_t queue_count;
    pthread_mutex_t mutex;
    pthread_cond_t has_work;
    pthread_cond_t idle;
    int stopping;
    int initialized_mutex;
    int initialized_has_work;
    int initialized_idle;
    int any_failure;
    lt_io_completion_fn completion;
    void *completion_user;
    lt_io_stats_t stats;
};

static int lt_pread_full(const lt_io_job_t *job, uint64_t *out_read) {
    uint64_t done = 0;
    unsigned char *destination = (unsigned char *)job->destination;
    while (done < job->bytes) {
        uint64_t remaining = job->bytes - done;
        size_t chunk = remaining > (uint64_t)SSIZE_MAX ? (size_t)SSIZE_MAX : (size_t)remaining;
        ssize_t got = pread(job->fd, destination + (size_t)done, chunk,
                            (off_t)(job->offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            *out_read = done;
            return errno ? -errno : -EIO;
        }
        if (got == 0) {
            *out_read = done;
            return -EIO;
        }
        done += (uint64_t)got;
    }
    *out_read = done;
    return 0;
}

static void *lt_io_worker(void *opaque) {
    lt_io_reader_t *reader = (lt_io_reader_t *)opaque;
    for (;;) {
        lt_io_job_t job;
        uint64_t bytes_read = 0;
        int status;
        pthread_mutex_lock(&reader->mutex);
        while (reader->queue_count == 0 && !reader->stopping)
            pthread_cond_wait(&reader->has_work, &reader->mutex);
        if (reader->queue_count == 0 && reader->stopping) {
            pthread_mutex_unlock(&reader->mutex);
            break;
        }
        job = reader->queue[reader->queue_head];
        reader->queue_head = (reader->queue_head + 1) % reader->queue_capacity;
        reader->queue_count--;
        reader->stats.queued_jobs--;
        reader->stats.inflight_jobs++;
        pthread_mutex_unlock(&reader->mutex);

        status = lt_pread_full(&job, &bytes_read);

        /* Account the result before the callback, but keep the job in-flight
         * until the callback returns. wait_idle therefore guarantees that
         * user completion work is finished and buffers may be reused. */
        pthread_mutex_lock(&reader->mutex);
        if (status == 0) {
            reader->stats.completed_jobs++;
            reader->stats.completed_bytes += bytes_read;
        } else {
            reader->stats.failed_jobs++;
            reader->any_failure = 1;
        }
        pthread_mutex_unlock(&reader->mutex);

        if (reader->completion)
            reader->completion(&job, status, bytes_read, reader->completion_user);

        pthread_mutex_lock(&reader->mutex);
        reader->stats.inflight_jobs--;
        if (reader->queue_count == 0 && reader->stats.inflight_jobs == 0)
            pthread_cond_broadcast(&reader->idle);
        pthread_mutex_unlock(&reader->mutex);
    }
    return NULL;
}

lt_io_reader_t *lt_io_reader_create(size_t workers,
                                    size_t queue_capacity,
                                    lt_io_completion_fn completion,
                                    void *completion_user) {
    lt_io_reader_t *reader;
    size_t created = 0;
    if (workers == 0 || queue_capacity == 0) return NULL;
    reader = (lt_io_reader_t *)calloc(1, sizeof(*reader));
    if (!reader) return NULL;
    reader->threads = (pthread_t *)calloc(workers, sizeof(*reader->threads));
    reader->queue = (lt_io_job_t *)calloc(queue_capacity, sizeof(*reader->queue));
    if (!reader->threads || !reader->queue) goto fail;
    reader->workers = workers;
    reader->queue_capacity = queue_capacity;
    reader->completion = completion;
    reader->completion_user = completion_user;
    if (pthread_mutex_init(&reader->mutex, NULL) != 0) goto fail;
    reader->initialized_mutex = 1;
    if (pthread_cond_init(&reader->has_work, NULL) != 0) goto fail;
    reader->initialized_has_work = 1;
    if (pthread_cond_init(&reader->idle, NULL) != 0) goto fail;
    reader->initialized_idle = 1;
    for (created = 0; created < workers; ++created) {
        if (pthread_create(&reader->threads[created], NULL, lt_io_worker, reader) != 0)
            goto fail_threads;
    }
    return reader;

fail_threads:
    pthread_mutex_lock(&reader->mutex);
    reader->stopping = 1;
    pthread_cond_broadcast(&reader->has_work);
    pthread_mutex_unlock(&reader->mutex);
    while (created > 0) {
        --created;
        pthread_join(reader->threads[created], NULL);
    }
fail:
    if (reader->initialized_idle) pthread_cond_destroy(&reader->idle);
    if (reader->initialized_has_work) pthread_cond_destroy(&reader->has_work);
    if (reader->initialized_mutex) pthread_mutex_destroy(&reader->mutex);
    free(reader->threads);
    free(reader->queue);
    free(reader);
    return NULL;
}

void lt_io_reader_destroy(lt_io_reader_t *reader) {
    size_t i;
    if (!reader) return;
    (void)lt_io_reader_wait_idle(reader);
    pthread_mutex_lock(&reader->mutex);
    reader->stopping = 1;
    pthread_cond_broadcast(&reader->has_work);
    pthread_mutex_unlock(&reader->mutex);
    for (i = 0; i < reader->workers; ++i) pthread_join(reader->threads[i], NULL);
    pthread_cond_destroy(&reader->idle);
    pthread_cond_destroy(&reader->has_work);
    pthread_mutex_destroy(&reader->mutex);
    free(reader->threads);
    free(reader->queue);
    free(reader);
}

int lt_io_reader_submit(lt_io_reader_t *reader, const lt_io_job_t *job) {
    if (!reader || !job || job->tensor_id == 0 || job->fd < 0 ||
        !job->destination || job->bytes == 0 || job->bytes > SIZE_MAX ||
        job->offset > (uint64_t)INT64_MAX ||
        job->bytes - 1 > (uint64_t)INT64_MAX - job->offset)
        return -1;
    pthread_mutex_lock(&reader->mutex);
    if (reader->stopping) {
        pthread_mutex_unlock(&reader->mutex);
        return -1;
    }
    if (reader->queue_count == reader->queue_capacity) {
        pthread_mutex_unlock(&reader->mutex);
        return 0;
    }
    reader->queue[reader->queue_tail] = *job;
    reader->queue_tail = (reader->queue_tail + 1) % reader->queue_capacity;
    reader->queue_count++;
    reader->stats.submitted_jobs++;
    reader->stats.submitted_bytes += job->bytes;
    reader->stats.queued_jobs++;
    pthread_cond_signal(&reader->has_work);
    pthread_mutex_unlock(&reader->mutex);
    return 1;
}

int lt_io_reader_wait_idle(lt_io_reader_t *reader) {
    int failed;
    if (!reader) return -1;
    pthread_mutex_lock(&reader->mutex);
    while (reader->queue_count != 0 || reader->stats.inflight_jobs != 0)
        pthread_cond_wait(&reader->idle, &reader->mutex);
    failed = reader->any_failure;
    pthread_mutex_unlock(&reader->mutex);
    return failed ? -1 : 0;
}

lt_io_stats_t lt_io_reader_stats(lt_io_reader_t *reader) {
    lt_io_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    if (!reader) return stats;
    pthread_mutex_lock(&reader->mutex);
    stats = reader->stats;
    pthread_mutex_unlock(&reader->mutex);
    return stats;
}
