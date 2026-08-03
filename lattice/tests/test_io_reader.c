#define _POSIX_C_SOURCE 200809L
#include "lattice/io_reader.h"

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        ++failures; \
    } \
} while (0)

typedef struct {
    pthread_mutex_t mutex;
    int callbacks;
    int failed_callbacks;
    uint64_t bytes;
} Completion;

static void completed(const lt_io_job_t *job, int status, uint64_t bytes_read, void *user) {
    Completion *completion = (Completion *)user;
    (void)job;
    pthread_mutex_lock(&completion->mutex);
    completion->callbacks++;
    completion->bytes += bytes_read;
    if (status != 0) completion->failed_callbacks++;
    pthread_mutex_unlock(&completion->mutex);
}

static void make_fixture(char path[256], unsigned char *data, size_t size) {
    int fd;
    size_t done = 0;
    snprintf(path, 256, "/tmp/lattice-io-%ld.bin", (long)getpid());
    for (done = 0; done < size; ++done) data[done] = (unsigned char)((done * 17u + 3u) & 0xffu);
    fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    CHECK(fd >= 0);
    if (fd < 0) return;
    done = 0;
    while (done < size) {
        ssize_t wrote = write(fd, data + done, size - done);
        CHECK(wrote > 0);
        if (wrote <= 0) break;
        done += (size_t)wrote;
    }
    CHECK(close(fd) == 0);
}

static void test_successful_reads(void) {
    unsigned char source[65536];
    unsigned char a[4096], b[8192], c[16384];
    char path[256];
    int fd;
    Completion completion;
    lt_io_reader_t *reader;
    lt_io_job_t jobs[3];
    lt_io_stats_t stats;
    memset(&completion, 0, sizeof(completion));
    pthread_mutex_init(&completion.mutex, NULL);
    make_fixture(path, source, sizeof(source));
    fd = open(path, O_RDONLY);
    CHECK(fd >= 0);
    reader = lt_io_reader_create(2, 8, completed, &completion);
    CHECK(reader != NULL);
    memset(jobs, 0, sizeof(jobs));
    jobs[0].tensor_id = 1; jobs[0].fd = fd; jobs[0].offset = 0; jobs[0].destination = a; jobs[0].bytes = sizeof(a);
    jobs[1].tensor_id = 2; jobs[1].fd = fd; jobs[1].offset = 7000; jobs[1].destination = b; jobs[1].bytes = sizeof(b);
    jobs[2].tensor_id = 3; jobs[2].fd = fd; jobs[2].offset = 30000; jobs[2].destination = c; jobs[2].bytes = sizeof(c);
    CHECK(lt_io_reader_submit(reader, &jobs[0]) == 1);
    CHECK(lt_io_reader_submit(reader, &jobs[1]) == 1);
    CHECK(lt_io_reader_submit(reader, &jobs[2]) == 1);
    CHECK(lt_io_reader_wait_idle(reader) == 0);
    CHECK(memcmp(a, source, sizeof(a)) == 0);
    CHECK(memcmp(b, source + 7000, sizeof(b)) == 0);
    CHECK(memcmp(c, source + 30000, sizeof(c)) == 0);
    pthread_mutex_lock(&completion.mutex);
    CHECK(completion.callbacks == 3);
    CHECK(completion.failed_callbacks == 0);
    CHECK(completion.bytes == sizeof(a) + sizeof(b) + sizeof(c));
    pthread_mutex_unlock(&completion.mutex);
    stats = lt_io_reader_stats(reader);
    CHECK(stats.submitted_jobs == 3);
    CHECK(stats.completed_jobs == 3);
    CHECK(stats.failed_jobs == 0);
    CHECK(stats.queued_jobs == 0 && stats.inflight_jobs == 0);
    lt_io_reader_destroy(reader);
    close(fd);
    unlink(path);
    pthread_mutex_destroy(&completion.mutex);
}

static void test_short_read_is_failure(void) {
    unsigned char source[1024], target[1024];
    char path[256];
    int fd;
    Completion completion;
    lt_io_reader_t *reader;
    lt_io_job_t job;
    lt_io_stats_t stats;
    memset(&completion, 0, sizeof(completion));
    pthread_mutex_init(&completion.mutex, NULL);
    make_fixture(path, source, sizeof(source));
    fd = open(path, O_RDONLY);
    reader = lt_io_reader_create(1, 2, completed, &completion);
    memset(&job, 0, sizeof(job));
    job.tensor_id = 9;
    job.fd = fd;
    job.offset = 700;
    job.destination = target;
    job.bytes = sizeof(target);
    CHECK(lt_io_reader_submit(reader, &job) == 1);
    CHECK(lt_io_reader_wait_idle(reader) == -1);
    stats = lt_io_reader_stats(reader);
    CHECK(stats.failed_jobs == 1);
    pthread_mutex_lock(&completion.mutex);
    CHECK(completion.callbacks == 1);
    CHECK(completion.failed_callbacks == 1);
    CHECK(completion.bytes == 324);
    pthread_mutex_unlock(&completion.mutex);
    lt_io_reader_destroy(reader);
    close(fd);
    unlink(path);
    pthread_mutex_destroy(&completion.mutex);
}

static void test_invalid_jobs(void) {
    lt_io_reader_t *reader = lt_io_reader_create(1, 1, NULL, NULL);
    lt_io_job_t job;
    memset(&job, 0, sizeof(job));
    CHECK(reader != NULL);
    CHECK(lt_io_reader_submit(reader, &job) == -1);
    lt_io_reader_destroy(reader);
}

int main(void) {
    test_successful_reads();
    test_short_read_is_failure();
    test_invalid_jobs();
    if (failures) {
        fprintf(stderr, "%d io reader test(s) failed\n", failures);
        return 1;
    }
    puts("io reader tests passed");
    return 0;
}
