#define _GNU_SOURCE
#include "trunk_stream.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef O_DIRECT
#define O_DIRECT 0
#endif

#define CERNO_HEADER_BYTES 64u
#define CERNO_ENTRY_BYTES 32u
#define CERNO_MAGIC "CERNTRK1"
#define CERNO_VERSION 1u

typedef struct {
    uint32_t layer_id;
    uint64_t offset;
    uint64_t length;
    uint32_t crc32;
    uint32_t io_length;
    unsigned char *resident;
} trunk_entry;

struct cerno_trunk_reader {
    int meta_fd;
    int data_fd;
    int direct_io_active;
    uint64_t alignment;
    uint32_t n_layers;
    trunk_entry *entries;
    uint64_t resident_budget_bytes;
    uint64_t resident_payload_bytes;
    uint32_t resident_layers;
    uint64_t streamed_payload_bytes_per_pass;
    uint64_t streamed_io_bytes_per_pass;
    uint64_t startup_payload_bytes;
    uint64_t io_bytes_requested;
    uint64_t payload_bytes_consumed;
    uint64_t read_calls;
    uint64_t resident_hits;
    uint64_t streamed_hits;
    uint64_t modeled_peak_working_bytes;
    size_t buffer_bytes;
    unsigned char *buffers[2];
    char error[256];
};

typedef struct {
    cerno_trunk_reader *reader;
    const trunk_entry *entry;
    unsigned char *buffer;
    int rc;
} prefetch_job;

static uint32_t rd32(const unsigned char *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const unsigned char *p) {
    return ((uint64_t)rd32(p)) | ((uint64_t)rd32(p + 4) << 32);
}

static int is_power_of_two_u64(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static void set_error(cerno_trunk_reader *reader, const char *fmt, ...) {
    if (!reader) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(reader->error, sizeof(reader->error), fmt, args);
    va_end(args);
}

static uint32_t crc32_ieee(const unsigned char *data, size_t size) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

static int pread_exact(
    int fd,
    void *buffer,
    size_t bytes,
    uint64_t offset,
    uint64_t *calls
) {
    unsigned char *dst = (unsigned char *)buffer;
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, dst + done, bytes - done, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        if (n == 0) {
            errno = EIO;
            return -1;
        }
        done += (size_t)n;
        if (calls) {
            *calls += 1;
        }
    }
    return 0;
}

static int read_streamed_entry(
    cerno_trunk_reader *reader,
    const trunk_entry *entry,
    unsigned char *buffer
) {
    const size_t request_bytes = reader->direct_io_active
        ? (size_t)entry->io_length
        : (size_t)entry->length;
    uint64_t calls = 0;
    if (pread_exact(
            reader->data_fd,
            buffer,
            request_bytes,
            entry->offset,
            &calls
        ) != 0) {
        set_error(
            reader,
            "pread failed for layer %u at offset %llu: %s",
            entry->layer_id,
            (unsigned long long)entry->offset,
            strerror(errno)
        );
        return -1;
    }
    if (crc32_ieee(buffer, (size_t)entry->length) != entry->crc32) {
        set_error(reader, "CRC mismatch for layer %u", entry->layer_id);
        return -1;
    }
    reader->read_calls += calls;
    reader->io_bytes_requested += request_bytes;
    reader->payload_bytes_consumed += entry->length;
    return 0;
}

static void *prefetch_main(void *arg) {
    prefetch_job *job = (prefetch_job *)arg;
    job->rc = read_streamed_entry(job->reader, job->entry, job->buffer);
    return NULL;
}

static void cleanup_reader(cerno_trunk_reader *reader) {
    if (!reader) {
        return;
    }
    if (reader->meta_fd >= 0) {
        close(reader->meta_fd);
    }
    if (reader->data_fd >= 0 && reader->data_fd != reader->meta_fd) {
        close(reader->data_fd);
    }
    if (reader->entries) {
        for (uint32_t i = 0; i < reader->n_layers; ++i) {
            free(reader->entries[i].resident);
        }
    }
    free(reader->entries);
    free(reader->buffers[0]);
    free(reader->buffers[1]);
    free(reader);
}

int cerno_trunk_open(
    cerno_trunk_reader **out,
    const char *path,
    uint64_t resident_budget_bytes,
    int direct_mode
) {
    if (!out || !path || direct_mode < 0 || direct_mode > 2) {
        errno = EINVAL;
        return -1;
    }
    *out = NULL;

    cerno_trunk_reader *reader = calloc(1, sizeof(*reader));
    if (!reader) {
        return -1;
    }
    reader->meta_fd = -1;
    reader->data_fd = -1;
    reader->resident_budget_bytes = resident_budget_bytes;
    reader->error[0] = '\0';

    reader->meta_fd = open(path, O_RDONLY);
    if (reader->meta_fd < 0) {
        set_error(reader, "open failed: %s", strerror(errno));
        cleanup_reader(reader);
        return -1;
    }

    unsigned char header[CERNO_HEADER_BYTES];
    if (pread_exact(reader->meta_fd, header, sizeof(header), 0, NULL) != 0) {
        set_error(reader, "header read failed: %s", strerror(errno));
        cleanup_reader(reader);
        return -1;
    }
    if (memcmp(header, CERNO_MAGIC, 8) != 0) {
        set_error(reader, "bad trunk magic");
        cleanup_reader(reader);
        errno = EINVAL;
        return -1;
    }
    const uint32_t version = rd32(header + 8);
    const uint32_t n_layers = rd32(header + 12);
    const uint64_t index_offset = rd64(header + 16);
    const uint64_t alignment = rd64(header + 24);
    if (
        version != CERNO_VERSION ||
        n_layers == 0 ||
        index_offset != CERNO_HEADER_BYTES ||
        !is_power_of_two_u64(alignment) ||
        alignment < sizeof(void *)
    ) {
        set_error(reader, "unsupported or malformed trunk header");
        cleanup_reader(reader);
        errno = EINVAL;
        return -1;
    }
    reader->alignment = alignment;
    reader->n_layers = n_layers;
    reader->entries = calloc(n_layers, sizeof(*reader->entries));
    if (!reader->entries) {
        cleanup_reader(reader);
        return -1;
    }

    struct stat st;
    if (fstat(reader->meta_fd, &st) != 0) {
        set_error(reader, "fstat failed: %s", strerror(errno));
        cleanup_reader(reader);
        return -1;
    }

    uint64_t previous_end = 0;
    unsigned char raw[CERNO_ENTRY_BYTES];
    for (uint32_t i = 0; i < n_layers; ++i) {
        const uint64_t entry_offset = index_offset + (uint64_t)i * CERNO_ENTRY_BYTES;
        if (pread_exact(reader->meta_fd, raw, sizeof(raw), entry_offset, NULL) != 0) {
            set_error(reader, "index read failed at layer %u", i);
            cleanup_reader(reader);
            return -1;
        }
        trunk_entry *entry = &reader->entries[i];
        entry->layer_id = rd32(raw + 0);
        const uint32_t flags = rd32(raw + 4);
        entry->offset = rd64(raw + 8);
        entry->length = rd64(raw + 16);
        entry->crc32 = rd32(raw + 24);
        entry->io_length = rd32(raw + 28);
        if (
            entry->layer_id != i ||
            flags != 0 ||
            entry->length == 0 ||
            entry->io_length < entry->length ||
            entry->offset % alignment != 0 ||
            entry->io_length % alignment != 0 ||
            entry->offset < previous_end ||
            entry->offset + entry->io_length > (uint64_t)st.st_size
        ) {
            set_error(reader, "invalid index entry for layer %u", i);
            cleanup_reader(reader);
            errno = EINVAL;
            return -1;
        }
        previous_end = entry->offset + entry->io_length;
    }

    int direct_fd = -1;
    if (direct_mode != 0 && O_DIRECT != 0) {
        direct_fd = open(path, O_RDONLY | O_DIRECT);
    }
    if (direct_mode == 1 && direct_fd < 0) {
        set_error(reader, "O_DIRECT open failed: %s", strerror(errno));
        cleanup_reader(reader);
        return -1;
    }
    if (direct_fd >= 0) {
        reader->data_fd = direct_fd;
        reader->direct_io_active = 1;
    } else {
        reader->data_fd = reader->meta_fd;
        reader->direct_io_active = 0;
    }

    uint64_t used = 0;
    for (uint32_t i = 0; i < n_layers; ++i) {
        trunk_entry *entry = &reader->entries[i];
        if (used + entry->length > resident_budget_bytes) {
            break;
        }
        entry->resident = malloc((size_t)entry->length);
        if (!entry->resident) {
            cleanup_reader(reader);
            return -1;
        }
        if (pread_exact(
                reader->meta_fd,
                entry->resident,
                (size_t)entry->length,
                entry->offset,
                NULL
            ) != 0) {
            set_error(reader, "resident read failed for layer %u", i);
            cleanup_reader(reader);
            return -1;
        }
        if (crc32_ieee(entry->resident, (size_t)entry->length) != entry->crc32) {
            set_error(reader, "resident CRC mismatch for layer %u", i);
            cleanup_reader(reader);
            errno = EIO;
            return -1;
        }
        used += entry->length;
        reader->startup_payload_bytes += entry->length;
        reader->resident_layers += 1;
    }
    reader->resident_payload_bytes = used;

    uint64_t max_stream_io = 0;
    for (uint32_t i = reader->resident_layers; i < n_layers; ++i) {
        trunk_entry *entry = &reader->entries[i];
        reader->streamed_payload_bytes_per_pass += entry->length;
        reader->streamed_io_bytes_per_pass += reader->direct_io_active
            ? entry->io_length
            : entry->length;
        if (entry->io_length > max_stream_io) {
            max_stream_io = entry->io_length;
        }
    }
    reader->buffer_bytes = (size_t)max_stream_io;
    if (reader->buffer_bytes > 0) {
        for (int i = 0; i < 2; ++i) {
            if (posix_memalign(
                    (void **)&reader->buffers[i],
                    (size_t)alignment,
                    reader->buffer_bytes
                ) != 0) {
                set_error(reader, "aligned buffer allocation failed");
                cleanup_reader(reader);
                errno = ENOMEM;
                return -1;
            }
        }
    }
    reader->modeled_peak_working_bytes =
        reader->resident_payload_bytes + 2u * (uint64_t)reader->buffer_bytes;

    *out = reader;
    return 0;
}

void cerno_trunk_close(cerno_trunk_reader *reader) {
    cleanup_reader(reader);
}

int cerno_trunk_for_each(
    cerno_trunk_reader *reader,
    cerno_trunk_callback callback,
    void *context
) {
    if (!reader || !callback) {
        errno = EINVAL;
        return -1;
    }

    for (uint32_t i = 0; i < reader->resident_layers; ++i) {
        trunk_entry *entry = &reader->entries[i];
        reader->resident_hits += 1;
        if (callback(entry->layer_id, entry->resident, (size_t)entry->length, context) != 0) {
            set_error(reader, "callback failed for resident layer %u", i);
            return -1;
        }
    }

    if (reader->resident_layers == reader->n_layers) {
        return 0;
    }

    uint32_t current = reader->resident_layers;
    unsigned current_buffer = 0;
    if (read_streamed_entry(
            reader,
            &reader->entries[current],
            reader->buffers[current_buffer]
        ) != 0) {
        return -1;
    }
    reader->streamed_hits += 1;

    while (current < reader->n_layers) {
        const uint32_t next = current + 1;
        pthread_t thread;
        prefetch_job job = {0};
        int thread_started = 0;

        if (next < reader->n_layers) {
            job.reader = reader;
            job.entry = &reader->entries[next];
            job.buffer = reader->buffers[current_buffer ^ 1u];
            job.rc = -1;
            if (pthread_create(&thread, NULL, prefetch_main, &job) != 0) {
                set_error(reader, "pthread_create failed at layer %u", next);
                return -1;
            }
            thread_started = 1;
        }

        trunk_entry *entry = &reader->entries[current];
        if (callback(
                entry->layer_id,
                reader->buffers[current_buffer],
                (size_t)entry->length,
                context
            ) != 0) {
            set_error(reader, "callback failed for streamed layer %u", current);
            if (thread_started) {
                pthread_join(thread, NULL);
            }
            return -1;
        }

        if (thread_started) {
            if (pthread_join(thread, NULL) != 0 || job.rc != 0) {
                if (job.rc == 0) {
                    set_error(reader, "pthread_join failed at layer %u", next);
                }
                return -1;
            }
            reader->streamed_hits += 1;
            current_buffer ^= 1u;
        }
        current = next;
    }
    return 0;
}

void cerno_trunk_get_stats(
    const cerno_trunk_reader *reader,
    cerno_trunk_stats *out
) {
    if (!reader || !out) {
        return;
    }
    out->resident_budget_bytes = reader->resident_budget_bytes;
    out->resident_payload_bytes = reader->resident_payload_bytes;
    out->streamed_payload_bytes_per_pass = reader->streamed_payload_bytes_per_pass;
    out->streamed_io_bytes_per_pass = reader->streamed_io_bytes_per_pass;
    out->startup_payload_bytes = reader->startup_payload_bytes;
    out->io_bytes_requested = reader->io_bytes_requested;
    out->payload_bytes_consumed = reader->payload_bytes_consumed;
    out->read_calls = reader->read_calls;
    out->resident_hits = reader->resident_hits;
    out->streamed_hits = reader->streamed_hits;
    out->modeled_peak_working_bytes = reader->modeled_peak_working_bytes;
    out->resident_layers = reader->resident_layers;
    out->total_layers = reader->n_layers;
    out->direct_io_active = reader->direct_io_active;
}

uint64_t cerno_trunk_prefix_payload_bytes(
    const cerno_trunk_reader *reader,
    uint32_t layer_count
) {
    if (!reader) {
        return 0;
    }
    if (layer_count > reader->n_layers) {
        layer_count = reader->n_layers;
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < layer_count; ++i) {
        total += reader->entries[i].length;
    }
    return total;
}

const char *cerno_trunk_error(const cerno_trunk_reader *reader) {
    if (!reader) {
        return "no reader";
    }
    return reader->error[0] ? reader->error : "no error";
}
