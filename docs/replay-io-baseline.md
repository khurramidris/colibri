# Replay storage-I/O baseline

The fixed-token `REPLAY=1` path now prints one machine-readable `REPLAY_IO v1` record for the same profiler-free teacher-forced decode window used by the throughput line.

Example shape:

```text
REPLAY decode: 16 tokens in 42.000s | 0.38 tok/s | expert hit 25.0%
REPLAY_IO v1 available=1 scope=teacher_forced_decode logical_expert_bytes=123 process_read_bytes=456 process_rchar=789 process_read_syscalls=10 physical_to_logical=3.707317 logical_bytes_per_token=7.688 physical_bytes_per_token=28.500
```

The numbers above are illustrative only; they are not a Colibrì or GLM-5.2 measurement.

## Definitions

- `logical_expert_bytes`: expert weight and scale bytes requested by Colibrì during the measured decode loop. This is the engine-side demand counter, independent of profiler timing.
- `process_read_bytes`: Linux `/proc/self/io` `read_bytes` delta. It is storage bytes attributed by the kernel to this process during the same window.
- `process_rchar`: Linux `/proc/self/io` `rchar` delta. It counts bytes returned by read-like system calls, including page-cache hits, and is not a physical-storage metric.
- `process_read_syscalls`: Linux `/proc/self/io` `syscr` delta.
- `physical_to_logical`: `process_read_bytes / logical_expert_bytes`. Values above 1 indicate storage read amplification relative to requested expert payload. Values below 1 are possible when the page cache or resident tiers satisfy part of the logical demand.
- `*_bytes_per_token`: the corresponding decode-window byte count divided by replayed tokens.

On non-Linux systems, or when `/proc/self/io` cannot be read, the record is emitted with `available=0`. The engine never substitutes an estimate for a missing kernel counter.

## Measurement boundary

The first replay step remains a warm-up and is excluded. Counter snapshots bracket only the timed teacher-forced continuation loop. The `/proc/self/io` reads themselves occur outside the throughput timer. `read_bytes` is process-attributed accounting, not block-device telemetry; unrelated reads performed by threads in the same process during the window are included.

## Baseline protocol

For a meaningful cold-storage baseline:

1. Record the engine commit, model/conversion fingerprint, kernel, filesystem, storage devices, mount options, RAM, thread count, cache capacity, and every environment variable.
2. Disable MTP, lossy routing changes, historical pin files, prefetch experiments, and GPU tiers unless they are the variable under test.
3. Use the same fixed prompt and continuation for every comparison.
4. Declare how the cold state was established. Do not claim a cold run merely because the Colibrì LRU cache started empty; the operating-system page cache is a separate tier.
5. Run enough repetitions to report the median and dispersion. Preserve the complete stdout, oracle artifact, and resource/evidence report.
6. Reject a performance result if routing, logits, token continuation, or other preregistered correctness gates changed.

This record establishes the baseline needed to distinguish a real reduction in storage traffic from a faster-looking run caused by warm cache state or altered semantics.
