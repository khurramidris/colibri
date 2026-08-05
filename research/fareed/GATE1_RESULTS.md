# Gate 1 — synthetic compressed-trunk streamer

Status: **PASS for mechanics; no real-model performance claim.**

## What was implemented

- a single binary trunk file with a fixed layer index;
- 4096-byte-aligned layer payloads and aligned direct-I/O lengths;
- exact payload lengths and CRC32 integrity checks;
- a declared resident-byte budget;
- deterministic prefix residency, matching Fareed's memory-depth dial;
- buffered `pread` for every non-resident payload in the Python oracle;
- one-layer asynchronous look-ahead with at most two streaming buffers;
- separate payload-byte, aligned-I/O-byte and startup accounting;
- a tiny per-row-int8 compressed matrix payload consumed directly from the
  same bytes in resident and streamed modes.

## Differential result

Local command:

```text
python3 research/fareed/synthetic_trunk.py \
  --layers 24 --width 48 --resident-layers 7 --passes 5
```

Observed:

```text
bit-identical passes             5 / 5
resident payload bytes           17,556
streamed payload bytes/pass      42,636
aligned direct-I/O bytes/pass    69,632
startup payload bytes            17,556
payload bytes requested total    213,180
expected payload bytes total     213,180
modeled O_DIRECT bytes total      348,160
buffered pread calls             85
modeled working bytes            25,748
```

The five streamed executions produced byte-identical final states to the fully
resident execution. Every non-resident payload was requested exactly once per
pass; resident layers were read once at startup and not reread during
execution.

The Python oracle uses buffered `pread`. Therefore `payload bytes requested`
is not claimed to equal storage-device physical traffic: the kernel page cache
may satisfy some reads. The container separately records an aligned
`io_length` for every layer, so a native O_DIRECT reader can request exactly
the modeled aligned byte count and measure it without conflating payload size
with physical I/O size.

## Test result

Fifteen deterministic tests passed locally across the synthetic reader and K3
manifest tools:

- index, padding and payload round trip;
- offset and direct-I/O-length alignment;
- whole-layer residency under a strict byte budget;
- resident/streamed bit identity;
- exact payload-request accounting across repeated passes;
- zero-resident and fully-resident extremes;
- bounded aligned double-buffer working-set accounting;
- corrupted payload rejection;
- malformed header rejection;
- closed-reader safety;
- K3 non-expert layer-span identification;
- exact byte-copy packing without re-encoding;
- routed-expert interleaving rejection;
- multiple-layer shard rejection;
- noncontiguous-layer rejection.

The branch CI repeats the tests and differential run on every change.

## What this proves

The core Fareed-to-Colibri mechanism is implementable:

1. existing compressed layer bytes can be packed once;
2. any whole-layer prefix can stay resident under a declared budget;
3. the remainder can be streamed in deterministic order;
4. streaming does not require changing the numerical representation;
5. requested payload bytes and aligned direct-I/O bytes can be accounted
   separately and exactly;
6. Colibri's K3 repacker layout can be audited for one contiguous non-expert
   span per layer without loading tensor data.

## What this does not prove

- a real 94-shard K3 repack has not yet been scanned in this branch;
- the K3 C engine is not yet wired to the packed trunk;
- native O_DIRECT buffers and C prefetch threads are not yet integrated;
- Python thread overlap is not a performance measurement;
- real hidden states, logits and tokens have not been compared;
- peak RSS has not been measured from a real engine process;
- no end-to-end speedup claim is justified yet.

## Decision

Proceed to Gate 2B: implement the native C reader against the same indexed
format, issue aligned O_DIRECT reads into two reusable buffers, and execute the
synthetic compressed matrices through resident and streamed paths. After that,
run the manifest tool on actual Colibri-repacked K3 shards and wire a truncated
real layer stack.
