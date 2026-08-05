# Gate 1 — synthetic compressed-trunk streamer

Status: **PASS for mechanics; no real-model performance claim.**

## What was implemented

- a single binary trunk file with a fixed layer index;
- 4096-byte-aligned layer payloads;
- exact layer lengths and CRC32 integrity checks;
- a declared resident-byte budget;
- deterministic prefix residency, matching Fareed's memory-depth dial;
- `pread` for every non-resident layer;
- one-layer asynchronous look-ahead with at most two streaming buffers;
- separate startup-read and per-pass physical-read accounting;
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
bit-identical passes       5 / 5
resident bytes             17,556
streamed bytes per pass    42,636
startup read bytes         17,556
per-pass read bytes total  213,180
expected read bytes total  213,180
physical read calls        85
modeled working bytes      22,572
```

The five streamed executions produced byte-identical final states to the fully
resident execution. Every non-resident layer was read exactly once per pass;
resident layers were read once at startup and not reread during execution.

## Test result

Ten deterministic tests passed locally:

- index and payload round trip;
- alignment;
- whole-layer residency under a strict byte budget;
- resident/streamed bit identity;
- exact prefetch/read accounting across repeated passes;
- zero-resident and fully-resident extremes;
- bounded double-buffer working-set accounting;
- corrupted payload rejection;
- malformed header rejection;
- closed-reader safety.

The branch CI repeats the tests and differential run on every change.

## What this proves

The core Fareed-to-Colibri mechanism is implementable:

1. existing compressed layer bytes can be packed once;
2. any whole-layer prefix can stay resident under a declared budget;
3. the remainder can be streamed in deterministic order;
4. streaming does not require changing the numerical representation;
5. physical bytes can be accounted exactly rather than inferred from logical
   tensor requests.

## What this does not prove

- actual K3 or GLM layer layouts are not integrated;
- `O_DIRECT`, aligned native buffers and a C kernel are not implemented;
- Python thread overlap is not a performance measurement;
- OS page-cache effects are not controlled;
- real hidden states, logits and tokens have not been compared;
- peak RSS has not been measured from a real process;
- no end-to-end speedup claim is justified yet.

## Decision

Proceed to Gate 2A: build a real Colibri trunk manifest from existing repacked
containers, identify every non-expert tensor belonging to each layer, and prove
that each layer can be reconstructed from one indexed byte range without
persistent expansion. Only then implement the native C double-buffer reader.
