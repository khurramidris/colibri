# GLM-5.2 pageable execution path

This diagram describes placement only. It does not permit router pruning, expert substitution or attention approximation.

```text
IMMUTABLE BACKING STORE                         ADAPTIVE FAST TIERS

+----------------------------------+            +------------------------------+
| NVMe expert container            |            | RAM expert cache             |
|                                  |  demand /  | - immutable expert payloads  |
| layer 3, expert 0: gate/up/down  |  prefetch  | - LRU/frequency metadata     |
| layer 3, expert 1: gate/up/down  +----------->| - in-flight read slots       |
| ...                              |            +---------------+--------------+
| layer 77, expert 255             |                            |
| optional MTP expert inventory    |                            | optional promote
+----------------------------------+                            v
                                                   +---------------------------+
                                                   | VRAM / accelerator tier   |
                                                   | only when end-to-end win  |
                                                   +---------------------------+

ALWAYS-RESIDENT MAIN-MODEL STATE

+--------------------------------------------------------------------------+
| embeddings/output | attention + DSA/IndexShare | 3 dense MLPs            |
| 75 routers        | 75 shared experts          | norms + quant scales     |
| compressed KV/indexer conversation state       | tokenizer/template IDs   |
+--------------------------------------------------------------------------+
```

```text
ONE DECODE TOKEN

input token
    |
    v
embedding
    |
    v
for layer = 0..77
    |
    +--> attention / architecture-native state update
    |
    +--> dense layer? ---- yes ----> resident dense MLP ------------------+
    |                                                                    |
    +--> sparse layer?                                                   |
           |                                                             |
           v                                                             |
      FP32 router                                                        |
      sigmoid + correction bias + exact top-8                            |
           |                                                             |
           v                                                             |
      unite duplicate (layer, expert) requests for the active batch      |
           |                                                             |
           v                                                             |
      lookup VRAM -> RAM -> in-flight request -> NVMe                     |
           |                                                             |
           +--> hit: use immutable resident payload                       |
           |                                                             |
           +--> miss: positional read, decode scales, install cache slot  |
           |                                                             |
           v                                                             |
      execute all 8 selected experts + resident shared expert             |
           |                                                             |
           v                                                             |
      exact weighted accumulation + residual <----------------------------+
    |
    v
final norm -> output projection -> sampling
    |
    +--> optional compatible MTP draft + main-model verification
    |
    v
next token + telemetry + reproducibility record
```

## What placement may change

- where an immutable expert payload resides;
- when a correct payload is read;
- whether a future correct request is prefetched;
- scheduling, overlap, batching and cache replacement;
- which exact backend computes a supported kernel.

## What placement may not change by default

- router scores, ordering or top-k;
- selected expert identities;
- expert tensor values or scale interpretation;
- shared-expert contribution;
- model topology, residuals or normalization;
- attention, RoPE, DSA/IndexShare or KV semantics;
- tokenizer, chat template or sampling policy.
