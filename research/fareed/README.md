# Fareed-inspired compressed trunk streaming

## Research question

Can Colibri retain its existing quantized numerics while lowering the minimum
RAM requirement by keeping only part of the dense/always-active trunk resident
and streaming the remaining layers from NVMe just before execution?

This transfers the systems lesson from `kimi-k3-in-c`: the trunk memory floor
can become a user-controlled dial. It does not copy Fareed's engine and does not
change Colibri's expert representation or quantization.

## Why this is the first target

Colibri already streams routed experts efficiently, but its non-expert trunk is
normally loaded into RAM. For Kimi K3, Colibri documents roughly 35 GiB for the
4-bit resident trunk and roughly 57 GiB for the 8-bit resident trunk. Fareed's
engine shows that a layer-addressable trunk can be partially pinned and the
remainder streamed while preserving identical output across memory budgets.

The proposed combination is therefore:

- Colibri's smaller, existing quantized trunk representation;
- Fareed-style layer-addressable packing and explicit residency control;
- sequential/direct reads, double buffering and next-layer prefetch;
- byte-identical execution relative to resident Colibri.

## Storage model

Let:

- `D` be the compressed trunk size;
- `R` be usable system RAM;
- `H` be RAM reserved for runtime state, KV/KDA state and expert slots;
- `P = max(0, R - H)` be the trunk bytes that can remain resident.

Then the minimum trunk traffic per decoded token is:

```
B_trunk = max(0, D - P)
```

This is an exact lower bound for a sequential transformer when each evicted
layer's trunk weights are used once per token and are not recomputed.

## Initial K3 planning point

Using only documented sizes, not a benchmark result:

- compressed Colibri trunk `D`: about 35 GiB at 4-bit;
- usable RAM `R`: 25 GiB;
- protected non-trunk budget `H`: 7 GiB;
- resident trunk `P`: 18 GiB;
- streamed trunk: about 17 GiB/token.

Fareed's low-memory exact path moves roughly 101 GiB of trunk plus roughly
24 GiB of experts per token. A compressed-streaming Colibri planning model is
therefore roughly 17 GiB of trunk plus the same order of expert traffic. This
suggests around 3x lower total storage traffic than exact BF16-style low-memory
streaming, before overlap and kernel effects. It does not imply a 3x end-to-end
speedup.

## Required implementation properties

1. A packed trunk index with one contiguous range per layer.
2. A deterministic residency planner under an explicit RAM budget.
3. Two aligned buffers for current-layer execution and next-layer prefetch.
4. `O_DIRECT` or equivalent explicit I/O when beneficial, with a buffered
   fallback.
5. No hidden expansion of streamed quantized weights into a larger persistent
   representation.
6. Physical read accounting, not logical-request accounting.
7. Resident and streamed modes must execute the same quantized bytes.

## Validation ladder

### Gate 0: accounting simulation

- enumerate every non-expert tensor by layer and physical bytes;
- predict resident and streamed bytes for each RAM budget;
- reject any plan that exceeds the declared RSS budget.

### Gate 1: synthetic layer file

- pack deterministic tensors into a layer-addressable file;
- compare resident and streamed matmul outputs bit-for-bit;
- verify read offsets, alignment, short-read handling and buffer reuse.

### Gate 2: truncated real model

- run several real layers in resident and streamed modes;
- require byte-identical hidden-state traces;
- record physical bytes, stall time and peak RSS.

### Gate 3: full model

- same generated tokens and logits as resident Colibri;
- repeated decode measurements under fixed RAM and cache budgets;
- compare against resident Colibri where it fits, OS swapping, and Fareed's
  exact low-memory engine on matched hardware where possible.

## Pass gates

- peak RSS within the declared budget, initially 25 GiB;
- streamed and resident Colibri outputs are bit-identical;
- at least 2x lower trunk traffic than exact low-memory streaming;
- at least 1.5x end-to-end speedup versus the strongest exact low-memory
  baseline on the same hardware;
- no uncounted page-cache, startup, repacking or decompression costs.

## Kill or narrow conditions

- the compressed trunk cannot be made layer-addressable without persistent
  expansion;
- I/O cannot overlap enough to beat exact low-memory streaming by 1.5x;
- non-trunk state leaves too little RAM for a useful resident prefix;
- the approach helps only one model because other MoEs have negligible trunks;
- performance is better only because the OS page cache violates the budget.

## Scope

This experiment does not yet attempt new quantization, progressive experts,
speculative decoding or approximate routing. The sole question is whether
explicitly streaming Colibri's existing compressed trunk creates a new and
useful memory-performance point.