# GLM-5.2 working-set feasibility contract

This note freezes the first-principles model, formulas, correctness invariants and experiment order for testing whether GLM-5.2 can execute with its routed expert weights treated as pageable state.

It is an **analysis and test contract**, not evidence that this branch has already reproduced the full 744B run. Real-checkpoint evidence must be emitted by the native engine and Lattice qualification workflow.

## Evidence labels

- **Model fact** — read from the official GLM-5.2 configuration or implementation.
- **Derived estimate** — arithmetic from model facts; the actual converted container must be measured.
- **Hypothesis** — prediction made before an experiment.
- **Measurement** — accepted only from a preserved run record containing model, engine, machine and command identity.

Primary architecture references:

- `zai-org/GLM-5.2/config.json`
- Hugging Face `modeling_glm_moe_dsa.py`
- Hugging Face GLM-MoE-DSA model documentation

The upstream Colibrì README is useful prior work, but its benchmark numbers remain external claims until reproduced on an identified machine and checkpoint.

## 1. Architecture decomposition

Official configuration values used here:

| Quantity | Value |
|---|---:|
| Hidden width | 6,144 |
| Decoder layers | 78 |
| Initial dense MLP layers | 3 |
| Sparse MoE layers in the main decoder | 75 |
| Routed experts per sparse layer | 256 |
| Routed experts selected per token/layer | 8 |
| Shared experts per sparse layer | 1 |
| Routed/shared expert intermediate width | 2,048 |
| Dense MLP intermediate width | 12,288 |
| Router computation | FP32 linear, sigmoid, correction bias, top-k, normalization, scale 2.5 |
| MTP layers | 1 auxiliary next-token-prediction layer |
| MLA compressed KV width | 512 latent + 64 RoPE = 576 elements/layer/token |
| DSA selected history positions | up to 2,048 |

### Universally required on every main-model token

These weights are not placement-optional under the default semantics:

1. token embedding and output projection;
2. all 78 attention blocks, norms and DSA/IndexShare machinery;
3. the first three dense MLPs;
4. all 75 router projections and FP32 correction biases;
5. all 75 shared experts;
6. tokenizer and chat-template behavior;
7. active conversation state, including the architecture-native KV/indexer state.

The exact resident byte count must be obtained from the converted checkpoint manifest. Nominal parameter arithmetic is not an acceptable substitute because quantization, tied or untied tensors, scale tensors, alignment and container headers alter the result.

### Conditionally required

The main decoder has:

```text
75 sparse layers × 256 routed experts = 19,200 routed expert instances
```

Each token invokes:

```text
75 sparse layers × top-8 = 600 routed expert calls
```

The auxiliary MTP layer is separate. If its architecture mirrors one sparse layer, it adds another 256 stored routed expert instances and eight routed calls per draft position. That explains a 19,456-expert storage inventory when main-model and MTP experts are counted together. The converter must prove this from tensor names rather than infer it.

MTP may accelerate verified generation, but it is not required for the first correct main-model token. It must never be silently folded into the main-model oracle because the standard Hugging Face GLM-MoE-DSA implementation explicitly omits the MTP layer.

## 2. Expert arithmetic

The official implementation stores a fused gate/up tensor and a down tensor. The equivalent scalar-weight count per expert is:

```text
2 × 6,144 × 2,048 + 6,144 × 2,048
= 37,748,736 weights
```

### Bytes per expert

| Representation | Derived payload per expert | Notes |
|---|---:|---|
| BF16/FP16 | 72.000 MiB | no scale overhead |
| FP8/INT8 raw | 36.000 MiB | scale metadata omitted |
| INT4 raw | 18.000 MiB | scale metadata omitted |
| group-64 INT4 + one FP16 scale/block | 19.125 MiB | analytical approximation; measure the real container |

For group-64 INT4, the scale overhead is:

```text
37,748,736 / 64 blocks × 2 bytes = 1.125 MiB
```

### Active MLP weights per main-model token

Routed experts:

```text
600 × 37,748,736 = 22,649,241,600 weights
```

Shared experts:

```text
75 × 37,748,736 = 2,831,155,200 weights
```

First three dense MLPs:

```text
3 × (3 × 6,144 × 12,288) = 679,477,248 weights
```

The derived active MLP total is therefore about 26.16B weights/token. The official model documentation reports approximately 40B active parameters/token; the remainder is attention, embeddings/output, routing, normalization, indexing and other always-active tensors. The manifest and execution trace remain authoritative.

## 3. Cold storage traffic

At the group-64 INT4 estimate, a zero-hit main-model decode token requires:

```text
600 × 19.125 MiB = 11,475 MiB
                    = 11.206 GiB
                    = 12.032 GB
```

Let:

- `C` = 12.032 GB of cold routed payload/token;
- `h` = useful expert hit rate, from 0 to 1;
- `A` = physical read amplification, at least 1.0;
- `B` = measured cold physical storage bandwidth in GB/s;
- `T` = storage-limited tokens/s.

Then:

```text
physical_bytes_per_token = C × (1 - h) × A
T <= B / physical_bytes_per_token
```

Prefetch can hide foreground wait but does not reduce physical bytes unless the prefetched expert is later reused. A prefetched expert that is evicted before use increases `A`.

### Idealized storage ceilings at A = 1.0

| Cold physical bandwidth | 0% hit | 50% hit | 75% hit | 90% hit |
|---:|---:|---:|---:|---:|
| 1 GB/s | 0.083 tok/s | 0.166 | 0.332 | 0.831 |
| 3 GB/s | 0.249 tok/s | 0.499 | 0.997 | 2.493 |
| 5 GB/s | 0.416 tok/s | 0.831 | 1.662 | 4.155 |
| 7 GB/s | 0.582 tok/s | 1.164 | 2.327 | 5.818 |

These are ceilings, not forecasts. Small random reads, queue-depth limits, filesystem behavior, decompression/dequantization, CPU contention and imperfect overlap all lower realized throughput.

## 4. RAM and compute ceilings

Storage is only one roofline.

If the model performs approximately 40B active weight uses per token, a four-bit path consumes roughly 20 GB/token before scale traffic and activation writes. Let `R` be measured sustainable RAM bandwidth available to the engine:

```text
T_ram <= R / active_weight_bytes_per_token
```

Let `M` be measured useful quantized matrix-vector multiply throughput in weight-MAC/s:

```text
T_compute <= M / active_weight_uses_per_token
```

The planner must measure both. Peak ISA FLOPs and vendor memory specifications are not evidence of sustainable decode throughput.

The useful end-to-end ceiling is bounded by the slowest non-overlapped stage:

```text
T <= min(T_storage, T_ram, T_compute, T_attention, T_transfer, ...)
```

## 5. Conversation state

The architecture produces a 512-element KV latent and a 64-element RoPE key component before expansion. An exact architecture-native compressed cache therefore needs 576 elements/layer/token, plus indexer state and metadata.

At two bytes/element across 78 layers:

```text
576 × 78 × 2 = 89,856 bytes/token
```

| Context | Compressed KV payload, excluding metadata/indexer state |
|---:|---:|
| 8,192 tokens | 0.686 GiB |
| 32,768 tokens | 2.742 GiB |
| 131,072 tokens | 10.969 GiB |
| 1,000,000 tokens | 83.685 GiB |

Therefore, a 25 GB machine can prove short-context execution but cannot keep a one-million-token conversation fully resident at FP16 compressed-KV precision. Long-context support requires a separate state hierarchy, lower-precision policy validated for quality, recomputation, or a declared context limit. It must not be confused with pageable weights.

The standard Transformers implementation expands key/value tensors before updating its generic cache. A native compressed cache is permitted only if its attention outputs match the trusted implementation within the declared tolerance.

## 6. Minimum semantic invariant

Placement may change latency. It must not change the function computed by default.

The invariant includes:

1. exact tensor decoding, shapes, ordering, scales, zero-points and alignment;
2. FP32 router projection and correction-bias interpretation;
3. sigmoid scoring, group masking, top-8 selection, probability normalization and routed scale 2.5;
4. selected expert identities and their output weights;
5. shared-expert addition;
6. layer order, residual paths, normalization and activation;
7. MLA/DSA/IndexShare attention semantics and causality;
8. RoPE convention and position accounting;
9. tokenizer, special-token and chat-template behavior;
10. sampling configuration;
11. MTP draft/verification compatibility when speculation is enabled.

Cache eviction, tier assignment, read scheduling and prefetch are outside the mathematical invariant only when they deliver the same tensors before use.

## 7. Correctness ladder

No end-to-end performance result is promotable until the earlier gates pass.

1. **Container decoding** — compare representative tensors and quantization metadata with the converter source.
2. **Kernel fixtures** — scalar oracle versus every enabled quantized matvec path, including awkward tails and scale groups.
3. **Router fixture** — compare FP32 logits, corrected scores, ordered selected experts and routing weights at every sparse layer.
4. **Expert fixture** — compare individual expert outputs before mixture accumulation.
5. **One-layer fixture** — attention, shared expert, routed mixture and residual output.
6. **Attention-state fixture** — compare incremental decode with full-prefix recomputation and the trusted reference.
7. **Teacher-forced logits** — compare selected logits, top-k IDs, moments and deterministic projections for a fixed continuation.
8. **Full generation** — deterministic token sequence under fixed sampling, tokenizer and template.

Token agreement alone is insufficient near decision boundaries. Complete logit equality is also not assumed across different numerical kernels; tolerances must be calibrated against real supported backends and recorded.

## 8. Closed-loop experiment ledger

Only one major variable changes per A/B pair. Each run declares cold/warm state, repeats, median/p95, physical bytes, cache telemetry and correctness status.

| Experiment | Pre-registered break-even condition |
|---|---|
| Baseline demand load | Complete a correct forward pass and preserve teacher-forced evidence; speed is not a gate. |
| Per-layer LRU | Keep only if useful-token throughput rises and physical read amplification does not rise enough to erase the gain. |
| Persistent hot pinning | Win only when the pinned set's measured reuse saves more read time than the RAM it removes from the adaptive cache. |
| Recency/frequency hybrid | Win only if its hit-rate gain exceeds metadata/update overhead and avoids tail-latency regressions. |
| Live repinning | Change placement only when expected saved read time exceeds migration bytes plus hysteresis margin. |
| Router-ahead prefetch | `accuracy × exposed_read_latency_saved` must exceed extra physical bytes, queue contention and cache pollution. |
| Batch unioning | Win when duplicate `(layer, expert)` demand occurs across positions; one physical load must serve all consumers. |
| Adjacent expert layout | Win when fewer I/O requests and better service time outweigh padding and conversion cost. |
| Direct I/O | Win only on measured cold physical reads; OS page-cache bandwidth is reported separately. |
| io_uring/batched I/O | Win when queueing and syscall savings exceed submission/completion overhead at the engine's real read size. |
| Two SSDs | Win when requests are independently serviceable and aggregate physical bandwidth rises without duplicated reads. |
| Hot-shard mirror | Win when mirror hit probability and device parallelism exceed mirror maintenance and capacity cost. |
| NUMA placement | Win when reduced remote-memory traffic exceeds placement and scheduling overhead. |
| Optional GPU tier | Win only after PCIe transfer, launch, synchronization and residency are included in end-to-end useful-token time. |
| Native MTP | Win only on accepted tokens per total main+draft work, not draft speed alone. |
| Grammar-forced draft | Win only for compatible structured output and after verification cost and acceptance are counted. |

## 9. Required telemetry

Every accepted run record must expose:

- exact model and converted-format identity;
- engine revision and full command;
- prompt/reference identity;
- tokens/s and time to first token;
- resident RAM and VRAM;
- logical requested expert bytes and physical storage bytes;
- hit, miss, replacement and pin counts by tier;
- prefetch issued, useful, late, unused and evicted-before-use counts;
- read-service time and foreground-visible I/O wait separately;
- dense, routing, expert lookup, dequant/matvec, attention, transfer, kernel, synchronization and sampling time;
- speculation proposed/accepted/rejected and verification cost;
- correctness gate outcomes.

## 10. Present evidence boundary

This branch currently contains:

- the native Colibrì and OLMoE engines;
- a resource planner, doctor, autotuning and serving components;
- exact top-k and numerical replay fixtures;
- an OLMoE token-exact acceptance controller;
- a GLM numerical qualification controller.

The permanent CI validates dependency-minimal builds, native fixtures and cross-platform controller behavior. It does **not** download or execute the GLM-5.2 checkpoint.

A frontier-scale success claim requires a preserved real-checkpoint session showing:

1. resident load within the declared machine budget;
2. all 78 layers executed;
3. all 75 main sparse layers routed top-8 without semantic pruning;
4. demand-loaded selected experts;
5. coherent generation;
6. teacher-forced agreement under calibrated tolerances;
7. truthful RSS, storage bytes and stage timing;
8. cold and warm results clearly separated.

Until that artifact exists, the correct project status is **implementation and analytical feasibility established; frontier reproduction pending**.
