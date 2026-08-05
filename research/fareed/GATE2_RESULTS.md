# Gates 2A–2C — native I/O, K3 manifest and real layer lifecycle

Status: **PASS for the pre-checkpoint integration gates; full-model runtime evidence remains outstanding.**

Commit under test: `7c5350cb00ddd117ace8b16092dde8cf9a37fa4b`

GitHub Actions run: `Fareed-inspired trunk streaming #28`, run id
`31048848708`, Ubuntu 24.04, completed successfully on 2026-08-05 UTC.

## Gate 2A — exact K3 trunk manifest

The manifest tool reads safetensors headers without loading tensor payloads and:

- recognizes real K3 layer tensor names with and without `language_model.`;
- excludes routed expert tensors;
- requires a single contiguous non-expert span for each layer;
- rejects expert interleaving, multiple layers in one layer shard, gaps and
  noncontiguous layer numbering;
- copies the existing compressed bytes into the indexed trunk without
  dequantization or re-encoding.

Five manifest tests passed.

Decision: **PASS for layout analysis.** A real 94-shard Colibri K3 repack still
has to be scanned before claiming that the production snapshot satisfies these
layout assumptions.

## Gate 2B — native buffered and O_DIRECT streaming

The native C reader was compiled with strict warnings and exercised against a
24-layer compressed fixture with seven resident layers and five repeated
passes.

Observed in CI:

```text
buffered: direct=0 resident=7/24 payload/pass=42636 io/pass=42636
          reads=85 peak=25748 bit-identical=5/5
direct-preferred: direct=1 resident=7/24 payload/pass=42636 io/pass=69632
                  reads=85 peak=25748 bit-identical=5/5
native trunk streaming tests: ok
```

This proves:

- the O_DIRECT path was genuinely active on the CI filesystem;
- aligned I/O bytes are counted separately from useful payload bytes;
- the reader holds a deterministic resident prefix and two reusable aligned
  buffers;
- resident, buffered-streamed and direct-streamed execution consume identical
  compressed payload bytes and produce byte-identical outputs;
- the next layer is read on a worker thread while the current layer callback
  executes.

Decision: **PASS for the native storage mechanism.** This is not a throughput
benchmark; the fixture is intentionally small and likely cache-resident.

## Gate 2C — genuine K3 layer eviction and reload

A deterministic miniature K3 snapshot was generated using Colibri's real:

- tensor names;
- repacked U8 plus `.qs` matrix format;
- `Layer`, `W` and loader structures;
- KDA recurrence and convolution state;
- AttnRes mixing;
- dense SiTU-GLU MLP;
- production forward kernels.

Configuration: three KDA/dense layers, hidden size 64, one injected input row,
no routed experts and no language-model head.

The same production code was run at three residency depths:

```text
resident 3/3 | transient loads 0
resident 1/3 | transient loads 2
resident 0/3 | transient loads 3
```

All three runs produced an exactly equal 768-byte hidden-state trace:

```text
3 layers x 64 float32 values x 4 bytes = 768 bytes
```

The workflow used `cmp`, so this is byte identity rather than a floating-point
tolerance. The recurrent KDA state remained resident while each transient
layer's weights were loaded immediately before execution and released
immediately afterwards.

Decision: **PASS for the real K3 weight lifecycle and numerical invariance.**

## Combined scientific conclusion

The original Fareed-to-Colibri hypothesis has passed its pre-checkpoint
feasibility gates:

1. Colibri's compressed layer representation can be moved without changing it.
2. A bounded, partially resident, double-buffered O_DIRECT reader works.
3. Real K3 layer weights can be evicted and reloaded around genuine forward
   execution without changing hidden states.
4. Persistent recurrent/KV state is separable from transient layer weights.

## Evidence boundary

The current lifecycle executable first calls the ordinary `model_init`, then
evicts the chosen suffix. It therefore proves correctness of eviction/reload,
not reduced startup peak RSS. The native packed reader and the real K3 lifecycle
are also still separate code paths.

No claim is yet justified about:

- a 25 GiB peak-RSS run;
- real 93-layer K3 hidden states, routes, logits or generated tokens;
- one-read production layer binding;
- SSD bandwidth or overlap efficiency on a full-size layer;
- end-to-end speedup versus Fareed or resident Colibri.

## Next gate

Combine the two passing components:

- pack each real Colibri K3 layer into a directly bindable execution image;
- skip allocation of nonresident layers during `model_init`;
- bind `W` and float views directly into the current packed layer buffer;
- prefetch the next packed layer while the current one executes;
- compare resident and streamed traces on a truncated real checkpoint;
- only then measure RSS, physical I/O and end-to-end latency.

Decision: **PROCEED.** The hypothesis is technically viable, but it has not yet
passed the real-checkpoint or performance gates.
