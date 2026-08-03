# Gemma Expert Atlas — Phase 0 protocol

Gemma Expert Atlas is a falsification harness for the proposed sequence:

> Gemma 4 baseline → hardware-aware planner → expert caching/prefetch → selective ternary experts.

It does **not** claim a speedup. It determines whether Gemma's measured native routing contains enough held-out locality and predictability to justify the next engineering phase.

## Scientific rules

1. Pin the model revision, runtime commit, quantization artifact, prompt suite, and hardware identity.
2. Capture the native router's top-k IDs and scores without altering router semantics.
3. Split by request identity. Static policy training and transition learning may use only the training requests.
4. Evaluate static placement, LRU, LFU, request-prefill LFU, and Belady on held-out requests.
5. Treat Belady as an unattainable upper bound, never as a deployable result.
6. Report bytes moved and modelled exposed transfer time together with hit rate.
7. Keep MTP off during the first atlas run. Add it later as a separate factor.
8. Publish negative results and all raw traces needed to reproduce the analysis.

## Canonical route schema

One JSON object per layer per token:

```json
{
  "schema": "lattice.route.v1",
  "run_id": "run-immutable-id",
  "request_id": "coding-017",
  "phase": "decode",
  "token": 12,
  "layer": 7,
  "experts": [4, 9, 17, 21, 56, 68, 71, 103],
  "scores": [0.18, 0.16, 0.14, 0.13, 0.12, 0.10, 0.09, 0.08],
  "router_ns": 42000,
  "expert_ns": 1900000
}
```

Every recorded token must contain all expected layers. Duplicate rows, non-finite scores, invalid expert IDs, incomplete tokens, and oversized JSON lines fail closed.

## GPU capture

The optional adapter discovers router modules through hooks and writes the canonical trace:

```bash
python tools/gemma_atlas/hf_capture.py \
  --model google/gemma-4-26B-A4B-it \
  --revision <immutable-commit> \
  --prompt-file examples/gemma-atlas/prompts.example.jsonl \
  --output gemma4-routes.jsonl \
  --n-layers 30 --n-experts 128 --top-k 8 \
  --load-in-4bit --dry-discover
```

Run discovery first. The adapter refuses ambiguous layer coverage and duplicate router outputs. A model-specific runtime patch may later replace this adapter, but must emit the same schema.

## Analysis

Copy the example manifest and replace:

- `model_revision` with the immutable model commit;
- `checkpoint_sha256` with the exact quantized checkpoint digest;
- `runtime_revision` with the immutable runtime commit;
- `workload_sha256` with the exact prompt-suite digest;
- `expert_bytes` with measured packed bytes for one routed expert in the exact checkpoint;
- `target_cache_capacity` with the number of experts **per layer** that fits the pre-registered 8GB deployment budget after dense weights, KV cache, and workspace;
- transfer profiles with measured effective bandwidth, fixed latency, and a disclosed overlap assumption.

```bash
python -m lattice.atlas analyze \
  --manifest gemma4-atlas.json \
  --trace gemma4-routes.jsonl \
  --output-dir atlas-output
```

Outputs are write-once by default:

- `atlas-evidence.json` — complete inputs, split assignment, metrics, simulations, gates, and decision;
- `atlas-report.md` — human-readable report with explicit limitations;
- `atlas-receipt.json` — hashes binding the evidence and report.

Verify retained outputs and any still-available source inputs:

```bash
python -m lattice.atlas verify --output-dir atlas-output --require-inputs
```

## Ternary probes

`tools/gemma_atlas/ternary_probe.py` evaluates one captured SwiGLU expert operator from NPZ matrices and activations. Its output is a `lattice.ternary.probe.v1` row that can be concatenated across sampled experts and supplied with `--ternary`.

This probe is only a local sensitivity screen. It cannot establish model-level quality. A positive result must still pass full perplexity, task, multilingual, coding, and reasoning evaluation after model conversion.

## Default go/no-go gates

Caching/prefetch proceeds only when all configured gates pass on held-out **decode** routes at the pre-registered `target_cache_capacity`. Prefill may warm dynamic caches but is not counted as decode performance:

- practical cache hit rate ≥ 60%;
- Belady hit rate ≥ 75%;
- modelled exposed-transfer reduction ≥ 40%;
- non-oracle predictor recall ≥ 85%;
- predictor overfetch ≤ 20%.

These are pre-registered defaults, not universal constants. Any change must be recorded in the manifest before analysis.

## Required next hardware experiment

A GO result only authorizes an 8GB hybrid A/B. It does not prove the A/B will win. The next run must compare identical model, prompts, tokens, context, and correctness checks with cache/prefetch disabled versus enabled, reporting throughput, TTFT, p50/p95 token latency, PCIe bytes, hit rate, prefetch precision/recall, power, and quality.
