# Lattice Qualification Plane: architecture and evidence contract

## Product thesis

Colibri proves that very large sparse models can execute across heterogeneous memory. Lattice addresses a different commercial question:

> Which exact model/runtime/hardware configuration should a company deploy for its own workload, and can that decision be reproduced and audited?

This is not another static planner. The generated Colibri plan is the baseline. Lattice creates fixed token replays from representative customer prompts, measures a bounded set of quality-preserving execution candidates, preserves every failed trial, applies explicit promotion gates, and emits a deployment profile tied to the exact bytes and workload that produced it.

## Architecture

```text
customer workload JSON
        │
        ▼
Colibri deep doctor ──► resource plan ──► model/runtime fingerprint
        │                                      │
        └───────────── qualification project ◄─┘
                               │
                               ▼
                  deterministic token replays
                               │
                rotated candidate × workload × repeat
                               │
                               ▼
                     immutable run evidence
                               │
              regression + gain + confidence gates
                               │
                               ▼
             promoted profile + qualification report
```

Lattice imports and relies on the fork's existing `c/resource_plan.py` and `c/doctor.py`. It drives the same engine binaries used by `coli`, and uses Colibri's `TOKENS`, `REF`, `REF_FORCE`, `REPLAY` and `PROF` instrumentation. It deliberately does not duplicate tensor placement or pretend to predict performance without measurement.

## Commands

### `init`

```bash
python3 -m lattice init --repo . --model /models/glm52_i4 \
  --suite examples/lattice-workload.json --workspace .lattice
```

`init` performs the following:

1. detects the model family and requires the currently proven GLM/`colibri` replay adapter;
2. executes Colibri's deep doctor;
3. captures Colibri's generated resource plan;
4. validates every Safetensors header and tensor offset;
5. fingerprints config/tokenizer/index files, shard metadata and deterministic payload samples;
6. hashes the launcher, engine and execution-critical support modules;
7. validates and fingerprints the workload suite;
8. writes `project.json`.

The model fingerprint is intentionally practical rather than a full hash of hundreds of gigabytes. It hashes all structural metadata and deterministic samples from every shard. A deployment requiring full cryptographic payload attestation should add an offline complete-shard manifest.

### `qualify`

```bash
python3 -m lattice qualify --workspace .lattice --repeats 3 --timeout 900
```

For each workload, Lattice first asks Colibri to generate one continuation while emitting prompt and generated token IDs. Those IDs become an immutable replay. Every candidate is then teacher-forced over the same token sequence, so scheduling is measured without changing sampling, routing inputs or the continuation.

Candidate order is rotated across workload and repeat to reduce simple order and warm-cache bias. Adaptive pinning, mutable conversation state and quality-affecting environment keys are stripped from the qualification environment.

Sessions are checkpointed before calibration, after every completed workload replay, and after every trial. If the process or machine is interrupted, resume the same session without recalibrating completed workloads:

```bash
python3 -m lattice qualify --workspace .lattice --resume <session-id>
```

Failed attempts remain immutable evidence. By default a resume retries failed and missing tasks but never reruns a successful task, so retries cannot inflate the minimum-run or confidence gates. Use `--no-retry-failed` to fill only tasks that were never attempted.

The built-in matrix is topology-aware and may include:

- generated Colibri baseline;
- physical-core and half-core thread counts;
- NUMA interleave on multi-socket hosts;
- I/O pipeline;
- direct I/O plus pipeline;
- value-preserving real pilot prefetch;
- Linux `io_uring` plus direct I/O;
- CUDA resident pipeline variants.

Only an allowlist of execution and placement controls is eligible. Quantization, `TOPK`, `TOPP`, cache-aware routing, model weights and sampling policy are excluded.

### `recommend`

```bash
python3 -m lattice recommend --workspace .lattice --session <id> \
  --min-runs 3 --min-gain 0.03 --max-regression 0.05 \
  --confidence 0.90 --require-confidence --hourly-cost 2.50
```

Promotion requires:

- a complete candidate × workload × repeat matrix;
- successful, untruncated replay telemetry;
- at least `min-runs` paired samples per workload;
- no workload regression worse than `max-regression`;
- workload-weighted geometric speedup of at least `min-gain`;
- optionally, a paired bootstrap lower confidence bound above zero gain.

Cost per million generated tokens uses the operator-supplied hardware cost and the weighted harmonic effective throughput, which reflects time spent across the workload mix.

### `verify`

`verify` does not trust `current-profile.json`. It recomputes current model and runtime identities, validates replay hashes and every run's candidate/task identity, checks the complete task matrix, reruns the selection calculation using the recorded promotion policy, and confirms the profile ID and winner.

### `report`

The Markdown report contains the qualified identities, workload mix, promotion policy, all candidate outcomes, confidence interval, estimated cost and deployment environment. Failed candidates are visible rather than silently removed.

### `env` and `launch`

`env` and `launch` re-run full evidence verification immediately before deployment. `env` emits the promoted quality-preserving environment as JSON or shell exports. `launch` invokes the fork's own `c/coli` entrypoint with the model and engine bound to the verified profile:

```bash
python3 -m lattice env --workspace .lattice --format shell
python3 -m lattice launch --workspace .lattice -- chat
python3 -m lattice launch --workspace .lattice -- serve --host 127.0.0.1 --port 8000
```

A model override is rejected. Adaptive KV persistence and expert-history learning were frozen during qualification; `--adaptive` is therefore an explicit production opt-in and is reported as a departure from the exact measured environment.

## Workspace layout

```text
.lattice/
  project.json
  replays/
  runs/
  sessions/
  profiles/
  current-profile.json
  reports/
```

Run and profile records are immutable. A workspace containing evidence cannot be force-reinitialized; use a new workspace to preserve provenance.

## Workload schema

```json
{
  "schema_version": 1,
  "name": "customer-support-production-mix",
  "default_context": 4096,
  "default_tokens": 32,
  "hourly_cost_usd": 2.5,
  "cases": [
    {
      "id": "ticket-summary",
      "prompt": "Summarize the following ticket...",
      "weight": 3,
      "context": 8192,
      "tokens": 64
    }
  ]
}
```

`weight` represents relative production frequency. `expected_contains` may be supplied for a basic calibration-time semantic assertion. Prompts are stored in the local project because they define the workload; do not commit a workspace containing confidential prompts.

## What this proves

A completed qualification proves that, for one exact model/runtime/hardware/workload identity:

- every measured candidate consumed the same replayed tokens;
- performance telemetry existed and was not truncated;
- the winner cleared explicit aggregate and per-workload gates;
- the profile can be recomputed from retained evidence;
- the estimated cost follows from a disclosed hourly cost assumption.

The v0.1 adapter is deliberately limited to GLM/`colibri`. Inkling, Kimi K3 and OLMoE remain Colibri engine capabilities, but they need dedicated Lattice calibration/replay adapters before Lattice can make the same qualification claim for them.

It does not prove:

- universal speedup on other hardware or prompts;
- semantic quality beyond deterministic replay and any supplied assertions;
- full cryptographic hashing of every tensor payload byte;
- datacenter availability, concurrency or SLA;
- a new inference kernel.

## Why this can become a company

The immediate product is qualification and deployment evidence. The compounding assets are:

1. a normalized corpus of model × hardware × workload × configuration observations;
2. a performance predictor trained on those observations;
3. engine selection across Colibri, llama.cpp, vLLM, SGLang and vendor runtimes;
4. continuous adaptation and drift detection in production;
5. fleet-level placement and procurement recommendations.

The open-source node creates trusted measurements. A commercial control plane can aggregate anonymized evidence, predict feasibility before hardware purchase, manage fleets, enforce deployment policy, and sell validated performance outcomes rather than raw infrastructure.
