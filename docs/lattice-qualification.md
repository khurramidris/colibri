# Lattice Qualification Plane: architecture and evidence contract

## Product thesis

Colibri demonstrates that very large sparse models can execute across heterogeneous memory. Lattice addresses a different commercial question:

> Which tested model/runtime/hardware configuration should a company deploy for its own workload, and can that decision be reproduced and audited?

This is not another static planner. The generated Colibri plan is the baseline. Lattice creates fixed token replays from representative customer prompts, measures a bounded set of reviewed execution candidates, preserves every failed trial, applies explicit promotion gates, and emits a deployment profile tied to the recorded fingerprints, workload and evidence root.

The current assurance level is **deterministic replay consistency**. It supports controlled throughput comparison under the same forced token path. It does not establish equal logits, probabilities, numerical error, free-running outputs or downstream model quality.

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
             SHA-256-sealed local run records
                               │
                 completed-session evidence root
                               │
              regression + gain + confidence gates
                               │
                               ▼
             evidence-bound profile + qualification report
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
2. chooses the maximum context required by the workload suite and uses it for deep doctor and planning;
3. captures Colibri's generated resource plan;
4. validates every Safetensors header and tensor offset;
5. fingerprints primary, split and mirror weight directories, including deterministic payload samples;
6. hashes the launcher, engine and selected execution-critical support modules;
7. fingerprints the CPU/GPU plan, storage-device identity and controlled qualification environment;
8. validates and fingerprints the workload suite;
9. writes `project.json`.

The model fingerprint is intentionally practical rather than a full hash of hundreds of gigabytes. It hashes structural metadata and deterministic samples from every shard. A deployment requiring full cryptographic payload attestation should add an offline complete-shard manifest.

The runtime and hardware fingerprints are also scoped fingerprints, not exhaustive remote attestation. They do not currently cover every linked library, driver, firmware version, thermal state or background process.

### `qualify`

```bash
python3 -m lattice qualify --workspace .lattice --repeats 3 --timeout 900
```

For each workload, Lattice first asks Colibri to generate one continuation while emitting prompt and generated token IDs. Those IDs become a hash-identified replay. Every candidate is then forced over the same token sequence, so execution scheduling can be compared without generation randomness.

This does **not** verify logit equality. A numerical correctness oracle remains a future engine-level requirement.

Candidate order is rotated across workload and repeat to reduce simple order and warm-cache bias. Each trial launches a fresh engine process, but operating-system page cache, machine temperature and competing system load can still affect results. Real qualification should therefore use at least three repeats, a quiet machine and disclosed hardware conditions.

Qualification uses an explicit reviewed allowlist. Unknown `COLI_*`, `OMP_*` and `GOMP_*` variables are stripped from ambient state and rejected when supplied through persisted overrides. Fixed semantic controls such as `COLI_POLICY=quality`, `DRAFT=0`, `KVSAVE=0`, `AUTOPIN=0` and `REPIN=0` are value-constrained and included in the execution fingerprint.

Sessions are checkpointed before calibration, after every completed workload replay, and after every trial. If the process or machine is interrupted, resume the same session without recalibrating completed workloads:

```bash
python3 -m lattice qualify --workspace .lattice --resume <session-id>
```

Failed attempts remain in the local evidence set. By default a resume retries failed and missing tasks but never reruns a successful task, so retries cannot inflate minimum-run or confidence gates. Use `--no-retry-failed` to fill only tasks that were never attempted.

The built-in matrix is topology-aware and may include:

- generated Colibri baseline;
- physical-core and half-core thread counts;
- NUMA interleave on multi-socket hosts;
- I/O pipeline;
- direct I/O plus pipeline;
- real pilot prefetch;
- Linux `io_uring` plus direct I/O;
- CUDA resident-pipeline variants.

Only reviewed execution and placement controls are eligible. Quantization changes, `TOPK`, `TOPP`, cache-aware routing, model weights and sampling policy are excluded from the candidate matrix.

### Evidence sealing

Each successful or failed run is stored as JSON with a `record_sha256` digest computed over the complete record except the digest field itself. Loading run evidence verifies that digest.

When a session completes, Lattice computes `evidence_root_sha256` over:

- the completed session definition;
- every replay hash;
- every sealed run ID and run digest.

The profile ID includes this evidence root and the exact selection policy. `verify` recomputes the root and winner before deployment.

This is **local tamper evidence**, not cryptographic immutability. A malicious administrator who can rewrite every related file can construct a new internally consistent workspace. The current system has no digital signature, external timestamp, TPM attestation, transparency log or write-once storage.

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

The statistics are useful screening gates, not a substitute for a full performance study. The current implementation does not yet perform thermal stabilization, power-state control, formal outlier handling, multiple-comparison correction or replication across machines.

Cost per million generated tokens uses the operator-supplied hardware cost and weighted harmonic effective throughput. It is explicitly a **decode-only** estimate: prompt prefill, idle capacity, batching, queueing and service overhead require a serving benchmark.

### `verify`

`verify` does not trust `current-profile.json`. It recomputes current model, runtime, hardware, storage-topology, maximum-context and controlled-environment fingerprints; validates replay hashes and every sealed run record; checks the complete task matrix and completed-session evidence root; reruns selection using the recorded policy; and confirms the profile ID and winner.

### `report`

The Markdown report contains the recorded fingerprints, assurance level, evidence root, workload mix, promotion policy, all candidate outcomes, confidence interval, estimated decode cost and deployment environment. Failed candidates are visible rather than silently removed.

### `env` and `launch`

`env` and `launch` rerun evidence verification immediately before deployment. `env` emits the measured promoted environment as JSON or shell exports. `launch` invokes the fork's own `c/coli` entrypoint with model and engine bound to the verified profile:

```bash
python3 -m lattice env --workspace .lattice --format shell
python3 -m lattice launch --workspace .lattice -- chat
python3 -m lattice launch --workspace .lattice -- serve --host 127.0.0.1 --port 8000
```

Model, memory, context, accelerator, policy, sampling and auto-tier overrides are rejected during verified launch. Chat is forced to a private local engine rather than auto-attaching to an unrelated server, and Colibri's independent saved tune profile is disabled.

Adaptive KV persistence and expert-history learning were frozen during qualification. `--adaptive` is therefore an explicit production departure from the measured environment.

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

The application refuses to overwrite sealed run records and completed sessions through normal APIs. A workspace containing evidence cannot be force-reinitialized; use a new workspace to preserve provenance. Qualification and deployment share one exclusive workspace lock, so a live launch cannot silently contaminate a benchmark on the same model and machine.

The files remain ordinary local files. Filesystem administrators are outside the current integrity threat model.

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

`weight` represents relative production frequency. `expected_contains` may be supplied for a basic calibration-time text assertion. Prompts are stored in the local project because they define the workload; do not commit a workspace containing confidential prompts.

## What this establishes

A completed qualification establishes that, for the recorded fingerprints and workload:

- every measured candidate consumed the same forced token sequence;
- performance telemetry existed and was not truncated;
- the selected candidate cleared explicit aggregate and per-workload throughput gates;
- run edits and ordinary evidence-set inconsistencies are detectable;
- the profile can be recomputed from the retained evidence root and selection policy;
- the decode-cost estimate follows from a disclosed hourly-cost assumption.

The v0.2 adapter is deliberately limited to GLM/`colibri`. Inkling, Kimi K3 and OLMoE remain Colibri engine capabilities, but they need dedicated Lattice calibration/replay adapters before Lattice can make the same measurement claim for them.

It does not establish:

- real-model performance until run with real weights and target hardware;
- universal speedup on other hardware or prompts;
- logit, probability, router or free-generation equivalence;
- downstream model quality;
- full cryptographic hashing of every tensor payload byte;
- protection against coordinated evidence rewriting by an administrator;
- datacenter availability, concurrency or SLA;
- a new inference kernel.

## Why this can become a company

The immediate product is workload-specific qualification and deployment evidence. The potential compounding assets are:

1. a normalized corpus of model × hardware × workload × configuration observations;
2. a performance predictor trained on real observations;
3. engine selection across Colibri, llama.cpp, vLLM, SGLang and vendor runtimes;
4. continuous adaptation and drift detection in production;
5. fleet-level placement and procurement recommendations.

Those are future product directions, not capabilities proven by the current prototype. The next decisive milestone is a sanitized qualification report generated with real Colibri weights on documented target hardware.
