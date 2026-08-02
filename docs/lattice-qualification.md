# Lattice Qualification Plane: architecture and evidence contract

## Product thesis

Colibri is the inference engine. Lattice is the qualification and deployment-evidence layer being built on top of it.

The bounded commercial question is:

> For this recorded model, runtime, machine and representative workload, which tested execution configuration produced the strongest replay throughput while satisfying the recorded numerical-consistency, regression and evidence policies?

Lattice does not replace Colibri's planner, doctor, model loader, kernels or server. It treats the generated Colibri resource plan as the baseline, creates deterministic workload replays, measures reviewed execution candidates, retains successful and failed evidence, applies explicit promotion gates and emits a deployment profile tied to the complete local evidence contract.

The current assurance level is **numerical replay consistency**. It is not complete logit equality, semantic equivalence or downstream model-quality validation.

## Architecture

```text
customer workload suite
        │
        ▼
Colibri deep doctor + resource plan
        │
        ▼
sampled model/runtime/hardware/environment identity
        │
        ▼
deterministic prompt + continuation token replay
        │
        ▼
candidate × workload × repeat
        │
        ├── phase 1: uninstrumented timed replay
        │
        └── phase 2: KV reset + numerical replay oracle
                         │
                         ▼
          private bounded oracle artifact
                         │
                         ▼
 sealed run records + completed-session evidence root
                         │
                         ▼
 numerical + regression + gain + confidence gates
                         │
                         ▼
 evidence-bound profile + report + guarded launch
```

## Numerical replay oracle v2

The GLM engine exposes an opt-in oracle only when Lattice sets `REPLAY_ORACLE=1`. Ordinary Colibri inference does not execute this path.

Each qualification trial runs two phases in the same fresh process:

1. **Performance phase.** Colibri prefills the recorded prompt and replays the recorded continuation without oracle computation. Published decode throughput and profiler telemetry come only from this phase.
2. **Validation phase.** Colibri resets KV state, repeats the same forced-token replay and computes a numerical sketch before each replay-step logit vector is freed.

Separating the phases prevents oracle computation and transport from contaminating the performance measurement it gates.

For every replay step, schema `coli-replay-oracle/2` records:

- forced token ID;
- exact top-1 token ID;
- exact top-2 token ID;
- the exact ordered top-k token-ID list;
- top-1 logit;
- forced-token logit;
- top-1 minus top-2 margin;
- finite-vector mean;
- finite-vector RMS;
- four deterministic signed projections over the complete finite logit vector;
- non-finite logit count.

The current versioned policy is:

```text
schema: coli-replay-oracle/2
measurement: separate_replay_pass
transport: private_file
representation: compact_rows
top-k identity: 8 ordered IDs, exact
forced/top-1/top-2 identity: exact
absolute numeric tolerance: 0.005
relative numeric tolerance: 0.0005
non-finite logits: reject
```

The exact policy is stored in the session, included in the session evidence root, copied into the promoted profile and included in profile identity. Any policy change requires requalification.

### What the oracle detects

The gate detects, among other things:

- changed forced-token identity;
- changed top prediction;
- changed second prediction;
- changed top-k membership or ordering;
- non-finite logits;
- selected-logit drift beyond tolerance;
- distribution-scale changes reflected in mean or RMS;
- tail changes reflected in the deterministic projections.

### What the oracle does not prove

It does not record every logit. Therefore it cannot prove complete vector equality. Different logit vectors can theoretically share the same compact sketch. It also does not prove:

- identical free-running generations;
- equivalent router internals not visible in the final logit sketch;
- equivalent downstream task quality;
- calibrated acceptability across all CPU, CUDA, Metal, HIP or Vulkan implementations.

The current tolerances are explicit engineering defaults. They still require empirical calibration using real supported models and hardware.

## Private bounded artifact transport

Verbose per-token oracle rows are not sent through stdout. Doing so would either overflow Lattice's 64 KiB process-output boundary or require weakening that safety control for long continuations.

Instead:

1. Lattice creates a private temporary directory and passes an artifact path through `REPLAY_ORACLE_OUT`.
2. The engine writes strict tab-separated compact rows to that file.
3. The engine flushes and closes the artifact before emitting a small `REPLAY_ORACLE_WRITTEN` marker to stdout.
4. Lattice requires a regular, non-symlink file.
5. Lattice enforces an independent 4 MiB artifact limit.
6. Lattice validates UTF-8, record shape, exact field count, summary consistency, top-k length and policy identity.
7. The temporary file is removed when the trial exits; its parsed compact representation becomes part of the sealed run record.

A regression test parses a complete 2,048-step artifact larger than the ordinary stdout limit. This demonstrates transport scalability without claiming a production frontier-model benchmark.

## `init`

```bash
python3 -m lattice init --repo . --model /models/glm52_i4 \
  --suite examples/lattice-workload.json --workspace .lattice
```

Initialization:

1. requires the currently proven GLM/`colibri` adapter;
2. selects the maximum context declared by the workload suite;
3. executes Colibri planning and deep doctor at that context;
4. validates Safetensors headers and tensor offsets;
5. fingerprints primary, split and mirror model locations using structural metadata and deterministic payload samples;
6. fingerprints the launcher, selected engine and execution-critical support modules;
7. fingerprints the CPU/GPU resource plan, storage-volume identity and controlled qualification environment;
8. validates and fingerprints the workload suite;
9. writes `project.json`.

These are scoped operational fingerprints, not exhaustive remote attestation. They do not currently cover every tensor byte, linked library, driver, firmware revision, thermal state, power limit or background process.

## Qualification environment

Lattice strips unreviewed execution variables from ambient state and uses an explicit allowlist. Unknown `COLI_*`, `OMP_*` and `GOMP_*` overrides are rejected rather than silently accepted.

Controls with semantic implications are fixed and value-constrained, including:

```text
COLI_POLICY=quality
DRAFT=0
KVSAVE=0
AUTOPIN=0
REPIN=0
```

Reviewed hardware and placement controls such as accelerator selectors, RAM budget, thread count, `PIN_GB`, pipeline, prefetch and direct-I/O controls are included in the execution fingerprint.

## `qualify`

```bash
python3 -m lattice qualify --workspace .lattice --repeats 3 --timeout 900
```

For each workload, Colibri first emits prompt and generated token IDs. Those IDs become a hash-identified replay. Each candidate then consumes the same forced-token path.

Candidate order rotates across workloads and repeats to reduce simple warm-cache and fixed-order bias. Every trial uses a fresh process. This still does not control every systems variable: OS page cache, temperature, power state and competing load can affect results.

Use at least three repeats on a quiet, documented machine for meaningful real-model qualification.

The topology-aware candidate matrix may include:

- generated Colibri baseline;
- physical-core and reduced-thread variants;
- NUMA interleave on suitable systems;
- I/O pipeline;
- direct I/O with pipeline;
- real pilot prefetch;
- Linux `io_uring` variants;
- CUDA resident-pipeline variants.

The matrix intentionally excludes model weights, quantization, sampling temperature, top-k/top-p sampling, router semantics and expert-selection policy.

## Recovery

Qualification checkpoints:

- session creation;
- every completed workload calibration;
- every successful or failed trial;
- interruption state.

Resume with:

```bash
python3 -m lattice qualify --workspace .lattice --resume <session-id>
```

Successful tasks are not repeated. Failed tasks may be retried, but duplicate successful observations for the same candidate/workload/repeat are rejected so retries cannot inflate evidence.

## Evidence sealing

Every run is stored with `record_sha256`, computed over the complete run record except the digest field.

A completed session carries `evidence_root_sha256` over:

- the completed session definition;
- numerical-oracle policy;
- candidate matrix and repeat count;
- replay hashes;
- every run ID and sealed run digest.

The promoted profile binds:

- session ID;
- evidence root;
- oracle policy;
- exact selection policy;
- winner;
- model/runtime/hardware/execution/workload fingerprints.

This is **local tamper evidence**, not cryptographic immutability. A malicious administrator capable of coordinated rewriting can construct a new internally consistent workspace. There is no digital signature, external timestamp, TPM attestation, transparency log or write-once storage yet.

## `recommend`

```bash
python3 -m lattice recommend --workspace .lattice --session <id> \
  --min-runs 3 --min-gain 0.03 --max-regression 0.05 \
  --confidence 0.90 --require-confidence --hourly-cost 2.50
```

Promotion requires:

- a completed candidate × workload × repeat matrix;
- valid, untruncated performance telemetry;
- a valid oracle artifact for every successful run;
- stable baseline numerical sketches across repeats;
- candidate numerical agreement with the paired baseline workload/repeat;
- minimum successful paired observations for every workload;
- no workload regression beyond the configured limit;
- sufficient workload-weighted aggregate gain;
- optionally, a bootstrap lower confidence bound above zero gain.

A faster candidate with an oracle mismatch is ineligible.

The cost estimate is decode-only. It is derived from the supplied hourly hardware cost and workload-weighted harmonic decode throughput. It excludes prefill, idle capacity, batching, queueing, service overhead and availability margins.

## Statistical limitations

The current selection statistics are useful screening gates, not a complete performance study. Outstanding methodological work includes:

- stratified bootstrap preserving workload composition;
- formal power analysis;
- thermal and power stabilization;
- explicit warm-up policy;
- robust outlier methodology;
- multiple-candidate correction;
- replication across independent machines.

These limitations are material and should be disclosed in investor or customer reports.

## `verify`

`verify` recomputes current model, runtime, hardware, topology, context, workload and environment fingerprints. It then validates:

- replay hashes;
- every sealed run digest;
- complete task coverage;
- session evidence root;
- numerical-oracle schema and policy;
- every successful oracle record;
- baseline stability;
- candidate numerical agreement;
- selection policy;
- winner;
- profile identity.

It does not merely trust `current-profile.json`.

## `report`

Reports include:

- assurance level;
- exact oracle schema, measurement method, transport and tolerances;
- qualified fingerprints;
- evidence root;
- workload mix;
- promotion policy;
- every candidate outcome;
- failures and oracle mismatches;
- replay-throughput confidence interval;
- decode-only cost estimate;
- promoted environment;
- scientific and threat-model limitations.

## `env` and `launch`

```bash
python3 -m lattice env --workspace .lattice --format shell
python3 -m lattice launch --workspace .lattice -- chat
python3 -m lattice launch --workspace .lattice -- serve --host 127.0.0.1 --port 8000
```

Both re-run verification immediately before use.

Verified launch rejects model, context, memory, accelerator, policy, sampling and auto-tier overrides; disables Colibri's independent saved tune profile; and prevents chat from auto-attaching to an unrelated server.

`--adaptive` is an explicit departure from the measured environment because persistent KV and expert-history learning are disabled during qualification.

## Workspace

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

Application APIs refuse to overwrite sealed runs and completed sessions. Qualification and deployment share an exclusive workspace lock. The files remain ordinary local files; filesystem administrators are outside the current integrity threat model.

## What v0.4 establishes

For the recorded fingerprints, workload and oracle policy, a completed qualification establishes that:

- candidates consumed the same forced token path;
- published throughput came from a separate uninstrumented pass;
- every successful run produced a bounded private numerical artifact;
- exact forced/top-1/top-2/top-k token identities agreed with paired baseline evidence;
- numeric sketch fields remained within the recorded tolerances;
- the winner cleared workload and aggregate performance gates;
- ordinary run/evidence edits are detectable;
- the winner is recomputable from the retained evidence and policies.

It does not establish:

- real-model performance until run with real weights and target hardware;
- universal speedup on other machines or workloads;
- complete logit equality;
- semantic or downstream-quality equivalence;
- calibrated cross-backend tolerances;
- full cryptographic hashing of every model byte;
- protection from coordinated administrator tampering;
- production concurrency, uptime or SLA.

The v0.4 adapter remains GLM-only. Inkling, Kimi K3 and OLMoE require dedicated replay and oracle adapters.

## Commercial path

The immediate product is workload-specific qualification and deployment evidence. The potential compounding assets are:

1. a normalized corpus of model × hardware × workload × configuration × numerical-drift observations;
2. empirically calibrated backend-specific numerical policies;
3. performance and feasibility prediction before hardware purchase;
4. engine selection across Colibri and other runtimes;
5. production drift detection and safe requalification;
6. fleet placement and procurement recommendations.

These remain future directions. The next decisive milestone is a sanitized real-model qualification on documented hardware, including observed oracle deltas across repeated CPU/GPU runs.
