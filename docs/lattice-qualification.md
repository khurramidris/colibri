# Lattice Qualification Plane: architecture and evidence contract

## Product thesis

Colibri demonstrates that very large sparse models can execute across heterogeneous memory. Lattice addresses a different commercial question:

> Which tested model/runtime/hardware configuration should a company deploy for its own workload, and can that decision be reproduced and audited?

The generated Colibri plan is the baseline. Lattice creates fixed-token replays from representative customer prompts, measures a bounded set of reviewed execution candidates, preserves failed trials, applies numerical and performance gates, and emits a deployment profile tied to the recorded fingerprints, workload, numerical-oracle policy and evidence root.

The current assurance level is **numerical replay consistency**. It is stronger than forced-token replay alone, but it is not complete logit equality, semantic equivalence or downstream model-quality validation.

## Architecture

```text
customer workload JSON
        │
        ▼
Colibri deep doctor ──► resource plan ──► sampled model/runtime identity
        │                                      │
        └───────────── qualification project ◄─┘
                               │
                               ▼
                  deterministic token replays
                               │
                  candidate × workload × repeat
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   uninstrumented timed replay       separate numerical replay
                                                │
                              top-k identity + logit sketch
              └────────────────┬────────────────┘
                               ▼
                sealed run records + evidence root
                               │
       numerical + regression + gain + confidence gates
                               │
                               ▼
             evidence-bound profile + qualification report
```

Lattice relies on the fork's existing `c/resource_plan.py`, `c/doctor.py` and engine binaries. It uses Colibri's `TOKENS`, `REF`, `REF_FORCE`, `REPLAY` and `PROF` instrumentation and adds an opt-in `REPLAY_ORACLE` path inside the GLM engine.

## Numerical replay oracle

During ordinary inference the oracle is disabled and adds no work.

During Lattice qualification, every candidate run has two phases inside a fresh engine process:

1. **Performance phase:** Colibri prefills the recorded prompt and replays the recorded continuation without numerical instrumentation. The published decode throughput and profiler output come from this phase.
2. **Validation phase:** Colibri resets KV state, repeats the same replay and computes a numerical sketch before each logit vector is freed.

Separating the phases prevents sketch computation and output from contaminating the measured throughput.

For every replay step, schema `coli-replay-oracle/1` records:

- forced token ID;
- exact top-1 and top-2 token IDs;
- a deterministic hash of the ordered top-k token IDs;
- top-1 and forced-token logits;
- top-1 minus top-2 margin;
- full finite-vector mean and RMS;
- four deterministic signed projections over the complete finite logit vector;
- non-finite count.

The current policy requires:

```text
measurement: separate_replay_pass
top-k identity: 8 tokens, exact
forced/top-1/top-2 identity: exact
absolute numeric tolerance: 0.005
relative numeric tolerance: 0.0005
non-finite logits: reject
```

The oracle policy is stored in the session, profile and profile ID. Changing the policy requires requalification.

This is a compact consistency sketch. It can detect many meaningful deviations, including changed top predictions, altered top-k membership, large selected-logit changes, distribution-scale changes and projection changes in the logit tail. It cannot prove that every logit is equal. Collisions and undetected differences remain theoretically possible. The initial tolerances are conservative engineering defaults and still require calibration on real supported CPU/GPU runs.

Baseline repeats must be mutually consistent under the same oracle policy. Each candidate is paired with the corresponding baseline workload/repeat. Any mismatch makes the candidate ineligible even when it is faster.

## Commands

### `init`

```bash
python3 -m lattice init --repo . --model /models/glm52_i4 \
  --suite examples/lattice-workload.json --workspace .lattice
```

`init`:

1. requires the currently proven GLM/`colibri` adapter;
2. plans and runs deep doctor at the maximum workload context;
3. validates Safetensors headers and tensor offsets;
4. fingerprints primary, split and mirror weight directories using structural metadata and deterministic payload samples;
5. fingerprints the launcher, engine and selected execution-critical modules;
6. fingerprints the CPU/GPU plan, storage-device identity and controlled qualification environment;
7. validates and fingerprints the workload suite;
8. writes `project.json`.

These are scoped practical fingerprints, not exhaustive remote attestation. They do not cover every model byte, linked library, driver, firmware version, power state, temperature or background process.

### `qualify`

```bash
python3 -m lattice qualify --workspace .lattice --repeats 3 --timeout 900
```

For each workload, Colibri first emits prompt and generated token IDs. Those IDs become a hash-identified replay. Each candidate consumes the same token sequence, completes an uninstrumented performance pass, then completes the separate numerical-validation pass.

Candidate order rotates across workloads and repeats to reduce simple order bias. Each trial launches a fresh process, but operating-system page cache, temperature and competing load can still affect results. Real qualification should use at least three repeats on a quiet, documented machine.

The qualification environment uses an explicit allowlist. Unknown `COLI_*`, `OMP_*` and `GOMP_*` overrides are stripped or rejected. Fixed controls such as `COLI_POLICY=quality`, `DRAFT=0`, `KVSAVE=0`, `AUTOPIN=0` and `REPIN=0` are value-constrained and fingerprinted.

Sessions checkpoint completed calibrations and every trial. Resume an interrupted session with:

```bash
python3 -m lattice qualify --workspace .lattice --resume <session-id>
```

Successful tasks are not rerun, preventing retries from inflating evidence. Failed attempts remain retained.

The topology-aware candidate matrix may include:

- generated Colibri baseline;
- physical-core and half-core thread counts;
- NUMA interleave;
- I/O pipeline;
- direct I/O plus pipeline;
- real pilot prefetch;
- Linux `io_uring` plus direct I/O;
- CUDA resident-pipeline variants.

Quantization, model weights, sampling, router semantics and expert-selection policy are excluded from the matrix.

### Evidence sealing

Every successful or failed run is stored with `record_sha256`. Completed sessions carry `evidence_root_sha256` over:

- the completed session definition, including numerical-oracle policy;
- every replay hash;
- every run ID and run digest.

Profiles bind the evidence root, oracle policy, selection policy and winner. `verify` recomputes all of them.

This is local tamper evidence, not immutability. A malicious administrator capable of coordinated rewriting can construct a new internally consistent workspace. There is no digital signature, external timestamp, TPM attestation, transparency log or write-once storage yet.

### `recommend`

```bash
python3 -m lattice recommend --workspace .lattice --session <id> \
  --min-runs 3 --min-gain 0.03 --max-regression 0.05 \
  --confidence 0.90 --require-confidence --hourly-cost 2.50
```

Promotion requires:

- a complete candidate × workload × repeat matrix;
- valid, untruncated throughput and numerical-oracle telemetry;
- a stable baseline numerical oracle;
- candidate oracle agreement with its paired baseline runs;
- at least `min-runs` paired samples per workload;
- no workload regression worse than `max_regression`;
- workload-weighted geometric gain of at least `min_gain`;
- optionally, a paired bootstrap lower bound above zero gain.

These are screening gates, not a full benchmarking study. The current method does not yet include formal power analysis, temperature/power stabilization, outlier methodology, multiple-comparison correction or replication across independent machines.

Cost per million generated tokens is a decode-only estimate derived from the operator-supplied hourly cost and weighted harmonic decode throughput. It excludes prefill, idle capacity, batching, queueing and service overhead.

### `verify`

`verify` recomputes current model, runtime, hardware, topology, context and environment fingerprints; validates replay hashes and run digests; checks the complete task matrix and evidence root; validates oracle policy and every successful oracle record; reruns candidate selection; and confirms the profile ID and winner.

### `report`

Reports include:

- exact numerical-oracle policy and tolerances;
- fingerprints and evidence root;
- workload and promotion policy;
- all candidate results, failures and oracle mismatches;
- throughput confidence interval;
- decode-only cost estimate;
- promoted environment;
- explicit scientific and threat-model boundaries.

### `env` and `launch`

```bash
python3 -m lattice env --workspace .lattice --format shell
python3 -m lattice launch --workspace .lattice -- chat
python3 -m lattice launch --workspace .lattice -- serve --host 127.0.0.1 --port 8000
```

Both commands verify the workspace immediately before use. Launch rejects model, memory, context, accelerator, policy, sampling and auto-tier overrides; disables Colibri's independent tune profile; and forces chat to use a private local engine rather than auto-attaching elsewhere.

`--adaptive` is an explicit departure from the frozen measured environment because persistent KV and expert-history learning are disabled during qualification.

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

Normal application APIs refuse to overwrite sealed runs and completed sessions. Qualification and deployment share an exclusive lock. Files remain ordinary local files; filesystem administrators are outside the current integrity threat model.

## What a completed qualification establishes

For the recorded fingerprints, workload and oracle policy:

- candidates consumed the same forced token sequence;
- published throughput came from an uninstrumented replay phase;
- each successful run completed a separate numerical-validation replay;
- exact token identities and the numerical sketch remained within the recorded policy;
- the selected candidate cleared per-workload and aggregate throughput gates;
- ordinary run/evidence edits are detectable;
- the winner is recomputable from the retained evidence root and policies.

It does not establish:

- real-model performance until run with real weights and target hardware;
- universal speedup on other machines or workloads;
- complete logit equality or formal numerical equivalence;
- free-running output or downstream quality equivalence;
- calibrated cross-backend tolerances until real CPU/GPU evidence exists;
- full hashing of every model byte;
- protection against coordinated administrator tampering;
- fleet availability, concurrency or SLA.

The v0.3 adapter remains GLM-only. Inkling, Kimi K3 and OLMoE require dedicated replay/oracle adapters.

## Commercial path

The immediate product is workload-specific qualification and evidence. The potential compounding assets are:

1. a normalized corpus of model × hardware × workload × configuration × numerical-drift observations;
2. calibrated backend-specific numerical policies;
3. performance and feasibility prediction before hardware purchase;
4. engine selection across Colibri and other runtimes;
5. production drift detection and safe requalification;
6. fleet placement and procurement recommendations.

These remain future directions. The next decisive milestone is a sanitized real-model qualification report on documented hardware, including observed numerical-oracle deltas across baseline repeats and candidate backends.
