# Lattice Qualification Plane: architecture and evidence contract

## Product thesis

Colibri is the inference engine. Lattice is the workload-specific qualification and deployment-evidence layer built on top of it.

The bounded question is:

> For this recorded model, runtime, machine and representative workload, which tested execution configuration produced the strongest replay throughput while satisfying the recorded numerical-consistency, regression and statistical policies?

Lattice does not replace Colibri's planner, doctor, model loader, kernels or server. It treats the generated Colibri resource plan as the baseline, creates deterministic workload replays, measures reviewed execution candidates, retains successful and failed evidence, applies explicit promotion gates and emits a deployment profile tied to the complete local evidence contract.

The current assurance level is **numerical replay consistency**. It is not complete logit equality, semantic equivalence or downstream model-quality validation.

## Qualification flow

```text
workload suite
    │
    ▼
Colibri deep doctor + resource plan
    │
    ▼
sampled model/runtime/hardware/environment identity
    │
    ▼
deterministic prompt + continuation replay
    │
    ▼
candidate × workload × repeat
    │
    ├── uninstrumented timed replay
    └── KV reset + numerical validation replay
                  │
                  ▼
         private bounded oracle artifact
                  │
                  ▼
     sealed runs + completed-session root
                  │
                  ▼
 numerical + paired statistical promotion gates
                  │
                  ▼
 evidence-bound profile, report and guarded launch
```

## Numerical replay oracle

During normal Colibri inference the oracle is disabled.

During Lattice qualification, every candidate trial has two phases in the same fresh process:

1. **Performance phase:** the fixed continuation is replayed without oracle computation. Published decode throughput and profiler data come only from this phase.
2. **Validation phase:** KV state is reset, the same continuation is replayed again and a numerical sketch is calculated before each logit vector is freed.

Separating the phases prevents numerical validation from contaminating the timing measurement.

Schema `coli-replay-oracle/2` records for every step:

- forced token ID;
- exact top-1 and top-2 IDs;
- exact ordered top-k ID list;
- top-1 and forced-token logits;
- top-1 margin;
- finite-vector mean and RMS;
- four deterministic signed full-vector projections;
- non-finite count.

The current policy is:

```text
measurement: separate_replay_pass
transport: private_file
representation: compact_rows
top-k identity: 8 ordered IDs, exact
absolute tolerance: 0.005
relative tolerance: 0.0005
non-finite logits: reject
```

The detailed rows are written to a controller-created private temporary artifact rather than stdout. Lattice requires a regular non-symlink UTF-8 file and enforces an independent 4 MiB limit. A 2,048-step regression artifact larger than the 64 KiB process-output ceiling is covered by tests.

This is a compact numerical sketch. It can detect many meaningful differences but cannot prove every logit is equal. The tolerances still require calibration on real supported CPU/GPU deployments.

## Initialization and fingerprints

```bash
python3 -m lattice init --repo . --model /models/glm52_i4 \
  --suite examples/lattice-workload.json --workspace .lattice
```

Initialization:

1. requires the currently proven GLM/`colibri` adapter;
2. plans and runs deep doctor at the maximum workload context;
3. validates Safetensors headers and tensor offsets;
4. fingerprints primary, split and mirror model locations using metadata and deterministic payload samples;
5. fingerprints the launcher, selected engine and execution-critical modules;
6. fingerprints the CPU/GPU plan, storage identity and controlled environment;
7. validates and fingerprints the workload suite;
8. writes `project.json`.

These are scoped operational fingerprints, not exhaustive attestation. They do not cover every model byte, library, driver, firmware revision, power state, temperature or background process.

## Controlled environment

Unknown `COLI_*`, `OMP_*` and `GOMP_*` qualification overrides are stripped or rejected. Fixed controls include:

```text
COLI_POLICY=quality
DRAFT=0
KVSAVE=0
AUTOPIN=0
REPIN=0
```

Reviewed placement controls such as accelerator selectors, RAM budget, thread count, `PIN_GB`, pipeline, prefetch and direct I/O are included in the execution fingerprint.

## Qualification and recovery

```bash
python3 -m lattice qualify --workspace .lattice --repeats 3 --timeout 900
```

Candidate order rotates across workloads and repeats to reduce simple order bias. Every trial uses a fresh process. OS page cache, temperature, power state and competing load can still affect results.

The candidate matrix may include thread-count, NUMA, pipeline, direct-I/O, pilot-prefetch, `io_uring` and CUDA resident-pipeline variants. It excludes model weights, quantization, sampling and router semantics.

Qualification checkpoints completed calibrations and every trial. Resume with:

```bash
python3 -m lattice qualify --workspace .lattice --resume <session-id>
```

Successful tasks are not repeated. Failed attempts remain retained. Duplicate successful evidence for a candidate/workload/repeat is rejected.

## Evidence sealing

Every run carries `record_sha256`. A completed session carries `evidence_root_sha256` over:

- session definition;
- workload and candidate matrix;
- oracle policy;
- replay hashes;
- every run ID and run digest.

The promoted profile binds the evidence root, oracle policy, statistical methodology, selection thresholds, winner and qualified fingerprints.

This is local tamper evidence, not immutability or remote attestation. A malicious administrator capable of coordinated rewriting remains outside the current threat model.

## Statistical methodology v2

Lattice v0.5 replaces the earlier pooled bootstrap with methodology identified as `lattice-statistics/2`.

### Pairing

Baseline and candidate measurements are paired by workload and repeat number. For every workload:

```text
paired ratio[r] = candidate tok/s[r] / baseline tok/s[r]
```

The workload point estimate is the median paired ratio. This avoids the potentially misleading ratio-of-independent-medians calculation.

The aggregate point estimate is the workload-weighted geometric mean of the workload median paired ratios. The declared workload weights remain fixed.

### Stratified bootstrap

Bootstrap resampling occurs independently within each workload:

1. resample that workload's paired ratios with replacement;
2. calculate its median resampled ratio;
3. combine all workload medians using the original workload weights;
4. repeat for 5,000 draws.

This prevents a workload with more observations from silently acquiring more influence than its declared production weight.

### Multiple candidates

Testing several candidates increases the chance that at least one appears favourable through noise. Lattice therefore applies a Bonferroni adjustment to the interval level:

```text
per-candidate confidence = 1 - (1 - requested family-wise confidence) / candidate count
```

For example, a requested 90% family-wise interval across five non-baseline candidates uses a 98% interval for each candidate.

The requested family-wise level, candidate count, adjusted per-candidate level, bootstrap method, sample count and point estimator are stored in `statistics_policy`, included in profile identity and recomputed during verification.

### Minimum repeats

Confidence-gated promotion requires at least three paired runs per workload. One or two runs may still be evaluated without `--require-confidence`, but must not be presented as confidence-qualified evidence.

### Remaining limitations

This is stronger screening methodology, not a complete performance study. It still lacks:

- formal statistical power analysis;
- thermal and power stabilization;
- explicit warm-up and cache-state protocol;
- robust outlier modelling;
- replication across independent machines;
- empirical validation of interval coverage on real inference noise.

These limitations must remain visible in customer and investor reports.

## Promotion

```bash
python3 -m lattice recommend --workspace .lattice --session <id> \
  --min-runs 3 --min-gain 0.03 --max-regression 0.05 \
  --confidence 0.90 --require-confidence --hourly-cost 2.50
```

Promotion requires:

- complete candidate × workload × repeat evidence;
- valid untruncated timing telemetry;
- a valid numerical artifact for every successful run;
- stable baseline oracle evidence;
- candidate numerical agreement with paired baseline runs;
- minimum paired observations for each workload;
- no workload regression beyond the threshold;
- sufficient weighted paired gain;
- when requested, an adjusted lower confidence bound above zero gain.

A faster candidate with an oracle mismatch is ineligible.

The cost estimate is decode-only and excludes prefill, idle capacity, batching, queueing, service overhead and availability margin.

## Verification

`verify` recomputes current fingerprints and validates:

- replay hashes;
- sealed run digests;
- task coverage;
- session evidence root;
- oracle policy and records;
- baseline numerical stability;
- candidate numerical agreement;
- statistical methodology and adjustment;
- selection thresholds;
- winner and profile identity.

It does not simply trust `current-profile.json`.

## Reporting

Reports disclose:

- numerical assurance level and tolerances;
- statistical point estimator, bootstrap, sample count and family-wise adjustment;
- fingerprints and evidence root;
- workload and selection policy;
- all candidate outcomes, failures and mismatches;
- adjusted replay interval;
- decode-only cost estimate;
- scientific and threat-model boundaries.

## Guarded deployment

```bash
python3 -m lattice env --workspace .lattice --format shell
python3 -m lattice launch --workspace .lattice -- chat
python3 -m lattice launch --workspace .lattice -- serve --host 127.0.0.1 --port 8000
```

Both commands verify the workspace immediately before use. Launch rejects model, context, memory, accelerator, policy, sampling and auto-tier overrides; disables Colibri's independent tune profile; and prevents chat from attaching to an unrelated server.

`--adaptive` is an explicit departure from the frozen measured environment.

## What v0.5 establishes

For the recorded fingerprints, workload, numerical policy and statistical policy, a completed qualification establishes that:

- candidates consumed the same forced-token path;
- throughput came from an uninstrumented pass;
- numerical validation occurred in a separate bounded-artifact pass;
- exact token identities and numerical sketches agreed within policy;
- paired workload ratios were evaluated under fixed workload weights;
- intervals preserved workload strata and adjusted across candidates;
- the winner cleared numerical, regression, gain and optional confidence gates;
- the result is recomputable from retained evidence.

It does not establish real-model performance until run with real weights and target hardware, universal speedup, complete logit equality, downstream-quality equivalence, calibrated cross-backend tolerances, cryptographic attestation or production SLA.

The adapter remains GLM-only. Inkling, Kimi K3 and OLMoE require dedicated replay and oracle adapters.

## Commercial path

The immediate product is workload-specific qualification and deployment evidence. Potential compounding assets include:

1. a normalized model × hardware × workload × configuration × drift corpus;
2. calibrated backend-specific numerical policies;
3. performance and feasibility prediction before hardware purchase;
4. cross-engine selection;
5. production drift detection and safe requalification;
6. fleet placement and procurement recommendations.

The next decisive milestone remains a sanitized real-model qualification on documented hardware, including repeated CPU/GPU numerical deltas and observed systems variance.
