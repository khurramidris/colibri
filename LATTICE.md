# Lattice Qualification Plane

Lattice is the commercial qualification and deployment layer being built on this Colibri fork.

Colibri is the inference engine: it moves frontier MoE weights across NVMe, RAM and VRAM, exposes hardware-aware planning, and provides replay instrumentation. Lattice turns those capabilities into customer-facing acceptance and qualification evidence.

> Given a model, a machine and a representative workload, determine whether the backend genuinely executes, which tested configuration is suitable to deploy, what it costs, and what retained evidence supports that decision.

The current implementation lives in [`lattice/`](lattice/) and is intentionally narrow. It does **not** claim a universal speedup, complete logit equality, semantic equivalence or downstream model-quality validation.

## Two evidence levels

### OLMoE real-model acceptance

`accept-olmoe` is the first manageable real-checkpoint gate. It:

- validates a converted OLMoE model and independent Transformers reference;
- fingerprints the model, runtime, machine/storage identity and reference;
- runs the native OLMoE C engine repeatedly under fixed conservative controls;
- requires exact continuation-token agreement on every run;
- captures load time, RSS, throughput and expert-cache telemetry;
- preserves sealed successful and failed run evidence;
- emits immutable JSON evidence and a Markdown acceptance report.

This establishes **token-exact reference replay for one recorded continuation**. It does not select or promote an optimized configuration.

```bash
python3 -m lattice accept-olmoe \
  --repo . \
  --model /models/olmoe-merged \
  --reference ./olmoe-reference.json \
  --repeats 3 \
  --output ./evidence/olmoe-acceptance.json
```

See [`docs/olmoe-acceptance.md`](docs/olmoe-acceptance.md).

### GLM numerical qualification

The GLM/`colibri` qualification adapter provides:

- deep Colibri preflight and plan capture;
- sampled primary/split/mirror model-payload, runtime, hardware and controlled-environment fingerprints;
- deterministic forced-token replay across a weighted workload suite;
- an uninstrumented timed pass followed by a separate numerical-validation pass;
- exact forced-token, top-1, top-2 and ordered top-k token-ID checks;
- non-finite rejection plus bounded comparisons of selected logits, moments and deterministic full-vector projections;
- a controller-created private oracle artifact independently capped at 4 MiB;
- topology-aware execution candidates and rotated order across workloads/repeats;
- SHA-256-sealed run records and a completed-session evidence root;
- exact per-repeat throughput pairing inside each workload;
- workload-stratified bootstrap intervals preserving declared workload weights;
- Bonferroni adjustment across non-baseline candidates;
- at least three paired repeats for confidence-gated promotion;
- per-workload regression and aggregate gain gates;
- an evidence-bound deployment profile, recomputed verification and guarded launch.

The qualification assurance level is **numerical replay consistency**, not complete logit equality. Numerical tolerances still require calibration against real supported CPU/GPU deployments.

```bash
python3 -m lattice init \
  --repo . \
  --model /models/glm52_i4 \
  --suite examples/lattice-workload.json \
  --workspace .lattice

python3 -m lattice qualify --workspace .lattice --repeats 3
python3 -m lattice recommend \
  --workspace .lattice \
  --session <session-id> \
  --min-runs 3 \
  --min-gain 0.03 \
  --max-regression 0.05 \
  --confidence 0.90 \
  --require-confidence
python3 -m lattice verify --workspace .lattice --deep
python3 -m lattice report --workspace .lattice --output qualification-report.md
```

## Scientific boundary

Performance promotion uses paired repeat ratios within each workload, a workload-weighted geometric aggregate, stratified bootstrap resampling and a family-wise candidate adjustment. This remains screening methodology rather than a complete benchmarking study. Thermal stabilization, formal power analysis, robust outlier modelling, multiple-machine replication and real-backend tolerance calibration remain open work.

The evidence files are locally tamper-evident, not cryptographically immutable. Coordinated rewriting by a malicious filesystem administrator remains outside the current threat model.

The full numerical qualification adapter remains GLM-only. OLMoE now has the narrower acceptance path; Inkling and Kimi K3 still require dedicated acceptance and replay-oracle adapters before Lattice will claim coverage.

Read [`docs/lattice-qualification.md`](docs/lattice-qualification.md) for the complete numerical and statistical qualification contract.
