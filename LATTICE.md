# Lattice Qualification Plane

Lattice is the commercial qualification and deployment layer being built on this Colibri fork.

Colibri is the inference engine: it moves frontier MoE weights across NVMe, RAM and VRAM, exposes hardware-aware planning, and provides an opt-in numerical replay oracle. Lattice turns those capabilities into a customer-facing qualification artifact:

> Given a model, a machine and a representative workload, determine which tested execution configuration is suitable to deploy, what it costs, and what retained evidence supports that decision.

The current implementation lives in [`lattice/`](lattice/) and is intentionally narrow. It does **not** claim a universal speedup, complete logit equality, semantic equivalence or downstream model-quality validation. It provides:

- deep Colibri preflight and plan capture for the currently proven GLM/`colibri` adapter;
- sampled primary/split/mirror model-payload, runtime, hardware and controlled-environment fingerprints;
- deterministic forced-token replay across a weighted workload suite;
- an uninstrumented timed pass followed by a separate numerical-validation pass;
- exact forced-token, top-1, top-2 and ordered top-k token-ID checks;
- non-finite rejection plus bounded comparisons of selected logits, moments and four deterministic full-vector projections;
- a controller-created private oracle artifact independently capped at 4 MiB;
- an explicit allowlist for reviewed qualification environment controls;
- topology-aware execution candidates and rotated order across workloads/repeats;
- SHA-256-sealed run records and a completed-session evidence root;
- paired per-repeat throughput ratios inside each workload;
- workload-stratified bootstrap intervals that preserve declared workload weights;
- Bonferroni adjustment across all non-baseline candidate comparisons;
- at least three paired repeats for confidence-gated promotion;
- per-workload regression and aggregate gain gates;
- decode-only cost estimates from an operator-supplied hourly cost;
- a deployment profile bound to evidence, numerical-oracle, statistical and selection policies;
- recomputed verification, guarded environment export and launch;
- an explicit customer-readable qualification report.

## Assurance level

Lattice v0.5 measures candidates while forcing the same prompt and continuation token IDs through each run. Colibri first performs an uninstrumented timed replay. It then resets KV state and performs a separate numerical-validation replay. Detailed oracle rows are written to a private bounded artifact; stdout receives only a compact publication marker.

This is **numerical replay consistency**, not complete logit equality. The numerical tolerances are explicit and versioned but still require calibration against real supported CPU/GPU deployments.

Performance promotion uses the median paired candidate/baseline ratio for each workload, then applies the declared workload weights through a geometric aggregate. Bootstrap resampling occurs independently within each workload, so a workload with more observations cannot silently dominate the interval. The requested family-wise confidence is adjusted across the complete candidate family using Bonferroni correction.

This remains screening methodology rather than a complete benchmarking study. Thermal stabilization, formal power analysis, robust outlier modelling, multiple-machine replication and real-backend tolerance calibration remain open work.

The evidence files are locally tamper-evident, not cryptographically immutable. Coordinated rewriting by a malicious filesystem administrator remains outside the current threat model.

## Demo

```bash
python3 -m lattice init \
  --repo . \
  --model /models/glm52_i4 \
  --suite examples/lattice-workload.json \
  --workspace .lattice

python3 -m lattice qualify --workspace .lattice --repeats 3
python3 -m lattice qualify --workspace .lattice --resume <session-id>

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
python3 -m lattice env --workspace .lattice --format shell
python3 -m lattice launch --workspace .lattice -- serve --port 8000
```

The v0.5 qualification adapter remains GLM-only. Inkling, Kimi K3 and OLMoE require dedicated replay and numerical-oracle adapters before Lattice will claim qualification coverage.

Read [`docs/lattice-qualification.md`](docs/lattice-qualification.md) for the complete architecture, evidence, numerical and statistical contract.
