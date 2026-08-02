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
- a controller-created private oracle artifact, independently capped at 4 MiB, so long validation traces do not weaken the 64 KiB process-output limit;
- a compact row representation tested with a full 2,048-step replay artifact;
- an explicit allowlist for reviewed qualification environment controls;
- hardware-specific scheduling and placement candidates only;
- rotated candidate order to reduce simple warm-cache/order bias;
- SHA-256-sealed run records and a session evidence root, including failed attempts;
- per-workload regression gates and paired bootstrap confidence intervals;
- decode-only cost-per-million-token estimates from an operator-supplied hourly cost;
- a deployment profile bound to the evidence root, numerical-oracle policy and selection policy;
- verification that recomputes the winner from retained run evidence;
- a customer-readable Markdown qualification report.

## Assurance level

Lattice v0.4 measures candidates while forcing the same prompt and continuation token IDs through each run. Colibri first performs an uninstrumented timed replay. It then resets KV state and performs a separate numerical-validation replay. Detailed oracle rows are written to a private bounded artifact; stdout receives only a compact publication marker.

This is **numerical replay consistency**, not complete logit equality. Exact token identities eliminate the earlier top-k hash-collision ambiguity, but the remaining numerical sketch cannot prove that every logit matches, that free-running generations are identical, or that downstream task quality is unchanged. The absolute and relative tolerances are explicit and versioned, but still require calibration against real supported CPU/GPU deployments.

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

The v0.4 qualification adapter remains GLM-only. Inkling, Kimi K3 and OLMoE require dedicated replay and numerical-oracle adapters before Lattice will claim qualification coverage.

Read [`docs/lattice-qualification.md`](docs/lattice-qualification.md) for the broader architecture, evidence contract and limitations. PR #3 documents the v0.4 exact-ID and bounded-artifact delta while that detailed document is consolidated after the dependent PR stack is merged.
