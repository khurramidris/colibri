# Lattice Qualification Plane

Lattice is the commercial qualification and deployment layer being built on this Colibri fork.

Colibri is the inference engine: it moves frontier MoE weights across NVMe, RAM and VRAM, exposes hardware-aware planning, and provides deterministic replay telemetry. Lattice turns those capabilities into a customer-facing qualification artifact:

> Given a model, a machine and a representative workload, determine which tested execution configuration is suitable to deploy, what it costs, and what retained evidence supports that decision.

The current implementation lives in [`lattice/`](lattice/) and is intentionally narrow. It does **not** claim a new kernel, universal speedup, logit equivalence or model-quality validation. It provides:

- deep Colibri preflight and plan capture for the currently proven GLM/`colibri` adapter;
- sampled primary/split/mirror model-payload, runtime, hardware and controlled-environment fingerprints;
- deterministic, forced-token replay across a weighted workload suite;
- an explicit allowlist for reviewed qualification environment controls;
- hardware-specific scheduling and placement candidates only;
- rotated candidate order to reduce simple warm-cache/order bias;
- SHA-256-sealed run records and a session evidence root, including failed attempts;
- per-workload regression gates and paired bootstrap confidence intervals;
- decode-only cost-per-million-token estimates from an operator-supplied hourly cost;
- a deployment profile whose identity is bound to the exact evidence root and selection policy;
- verification that recomputes the winner from retained run evidence;
- a customer-readable Markdown qualification report.

## What the assurance level means

Lattice v0.2 measures candidates while forcing the same prompt and continuation token IDs through each run. This controls generation randomness and makes execution timing more comparable. It does **not** prove equal logits, probabilities, numerical error, router decisions or downstream task quality. Until Colibri exposes an appropriate numerical oracle for qualification runs, reports should describe this as **deterministic replay consistency**, not semantic equivalence.

The evidence files are locally tamper-evident, not cryptographically immutable. Each run has a digest and each completed session has a root binding its run digests, replay hashes and session definition. Coordinated rewriting of ordinary local files by a malicious administrator is outside the current threat model.

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

The v0.2 qualification adapter is intentionally GLM-only. Inkling, Kimi K3 and OLMoE support requires dedicated deterministic replay adapters before Lattice will claim qualification coverage.

Read [`docs/lattice-qualification.md`](docs/lattice-qualification.md) for the architecture, evidence contract and limitations.
