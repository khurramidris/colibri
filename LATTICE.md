# Lattice Qualification Plane

Lattice is the commercial product layer being built on this Colibri fork.

Colibri is the inference engine: it moves frontier MoE weights across NVMe, RAM and VRAM, exposes hardware-aware planning, and provides deterministic replay telemetry. Lattice turns those capabilities into a customer-facing qualification artifact:

> Given a model, a machine and a representative workload, prove which quality-preserving configuration is safe to deploy, what it costs, and what evidence supports that decision.

The first implementation lives in [`lattice/`](lattice/) and is intentionally narrow. It does **not** claim a new kernel or universal speedup. It provides:

- deep Colibri preflight and plan capture for the currently proven GLM/`colibri` adapter;
- sampled primary/split/mirror model-payload, runtime, hardware and controlled-environment fingerprints;
- deterministic, teacher-forced replay across a weighted workload suite;
- hardware-specific scheduling and placement candidates only;
- rotated candidate order to reduce warm-cache/order bias;
- immutable run evidence, including failures;
- per-workload regression gates and paired bootstrap confidence intervals;
- decode-only cost-per-million-token estimates from an operator-supplied hourly cost;
- an immutable promoted deployment profile;
- verification that recomputes the winner from raw run evidence;
- a customer/investor-readable Markdown qualification report.

## Demo

```bash
python3 -m lattice init \
  --repo . \
  --model /models/glm52_i4 \
  --suite examples/lattice-workload.json \
  --workspace .lattice

python3 -m lattice qualify --workspace .lattice --repeats 3

# If calibration or a replay run is interrupted, resume the same evidence session:
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

# Inspect or launch the exact verified profile:
python3 -m lattice env --workspace .lattice --format shell
python3 -m lattice launch --workspace .lattice -- serve --port 8000
```

The v0.1 qualification adapter is intentionally GLM-only. Inkling, Kimi K3 and OLMoE support requires dedicated deterministic replay adapters before Lattice will claim qualification coverage.

Read [`docs/lattice-qualification.md`](docs/lattice-qualification.md) for the architecture, evidence contract and limitations.
