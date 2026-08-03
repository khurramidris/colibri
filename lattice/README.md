# Lattice Runtime

Lattice is an experimental **storage-native exact inference substrate** inside this Colibri fork.
It treats NVMe, RAM and VRAM as one planned hierarchy, then schedules tensor movement through
three paths: demand, proactive prediction and low-priority speculation.

This directory is intentionally self-contained. Existing Colibri engines are not changed by
this first milestone, so no new code can silently alter their model arithmetic.

## What exists now

- A dependency-free C library for exact tensor-class placement across NVMe, RAM and VRAM.
- Hard feasibility rules, memory reserves and deterministic plans.
- Predicted storage bytes per token and transfer/compute latency accounting.
- A three-path tensor transfer scheduler with deduplication and request promotion.
- Request-specific expert signatures derived from prefill routing observations.
- A 4096-byte-aligned, block-addressable native MXFP4 expert file format.
- A resumable packer that preserves native packed bytes without dequantizing.
- A safetensors header scanner that creates model manifests without loading weights.
- Reproducible RAM/VRAM Pareto sweeps.
- C unit tests, Python fixture tests, Linux/macOS CI and ASan/UBSan.

This is the measurement and execution-control foundation. It is **not yet evidence of an
end-to-end model speedup**. That requires model adapters, real weights and controlled hardware
runs under the protocol in `docs/EXPERIMENT_PROTOCOL.md`.

## Build and test

```bash
make -C lattice check
python3 -m unittest discover -s lattice/tests -p 'test_*.py' -v
```

Run the synthetic constrained-MoE example:

```bash
make -C lattice demo
```

Plan the included manifest and write machine-readable output:

```bash
lattice/build/lattice-plan \
  --manifest lattice/examples/small-moe.tsv \
  --ram-gib 8 \
  --vram-gib 4 \
  --nvme-gbps 5 \
  --json lattice/build/plan.json
```

## Scan a real safetensors checkpoint

The scanner reads only model headers and `config.json`:

```bash
python3 lattice/tools/scan_safetensors.py /models/my-moe \
  --kv-cache-gib 2 \
  -o lattice/build/my-moe.tsv

lattice/build/lattice-plan \
  --manifest lattice/build/my-moe.tsv \
  --ram-gib 16 --vram-gib 8 --nvme-gbps 7
```

The generated costs are a starting model. Measured CPU/GPU costs and real route traces must
replace defaults before performance claims are made.

## Generate a memory/latency frontier

```bash
python3 lattice/tools/sweep.py \
  --manifest lattice/build/my-moe.tsv \
  --ram-gib 4,8,16,32 \
  --vram-gib 0,4,8,16 \
  --output lattice/build/frontier.csv
```

## Storage-aware sub-expert blocks

`pack_expert_blocks.py` creates one fixed-record file per sparse layer. Every record contains:

1. the selected W1 output rows,
2. their native scale bytes,
3. the matching W3 output rows and scales,
4. the corresponding W2 input columns and scales,
5. zero padding to a 4096-byte boundary.

Adjacent selected blocks coalesce into a single aligned read. A skipped block therefore avoids
storage, host-memory, device-transfer and compute work together.

Dry-run the geometry first:

```bash
python3 lattice/tools/pack_expert_blocks.py \
  /models/Kimi-K3 /models/Kimi-K3-lattice-blocks \
  --layers 3 --block-channels 256 --dry-run
```

Then pack a layer:

```bash
python3 lattice/tools/pack_expert_blocks.py \
  /models/Kimi-K3 /models/Kimi-K3-lattice-blocks \
  --layers 3 --block-channels 256
```

The packer is resumable and fails closed when tensor geometry, dtype, offsets or output size do
not match the expected native MXFP4 layout.

## Exact and adaptive modes are separate

### Exact mode

Placement, caching, prefetching, overlap and speculation may change latency only. The original
router decisions, weight bytes and arithmetic remain unchanged. Incorrect predictions become
cache misses, not incorrect tokens.

### Adaptive mode

Token-level channel-block selection intentionally changes computation. It must use separate
quality gates, model identifiers and benchmark reports. Adaptive results must never be presented
as byte-identical execution of the unmodified model.

## Near-term gates

1. **Planner calibration:** predicted bytes and transfer time match measured values within a
   declared error band.
2. **Small-MoE exact adapter:** identical greedy tokens and bounded logit error under several
   forced RAM/VRAM budgets.
3. **Predictive movement:** fewer synchronous miss bytes with no semantic change.
4. **Block format kernel:** partial native-MXFP4 expert output matches the corresponding slices
   of full expert execution.
5. **Adaptive research:** real end-to-end speedup after router overhead, with a published quality
   frontier.
6. **Second architecture:** demonstrate that the scheduler is not a one-model trick.
7. **Kimi K3:** only after the earlier gates pass.

## Directory map

```text
lattice/
├── include/lattice/       public C APIs
├── src/                   planner, scheduler and research primitives
├── tests/                 dependency-free and synthetic-fixture gates
├── tools/                 model scanning, packing and Pareto sweeps
├── examples/              transparent manifests
└── docs/                  architecture and experiment protocol
```

## Scientific rule

No optimization is accepted because it sounds plausible. It must expose a mechanism, a baseline,
a correctness boundary and a reproducible measurement. Failed hypotheses remain part of the
record.
