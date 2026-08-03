# BATS-MoE research track

BATS-MoE is the byte-aware scheduling layer for Colibri's storage-resident MoE runtime. It is not a new router and it does not change model arithmetic.

## Hypothesis

For speculative verification and predictive prefetch, the useful cost is the *marginal exposed transfer time* of the union of required experts under the current placement—not expert count and not nominal FLOPs.

A route touching six already resident experts may be cheaper than a route requiring two cold NVMe experts. The decision must therefore change as residency, in-flight reads, queue pressure, bandwidth, and overlap change.

## What this first change contains

- `c/bats.h`: dependency-free, deterministic C primitives for calibrated transfer-time prediction, in-flight and resident experts, unique expert-union costing, budgeted candidate selection, and PILOT admission.
- `c/olmoe_bats.c`: exact OLMoE shadow runtime that logs predicted transfer cost beside measured expert-cache miss time without changing routing or arithmetic.
- `c/tests/test_bats.c`: synthetic falsification tests for the central claims.
- `c/tools/bats_replay.py`: independent JSONL trace oracle for offline analysis.
- `c/tools/bats_route_trace.py`: adapter from Colibri's existing `ROUTE_TRACE` stream to BATS shadow-planning records.
- `c/tools/bats_shadow_report.py`: calibration report with error, bias, correlation, effective observed bandwidth, and per-layer results.
- `.github/workflows/bats.yml`: builds and runs the isolated gates on Linux.

The stock binaries and default Colibri execution remain unchanged. The separate OLMoE binary is shadow-only: it observes and measures but does not yet make placement or speculative-verification decisions.

## Build and run the exact OLMoE shadow runtime

From `c/`:

```bash
make -f Makefile.bats olmoe-bats
```

With `BATS_SHADOW_OUT` unset, `olmoe_bats` delegates to the stock OLMoE entry point. Set it to enable exact shadow instrumentation:

```bash
SNAP=/models/olmoe \
BATS_SHADOW_OUT=/tmp/bats-shadow.jsonl \
BATS_NVME_GBPS=4.0 \
BATS_NVME_FIXED_US=80 \
BATS_NVME_QUEUE_US=20 \
BATS_NVME_OVERLAP=0.0 \
./olmoe_bats 16 8 ref.json
```

The process still performs the normal reference-token comparison. Shadow mode returns a non-zero status when generated tokens do not match the supplied reference. Each JSONL row records:

- layer and batch row;
- selected experts and routing gates;
- experts resident before execution;
- marginal cold bytes after batch-union reuse;
- predicted exposed transfer time;
- measured total `expert_get` time;
- measured time spent on actual cache misses.

Generate a calibration report:

```bash
python3 tools/bats_shadow_report.py /tmp/bats-shadow.jsonl --pretty
```

The report separates cache-only rows from miss rows and calculates overall and per-layer prediction ratio, mean error, MAE, RMSE, MAPE, Pearson correlation, observed time per miss, and an effective bandwidth that deliberately folds fixed latency and queueing into one descriptive rate.

`CHAT=1` currently delegates to stock chat and does not produce a shadow log. The reference and `PPL=1` experiment paths are instrumented first because they provide deterministic correctness or quality gates.

## Trace schema

Each planner JSONL record contains:

```json
{
  "hardware": {
    "nvme": {"bandwidth_gbps": 4.0, "fixed_us": 80, "queue_us": 20, "overlap": 0.2}
  },
  "experts": [
    {"id": 7, "bytes": 19000000, "tier": "nvme"},
    {"id": 8, "bytes": 19000000, "tier": "exec", "resident_in_exec": true}
  ],
  "candidates": [
    {"id": 0, "expected_accepted_tokens": 1.7, "verify_compute_us": 90, "experts": [7, 8]}
  ],
  "max_candidates": 1,
  "budget_us": 6000
}
```

Replay with:

```bash
python3 c/tools/bats_replay.py c/tests/bats_trace.jsonl --pretty
```

## Use existing Colibri route traces

Capture routing with the engine's existing telemetry:

```bash
ROUTE_TRACE=/tmp/routes.txt ...
```

Convert that stream to BATS records. `--n-experts` is the model's per-layer expert count; expert ids are globalized internally as `layer * n_experts + expert` so experts from different layers never collide.

```bash
python3 c/tools/bats_route_trace.py /tmp/routes.txt \
  --n-experts 64 \
  --expert-bytes 19000000 \
  --default-tier nvme \
  --bandwidth-gbps 4.0 \
  --fixed-us 80 \
  --queue-us 20 \
  > /tmp/bats.jsonl

python3 c/tools/bats_replay.py /tmp/bats.jsonl
```

An optional residency file records the current snapshot as:

```text
# layer expert tier [remaining_us]
2 17 exec
2 31 nvme 340.0
```

Pass it with `--residency`. A fourth value means the load is in flight and supplies its predicted remaining time.

For speculative-decoding analysis, pass `--acceptance` with JSONL annotations:

```json
{"call": 3, "row": 0, "expected_accepted_tokens": 1.7, "verify_compute_us": 90, "kv_us": 12}
```

Without an acceptance file, the adapter deliberately assigns every row a benefit of `1.0`. That mode is valid only for transfer-cost shadow analysis; it must not be presented as an acceptance-aware speculative-decoding result.

## Runtime integration sequence

1. Calibrate the transfer model from exact OLMoE shadow logs.
2. Snapshot in-flight queue state rather than treating every non-resident expert as a fresh NVMe read.
3. Emit per-candidate expert unions and acceptance estimates from the existing MTP verification path.
4. Run BATS in the production engine's in-process shadow mode and compare its proposed decisions with measured outcomes.
5. Enable opt-in `BATS=1` only after shadow ranking is stable on held-out workloads.
6. A/B against current PILOT/LFRU on cold and warm cache states.

## Required measurements

Every result must include commit, model/container, quantization, hardware, prompt set, cache state, draft depth, acceptance, expert hit rate, bytes read, TTFT, TPOT, throughput, and token/quality equivalence.

The first success threshold is not a headline speed claim. It is:

- transfer-time prediction error low enough to rank alternatives reliably;
- fewer NVMe bytes per accepted token than expert-count selection;
- no semantic difference in exact mode;
- a controlled end-to-end win over the strongest existing policy.
