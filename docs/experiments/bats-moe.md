# BATS-MoE research track

BATS-MoE is the byte-aware scheduling layer for Colibri's storage-resident MoE runtime. It is not a new router and it does not change model arithmetic.

## Hypothesis

For speculative verification and predictive prefetch, the useful cost is the *marginal exposed transfer time* of the union of required experts under the current placement—not expert count and not nominal FLOPs.

A route touching six already resident experts may be cheaper than a route requiring two cold NVMe experts. The decision must therefore change as residency, in-flight reads, queue pressure, bandwidth, and overlap change.

## What this first change contains

- `c/bats.h`: dependency-free, deterministic C primitives for calibrated transfer-time prediction, in-flight and resident experts, unique expert-union costing, budgeted candidate selection, and PILOT admission.
- `c/tests/test_bats.c`: synthetic falsification tests for the central claims.
- `c/tools/bats_replay.py`: independent JSONL trace oracle for offline analysis.
- `.github/workflows/bats.yml`: builds and runs the isolated gate on Linux.

This is intentionally an isolated foundation. Default Colibri execution remains unchanged until a runtime integration can be measured against the existing PILOT and LFRU policies.

## Trace schema

Each JSONL record contains:

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

## Runtime integration sequence

1. Emit per-candidate expert unions from the existing MTP verification path.
2. Snapshot expert tier, resident state, in-flight state, bytes, and queue delay.
3. Run BATS in shadow mode and log its decision without changing execution.
4. Compare predicted cost with measured cost and calibrate the hardware profile.
5. Enable `BATS=1` only after shadow replay is stable.
6. A/B against current PILOT/LFRU on held-out prompts and cold/warm cache states.

## Required measurements

Every result must include commit, model/container, quantization, hardware, prompt set, cache state, draft depth, acceptance, expert hit rate, bytes read, TTFT, TPOT, throughput, and token/quality equivalence.

The first success threshold is not a headline speed claim. It is:

- transfer-time prediction error low enough to rank alternatives reliably;
- fewer NVMe bytes per accepted token than expert-count selection;
- no semantic difference in exact mode;
- a controlled end-to-end win over the strongest existing policy.
