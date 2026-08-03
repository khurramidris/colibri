# BATS-MoE research track

BATS-MoE is Colibri's byte-aware scheduling research layer for storage-resident mixture-of-experts inference. It does not replace the router and the exact path does not change model arithmetic.

## Hypothesis

For speculative verification and predictive prefetch, the useful cost is the *marginal exposed transfer time* of the union of required experts under the current placement—not expert count or nominal FLOPs.

A route touching six already resident experts may be cheaper than one requiring two cold NVMe experts. The decision must therefore change with residency, in-flight work, queue pressure, measured bandwidth, and overlap.

## Components

- `c/bats.h`: dependency-free deterministic C cost model, union-aware candidate planner, and prefetch-admission primitive.
- `c/olmoe_bats.c`: exact OLMoE shadow runtime. It logs predicted expert-transfer cost beside measured cache-miss time without changing routing or arithmetic.
- `c/tools/bats_calibrate.py`: target-machine storage profiler and effective fixed-latency/bandwidth fit.
- `c/tools/bats_shadow_report.py`: overall and per-layer calibration report.
- `c/tools/bats_replay.py`: independent JSONL planning oracle.
- `c/tools/bats_route_trace.py`: adapter from the existing Colibri `ROUTE_TRACE` format.
- `c/tests/test_bats*.{c,py}`: deterministic falsification, parser, replay, calibration, and reporting tests.
- `.github/workflows/bats.yml`: isolated Linux gate, including compilation of the exact OLMoE shadow binary.

The stock Colibri binaries and default execution path remain unchanged. The alternate OLMoE binary is currently observational: it measures but does not yet alter placement, prefetch admission, or speculative verification.

## 1. Calibrate the target storage path

Run the profiler against a large model or expert file located on the same storage device used for inference:

```bash
cd c
python3 tools/bats_calibrate.py /models/olmoe/model-00001-of-00008.safetensors \
  --sizes 64KiB,1MiB,8MiB,32MiB \
  --samples 9 \
  --cold-hint \
  --pretty
```

To emit environment variables:

```bash
eval "$(python3 tools/bats_calibrate.py \
  /models/olmoe/model-00001-of-00008.safetensors \
  --cold-hint --shell)"
```

The resulting profile is an effective single-read model:

```text
time_us = fixed_us + bytes / (bandwidth_gbps * 1000)
```

The profiler reports medians and tail samples at multiple read sizes. `POSIX_FADV_DONTNEED` is only a cache-drop hint, not proof of physical-media reads. Queue delay and useful overlap must be learned from the in-engine shadow trace under realistic concurrency.

## 2. Build and run the exact OLMoE shadow runtime

```bash
make -f Makefile.bats olmoe-bats
```

With `BATS_SHADOW_OUT` unset, `olmoe_bats` delegates to the stock OLMoE entry point. Set it to enable instrumentation:

```bash
SNAP=/models/olmoe \
BATS_SHADOW_OUT=/tmp/bats-shadow.jsonl \
./olmoe_bats 16 8 ref.json
```

Optional overrides:

```bash
BATS_NVME_GBPS=4.0
BATS_NVME_FIXED_US=80
BATS_NVME_QUEUE_US=20
BATS_NVME_OVERLAP=0.0
```

The reference path still compares generated and expected token IDs. The custom shadow path exits non-zero on mismatch. `PPL=1` is also instrumented. `CHAT=1` currently delegates to stock chat and does not emit a shadow log.

Each JSONL row contains:

- token position, layer, and batch row;
- selected experts and gates;
- experts resident before execution;
- marginal cold bytes after union reuse;
- predicted exposed transfer time;
- measured total `expert_get` time;
- measured time and count for actual cache misses.

## 3. Measure calibration quality

```bash
python3 tools/bats_shadow_report.py /tmp/bats-shadow.jsonl --pretty
```

The report includes prediction-to-actual ratio, mean error, MAE, RMSE, MAPE, Pearson correlation, observed time per miss, descriptive effective bandwidth, and per-layer breakdowns.

These statistics validate the *cost model*. They do not establish an end-to-end BATS speedup. That requires an active policy A/B test with identical model, quantization, prompts, cache state, and correctness gates.

## 4. Replay planner records

A planner JSONL record contains hardware costs, expert states, candidate expert sets, expected accepted tokens, and an optional budget. Experts in flight may report both `remaining_us` and `remaining_bytes`; BATS charges only the remaining work.

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

```bash
python3 tools/bats_replay.py tests/bats_trace.jsonl --pretty
```

## 5. Convert existing route traces

Capture routing with the normal engine:

```bash
ROUTE_TRACE=/tmp/routes.txt ...
```

Convert it:

```bash
python3 tools/bats_route_trace.py /tmp/routes.txt \
  --n-experts 64 \
  --expert-bytes 19000000 \
  --default-tier nvme \
  --bandwidth-gbps 4.0 \
  --fixed-us 80 \
  --queue-us 20 \
  > /tmp/bats-planner.jsonl

python3 tools/bats_replay.py /tmp/bats-planner.jsonl
```

The adapter globalizes expert IDs as `layer * n_experts + expert`, preventing experts in different layers from colliding.

An optional residency snapshot uses:

```text
# layer expert tier [remaining_us]
2 17 exec
2 31 nvme 340.0
```

For speculative-decoding analysis, optional JSONL acceptance annotations use:

```json
{"call": 3, "row": 0, "expected_accepted_tokens": 1.7, "verify_compute_us": 90, "kv_us": 12}
```

Without acceptance annotations, every row receives a benefit of `1.0`. That is suitable only for transfer-cost shadow analysis and must not be reported as acceptance-aware speculative decoding.

## Integration sequence

1. Calibrate transfer cost on real OLMoE shadow traces.
2. Expose exact in-flight queue state and remaining work from the runtime.
3. Emit candidate unions, acceptance estimates, and verification timing from the MTP path.
4. Run BATS proposals in-process in shadow mode against measured outcomes.
5. Enable an opt-in active policy only after held-out ranking accuracy is stable.
6. A/B against current PILOT/LFRU under cold and warm cache states.

## Required evidence

Every performance result must record commit, model/container, quantization, hardware, prompt set, cache state, draft depth, acceptance, expert hit rate, bytes read, TTFT, TPOT, throughput, and token/quality equivalence.

The first success criteria are:

- transfer-cost ranking accurate enough to choose alternatives reliably;
- fewer NVMe bytes per accepted token than expert-count selection;
- no semantic difference in exact mode;
- a controlled end-to-end improvement over the strongest existing policy.
