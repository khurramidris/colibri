# Experiment Protocol

This protocol exists to prevent an attractive systems demo from being mistaken for a validated
inference result.

## 1. Pre-register the claim

Before running an optimization, record:

- hypothesis,
- mechanism expected to cause the improvement,
- primary metric,
- secondary metrics,
- correctness boundary,
- baseline commit and configuration,
- target commit and configuration,
- hardware and thermal conditions,
- success and failure thresholds.

A result that changes its primary metric after seeing the data is exploratory, not confirmatory.

## 2. Freeze artifacts

Every report must identify:

- model repository and exact checkpoint revision,
- hashes of configuration, tokenizer and conversion manifests,
- runtime commit,
- compiler and flags,
- kernel/backend versions,
- generated placement plan,
- block-pack receipts,
- prompt corpus revision,
- route trace and random seed.

Large weight files may use published shard hashes instead of duplicating them.

## 3. Hardware disclosure

Record at minimum:

- CPU model, physical cores, SMT state and frequency policy,
- RAM capacity, channel configuration and enforced process/cgroup limit,
- GPU model, driver, VRAM and power limit,
- PCIe generation and negotiated link width,
- every storage device, filesystem and mount option,
- whether files share a physical device,
- ambient/initial temperature when throttling is plausible,
- operating system and kernel.

“Consumer workstation” is not a hardware specification.

## 4. Memory enforcement

A low-memory claim must enforce the limit rather than merely observe a small initial allocation.
Use an operating-system mechanism such as a cgroup/job limit where available.

Report:

- peak process RSS,
- cgroup/job peak memory,
- GPU peak allocation and driver-reported use,
- filesystem page-cache behavior,
- whether direct I/O was used,
- swap configuration and swap traffic.

Cold-cache and warm-cache runs are separate experiments.

## 5. Timing boundaries

Measure and report separately:

- model initialization,
- repacking/conversion,
- prompt tokenization,
- prefill,
- time to first generated token,
- steady-state decode,
- synchronous storage stall,
- total storage read time,
- host-to-device transfer,
- attention/recurrent computation,
- MoE computation,
- output head,
- shutdown/persistence.

Do not divide prompt processing into decode throughput or hide conversion in a one-time setup
without saying so.

## 6. Byte accounting

For storage-native inference, primary physical metrics include:

- bytes requested from each storage tier per token,
- bytes actually completed by each device,
- read amplification,
- bytes copied into RAM,
- bytes transferred over PCIe,
- cache hit/miss counts and byte-weighted hit rate,
- proactive precision and recall,
- speculative waste,
- demand requests delayed by predictions.

Theoretical FLOPs are secondary when the workload is movement-bound.

## 7. Correctness gates

### Exact mode

Required gates:

1. tensor-container and scale interpretation tests,
2. layer-local hidden-state comparisons,
3. teacher-forced logit comparisons,
4. greedy-token equality over a fixed corpus,
5. generation equality across memory budgets,
6. state-carrying incremental-decode equality,
7. checks for NaN/Inf and deterministic repeatability.

Tolerances must be declared before evaluation. “Looks correct” is not a gate.

### Adaptive mode

Adaptive mode changes computation and must report:

- benchmark quality at every sparsity/budget point,
- calibration and evaluation corpora separated,
- router overhead,
- per-layer and cumulative output error,
- task categories that regress,
- fallback rate to full width,
- model identity clearly distinguished from exact mode.

## 8. Baselines

Use the strongest relevant and reproducible baselines, not only an earlier internal build.
For the same model and hardware compare where applicable:

- upstream Colibri configuration,
- Fareed-style low-memory configuration,
- fully resident execution,
- demand-only streaming,
- standard LRU/LFU cache,
- prediction disabled,
- block execution disabled,
- CPU-only and available accelerator paths.

Each ablation changes one mechanism at a time.

## 9. Run design

- At least one warm-up run that is not reported as a sample.
- At least five measured runs for stable microbenchmarks.
- At least three full-model runs when each run is very expensive.
- Report median, range and a dispersion statistic.
- Randomize configuration order where temperature or cache state could bias results.
- Keep prompt order fixed or explicitly randomized with a recorded seed.
- Sustain decode long enough to expose cache cycling and thermal throttling.

## 10. Small-model hypothesis ladder

### Gate A — synthetic components

- planner feasibility and monotonicity,
- scheduler priority/deduplication,
- exact block-format round trips,
- storage ranges and bytes checked against fixtures.

### Gate B — one real MoE layer

- full expert versus sum of all block contributions,
- selected-block output versus reference partial contribution,
- actual partial read bytes,
- router overhead isolated.

### Gate C — several consecutive layers

- hidden-state error accumulation,
- transfer/computation overlap,
- prediction precision and miss latency.

### Gate D — full smaller MoE

- exact execution under multiple enforced budgets,
- memory/latency Pareto frontier,
- predictive exact mode,
- adaptive quality/speed frontier.

### Gate E — second architecture

Port without redesigning the planner or scheduler interfaces. Architecture-specific kernels are
allowed; architecture-specific control-plane assumptions must be documented.

### Gate F — Kimi K3

Only after Gates A–E pass. The Kimi result must include exact and adaptive modes separately.

## 11. Minimum breakthrough thresholds

These are project targets, not current claims.

### Exact runtime

At least one of:

- 2× decode throughput under the same enforced memory budget,
- 50% lower required fast memory at comparable speed,
- 40% lower storage bytes per token with a meaningful end-to-end gain.

### Adaptive block execution

All of:

- at least 35% lower expert weight traffic,
- at least 1.5× end-to-end decode acceleration after router overhead,
- negligible or explicitly bounded quality loss on held-out coding, reasoning and language tasks,
- no reliance on one prompt family.

Failure to meet these thresholds is still useful research if the mechanism is measured honestly.

## 12. Publication table

Every public benchmark table should contain:

| Field | Required |
|---|---|
| Hardware and power limits | Yes |
| Model/checkpoint hash | Yes |
| Runtime commit | Yes |
| Exact/adaptive label | Yes |
| RAM/VRAM limits and peaks | Yes |
| Cold/warm cache state | Yes |
| Prompt and generated token counts | Yes |
| TTFT, prefill and decode | Yes |
| Storage and PCIe bytes/token | Yes |
| Correctness/quality gate | Yes |
| Repetitions and variance | Yes |

## 13. Claim language

Allowed after evidence:

- “On the disclosed system and workload, configuration X reduced measured NVMe bytes/token by Y.”
- “Exact mode produced identical greedy tokens on corpus Z under budgets A–D.”
- “Adaptive mode achieved this quality/speed frontier.”

Not allowed:

- universal speed claims from one machine,
- kernel speedups presented as end-to-end speedups,
- approximate execution described as the unmodified model,
- extrapolated Kimi performance from a smaller model,
- peak device bandwidth used as measured application bandwidth.
