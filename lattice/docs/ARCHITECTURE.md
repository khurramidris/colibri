# Lattice Architecture

## Thesis

Sparse-model runtimes usually optimize model arithmetic, expert caching, device placement and
storage I/O as separate subsystems. Lattice tests a different proposition:

> The useful unit of inference optimization is the complete path a tensor takes from persistent
> storage to arithmetic, under a token-specific probability of use.

The system therefore plans and measures bytes as carefully as FLOPs.

## Invariants

1. **Exact placement is semantics-preserving.** Moving a tensor between tiers must not change its
   bytes, shape, quantization interpretation, routing or arithmetic.
2. **Predictions cannot affect correctness.** A bad prediction produces an urgent read, not an
   altered token.
3. **Adaptive execution is a different mode.** Width pruning is explicitly approximate and has
   separate model identities and quality reports.
4. **Plans are inspectable.** Every allocation, predicted byte count and assumed bandwidth is
   serializable.
5. **Unsupported plans fail closed.** An adapter must reject placements it cannot actually enact.
6. **Measurement precedes optimization.** A planner parameter is calibrated from a probe or a
   trace, never selected only because it gives an attractive estimate.

## System split

### Control plane

The control plane runs infrequently and may be more expensive:

- inventory checkpoint tensors,
- characterize hardware,
- ingest route traces,
- estimate tensor reuse and execution costs,
- produce a placement plan,
- generate runtime limits and block-layout manifests,
- compare measured performance with predictions.

### Data plane

The data plane runs for every request and token:

- record prefill routing evidence,
- submit proactive tensor reads,
- promote predicted reads when routing confirms demand,
- fill unused bandwidth with bounded speculation,
- cancel stale speculative work,
- execute exact tensors from the selected tier,
- record bytes, stalls, hits and timing.

The first implementation keeps these components small and embeddable in C runtimes.

## Exact placement planner

The planner operates on tensor classes. A class describes equally sized tensors ordered from
hottest to coldest and includes:

- semantic role,
- bytes per tensor,
- tensor count,
- expected touches per token,
- a hotness-skew model,
- measured CPU/GPU cost per touch,
- streamability and device capability,
- hard RAM/VRAM requirements.

The planner first admits hard requirements, then assigns remaining capacity by marginal avoided
latency per byte. It outputs:

- counts and bytes in VRAM, RAM and NVMe,
- expected hotness mass in each tier,
- expected NVMe bytes per token,
- estimated movement and compute time,
- a deterministic `lattice.plan.v1` document.

The current solver is deliberately simple and auditable. Later solvers may add integer
programming, multi-drive topology and online re-optimization, but they must preserve the same
validation contract.

## Three-path movement scheduler

Each tensor read enters one of three paths:

1. **Demand:** routing has confirmed that the tensor is required now.
2. **Proactive:** request-local evidence predicts that the tensor will be required soon.
3. **Speculative:** lower-confidence work may consume otherwise idle I/O capacity.

Ordering is deterministic: path priority, earliest deadline, confidence and insertion sequence.
Duplicate tensor IDs are merged. A demand request promotes an existing prediction rather than
issuing another read. In-flight byte limits prevent predictions from starving urgent work.

This policy is inspired by selective-transfer systems but applies to model tensors and expert
blocks within one heterogeneous machine.

## Request-specific expert signatures

During prefill, Lattice records expert selections and optional gate weights by layer. The
resulting signature ranks likely decode-time experts for that request. It can:

- seed the RAM/VRAM cache,
- submit proactive reads,
- be decayed across conversation turns,
- be persisted in a sparse portable format.

The signature never replaces the model router. Router output remains authoritative.

## Block-addressable experts

Ordinary MoE execution selects an expert but still loads and computes the full intermediate
width. Lattice's adaptive research branch introduces a second hierarchy:

```text
token
  -> selected experts
      -> selected intermediate-channel blocks
```

For native Kimi-style MXFP4 experts, a block record contains W1 and W3 rows plus the matching W2
columns. The block width is aligned to the quantization group. Records are padded to 4096 bytes,
so adjacent selected blocks can become one direct-I/O read.

The v1 file layout is fixed-size and arithmetic-free: the packer copies packed nibbles and scale
bytes exactly. This lets correctness tests isolate the layout from router training and kernels.

A future partial-expert kernel must prove:

- each block output equals the matching contribution from full execution,
- summing all blocks reproduces the full expert within the declared floating-point tolerance,
- block order does not alter scale interpretation,
- selected bytes and measured I/O match the file ranges exactly.

## Adapter contract

A model adapter must provide:

- tensor inventory and semantic roles,
- exact source offsets and formats,
- routing observations,
- tier load/unload hooks,
- execution hooks for resident and staged tensors,
- state-size accounting,
- correctness oracle integration.

An adapter advertises capabilities such as dense streaming, routed-expert streaming, GPU staging
and block execution. The control plane may only produce plans within that capability set.

Initial adapter order:

1. a smaller conventional MoE,
2. a structurally different hybrid/recurrent MoE,
3. Kimi K3.

Kimi remains the flagship scale demonstration, not the first debugging environment.

## Calibration loop

For each hardware profile:

1. measure sequential and concurrent storage reads,
2. measure RAM and host-to-device bandwidth,
3. benchmark exact tensor kernels at real shapes,
4. run a trace through the scheduler,
5. compare predicted and observed bytes and latency,
6. fit only parameters with an identifiable physical meaning,
7. rerun held-out traces.

A plan is not validated if it only predicts the trace used to fit it.

## Future components

- topology-aware multi-NVMe striping,
- asynchronous direct-I/O executor,
- rotating GPU staging buffers,
- CPU/GPU expert splitting,
- lossless MoE-aware speculative decoding,
- learned block routers and partial-expert kernels,
- semantic KV retention,
- distributed prefill/decode and remote tensor tiers.

These are ordered by dependency. Dynamic width is not allowed to hide weaknesses in the exact
runtime.
