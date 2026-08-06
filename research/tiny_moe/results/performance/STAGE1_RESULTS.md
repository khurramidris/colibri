# Stage 1: Native Windows F16 streaming

Status: timing matrix complete and routing audit exact.

The experiment used the frozen Tiny-MoE checkpoint, prompt IDs `[1, 2, 3,
4]`, one PyTorch CPU thread, MKL-DNN disabled, F16 weights, and the same
greedy execution path for every mode. Each of the 18 cases has one warm-up and
10 measured repetitions. Windows filesystem cache flushing was not attempted:
startup is process-cold and repetitions are OS-cache-warm.

## Proven

- All six modes reached final logits and generated 64 greedy tokens.
- Generated tokens are identical across all six modes for prefill, one-token
  decode, and the 64-token workload.
- The separate 65-step routing audit is exact across all modes: all Top-2
  selection hashes and router-weight hashes match at every prefill/decode
  step.
- Every nonresident F16 layer is read once per forward step. The measured
  decode read counts are 14, 13, 12, and 7 for zero, one, two, and seven
  resident layers respectively.
- Useful payload and read-request accounting are present in every raw case;
  fully resident has zero trunk reads.

## Measured

Mean latency from the 10 repetitions is shown below. Decode and generation are
milliseconds per generated token; prefill is total seconds.

| mode | resident layers | prefill s | decode1 ms/token | generation64 ms/token |
|---|---:|---:|---:|---:|
| fully resident | 14 | 1.651 | 527.1 | 834.4 |
| zero, no prefetch | 0 | 2.090 (+26.6%) | 784.6 (+48.9%) | 889.4 (+6.6%) |
| zero, prefetch | 0 | 2.484 (+50.4%) | 748.0 (+41.9%) | 789.7 (-5.4%) |
| one, prefetch | 1 | 2.941 (+78.1%) | 834.2 (+58.3%) | 840.6 (+0.8%) |
| two, prefetch | 2 | 3.054 (+85.0%) | 1095.2 (+107.8%) | 793.8 (-4.9%) |
| 50%, prefetch | 7 | 1.862 (+12.8%) | 1063.2 (+101.7%) | 726.9 (-12.9%) |

For zero-resident decode, prefetch reduced mean latency by 4.9%; for the
64-token workload it reduced mean latency by 12.6%. These are nominal means,
not claims of a reliable speedup: generation wall-time coefficient of variation
was 11.3% to 37.4% across modes, including 37.4% for fully resident.

Useful F16 trunk bytes per decode token were 58,290,232 with zero resident
layers, 54,126,644 with one, 49,963,056 with two, and 29,145,116 with seven.
The corresponding read requests per token were 14, 13, 12, and 7. Prefetch
hid approximately 98.3% to 98.8% of measured load time in short cases and
94.4% to 98.7% in the 64-token cases. The no-prefetch control intentionally
hid 0%.

The raw memory audit reported unique PyTorch tensor bytes of 445,019,224 for
fully resident, 415,874,108 for seven resident layers, 395,056,168 for two,
390,892,580 for one, and 386,728,992 for zero. These are audits of the child
process after execution; RSS remains subject to allocator and framework
effects. No claim is made that Windows steady-state RSS is monotonic from this
small matrix.

## Modeled

- The expected byte reduction from residency is linear in the packed F16
  non-expert trunk and is consistent with the measured payload counts.
- A compressed artifact with the same layer boundaries should reduce payload
  bytes in proportion to its useful packed size, but its conversion and matrix
  costs are not modeled as speed results here.

## Assumed

- The process-cold/OS-cache-warm classification is the only defensible Windows
  cache classification for this run; no reliable filesystem cache flush was
  available.
- The fixed prompt and deterministic CPU settings are representative only of
  this protocol, not of all Tiny-MoE workloads.

## Not yet tested

- Int8 or Int4 compressed execution and quality.
- Buffered versus O_DIRECT PyTorch model execution on Windows.
- A statistically stable speed conclusion; the observed timing variance is too
  high for a strong F16 streaming speedup claim.
- Whether Tiny-MoE is large enough for storage traffic to dominate on this
  laptop.

Raw case JSON, routing-audit JSON, environment details, and machine-readable
aggregates are in this directory. `STAGE1_ANALYSIS.json` is the validator
output; `PERFORMANCE_MATRIX.json` is the complete case index.
