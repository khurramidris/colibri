# Synthetic stress-screen results

Status: **PBE1 KILL as the primary format; broader Mach-inspired hypothesis NARROW.**

These results are deterministic synthetic screening, not evidence about a real model.
They are intended to eliminate weak representations before checkpoint work.

## Scope

- 80 matrices: eight distributions x ten seeds.
- Matrix shape: 64 x 512, grouped by 64.
- Distributions: Gaussian, Laplace, Student-t(3), 1% and 5% outlier mixtures,
  row-scale heterogeneity, low-rank-plus-noise, and blockwise heterogeneity.
- Activation probes: Gaussian, Laplace, 10% sparse, anisotropic and correlated.
- Baseline quality: Colibri int3-g64 reference approximation.
- Baseline physical size: Colibri E8/IQ3, 3.0625 bits/weight.

## Result 1: the current PBE1 candidate fails the quality gate

PBE1 with two sparse corrections per 64-weight group uses 2.125 effective
bits/weight and therefore cuts the expert payload by 30.6% versus E8/IQ3.
However:

- median Gaussian-activation matvec error was about 1.90x the int3 error;
- the 95th percentile was about 2.14x;
- it beat int3 in only 8.75% of cases;
- median physical rate required to match int3 was 6.875 bits/weight;
- only 8.75% of cases matched int3 at <=2.2 bits/weight;
- only 25% matched at <=2.5 bits/weight.

Conclusion: the binary base plus sparse scalar residual design is too weak.
Do not implement its production kernel.

## Result 2: ordinary grouped 2-bit quantization is also insufficient

At 2.125 bits/weight with power-of-two scales, the median matvec error was
about 1.48x int3 and it never beat int3 across the 80-case suite. With fp16
scales at 2.25 bits/weight, median error remained about 1.38x and it again
never beat int3.

This rejects the idea that the gain can come from merely changing Colibri to
an ordinary 2-bit grouped format.

## Result 3: learned structured representations improve the frontier

Exploratory residual-vector-quantization screens found:

- 4 stages, 2.0 index bits/weight with shared codebooks: median matvec error
  about 1.27x int3; only 3.3% of cases beat int3.
- 5 stages, 2.5 index bits/weight with shared codebooks: median error roughly
  matched int3, but only 50% of cases beat it.

A block-Hadamard rotation followed by learned entropy-coded scalar levels was
more encouraging:

| Levels | Median effective bpw | Median error / int3 | Beat-int3 rate |
|---:|---:|---:|---:|
| 4 | 2.19 | 1.17x | 37.5% |
| 5 | 2.51 | 0.96x | 50.0% |
| 6 | 2.77 | 0.82x | 98.4% |
| 7 | 3.00 | 0.71x | 100% |

This is not yet a deployable format: entropy decoding, random access, kernel
cost and container alignment are not included. It does show that rotation and
learned codebooks are the ingredients worth pursuing, rather than PBE1.

## Result 4: information theory says the broader target is possible but tight

For an ideal Gaussian source, the rate-distortion lower bound gives relative
RMSE approximately 2^-R. At 2.125 bits/weight this is about 0.229. The median
int3 relative weight error in the Gaussian screens was about 0.251, implying an
ideal minimum rate near 1.99 bits/weight to match it.

Therefore 2.0-2.2 bits is not mathematically impossible, but there is very
little implementation margin. A 1.7-bit universal target would have ideal
Gaussian relative RMSE about 0.308 and is unlikely to match int3 without
exploiting additional model structure or task sensitivity.

## Result 5: the systems gain is real if quality is solved

For a 2.125-bit representation:

- direct payload reduction versus E8/IQ3: 30.6%;
- cache capacity multiplier: 1.44x;
- simulated LRU physical expert-traffic reduction: 31-66%, depending on cache
  size and routing concentration;
- roofline speedup with a parity-speed kernel: about 1.18x at 50% I/O-bound,
  1.27x at 70%, 1.35x at 85%, and 1.41x at 95%;
- with a 30% slower expert kernel: about 1.10x, 1.18x, 1.27x and 1.38x in the
  same regimes.

Thus the runtime opportunity remains meaningful, but only after a stronger
representation preserves quality.

## Scientific decision

- **KILL:** PBE1 as the primary Cerno expert representation.
- **NARROW:** continue the broader Mach-inspired direction.
- Next candidate: rotated, learned additive/trellis codebooks with activation-
  aware calibration, exact physical accounting and direct packed execution.
- Initial practical target: 2.5-2.8 effective bits/weight, then attempt to push
  toward 2.2 only if the quality margin and kernel cost support it.
- Do not claim a result until the same gates pass on real expert tensors,
  held-out activations, full-model logits/routes and end-to-end timings.
