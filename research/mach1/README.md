# Mach-inspired progressive expert experiment

This directory is the first falsification harness for applying a **public
systems lesson** from Syzygy Mach-1 to Colibri: expert representation, storage,
and execution should be co-designed below ordinary 3–4 bit quantization.

It is **not** a port or reconstruction of Mach-1. No unpublished algorithm,
format, code, or performance result is assumed.

## Hypothesis

A streamed MoE expert can be represented as a compact base plus optional
refinement data, executed directly from that representation, and beat
Colibri's strongest current ~3-bit baselines under equal quality and hardware
budgets.

The first candidate, PBE1, uses:

- one packed binary sign plane;
- one fp16 base scale per group;
- a variable sparse correction stream;
- one fp16 correction scale and one uint8 correction count per group;
- one uint8 index and one int8 residual coefficient per selected weight.

For group size 64, its idealized effective rate is approximately:

```
1.625 + 16 * correction_fraction bits/weight
```

before final container/shard alignment. At 1%, 3%, and 5% corrections this
is approximately 1.785, 2.105, and 2.425 bits/weight.

## Why start offline

A new kernel is unjustified until the representation itself survives three
questions:

1. Does it remain at or below 2.2 **physical** bits/weight after every scale,
   count, correction, and padding byte is included?
2. Does it preserve expert matvecs and full-model logits closely enough?
3. Is the representation simple enough for a direct CPU/GPU kernel that does
   not erase the I/O gain?

This harness answers the first question exactly and screens the second on
isolated matrices. It does not claim end-to-end evidence.

## Run

```bash
python3 -m pip install numpy
python3 research/mach1/progressive_expert.py \
  --rows 128 --cols 1024 \
  --fractions 0,0.01,0.03,0.05,0.07 \
  --json-out out/mach1-screening.json

python3 -m unittest discover -s research/mach1 -p 'test_*.py'
```

For a real expert projection:

```bash
python3 research/mach1/progressive_expert.py \
  --weights path/to/expert_projection.npy \
  --activation-samples 256 \
  --json-out out/real-expert-screening.json
```

## Preregistered decision gates

Proceed to a C reference kernel only when a representative real-weight sample
meets all of these:

- effective payload <= 2.2 bits/weight;
- >= 25% fewer physical bytes than Colibri E8/IQ3 (3.0625 bpw);
- isolated matvec error is competitive enough to justify full-logit testing;
- no hidden expansion to int8/fp16 is required before multiplication.

Proceed to an integrated Colibri runtime only when the C kernel then meets:

- <= 30% kernel slowdown relative to the strongest matched baseline;
- >= 1.25x end-to-end decode speedup in an expert-I/O-bound regime;
- strict full-logit, routing, perplexity, and held-out task gates;
- all conversion time, metadata, alignment, cache, and fallback bytes counted.

Kill or narrow the hypothesis when correction data pushes the payload above
2.5 bpw, quality requires most experts to remain at 3–4 bits, or gains vanish
against E8/IQ3 rather than only against int4.

## Evidence boundary

Synthetic matrices are smoke tests. A scientific result requires:

- real weights from at least two MoE families;
- held-out activation traces;
- full-model logit and route comparisons;
- repeated end-to-end measurements under fixed RAM/VRAM/SSD budgets.
