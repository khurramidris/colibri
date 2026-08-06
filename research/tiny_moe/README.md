# Gate 3 — Tiny-MoE real trunk streaming

This directory is an isolated Gate 3 research implementation. It tests the
Fareed hypothesis on the genuinely trained `AbdelrhmanEbied/Tiny-MoE` base
checkpoint, not on Kimi K3 and not only on synthetic fixtures.

## Frozen experiment protocol v1

Frozen before runtime implementation on 2026-08-06 (Asia/Karachi):

- checkpoint: `AbdelrhmanEbied/Tiny-MoE`, `base/model.safetensors`, Hub commit
  `bb4d6474d9123ce6e1523b340bb6f8dd0877e56a`;
- source reference: `AbdelrhmanEbied/Tiny-MoE` commit
  `e380414b734c5a0834c9e7783d6f44a84728f5f2`;
- tokenizer: the source project uses `mistralai/Mistral-7B-v0.1`; only its
  tokenizer files are downloaded locally;
- precision, weights, routing, prompt token IDs, thread count, and PyTorch
  backend are held constant across residency modes;
- deterministic CPU inference, one PyTorch thread, MKL-DNN disabled after the
  measured diagnostic showed framework-side weight packing obscured residency,
  `torch.inference_mode()`, no autocast, no `torch.compile`, no weight
  absorption, fixed prompt token IDs, greedy argmax;
- required residency modes: all trunk layers, 50% of trunk layers, two layers,
  one layer, and zero layers;
- resident layers are the numeric prefix of the non-routed layer trunk. Routed
  experts are never packed into the trunk and are loaded as a separate working
  set for this first small-model experiment;
- comparison gates: layer hidden states, router top-k IDs and weights, final
  logits, greedy tokens, and persistent MLA cache state, repeated three times;
- output criterion: byte identity where the backend permits it. Any observed
  nondeterminism must be recorded with its operation and preregistered absolute
  and relative tolerances; tolerances may not be weakened after results.

The protocol is intentionally correctness-first. It does not preregister a
speedup claim.

## Architecture discovered from source and safetensors header

The author’s native PyTorch implementation is the executable reference. The
Hub artifact is not a Transformers checkpoint: it has no `config.json`,
`tokenizer` files, or `AutoModel` class, and its MLA implementation is custom.
Transformers is therefore not a trusted model reference for this checkpoint;
it is used only to load the source project’s Mistral tokenizer when available.

The base checkpoint has 213 F16 tensors and is 476,168,928 bytes. The model is:

| component | value |
|---|---:|
| layers | 14 |
| vocabulary | 32,000 |
| hidden size | 512 |
| attention heads | 8 |
| MLA KV rank | 96 |
| MLA non-RoPE / RoPE dimensions | 48 / 16 |
| routed experts per layer | 8 |
| shared experts per layer | 1 |
| routed experts per token | 2 |
| expert intermediate size | 1,024 |
| context | 512 base (2,048 YaRN variant) |
| weight tying | enabled by the source model |
| checkpoint dtype | F16 |

Per layer, routed `layers.N.moe.w13` and `layers.N.moe.w2` are expert working
set tensors. The attention tensors, norms, layer scales, router, and shared
expert tensors are the non-routed trunk. Embeddings, final norm, tied output
head, tokenizer/configuration, and MLA KV/position caches are persistent state.
`model_inspection.py` and `MANIFEST.json` are the authority for exact byte
counts and tensor ranges.

## Files

- `model_inspection.py`: range-only safetensors inspection and architecture
  manifest generation;
- `trunk_packer.py`: exact-byte, 4096-byte-aligned, layer-addressable trunk;
- `reference_runner.py`: fully resident author-source reference and traces;
- `streamed_runner.py`: bounded prefix residency, two reusable buffers,
  optional prefetch, and direct tensor views over current layer bytes;
- `accounting.py`: RSS, payload/request bytes, read calls, cache state, and
  timing records;
- `benchmark.py`: fixed-token A–E matrix and result serialization;
- `tests/`: unit and differential tests for format, ranges, and runtime;
- `results/`: raw run outputs and manifests;
- `GATE3_RESULTS.md`: evidence ledger; every claim is labeled PROVEN,
  MEASURED, MODELED, ASSUMED, or NOT YET TESTED.

## Reproduction outline

```powershell
python research/tiny_moe/model_inspection.py `
  --checkpoint research/tiny_moe/artifacts/checkpoint/base/model.safetensors `
  --output research/tiny_moe/MANIFEST.json
python research/tiny_moe/trunk_packer.py `
  --checkpoint research/tiny_moe/artifacts/checkpoint/base/model.safetensors `
  --manifest research/tiny_moe/MANIFEST.json `
  --output research/tiny_moe/artifacts/base.trunk.bin `
  --index research/tiny_moe/artifacts/base.trunk.json
python -m pytest -q research/tiny_moe/tests
python research/tiny_moe/benchmark.py --manifest research/tiny_moe/MANIFEST.json
```

The checkpoint and local tokenizer are ignored from version control but their
sizes, source revisions, and SHA-256 values are recorded in the manifest and
results. No model payload is silently fetched by a timed runner.
