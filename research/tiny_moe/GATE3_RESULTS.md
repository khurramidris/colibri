# Gate 3 results — Tiny-MoE real trunk streaming

Date: 2026-08-06  
Branch: `research/tiny-moe-real-trunk-streaming`  
Checkpoint: `AbdelrhmanEbied/Tiny-MoE`, `base/model.safetensors`  

## Verdict

The real-checkpoint correctness hypothesis is proven for the tested protocol:
fully resident, 50%, 2-layer, 1-layer, and zero-layer trunk residency all
reached final logits and greedy generated tokens with identical hidden states,
router selections, router weights, persistent-state hashes, and logits across
three repetitions. The resource gate is only partially closed: declared
resident/tensor bytes track the residency budget, but process-level steady-state
RSS did not show a clean monotonic decrease, so this is not a claim of a fully
qualified low-RSS production runtime.

## Frozen protocol

- Prompt token IDs: `[1, 2, 3, 4]`.
- Two greedy generated tokens; reference output: `[28723, 675]`.
- CPU, one PyTorch thread, F16 checkpoint weights, no autocast, no compilation,
  no weight absorption, MKL-DNN disabled.
- Prefetch enabled, two reusable transient layer buffers, buffered I/O on
  Windows.
- The author’s native PyTorch implementation is the trusted resident reference;
  this checkpoint does not provide a Transformers configuration or model-code
  artifact suitable for an independent Transformers load.
- Raw matrix artifact: `results/matrix_final/GATE3_MATRIX.json`.

## PROVEN

1. The checkpoint is genuinely trained model data, not a synthetic fixture.
2. The inspected architecture has 14 layers, hidden size 512, vocabulary
   32,000, eight routed experts per layer, one shared expert per layer, Top-2
   routing, MLA KV rank 96, and F16 tensors.
3. Routed expert tensors are excluded from the packed trunk. The trunk contains
   only deterministic non-expert layer tensors; each layer has 4,163,588 useful
   bytes. Routed experts total 352,321,536 bytes separately.
4. All five residency modes passed the same comparison against the resident
   reference in all three repetitions:
   - all 14 hidden-state hashes at every trace step;
   - Top-2 expert ID hashes;
   - router-weight hashes;
   - persistent MLA KV/position-state hashes;
   - final-logit hashes;
   - greedy tokens and trace length.
5. Every streamed nonresident layer was read exactly once per forward pass:
   one prefill read and one read for each of the two decode tokens.
6. The packed format validates magic, version, file bounds, alignment,
   non-overlapping ranges, contiguous tensor payloads, tensor names, and
   exclusion of routed experts.
7. The resident/streamed implementation does not instantiate the author’s
   full model. The final audit reported zero Python mmap objects and zero
   checkpoint-file mappings in every mode. Exact unique PyTorch tensor bytes
   decreased with the declared trunk budget.

## MEASURED

The values below are from the 15-run final matrix (five modes × three
repetitions). RSS is the child-process peak sampled before the ready marker
(`startup`) and after it (`steady`). `payload` and `requested` are the actual
buffered-read accounting values.

| mode | resident layers | resident trunk bytes | persistent state bytes | routed experts bytes | transient buffers | startup RSS MB | steady RSS MB | payload bytes/trace | total reads/trace | avg load wait ms | avg compute s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fully resident | 14 | 58,290,232 | 34,374,656 | 352,321,536 | 0 | 594.0–594.4 | 888.0–910.8 | 58,290,232 | 14 | 0.000 | 2.839 |
| 50% | 7 | 29,145,116 | 34,374,656 | 352,321,536 | 8,331,264 | 566.4 | 883.5–931.7 | 116,580,464 | 28 | 0.447 | 3.076 |
| 2 layers | 2 | 8,327,176 | 34,374,656 | 352,321,536 | 8,331,264 | 546.3–547.3 | 882.1–923.8 | 158,216,344 | 38 | 0.737 | 2.756 |
| 1 layer | 1 | 4,163,588 | 34,374,656 | 352,321,536 | 8,331,264 | 542.0–542.7 | 902.4–931.4 | 166,543,520 | 40 | 0.719 | 2.544 |
| 0 layers | 0 | 0 | 34,374,656 | 352,321,536 | 8,331,264 | 538.3–538.7 | 880.3–906.8 | 174,870,696 | 42 | 0.785 | 2.728 |

For the two generated decode tokens, each nonresident layer is read once per
token. The exact phase and per-layer values are retained in the JSON artifact;
the derived per-decode values are:

| mode | nonresident layers | payload per decode token |
|---|---:|---:|
| fully resident | 0 | 0 |
| 50% | 7 | 29,145,116 |
| 2 layers | 12 | 49,963,056 |
| 1 layer | 13 | 54,126,644 |
| 0 layers | 14 | 58,290,232 |

The buffered Windows path requested exactly the useful payload bytes. The
packed index records 4,165,632 aligned I/O bytes per layer (58,318,848 bytes
for all 14 layers); physical O_DIRECT traffic for this real checkpoint was not
run on Windows and is not substituted with buffered numbers.

Audit invariants in every final case were: `python_mmap_objects = 0`, no
checkpoint-file mapping, and no full checkpoint mapping retained by psutil.
Unique tensor bytes were 445,491,224 / 416,346,108 / 395,528,168 /
391,364,580 / 387,200,992 bytes for full / 50% / 2 / 1 / 0 layers.

The startup RSS and unique tensor audit fall with residency. The sampled
steady-state RSS ranges overlap substantially, so predictable peak-RSS
reduction is **not** proven. This is the principal remaining resource issue.

## MODELED

- A streamed pass has one read for each nonresident layer. For this two-token
  trace that is 3 reads per nonresident layer: prefill plus two decode steps.
- Direct-I/O requests would use the index’s 4,165,632-byte aligned extent per
  layer; this is a layout-derived request size, not a measured device-traffic
  number for Tiny-MoE.
- The intended working-set model is persistent state + resident trunk + routed
  expert working set + two maximum layer buffers + execution scratch.
- Prefetch can overlap the next layer’s read with current-layer compute, but the
  measured wait time is not a speedup claim.

## ASSUMED

- Windows `psutil.Process(...).memory_info().rss` is an adequate process-level
  RSS observation for this experiment; it does not decompose allocator arenas,
  driver memory, or physical page-cache residency.
- The author’s PyTorch implementation is the best available trusted reference
  because the Hub checkpoint has no usable Transformers config/model-code
  bundle.
- Buffered-read `requested_bytes` is a request/accounting value, not proof of
  physical SSD bytes.

## NOT YET TESTED

- Real Tiny-MoE O_DIRECT execution on Linux; the code supports it, but the WSL
  environment lacks the PyTorch installation needed for the real-model run.
- Cold-cache versus warm-cache real-model runs, direct physical-device traffic,
  and page-cache eviction control.
- No-prefetch, one-buffer versus two-buffer, prefill versus decode performance
  matrix, and thread-count/hardware sensitivity.
- An independent Transformers resident implementation.
- Production integration or headline speedup.

## Negative diagnostic retained

An earlier matrix with MKL-DNN enabled produced correct logits but large,
residency-insensitive process RSS and framework-side CPU packing. It is retained
under `results/matrix/` and is not used for the final claim. Gate 3 therefore
freezes MKL-DNN disabled so the runtime’s tensor accounting is not obscured by
that backend allocation behavior.

## Reproduction and checksums

```text
python research/tiny_moe/benchmark.py \
  --checkpoint research/tiny_moe/artifacts/checkpoint/base/model.safetensors \
  --trunk research/tiny_moe/artifacts/base.trunk.bin \
  --reference research/tiny_moe/results/reference_f16_decode2_no_mkldnn.json \
  --output-dir research/tiny_moe/results/matrix_final \
  --input-ids '[1, 2, 3, 4]' --new-tokens 2 --repeats 3 --disable-mkldnn
```

```text
checkpoint SHA-256: 8f23864c7576b8be1e23cba99432aca6eba161d6fbfde5536c62adc41da3845f
packed trunk SHA-256: 79bd10beb1c4f7aa7aa31ad8eeb4f03f5cae1b735c9e5a10498e2dd65ae6854f
packed index SHA-256: 4b20c0194b1e16e6a1972955ed055fed4bc46a4735eaccd1ba422ff2eadea5fd
```

## Test evidence

- `python -m pytest -q research/tiny_moe/tests`: 4 passed.
- `python -m pytest -q research/fareed`: 20 passed on Windows after the narrow
  `pread`/`pwrite` compatibility fallback.
- WSL `python3 -m unittest discover -s research/fareed -p 'test_*.py'`: 20
  passed.
- Linux native reader: strict GCC build and buffered/direct-preferred tests
  passed, including 5/5 bit-identical execution checks per mode.
- Project-owned Tiny-MoE code: Black, Ruff, and targeted mypy passed.
