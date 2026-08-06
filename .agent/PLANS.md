# ExecPlan: Tiny-MoE performance and compressed-trunk comparison

Use this file as durable working memory for the current long-running task.
Keep it concise, current, and self-contained.

## Goal

- Determine whether F16 layer streaming has a measurable cost/benefit and
  whether Int8 or Int4 Colibri-style trunk compression improves that tradeoff
  on the native Windows laptop without unacceptable model divergence.

## Constraints

- Preserve the Fareed branch and all existing evidence; do not rewrite prior
  results.
- First experiment changes residency only: same weights, routing, precision,
  token IDs, backend, and execution settings.
- No Kimi K3; do not silently substitute another model if Tiny-MoE is
  incompatible.
- No success claim from synthetic fixtures; failed/negative results remain.
- Keep research implementation isolated under research/tiny_moe until gates
  justify integration.
- Do not redo the completed Gate 3 correctness matrix.
- Native Windows is primary for PyTorch performance; WSL is limited to Linux
  reader/O_DIRECT experiments.
- Never claim physical cold-cache behavior without a documented cache flush.

## Relevant Files

- .agent/PLANS.md - durable Gate 3 execution state and resume point.
- research/fareed/ - prior implementation/evidence boundary to audit and reuse.
- research/fareed/GATE1_RESULTS.md - prior Gate 1 evidence.
- research/fareed/GATE2_RESULTS.md - prior Gate 2 evidence.
- research/tiny_moe/ - all new inspection, packing, runners, accounting, tests,
  and results.
- research/tiny_moe/results/performance/ - Stage 1/2 raw benchmark artifacts.
- research/tiny_moe/compressed_trunk.py - compressed artifact builder/reader.
- research/tiny_moe/compressed_runner.py - compressed execution path.
- research/tiny_moe/performance_benchmark.py - warm-up/repetition matrix.
- research/tiny_moe/PERFORMANCE_RESULTS.md - final measured decision.
- draft PR #12 - current review history and target branch.

## Plan

1. Freeze the Stage 1 Windows protocol and hardware/cache classification.
2. Benchmark existing F16 full-resident and five streaming residency modes for
   prefill, one-token decode, and >=64-token greedy generation with one warm-up
   plus >=10 measured repetitions; retain raw JSON.
3. Validate the packed-file reader under WSL buffered and O_DIRECT modes only.
4. Build exact-value Int8 per-row and Int4 group-64 non-expert trunk artifacts,
   with checksums and range validation; do not quantize routed experts.
5. Implement direct compressed-layer execution with explicit conversion and
   scratch accounting, never a full-F16 trunk copy.
6. Benchmark compressed modes and quality against the existing F16 reference;
   classify PASS/NARROW/KILL/INCONCLUSIVE from measurements.
7. Run tests/format/lint/type/native checks, write PERFORMANCE_RESULTS.md, and
   update PR #12 with raw artifacts and commands.

## Progress

- Repository cloned at `f6646c1` on
  `research/fareed-compressed-trunk-streaming`; plan initialized.
- PR #11 is open/draft, targets `main`, and records the Fareed Gate 1/2
  evidence. Existing working tree change is only `.agent/PLANS.md`.
- Existing Windows run: `python -m pytest -q research\\fareed` = 11 passed,
  9 failed, 5 errors. Failures use unavailable `os.pread`/`os.pwrite`; several
  teardowns retain open files. Do not rewrite prior evidence for this host
  mismatch.
- Working branch `research/tiny-moe-real-trunk-streaming` created.
- Tiny-MoE source inspected at `e380414b`; base checkpoint downloaded locally
  only, tokenizer files downloaded only, and exact header manifest generated.
- `model_inspection.py` produced 213 tensors / 14 layers and exact category
  totals; base checkpoint SHA-256 is
  `8f23864c7576b8be1e23cba99432aca6eba161d6fbfde5536c62adc41da3845f`.
- Frozen protocol and architecture boundary are documented in
  `research/tiny_moe/README.md`.
- `trunk_packer.py` produced a 58,388,480-byte aligned trunk with 58,290,232
  useful payload bytes and 58,318,848 aligned per-layer I/O bytes. One index
  sizing bug was fixed before validation; no prior evidence was changed.
- Vendored only the author inference Python files under
  `research/tiny_moe/vendor/tiny_moe` for reproducible reference loading. The
  strict author model load and CPU forward completed; initial artifact write
  failed only because `results/` was absent, now fixed.
- The first custom smoke exposed the upstream F32 default; both reference and
  streamed paths now preserve F16 checkpoint execution. A one-step, four-token
  full-resident comparison completed with identical final-logit SHA-256
  `70684cb154583d51740ae9e2124d9f23796e67eef973811ec95e2ad97e31b8f7` and
  identical first two and all 14 per-layer hidden-state hashes.
- Final Gate 3 matrix completed: five modes (14/7/2/1/0 resident layers), three
  repetitions each, two generated tokens, and all comparisons identical.
- MKL-DNN is disabled in the final protocol after the enabled diagnostic showed
  residency-insensitive framework CPU packing; that negative matrix remains in
  `results/matrix/`.
- Final audits found zero Python mmap objects and zero checkpoint-file mappings;
  unique tensor bytes decreased with the declared resident budget. Startup RSS
  decreased, but steady-state RSS ranges overlapped, so peak-RSS reduction is
  not claimed as proven.
- The packed trunk was regenerated and validated after the index-shape fix;
  source, packed trunk, and packed index checksums are in GATE3_RESULTS.md.
- Added a narrow Windows `pread`/`pwrite` fallback and constructor cleanup to
  Fareed's Python synthetic reader. Windows pytest is now 20/20; WSL unittest
  is 20/20; Linux native strict GCC plus buffered/direct-preferred tests pass.
- Project-owned Tiny-MoE code passes Black, Ruff, and targeted mypy; Tiny-MoE
  tests pass 4/4.
- New performance stage requested by user; completed correctness artifacts are
  frozen and must not be regenerated.
- Added a non-capturing `_step(..., capture=False)` path and
  `performance_benchmark.py`. The native Windows Stage 1 matrix has all 18
  cases complete: six modes across prefill, one-token decode, and 64-token
  generation, each with one warm-up and ten measured repetitions.
- Added a non-timing 65-step routing audit. All six modes have identical
  generated tokens, Top-2 selection hashes, and router-weight hashes at every
  step. The timing matrix was not rerun.
- Added `results/performance/STAGE1_ANALYSIS.json`,
  `PERFORMANCE_MATRIX.json`, raw case JSON, routing-audit JSON, and the
  human-readable `STAGE1_RESULTS.md`.
- Added initial `compressed_trunk.py` and `compressed_runner.py` scaffolding for
  Int8 per-row and Int4 group-64 artifacts; compression benchmarks and quality
  tests have not started.

## Discoveries

- Fareed Gate 1: synthetic mechanics pass; no real-model claim.
- Fareed Gate 2: native buffered/O_DIRECT and miniature real-K3 lifecycle pass;
  full checkpoint, startup RSS, production binding, and end-to-end speed remain
  open.
- The inherited reader assumes POSIX `pread`/`pwrite`, so Windows validation
  needs a compatibility boundary or a Linux/WSL execution environment.
- Tiny-MoE's source uses F16 weights but float32 MLA KV/position buffers when
  moved to CPU without an explicit dtype conversion. The source loader also
  silently promotes F16 checkpoint parameters to F32 unless the model is
  explicitly converted first. Gate 3 freezes explicit F16 model parameters and
  MLA cache buffers in both paths.
- Checkpoint tensor ordering is lexicographic and interleaves routed expert
  tensors with non-expert tensors. The new trunk is therefore a repacked exact
  byte copy of only non-routed tensors, not a raw source span.
- Tiny-MoE is small enough that framework/allocation overhead may dominate
  storage traffic; Stage 1 must test for an INCONCLUSIVE or NARROW result before
  interpreting compressed-byte reductions as speedups.

## Decisions

- Use the requested branch name research/tiny-moe-real-trunk-streaming.
- Treat the Transformers path as a reference, not as proof that a custom
  streamed path is correct.
- Treat the completed `results/matrix_final` and Gate 3 report as immutable
  correctness evidence; performance runs use new artifacts/directories.
- Use F16, Int8 per-row, and Int4 group-64 only for non-expert trunk tensors;
  retain routed expert weights unchanged.

## Verification

- Fareed tests pass on Windows and WSL after the narrow portability boundary;
  Linux native compiler and buffered/direct-preferred workflow pass.
- Inspection, packing, repeated real-checkpoint parity, generation, RSS, I/O,
  and complete test gates are documented in GATE3_RESULTS.md. Correctness is
  proven; steady-state RSS monotonicity and real-model O_DIRECT/page-cache
  gates remain open.
- Native Windows Stage 1 timing and route validation are complete. The timing
  data are noisy; F16 streaming has no reliable speedup claim and compressed
  comparison remains open.
- Compressed artifact correctness and Stage 2 performance are not yet run.

## Resume

- Current status: Stage 1 is validated and ready for an isolated commit; no
  benchmark process is active.
- Next action: commit only Stage 1 code/evidence, push it to PR #12, then
  complete and test the Int8/Int4 compressed path.
