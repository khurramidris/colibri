# ExecPlan: Tiny-MoE real trunk streaming Gate 3

Use this file as durable working memory for the current long-running task.
Keep it concise, current, and self-contained.

## Goal

- Establish or falsify Gate 3 for a genuinely trained AbdelrhmanEbied/Tiny-MoE
  checkpoint: a numerically identical fully resident and layer-streamed path,
  with measured memory/I/O accounting and reproducible evidence.

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

## Relevant Files

- .agent/PLANS.md - durable Gate 3 execution state and resume point.
- research/fareed/ - prior implementation/evidence boundary to audit and reuse.
- research/fareed/GATE1_RESULTS.md - prior Gate 1 evidence.
- research/fareed/GATE2_RESULTS.md - prior Gate 2 evidence.
- research/tiny_moe/ - all new inspection, packing, runners, accounting, tests,
  and results.
- draft PR #11 - review history to inspect and update only after evidence.

## Plan

1. Audit branch, latest commit, PR #11, Fareed code/results, and existing tests;
   create the dedicated working branch without altering prior evidence.
2. Inspect Tiny-MoE metadata/config/custom code/tokenizer/checkpoint index and
   tensor names without loading the full checkpoint; freeze the architecture,
   tensor manifest, source checksums, and experiment plan.
3. Download only required model files and build machine-readable MANIFEST.json.
4. Implement isolated layer-addressable trunk packing, validation, accounting,
   resident reference, streamed runner, and deterministic tests.
5. Run Gate 3 modes A-E with layer/token comparisons, RSS/I/O/cache metrics,
   repeated deterministic executions, and trusted Transformers reference where
   feasible.
6. Iterate only on measured blockers; run formatting, lint/type/native checks;
   write GATE3_RESULTS.md with PROVEN/MEASURED/MODELED/ASSUMED/NOT YET TESTED.
7. Update PR #11 or open a clearly linked draft PR targeting the Fareed branch
   only after the evidence and reviewable commits exist.

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

## Decisions

- Use the requested branch name research/tiny-moe-real-trunk-streaming.
- Treat the Transformers path as a reference, not as proof that a custom
  streamed path is correct.

## Verification

- Fareed tests pass on Windows and WSL after the narrow portability boundary;
  Linux native compiler and buffered/direct-preferred workflow pass.
- Inspection, packing, repeated real-checkpoint parity, generation, RSS, I/O,
  and complete test gates are documented in GATE3_RESULTS.md. Correctness is
  proven; steady-state RSS monotonicity and real-model O_DIRECT/page-cache
  gates remain open.

## Resume

- Current status: Gate 3 report is written. Correctness passes for the real
  Tiny-MoE checkpoint; resource/performance gates are explicitly bounded by the
  report's caveats.
- Next action: inspect the final diff, run final checks, create small reviewable
  commits, and update the existing draft PR history under the original task
  scope.
