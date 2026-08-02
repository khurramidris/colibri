# OLMoE real-model acceptance

This is the first deliberately manageable real-model gate for Lattice and Colibri.

OLMoE is approximately 7B total parameters with about 1B active parameters. It is small enough to download, convert and execute on ordinary development hardware, while still exercising Colibri's routed-expert streaming path.

The acceptance question is narrow:

> Does this exact converted OLMoE checkpoint, engine binary, machine and controlled runtime reproduce an independently generated greedy continuation on every requested repeat, while emitting complete memory and throughput telemetry?

A successful result proves real checkpoint execution and token-level agreement for the recorded reference. It does **not** prove complete logit equality, general model quality, optimal performance or production readiness.

## 1. Build and test Colibri

```bash
make -C c olmoe
make -C c check
python -m unittest discover -s lattice/tests -v
```

Do not download or convert the full model until the local engine and controller tests pass.

## 2. Create an independent reference

Install the offline reference dependencies:

```bash
python -m pip install torch transformers accelerate safetensors
```

Generate a short greedy continuation with Hugging Face Transformers:

```bash
python c/tools/make_olmoe_reference.py \
  --model allenai/OLMoE-1B-7B-0125-Instruct \
  --revision <IMMUTABLE_HUGGING_FACE_COMMIT> \
  --prompt "Explain why the sky appears blue in two concise sentences." \
  --tokens 32 \
  --device auto \
  --output olmoe-reference.json
```

Use an immutable model revision whenever possible. The output records the model identifier, revision, library versions, prompt hash, template mode and exact token IDs. The C engine does not participate in generating this reference.

The reference file contains `prompt_ids` and `full_ids`. Additional provenance fields are retained but ignored by the C harness.

## 3. Convert the model to Colibri's merged expert format

Install conversion dependencies if they are not already present:

```bash
python -m pip install torch safetensors huggingface_hub
```

The streaming converter processes one source shard at a time, deletes it after extraction and can resume an interrupted conversion:

```bash
python c/tools/convert_olmoe_merged.py \
  --repo allenai/OLMoE-1B-7B-0125-Instruct \
  --out /models/olmoe-merged \
  --min-free-gb 10
```

Preserve the exact Colibri commit, converter command and model revision in the experiment notes.

## 4. Run acceptance

```bash
python -m lattice accept-olmoe \
  --repo . \
  --model /models/olmoe-merged \
  --reference ./olmoe-reference.json \
  --cache-cap 16 \
  --quant-bits 8 \
  --repeats 3 \
  --timeout 1800 \
  --output ./evidence/olmoe-acceptance.json \
  --report ./evidence/olmoe-acceptance.md
```

The command returns success only when every requested run:

- exits normally;
- stays within the bounded process-output contract;
- emits complete load, RSS, expert-cache and speed telemetry;
- generates the expected continuation length;
- matches every independently generated continuation token.

The JSON evidence contains:

- sampled model fingerprint;
- runtime fingerprint;
- machine and model-volume identity;
- reference SHA-256;
- controlled OLMoE environment;
- sealed successful or failed run records;
- median and range for throughput, peak RSS, load RSS, load time and expert hit rate;
- an evidence root over the completed record and run digests.

Output files are immutable through the normal command path. Use a new output path for a changed model, engine, machine, reference or configuration.

## Controlled baseline

The first acceptance fixes potentially adaptive OLMoE controls:

```text
PILOT=0
WIDE=1
HOT=0
WARMUP=5
SMOOTH=0.3
CONF_LIMIT=0.92
PILOT_EVICT_GUARD=1
EXPERT_DROP=0
IDOT=0
```

`IDOT=0` deliberately selects the conservative scalar arithmetic path for the first token-exact gate. Throughput from this acceptance should not be presented as the fastest Colibri result. Optimized vector paths and adaptive policies require a subsequent paired qualification with an appropriate numerical contract.

A fixed OpenMP thread count may be supplied with `--threads`. Otherwise Colibri's engine-level thread tuning remains active.

## Acceptance versus qualification

`accept-olmoe` is not the same as Lattice's GLM `qualify` workflow.

| Acceptance | Qualification |
|---|---|
| Establishes that one real backend executes a recorded reference | Compares several candidate configurations |
| Requires token-exact continuation agreement | Uses a numerical replay oracle and promotion policy |
| Produces an acceptance record | Produces a deployment profile |
| Does not select a winner | May promote a measured candidate |
| Does not claim optimality | Applies regression, gain and confidence gates |

We are adding acceptance first because OLMoE does not yet expose the complete GLM replay-oracle protocol. Removing the GLM-only qualification guard before that protocol exists would exaggerate what Lattice can verify.

## What a successful run means

A real accepted record supports this statement:

> On the recorded machine, the recorded Colibri OLMoE engine and converted checkpoint reproduced the supplied independent greedy continuation exactly across the requested runs, with the reported resident memory, cache and throughput measurements.

It does not support these statements:

- Colibri is faster than another runtime.
- The model is generally quality-equivalent to the original checkpoint.
- The reported speed is production serving performance.
- The chosen cache or thread configuration is optimal.
- OLMoE is now covered by full Lattice numerical qualification.

The next step after a real accepted record is to add an OLMoE numerical replay adapter, then compare controlled execution candidates under Lattice's paired qualification methodology.
