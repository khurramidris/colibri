#!/usr/bin/env python3
"""Create an independent greedy OLMoE reference continuation with Transformers.

The output is consumed by ``python -m lattice accept-olmoe`` and deliberately
contains the prompt token IDs plus the complete prompt+continuation path. The C
engine never participates in producing this reference.

Example:
  python c/tools/make_olmoe_reference.py \
    --model allenai/OLMoE-1B-7B-0125-Instruct \
    --prompt "Explain why the sky appears blue in two sentences." \
    --tokens 32 --output olmoe-reference.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

try:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    sys.exit(
        f"Missing dependency: {exc}. Install with: "
        "pip install torch transformers accelerate safetensors"
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an independent greedy OLMoE token reference."
    )
    parser.add_argument("--model", required=True, help="Hugging Face repo ID or local model directory")
    parser.add_argument("--revision", help="immutable Hugging Face revision or commit")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="model placement; auto uses Accelerate device_map=auto",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--plain-prompt",
        action="store_true",
        help="tokenize prompt directly instead of using the tokenizer chat template",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> int:
    args = parse_args()
    if not 1 <= args.tokens <= 2048:
        sys.exit("--tokens must be between 1 and 2048")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        sys.exit(f"refusing to overwrite existing reference: {output}")

    common = {
        "revision": args.revision,
        "trust_remote_code": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **common)
    model_kwargs = {
        **common,
        "torch_dtype": resolve_dtype(args.dtype),
        "low_cpu_mem_usage": True,
    }
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.device != "auto":
        model = model.to(args.device)
    model.eval()

    if not args.plain_prompt and getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        template_mode = "tokenizer_chat_template"
    else:
        encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"]
        template_mode = "plain_prompt"

    input_device = next(model.parameters()).device
    input_ids = input_ids.to(input_device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_ids = [int(value) for value in input_ids[0].detach().cpu().tolist()]
    full_ids = [int(value) for value in generated[0].detach().cpu().tolist()]
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        sys.exit("Transformers did not produce a valid prompt+continuation sequence")
    if len(full_ids) > 4096:
        sys.exit("reference exceeds the current OLMoE engine 4096-token limit")

    payload = {
        "schema_version": 1,
        "generator": "transformers-greedy",
        "model": args.model,
        "revision": args.revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": args.device,
        "dtype": args.dtype,
        "template_mode": template_mode,
        "prompt_sha256": sha256_text(args.prompt),
        "requested_new_tokens": args.tokens,
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "output": str(output),
        "prompt_tokens": len(prompt_ids),
        "continuation_tokens": len(full_ids) - len(prompt_ids),
        "template_mode": template_mode,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
