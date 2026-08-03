#!/usr/bin/env python3
"""Capture canonical router traces from a Transformers MoE model.

This is an optional GPU-side adapter. It deliberately discovers router modules
at runtime and fails closed if the observed tensor shape or layer coverage is
ambiguous. It does not alter routing decisions.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


def _require_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("hf_capture.py requires torch and transformers") from error
    return torch, AutoModelForCausalLM, AutoTokenizer


def _tensor_from_output(output: Any, torch, n_experts: int):
    candidates = []
    if torch.is_tensor(output):
        candidates.append(output)
    elif isinstance(output, (tuple, list)):
        candidates.extend(item for item in output if torch.is_tensor(item))
    elif isinstance(output, dict):
        candidates.extend(item for item in output.values() if torch.is_tensor(item))
    matches = [item for item in candidates if item.ndim >= 2 and item.shape[-1] == n_experts]
    if len(matches) != 1:
        return None
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="immutable HF revision/commit")
    parser.add_argument("--prompt-file", type=Path, required=True, help="JSONL: {request_id,prompt,max_new_tokens}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--router-regex", default=r"(?:router|gate)$")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dry-discover", action="store_true")
    args = parser.parse_args()

    torch, AutoModelForCausalLM, AutoTokenizer = _require_dependencies()
    kwargs: dict[str, Any] = {
        "revision": args.revision,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, args.dtype)
    if args.load_in_4bit:
        kwargs["load_in_4bit"] = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()

    pattern = re.compile(args.router_regex)
    modules = [(name, module) for name, module in model.named_modules() if pattern.search(name)]
    if args.dry_discover:
        for name, module in modules:
            print(f"{name}\t{type(module).__name__}")
        return 0
    if not modules:
        raise SystemExit("no router modules matched; use --dry-discover and refine --router-regex")

    layer_pattern = re.compile(r"(?:layers?|blocks?)\.(\d+)")
    module_layers: dict[str, int] = {}
    for name, _ in modules:
        match = layer_pattern.search(name)
        if match:
            module_layers[name] = int(match.group(1))
    if set(module_layers.values()) != set(range(args.n_layers)):
        raise SystemExit(
            "router module discovery did not produce exactly one or more candidates for every expected layer"
        )

    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite trace: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {"request_id": "", "run_id": f"hf-{int(time.time())}", "call": -1, "seq_len": 0}
    seen_per_call: set[tuple[int, int]] = set()

    with output.open("x", encoding="utf-8") as handle, ExitStack() as stack:
        def model_pre_hook(module, args_, kwargs):
            state["call"] += 1
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args_:
                input_ids = args_[0]
            state["seq_len"] = int(input_ids.shape[-1]) if torch.is_tensor(input_ids) else 1
            seen_per_call.clear()

        stack.callback(model.register_forward_pre_hook(model_pre_hook, with_kwargs=True).remove)

        def make_hook(name: str, layer: int):
            def hook(module, args_, output_):
                logits = _tensor_from_output(output_, torch, args.n_experts)
                if logits is None:
                    return
                marker = (state["call"], layer)
                if marker in seen_per_call:
                    raise RuntimeError(f"ambiguous duplicate router output for layer {layer} in call {state['call']}")
                seen_per_call.add(marker)
                flat = logits.detach().float().reshape(-1, args.n_experts)
                values, indices = torch.topk(flat, k=args.top_k, dim=-1)
                phase = "prefill" if state["seq_len"] > 1 else "decode"
                for row in range(flat.shape[0]):
                    token = row if phase == "prefill" else state["call"]
                    record = {
                        "schema": "lattice.route.v1",
                        "run_id": state["run_id"],
                        "request_id": state["request_id"],
                        "phase": phase,
                        "token": token,
                        "layer": layer,
                        "experts": indices[row].tolist(),
                        "scores": values[row].tolist(),
                        "source_module": name,
                    }
                    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            return hook

        for name, module in modules:
            if name not in module_layers:
                continue
            stack.callback(module.register_forward_hook(make_hook(name, module_layers[name])).remove)

        with args.prompt_file.open(encoding="utf-8") as prompts:
            for line_number, raw in enumerate(prompts, 1):
                if not raw.strip():
                    continue
                item = json.loads(raw)
                request_id = item.get("request_id") or f"request-{line_number}"
                prompt = item["prompt"]
                max_new_tokens = int(item.get("max_new_tokens", 64))
                state["request_id"] = request_id
                state["call"] = -1
                inputs = tokenizer(prompt, return_tensors="pt")
                device = model.get_input_embeddings().weight.device
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.inference_mode():
                    model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
