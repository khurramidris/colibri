"""Fully resident Tiny-MoE reference runner and layer/token trace capture."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


@dataclass
class StepTrace:
    hidden_states: list[torch.Tensor]
    router_ids: list[torch.Tensor]
    router_weights: list[torch.Tensor]
    persistent: list[dict[str, Any]]
    logits: torch.Tensor


@dataclass
class GenerationTrace:
    steps: list[StepTrace]
    token_ids: list[int]


def tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _persistent_trace(model: torch.nn.Module) -> list[dict[str, Any]]:
    result = []
    for layer_id, layer in enumerate(model.layers):
        attention = layer.attn
        cache_len = int(attention.cache_len)
        result.append(
            {
                "layer_id": layer_id,
                "cache_len": cache_len,
                "kv_shape": list(attention.kv_cache[:, :cache_len].shape),
                "pe_shape": list(attention.pe_cache[:, :cache_len].shape),
                "kv_sha256": tensor_digest(attention.kv_cache[:, :cache_len]),
                "pe_sha256": tensor_digest(attention.pe_cache[:, :cache_len]),
            }
        )
    return result


class TraceCollector:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.hidden_states: list[torch.Tensor] = []
        self.router_ids: list[torch.Tensor] = []
        self.router_weights: list[torch.Tensor] = []
        self.handles = []
        for layer in model.layers:
            self.handles.append(layer.register_forward_hook(self._hidden_hook))
            self.handles.append(
                layer.moe.router.register_forward_hook(self._router_hook)
            )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def begin(self) -> None:
        self.hidden_states = []
        self.router_ids = []
        self.router_weights = []

    def _hidden_hook(
        self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        self.hidden_states.append(output.detach().clone().cpu())

    def _router_hook(
        self, module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor
    ) -> None:
        # The source implementation converts router logits to float32 before
        # top-k, so duplicate that exact boundary rather than hook semantics.
        parent = module
        while not hasattr(parent, "num_experts_per_token"):
            parent = getattr(parent, "_colibri_parent_moe", None)
            if parent is None:
                break
        top_k = int(getattr(parent, "num_experts_per_token", 2))
        logits = output.detach().to(torch.float32)
        topk_logits, topk_ids = torch.topk(logits, top_k, dim=-1)
        self.router_ids.append(topk_ids.cpu().clone())
        self.router_weights.append(torch.softmax(topk_logits, dim=-1).cpu().clone())


def _attach_router_parent(model: torch.nn.Module) -> None:
    for layer in model.layers:
        layer.moe.router._colibri_parent_moe = layer.moe


def load_author_reference(
    checkpoint: str | Path,
    source_root: str | Path,
    *,
    max_seq_len: int = 512,
) -> tuple[torch.nn.Module, TraceCollector, dict[str, Any]]:
    """Load the author model strictly, then drop the temporary state dict."""
    source_root = Path(source_root).resolve()
    if not (source_root / "inference" / "model_inference.py").is_file():
        raise FileNotFoundError(
            f"author inference source not found under {source_root}"
        )
    sys.path.insert(0, str(source_root))
    config_module = importlib.import_module("inference.inference_configs")
    model_module = importlib.import_module("inference.model_inference")
    config = config_module.ModelConfig()
    config.factor = 1.0
    config.rope_type = "default"
    config.max_seq_len = max_seq_len
    config.attn_impl = "sdpa"
    config.absorb_weights = False
    model = model_module.Transformer(config)
    # The upstream constructor defaults to F32 and load_state_dict would
    # silently cast the F16 checkpoint. Preserve the checkpoint precision for
    # this frozen experiment; the streamed executor uses the same F16 values.
    model.half()
    state_dict = load_file(str(checkpoint), device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load reported incompatible keys: {incompatible}")
    del state_dict
    model.eval()
    _attach_router_parent(model)
    collector = TraceCollector(model)
    return (
        model,
        collector,
        {
            "source_root": str(source_root),
            "config": {key: value for key, value in vars(config).items()},
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
            ),
            "buffer_bytes": sum(
                buffer.numel() * buffer.element_size() for buffer in model.buffers()
            ),
        },
    )


def _capture_step(
    model: torch.nn.Module, collector: TraceCollector, input_ids: torch.Tensor
) -> StepTrace:
    collector.begin()
    logits = model(input_ids, return_last_only=False).detach().cpu().clone()
    if len(collector.hidden_states) != len(model.layers) or len(
        collector.router_ids
    ) != len(model.layers):
        raise RuntimeError("trace hooks did not observe every layer/router")
    return StepTrace(
        hidden_states=collector.hidden_states,
        router_ids=collector.router_ids,
        router_weights=collector.router_weights,
        persistent=_persistent_trace(model),
        logits=logits,
    )


@torch.inference_mode()
def run_generation_trace(
    model: torch.nn.Module,
    collector: TraceCollector,
    input_ids: torch.Tensor,
    new_tokens: int = 3,
) -> GenerationTrace:
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("fixed-token runner expects input_ids with shape [1, T]")
    model.reset_cache()
    steps = [_capture_step(model, collector, input_ids)]
    token_ids: list[int] = []
    current = torch.argmax(steps[-1].logits[:, -1, :], dim=-1).view(1, 1)
    for _ in range(new_tokens):
        token_ids.append(int(current.item()))
        steps.append(_capture_step(model, collector, current))
        current = torch.argmax(steps[-1].logits[:, -1, :], dim=-1).view(1, 1)
    return GenerationTrace(steps=steps, token_ids=token_ids)


def trace_summary(trace: GenerationTrace) -> dict[str, Any]:
    return {
        "token_ids": trace.token_ids,
        "steps": [
            {
                "hidden_sha256": [
                    tensor_digest(tensor) for tensor in step.hidden_states
                ],
                "router_ids_sha256": [
                    tensor_digest(tensor) for tensor in step.router_ids
                ],
                "router_weights_sha256": [
                    tensor_digest(tensor) for tensor in step.router_weights
                ],
                "persistent": step.persistent,
                "logits_shape": list(step.logits.shape),
                "logits_sha256": tensor_digest(step.logits),
            }
            for step in trace.steps
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--input-ids", required=True, help="JSON list of integer token IDs"
    )
    parser.add_argument("--new-tokens", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help="diagnostic only; Gate 3 keeps MKL-DNN disabled",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = args.enable_mkldnn
    model, collector, metadata = load_author_reference(
        args.checkpoint, args.source_root
    )
    try:
        ids = torch.tensor([json.loads(args.input_ids)], dtype=torch.long)
        trace = run_generation_trace(model, collector, ids, args.new_tokens)
        output = {"metadata": metadata, "trace": trace_summary(trace)}
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {"output": args.output, "tokens": trace.token_ids}, sort_keys=True
            )
        )
    finally:
        collector.close()


if __name__ == "__main__":
    main()
