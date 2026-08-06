"""Tiny-MoE execution with bounded non-expert trunk residency."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from io import FileIO
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from accounting import rss_bytes
from reference_runner import GenerationTrace, StepTrace, tensor_digest
from safetensors import safe_open
from trunk_packer import ALIGNMENT, read_index, validate_index


def _aligned_view(size: int) -> tuple[bytearray, memoryview]:
    raw = bytearray(size + ALIGNMENT)
    address = ctypes.addressof(ctypes.c_char.from_buffer(raw))
    start = (-address) % ALIGNMENT
    return raw, memoryview(raw)[start : start + size]


class ReadAccounting:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.read_calls = 0
        self.payload_bytes = 0
        self.requested_bytes = 0
        self.startup_payload_bytes = 0
        self.startup_requested_bytes = 0
        self.token_payload_bytes = 0
        self.token_requested_bytes = 0
        self.read_sizes: list[int] = []
        self.read_events: list[dict[str, Any]] = []
        self.wait_seconds = 0.0
        self.compute_seconds = 0.0

    def record(
        self,
        layer_id: int,
        payload_bytes: int,
        requested_bytes: int,
        phase: str,
        load_seconds: float,
    ) -> None:
        with self.lock:
            self.read_calls += 1
            self.payload_bytes += payload_bytes
            self.requested_bytes += requested_bytes
            self.read_sizes.append(requested_bytes)
            self.read_events.append(
                {
                    "layer_id": layer_id,
                    "phase": phase,
                    "payload_bytes": payload_bytes,
                    "requested_bytes": requested_bytes,
                    "load_seconds": load_seconds,
                }
            )
            if phase == "startup":
                self.startup_payload_bytes += payload_bytes
                self.startup_requested_bytes += requested_bytes
            elif phase in {"prefill", "decode"}:
                self.token_payload_bytes += payload_bytes
                self.token_requested_bytes += requested_bytes
            else:
                raise ValueError(f"unknown read phase {phase}")

    def record_wait(self, seconds: float) -> None:
        with self.lock:
            self.wait_seconds += seconds

    def record_compute(self, seconds: float) -> None:
        with self.lock:
            self.compute_seconds += seconds

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "read_calls": self.read_calls,
                "payload_bytes": self.payload_bytes,
                "requested_bytes": self.requested_bytes,
                "startup_payload_bytes": self.startup_payload_bytes,
                "startup_requested_bytes": self.startup_requested_bytes,
                "token_payload_bytes": self.token_payload_bytes,
                "token_requested_bytes": self.token_requested_bytes,
                "read_sizes": list(self.read_sizes),
                "read_events": list(self.read_events),
                "load_wait_seconds": self.wait_seconds,
                "compute_seconds": self.compute_seconds,
            }


class LayerReader:
    def __init__(
        self,
        path: str | Path,
        index: dict[str, Any],
        *,
        io_mode: str = "buffered",
        max_buffers: int = 2,
    ) -> None:
        self.path = Path(path)
        self.index = index
        self.io_mode = io_mode
        self.accounting = ReadAccounting()
        self.max_io_bytes = max(layer["io_bytes"] for layer in index["layers"])
        self.buffers = [_aligned_view(self.max_io_bytes) for _ in range(max_buffers)]
        self.handles: list[FileIO | None] = [
            cast(FileIO, self.path.open("rb", buffering=0)) for _ in range(max_buffers)
        ]
        self.direct_fds: list[int | None] = [None] * max_buffers
        if io_mode == "direct":
            if os.name == "nt" or not hasattr(os, "O_DIRECT"):
                self.close()
                raise RuntimeError("O_DIRECT is unavailable on this platform")
            for handle_id in range(max_buffers):
                handle = self.handles[handle_id]
                assert handle is not None
                handle.close()
                self.handles[handle_id] = None  # type: ignore[assignment]
                self.direct_fds[handle_id] = os.open(
                    str(self.path), os.O_RDONLY | os.O_DIRECT
                )
        elif io_mode != "buffered":
            self.close()
            raise ValueError(f"unknown io mode {io_mode}")

    def close(self) -> None:
        for handle in self.handles:
            if handle is not None:
                handle.close()
        for fd in self.direct_fds:
            if fd is not None:
                os.close(fd)

    def _read_buffered(self, buffer_id: int, offset: int, length: int) -> None:
        handle = self.handles[buffer_id]
        if handle is None:
            raise RuntimeError("buffered handle is closed")
        handle.seek(offset)
        target = self.buffers[buffer_id][1][:length]
        cursor = 0
        while cursor < length:
            count = handle.readinto(target[cursor:])
            if not count:
                raise ValueError(f"short trunk read at {offset + cursor}")
            cursor += count

    def _read_direct(self, buffer_id: int, offset: int, length: int) -> None:
        fd = self.direct_fds[buffer_id]
        if fd is None:
            raise RuntimeError("direct fd is closed")
        os.lseek(fd, offset, os.SEEK_SET)
        target = self.buffers[buffer_id][1][:length]
        cursor = 0
        while cursor < length:
            count = os.readv(fd, [target[cursor:]])  # type: ignore[attr-defined]
            if not count:
                raise ValueError(f"short direct trunk read at {offset + cursor}")
            cursor += count

    def read_layer(
        self, layer_id: int, buffer_id: int, *, phase: str = "decode"
    ) -> dict[str, Any]:
        layer = self.index["layers"][layer_id]
        if layer["layer_id"] != layer_id:
            raise ValueError(f"index layer mismatch: expected {layer_id}")
        length = (
            layer["io_bytes"] if self.io_mode == "direct" else layer["payload_bytes"]
        )
        started = time.perf_counter()
        if self.io_mode == "direct":
            self._read_direct(buffer_id, layer["payload_offset"], length)
        else:
            self._read_buffered(buffer_id, layer["payload_offset"], length)
        elapsed = time.perf_counter() - started
        self.accounting.record(layer_id, layer["payload_bytes"], length, phase, elapsed)
        result = dict(layer)
        result["_load_seconds"] = elapsed
        return result

    def tensors_from_buffer(
        self, layer: dict[str, Any], buffer_id: int
    ) -> dict[str, torch.Tensor]:
        storage = self.buffers[buffer_id][1]
        result: dict[str, torch.Tensor] = {}
        dtype_map = {"F16": torch.float16, "F32": torch.float32, "BF16": torch.bfloat16}
        for spec in layer["tensors"]:
            dtype = dtype_map.get(spec["dtype"])
            if dtype is None:
                raise ValueError(f"unsupported execution dtype {spec['dtype']}")
            elements = 1
            for dimension in spec["shape"]:
                elements *= dimension
            result[spec["name"]] = torch.frombuffer(
                storage,
                dtype=dtype,
                count=elements,
                offset=spec["local_offset"],
            ).reshape(spec["shape"])
        return result


class TinyMoeRuntime:
    """Reference-math executor whose current layer weights are byte-buffer views."""

    def __init__(
        self,
        checkpoint: str | Path,
        trunk: str | Path,
        *,
        resident_layers: int,
        io_mode: str = "buffered",
        prefetch: bool = True,
        max_seq_len: int = 512,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.trunk = Path(trunk)
        self.index = read_index(self.trunk)
        validate_index(self.index, self.trunk.stat().st_size)
        layer_count = len(self.index["layers"])
        if resident_layers < 0 or resident_layers > layer_count:
            raise ValueError(f"resident_layers must be in [0, {layer_count}]")
        self.resident_layers = resident_layers
        self.prefetch = prefetch
        self.io_mode = io_mode
        self.max_seq_len = max_seq_len
        self.hidden_size = 512
        self.num_heads = 8
        self.kv_rank = 96
        self.qk_nope = 48
        self.qk_rope = 16
        self.top_k = 2
        self.num_experts = 8
        self.intermediate = 1024
        self.eps = 1e-6
        self.softmax_scale = (self.qk_nope + self.qk_rope) ** -0.5
        self.rope = self._make_rope(max_seq_len)
        self.accounting = ReadAccounting()
        self.reader: LayerReader | None = None
        self.resident: dict[int, tuple[bytearray, dict[str, torch.Tensor]]] = {}
        self.experts: list[dict[str, torch.Tensor]] = []
        self.last_route_trace: dict[str, list[str]] = {
            "router_ids_sha256": [],
            "router_weights_sha256": [],
        }
        self.embed_tokens: torch.Tensor
        self.norm: torch.Tensor
        self._load_persistent_and_experts()
        self._load_resident_layers()
        self.kv_cache = [
            torch.zeros(1, max_seq_len, self.kv_rank, dtype=torch.float16)
            for _ in range(layer_count)
        ]
        self.pe_cache = [
            torch.zeros(1, max_seq_len, self.qk_rope, dtype=torch.float16)
            for _ in range(layer_count)
        ]
        self.cache_len = [0] * layer_count

    @staticmethod
    def _make_rope(max_seq_len: int) -> torch.Tensor:
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float32), inv_freq)
        return torch.polar(torch.ones_like(freqs), freqs)

    def _load_tensor_copy(self, handle: Any, name: str) -> torch.Tensor:
        return handle.get_tensor(name).clone()

    def _load_persistent_and_experts(self) -> None:
        with safe_open(str(self.checkpoint), framework="pt", device="cpu") as handle:
            self.embed_tokens = self._load_tensor_copy(handle, "embed_tokens.weight")
            lm_head = self._load_tensor_copy(handle, "lm_head.weight")
            if not torch.equal(self.embed_tokens, lm_head):
                raise ValueError(
                    "checkpoint claims tied embeddings but embed/lm_head bytes differ"
                )
            del lm_head
            self.norm = self._load_tensor_copy(handle, "norm.weight")
            for layer_id in range(len(self.index["layers"])):
                self.experts.append(
                    {
                        f"layers.{layer_id}.moe.w13": self._load_tensor_copy(
                            handle, f"layers.{layer_id}.moe.w13"
                        ),
                        f"layers.{layer_id}.moe.w2": self._load_tensor_copy(
                            handle, f"layers.{layer_id}.moe.w2"
                        ),
                    }
                )

    def _load_resident_layers(self) -> None:
        if self.resident_layers == 0:
            return
        with self.trunk.open("rb", buffering=0) as handle:
            for layer_id in range(self.resident_layers):
                layer = self.index["layers"][layer_id]
                storage = bytearray(layer["payload_bytes"])
                handle.seek(layer["payload_offset"])
                view = memoryview(storage)
                cursor = 0
                while cursor < len(storage):
                    count = handle.readinto(view[cursor:])
                    if not count:
                        raise ValueError("short resident trunk read")
                    cursor += count
                self.resident[layer_id] = (
                    storage,
                    self._tensors_from_storage(layer, view),
                )
                self.accounting.record(
                    layer_id,
                    layer["payload_bytes"],
                    layer["payload_bytes"],
                    "startup",
                    0.0,
                )

    @staticmethod
    def _tensors_from_storage(
        layer: dict[str, Any], storage: memoryview | bytearray
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        dtype_map = {"F16": torch.float16, "F32": torch.float32, "BF16": torch.bfloat16}
        for spec in layer["tensors"]:
            elements = 1
            for dimension in spec["shape"]:
                elements *= dimension
            dtype = dtype_map.get(spec["dtype"])
            if dtype is None:
                raise ValueError(f"unsupported dtype {spec['dtype']}")
            result[spec["name"]] = torch.frombuffer(
                storage, dtype=dtype, count=elements, offset=spec["local_offset"]
            ).reshape(spec["shape"])
        return result

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def reset_cache(self) -> None:
        self.cache_len = [0] * len(self.cache_len)

    def _resident_weights(self, layer_id: int) -> dict[str, torch.Tensor]:
        return self.resident[layer_id][1]

    def _rope_apply(
        self, x: torch.Tensor, position_ids: torch.Tensor | None
    ) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(2)
            squeeze = True
        elif x.dim() == 4:
            squeeze = False
        else:
            raise ValueError("RoPE input rank")
        bsz, seq_len, heads, dim = x.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long)
        if position_ids.dim() == 1:
            freqs = self.rope.index_select(0, position_ids).unsqueeze(0).unsqueeze(2)
        elif position_ids.dim() == 2:
            if tuple(position_ids.shape) != (bsz, seq_len):
                raise ValueError("position_ids shape")
            freqs = self.rope[position_ids].unsqueeze(2)
        else:
            raise ValueError("position_ids rank")
        x_complex = torch.view_as_complex(
            x.float().reshape(bsz, seq_len, heads, dim // 2, 2)
        )
        result = torch.view_as_real(x_complex * freqs).flatten(-2).to(x.dtype)
        return result.squeeze(2) if squeeze else result

    def _attention(
        self,
        layer_id: int,
        x: torch.Tensor,
        weights: dict[str, torch.Tensor],
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prefix = f"layers.{layer_id}.attn."
        w_in = weights[prefix + "w_in.weight"]
        w_uq_qr = weights[prefix + "w_uq_qr.weight"]
        w_uk = weights[prefix + "w_uk"]
        w_uv = weights[prefix + "w_uv"]
        w_kr = weights[prefix + "w_kr.weight"]
        w_o = weights[prefix + "w_o.weight"]
        bsz, seq_len, _ = x.shape
        start = self.cache_len[layer_id]
        end = start + seq_len
        if end > self.max_seq_len:
            raise ValueError("sequence exceeds configured cache")
        c_in = F.linear(x, w_in)
        c_kv, c_q = c_in.split(self.kv_rank, dim=-1)
        q_proj = F.linear(c_q, w_uq_qr).reshape(
            bsz, seq_len, self.num_heads, self.qk_nope + self.qk_rope
        )
        q_nope, q_rope = q_proj.split([self.qk_nope, self.qk_rope], dim=-1)
        q_rope = self._rope_apply(q_rope, position_ids)
        q_nope = q_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)
        self.kv_cache[layer_id][:bsz, start:end] = c_kv
        k_rope_cur = self._rope_apply(
            F.linear(x, w_kr).unsqueeze(2), position_ids
        ).squeeze(2)
        self.pe_cache[layer_id][:bsz, start:end] = k_rope_cur
        self.cache_len[layer_id] = end
        past_c_kv = self.kv_cache[layer_id][:bsz, :end]
        past_pe = self.pe_cache[layer_id][:bsz, :end]
        q_lat = q_nope @ w_uk
        k_lat = past_c_kv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        k_rope = past_pe.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        value = past_c_kv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        query = torch.cat([q_lat, q_rope], dim=-1)
        key = torch.cat([k_lat, k_rope], dim=-1)
        attn_lat = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=(seq_len > 1 and position_ids is None),
            scale=self.softmax_scale,
        )
        attn_out = attn_lat @ w_uv
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        return F.linear(attn_out, w_o)

    def _moe(
        self, layer_id: int, x: torch.Tensor, weights: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix = f"layers.{layer_id}.moe."
        bsz, seq_len, hidden = x.shape
        flat = x.reshape(-1, hidden)
        shared_w13 = weights[prefix + "shared_w13.weight"]
        shared_w2 = weights[prefix + "shared_w2.weight"]
        gate_shared, up_shared = F.linear(flat, shared_w13).chunk(2, dim=-1)
        shared_out = F.linear(F.silu(gate_shared) * up_shared, shared_w2)
        router_logits = F.linear(flat, weights[prefix + "router.weight"]).to(
            torch.float32
        )
        topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)
        expert_ids = topk_indices.reshape(-1)
        token_ids = (
            torch.arange(flat.shape[0])
            .unsqueeze(1)
            .expand(flat.shape[0], self.top_k)
            .reshape(-1)
        )
        gates = topk_weights.reshape(-1).to(flat.dtype)
        expert_ids, sort_perm = torch.sort(expert_ids)
        token_ids = token_ids[sort_perm]
        gates = gates[sort_perm]
        counts = torch.bincount(expert_ids, minlength=self.num_experts)
        max_count = int(counts.max().item())
        valid_mask = torch.arange(max_count).unsqueeze(0) < counts.unsqueeze(1)
        packed_inputs = flat.new_zeros((self.num_experts, max_count, hidden))
        packed_inputs[valid_mask] = flat[token_ids]
        expert = self.experts[layer_id]
        proj = torch.matmul(packed_inputs, expert[f"layers.{layer_id}.moe.w13"])
        gate, up = proj.chunk(2, dim=-1)
        packed_outputs = torch.matmul(
            F.silu(gate) * up, expert[f"layers.{layer_id}.moe.w2"]
        )
        valid_outputs = packed_outputs[valid_mask]
        routed_out = flat.new_zeros((flat.shape[0], hidden))
        routed_out.index_add_(0, token_ids, valid_outputs * gates.unsqueeze(-1))
        return (
            (shared_out + routed_out).view(bsz, seq_len, hidden),
            topk_indices.view(bsz, seq_len, self.top_k),
            topk_weights.view(bsz, seq_len, self.top_k),
        )

    def _forward_layer(
        self,
        layer_id: int,
        x: torch.Tensor,
        weights: dict[str, torch.Tensor],
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix = f"layers.{layer_id}."
        h = x + weights[prefix + "ls1"] * self._attention(
            layer_id,
            F.rms_norm(
                x, (self.hidden_size,), weights[prefix + "attn_norm.weight"], self.eps
            ),
            weights,
            position_ids,
        )
        moe_out, route_ids, route_weights = self._moe(
            layer_id,
            F.rms_norm(
                h, (self.hidden_size,), weights[prefix + "ffn_norm.weight"], self.eps
            ),
            weights,
        )
        return h + weights[prefix + "ls2"] * moe_out, route_ids, route_weights

    def _persistent_trace(self) -> list[dict[str, Any]]:
        return [
            {
                "layer_id": layer_id,
                "cache_len": self.cache_len[layer_id],
                "kv_shape": list(
                    self.kv_cache[layer_id][:, : self.cache_len[layer_id]].shape
                ),
                "pe_shape": list(
                    self.pe_cache[layer_id][:, : self.cache_len[layer_id]].shape
                ),
                "kv_sha256": tensor_digest(
                    self.kv_cache[layer_id][:, : self.cache_len[layer_id]]
                ),
                "pe_sha256": tensor_digest(
                    self.pe_cache[layer_id][:, : self.cache_len[layer_id]]
                ),
            }
            for layer_id in range(len(self.cache_len))
        ]

    def _step(
        self,
        input_ids: torch.Tensor,
        phase: str,
        *,
        capture: bool = True,
        capture_routes: bool = False,
    ) -> StepTrace:
        x = F.embedding(input_ids, self.embed_tokens)
        hidden_states: list[torch.Tensor] = []
        route_ids: list[torch.Tensor] = []
        route_weights: list[torch.Tensor] = []
        route_id_hashes: list[str] = []
        route_weight_hashes: list[str] = []
        if self.resident_layers < len(self.index["layers"]) and self.reader is None:
            self.reader = LayerReader(self.trunk, self.index, io_mode=self.io_mode)
            self.reader.accounting = self.accounting
        reader = self.reader
        futures: dict[int, Future[dict[str, Any]]] = {}
        executor = ThreadPoolExecutor(max_workers=1) if self.prefetch else None
        try:
            buffer_id = 0
            for layer_id in range(len(self.index["layers"])):
                if layer_id < self.resident_layers:
                    weights = self._resident_weights(layer_id)
                else:
                    if executor is not None and layer_id in futures:
                        started_wait = time.perf_counter()
                        layer = futures.pop(layer_id).result()
                        self.accounting.record_wait(time.perf_counter() - started_wait)
                    else:
                        assert reader is not None
                        layer = reader.read_layer(layer_id, buffer_id, phase=phase)
                    assert reader is not None
                    weights = reader.tensors_from_buffer(layer, buffer_id)
                    next_layer = layer_id + 1
                    if executor is not None and next_layer < len(self.index["layers"]):
                        next_buffer = 1 - buffer_id
                        futures[next_layer] = executor.submit(
                            reader.read_layer, next_layer, next_buffer, phase=phase
                        )
                compute_started = time.perf_counter()
                x, ids, weights_for_route = self._forward_layer(layer_id, x, weights)
                self.accounting.record_compute(time.perf_counter() - compute_started)
                if capture:
                    hidden_states.append(x.detach().cpu().clone())
                    route_ids.append(ids.detach().cpu().clone())
                    route_weights.append(weights_for_route.detach().cpu().clone())
                if capture_routes:
                    route_id_hashes.append(tensor_digest(ids))
                    route_weight_hashes.append(tensor_digest(weights_for_route))
                if layer_id >= self.resident_layers:
                    buffer_id = 1 - buffer_id
            logits = (
                F.linear(
                    F.rms_norm(x, (self.hidden_size,), self.norm, self.eps),
                    self.embed_tokens,
                )
                .detach()
                .cpu()
                .clone()
            )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        self.last_route_trace = {
            "router_ids_sha256": route_id_hashes,
            "router_weights_sha256": route_weight_hashes,
        }
        return StepTrace(
            hidden_states,
            route_ids,
            route_weights,
            self._persistent_trace() if capture else [],
            logits,
        )

    @torch.inference_mode()
    def run_generation_trace(
        self, input_ids: torch.Tensor, new_tokens: int = 3
    ) -> GenerationTrace:
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError("fixed-token runner expects input_ids with shape [1, T]")
        self.reset_cache()
        steps = [self._step(input_ids, "prefill")]
        token_ids: list[int] = []
        current = torch.argmax(steps[-1].logits[:, -1, :], dim=-1).view(1, 1)
        for _ in range(new_tokens):
            token_ids.append(int(current.item()))
            steps.append(self._step(current, "decode"))
            current = torch.argmax(steps[-1].logits[:, -1, :], dim=-1).view(1, 1)
        return GenerationTrace(steps, token_ids)

    def metadata(self) -> dict[str, Any]:
        kv_cache_bytes = sum(
            cache.numel() * cache.element_size() for cache in self.kv_cache
        )
        pe_cache_bytes = sum(
            cache.numel() * cache.element_size() for cache in self.pe_cache
        )
        persistent_weight_bytes = (
            self.embed_tokens.numel() * self.embed_tokens.element_size()
            + self.norm.numel() * self.norm.element_size()
        )
        return {
            "resident_layers": self.resident_layers,
            "resident_trunk_bytes": sum(
                self.index["layers"][i]["payload_bytes"]
                for i in range(self.resident_layers)
            ),
            "persistent_weight_bytes": persistent_weight_bytes,
            "kv_cache_bytes": kv_cache_bytes,
            "pe_cache_bytes": pe_cache_bytes,
            "persistent_state_bytes": persistent_weight_bytes
            + kv_cache_bytes
            + pe_cache_bytes,
            "routed_expert_bytes": sum(
                t.numel() * t.element_size()
                for layer in self.experts
                for t in layer.values()
            ),
            "transient_buffer_bytes": (
                0
                if self.resident_layers == len(self.index["layers"])
                else 2 * max(layer["io_bytes"] for layer in self.index["layers"])
            ),
            "prefetch": self.prefetch,
            "io_mode": self.io_mode,
            "trunk_file_bytes": self.trunk.stat().st_size,
        }

    def memory_audit(self) -> dict[str, Any]:
        tensor_bytes = 0
        tensor_objects = 0
        seen: set[tuple[int, int]] = set()
        for obj in gc.get_objects():
            if type(obj) is torch.Tensor:
                tensor_objects += 1
                try:
                    key = (obj.data_ptr(), obj.numel() * obj.element_size())
                    if key not in seen:
                        seen.add(key)
                        tensor_bytes += key[1]
                except (AttributeError, RuntimeError, TypeError):
                    pass
        mmap_objects = sum(
            1 for obj in gc.get_objects() if type(obj).__module__ == "mmap"
        )
        checkpoint_mappings: list[dict[str, Any]] = []
        try:
            import psutil

            checkpoint_path = str(self.checkpoint.resolve()).lower()
            for mapping in psutil.Process().memory_maps(grouped=False):
                if (
                    mapping.path
                    and str(Path(mapping.path).resolve()).lower() == checkpoint_path
                ):
                    checkpoint_mappings.append(
                        {
                            "path": mapping.path,
                            "rss": mapping.rss,
                            "private": getattr(mapping, "private", None),
                        }
                    )
        except (ImportError, OSError):
            checkpoint_mappings = []
        return {
            "python_tensor_objects": tensor_objects,
            "python_tensor_unique_bytes": tensor_bytes,
            "python_mmap_objects": mmap_objects,
            "declared_resident_trunk_bytes": self.metadata()["resident_trunk_bytes"],
            "declared_routed_expert_bytes": self.metadata()["routed_expert_bytes"],
            "declared_persistent_state_bytes": self.metadata()[
                "persistent_state_bytes"
            ],
            "declared_transient_buffer_bytes": self.metadata()[
                "transient_buffer_bytes"
            ],
            "rss_at_audit_bytes": rss_bytes(),
            "checkpoint_file_mappings": checkpoint_mappings,
        }


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
    parser.add_argument("--trunk", required=True)
    parser.add_argument("--resident-layers", type=int, required=True)
    parser.add_argument("--input-ids", required=True)
    parser.add_argument("--new-tokens", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--io-mode", choices=("buffered", "direct"), default="buffered")
    parser.add_argument("--ready-file")
    parser.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help="diagnostic only; Gate 3 keeps MKL-DNN disabled",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = args.enable_mkldnn
    runtime = TinyMoeRuntime(
        args.checkpoint,
        args.trunk,
        resident_layers=args.resident_layers,
        io_mode=args.io_mode,
        prefetch=not args.no_prefetch,
    )
    try:
        if args.ready_file:
            ready_path = Path(args.ready_file)
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text(
                json.dumps({"rss_bytes": rss_bytes(), "metadata": runtime.metadata()})
                + "\n",
                encoding="utf-8",
            )
        trace = runtime.run_generation_trace(
            torch.tensor([json.loads(args.input_ids)], dtype=torch.long),
            args.new_tokens,
        )
        output = {
            "metadata": runtime.metadata(),
            "accounting": runtime.accounting.snapshot(),
            "memory_audit": runtime.memory_audit(),
            "trace": trace_summary(trace),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "output": args.output,
                    "tokens": trace.token_ids,
                    "accounting": runtime.accounting.snapshot(),
                },
                sort_keys=True,
            )
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
