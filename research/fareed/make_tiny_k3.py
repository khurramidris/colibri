#!/usr/bin/env python3
"""Generate a deterministic miniature Kimi K3 snapshot for lifecycle tests.

The snapshot uses Colibri's real tensor names and repacked U8 + `.qs` matrix
format. It has three dense KDA layers, no routed experts, no embedding and no
head. K3_X0 supplies the input, so both resident and streamed executables run
the genuine KDA, AttnRes and dense-MLP code while remaining small enough for CI.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def _quantize_rows(matrix: np.ndarray) -> tuple[bytes, bytes]:
    matrix = np.asarray(matrix, dtype=np.float32)
    amax = np.max(np.abs(matrix), axis=1)
    scale = np.maximum(amax / 127.0, np.float32(1.0e-12)).astype(np.float32)
    q = np.clip(np.rint(matrix / scale[:, None]), -127, 127).astype(np.int8)
    return q.view(np.uint8).tobytes(), scale.tobytes()


def _f32(value: np.ndarray) -> bytes:
    return np.asarray(value, dtype="<f4").tobytes()


def _add_matrix(
    tensors: list[tuple[str, str, list[int], bytes]],
    name: str,
    matrix: np.ndarray,
) -> None:
    q, scales = _quantize_rows(matrix)
    rows, cols = map(int, matrix.shape)
    tensors.append((name, "U8", [rows, cols], q))
    tensors.append((name + ".qs", "F32", [rows], scales))


def _write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    payloads: list[bytes] = []
    for name, dtype, shape, payload in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-(8 + len(raw))) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for payload in payloads:
            handle.write(payload)


def build_snapshot(output: Path, *, layers: int, hidden: int, seed: int) -> None:
    if hidden < 32 or hidden % 32:
        raise ValueError("hidden must be a multiple of 32")
    if layers < 1:
        raise ValueError("layers must be positive")

    rng = np.random.default_rng(seed)
    config = {
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "vocab_size": 128,
        "first_k_dense_replace": layers,
        "intermediate_size": hidden,
        "num_attention_heads": 1,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 16,
        "v_head_dim": 16,
        "num_experts": 1,
        "num_experts_per_token": 1,
        "moe_intermediate_size": 32,
        "routed_expert_hidden_size": 32,
        "num_shared_experts": 1,
        "attn_res_block_size": 2,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "rms_norm_eps": 1.0e-5,
        "linear_attn_config": {
            "num_heads": 1,
            "head_dim": hidden,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": list(range(1, layers + 1)),
        },
        "bos_token_id": 0,
        "eos_token_id": 1,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tensors: list[tuple[str, str, list[int], bytes]] = []
    eye = np.eye(hidden, dtype=np.float32)
    ones = np.ones(hidden, dtype=np.float32)
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        jitter = np.float32(0.012 + layer * 0.002)

        tensors.extend(
            [
                (f"{prefix}.input_layernorm.weight", "F32", [hidden], _f32(ones)),
                (f"{prefix}.post_attention_layernorm.weight", "F32", [hidden], _f32(ones)),
                (f"{prefix}.self_attention_res_norm.weight", "F32", [hidden], _f32(ones)),
                (f"{prefix}.self_attention_res_proj.weight", "F32", [hidden], _f32(ones * 0.04)),
                (f"{prefix}.mlp_res_norm.weight", "F32", [hidden], _f32(ones)),
                (f"{prefix}.mlp_res_proj.weight", "F32", [hidden], _f32(ones * 0.03)),
            ]
        )

        for role, diagonal in (
            ("q_proj", 0.72),
            ("k_proj", 0.68),
            ("v_proj", 0.64),
            ("g_proj", 0.20),
            ("o_proj", 0.55),
        ):
            noise = rng.normal(0.0, jitter, (hidden, hidden)).astype(np.float32)
            _add_matrix(
                tensors,
                f"{prefix}.self_attn.{role}.weight",
                eye * np.float32(diagonal + 0.01 * layer) + noise,
            )

        taps = np.zeros((hidden, 4), dtype=np.float32)
        taps[:, -1] = 0.85
        taps[:, -2] = 0.10
        taps[:, -3] = 0.04
        taps[:, -4] = 0.01
        for role in ("q", "k", "v"):
            tensors.append(
                (
                    f"{prefix}.self_attn.{role}_conv1d.weight",
                    "F32",
                    [hidden, 1, 4],
                    _f32(taps),
                )
            )

        tensors.extend(
            [
                (
                    f"{prefix}.self_attn.f_a_proj.weight",
                    "F32",
                    [hidden, hidden],
                    _f32(eye * 0.08),
                ),
                (
                    f"{prefix}.self_attn.f_b_proj.weight",
                    "F32",
                    [hidden, hidden],
                    _f32(eye * 0.07),
                ),
                (
                    f"{prefix}.self_attn.b_proj.weight",
                    "F32",
                    [1, hidden],
                    _f32(np.full((1, hidden), 0.01, dtype=np.float32)),
                ),
                (
                    f"{prefix}.self_attn.dt_bias",
                    "F32",
                    [hidden],
                    _f32(np.full(hidden, -1.5 + 0.05 * layer, dtype=np.float32)),
                ),
                (
                    f"{prefix}.self_attn.o_norm.weight",
                    "F32",
                    [hidden],
                    _f32(ones),
                ),
                (
                    f"{prefix}.self_attn.A_log",
                    "F32",
                    [hidden],
                    _f32(np.zeros(hidden, dtype=np.float32)),
                ),
            ]
        )

        for role, diagonal in (
            ("gate_proj", 0.45),
            ("up_proj", 0.40),
            ("down_proj", 0.50),
        ):
            noise = rng.normal(0.0, jitter, (hidden, hidden)).astype(np.float32)
            _add_matrix(
                tensors,
                f"{prefix}.mlp.{role}.weight",
                eye * np.float32(diagonal + 0.01 * layer) + noise,
            )

    shard = output / "model-00001-of-00001.safetensors"
    _write_safetensors(shard, tensors)
    weight_map = {name: shard.name for name, _, _, _ in tensors}
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": weight_map,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    x0 = np.linspace(-0.35, 0.40, hidden, dtype=np.float32)[None, :]
    x0 += rng.normal(0.0, 0.01, x0.shape).astype(np.float32)
    (output / "x0.f32").write_bytes(x0.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    build_snapshot(args.output, layers=args.layers, hidden=args.hidden, seed=args.seed)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
