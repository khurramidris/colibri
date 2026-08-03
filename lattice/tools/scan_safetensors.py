#!/usr/bin/env python3
"""Build a Lattice tensor-class manifest from a Hugging Face safetensors model.

The scanner reads only safetensors headers and config.json. It never maps or
loads tensor payloads, so it is safe to run on very large checkpoints. Output is
TSV accepted by lattice-plan.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HEADER_LIMIT = 256 * 1024 * 1024
NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Tensor:
    name: str
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass
class Group:
    role: str
    nbytes: int
    tensors: list[Tensor]


def nested_find(root: Any, keys: Iterable[str]) -> Any | None:
    wanted = set(keys)
    stack = [root]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key in wanted:
                    return child
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
    return None


def read_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_safetensors_header(path: Path) -> list[Tensor]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as fp:
            prefix = fp.read(8)
            if len(prefix) != 8:
                raise ValueError("file is shorter than the 8-byte header length")
            (header_size,) = struct.unpack("<Q", prefix)
            if header_size == 0 or header_size > HEADER_LIMIT:
                raise ValueError(f"header length {header_size} is outside the accepted range")
            if 8 + header_size > file_size:
                raise ValueError("declared header extends beyond end of file")
            payload = fp.read(header_size)
            if len(payload) != header_size:
                raise ValueError("short header read")
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc

    try:
        header = json.loads(payload.decode("utf-8").rstrip(" \t\r\n\0"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"header in {path} is not an object")

    data_size = file_size - 8 - header_size
    tensors: list[Tensor] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(meta, dict):
            raise ValueError(f"malformed tensor entry in {path}")
        offsets = meta.get("data_offsets")
        shape = meta.get("shape")
        dtype = meta.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(v, int) for v in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_size
            or not isinstance(shape, list)
            or not all(isinstance(v, int) and v >= 0 for v in shape)
            or not isinstance(dtype, str)
        ):
            raise ValueError(f"malformed metadata for tensor {name!r} in {path}")
        tensors.append(
            Tensor(
                name=name,
                nbytes=offsets[1] - offsets[0],
                dtype=dtype,
                shape=tuple(shape),
            )
        )
    return tensors


def classify(name: str) -> str:
    lower = name.lower()
    if "embed_tokens" in lower or "word_embeddings" in lower:
        return "embedding"
    if "lm_head" in lower or "output_projection" in lower:
        return "embedding"
    if re.search(r"(?:^|\.)(?:experts|expert)\.\d+(?:\.|$)", lower) or ".block_sparse_moe.experts." in lower:
        return "routed_expert"
    if "shared_expert" in lower or "shared_experts" in lower:
        return "shared_expert"
    if "self_attn" in lower or ".attention." in lower or ".attn." in lower:
        return "attention"
    if "router" in lower or ".gate." in lower or lower.endswith("gate.weight"):
        return "norm"
    if "norm" in lower or lower.endswith("bias"):
        return "norm"
    if ".mlp." in lower or "feed_forward" in lower or "ffn" in lower:
        return "dense"
    return "other"


def safe_name(text: str) -> str:
    cleaned = NAME_RE.sub("_", text).strip("_.-")
    return cleaned[:80] or "tensor_class"


def role_flags(role: str) -> str:
    if role == "norm":
        return "ram_required,gpu"
    return "streamable,gpu"


def build_groups(tensors: list[Tensor]) -> list[Group]:
    buckets: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    for tensor in tensors:
        buckets[(classify(tensor.name), tensor.nbytes)].append(tensor)
    groups = [Group(role=role, nbytes=nbytes, tensors=members) for (role, nbytes), members in buckets.items()]
    groups.sort(key=lambda g: (g.role, -g.nbytes, g.tensors[0].name))
    return groups


def manifest_rows(
    groups: list[Group],
    n_experts: int,
    top_k: int,
    expert_skew: float,
) -> list[str]:
    rows: list[str] = []
    role_index: dict[str, int] = defaultdict(int)
    for group in groups:
        role_index[group.role] += 1
        count = len(group.tensors)
        if group.role == "routed_expert":
            if n_experts <= 0 or top_k <= 0:
                raise ValueError(
                    "routed expert tensors were found but num_experts/top_k could not be inferred; "
                    "pass --num-experts and --top-k"
                )
            touches = count * top_k / n_experts
            skew = expert_skew
        else:
            touches = float(count)
            skew = 0.0
        name = safe_name(f"{group.role}_{role_index[group.role]}_{group.nbytes}b")
        rows.append(
            "\t".join(
                [
                    name,
                    group.role,
                    str(group.nbytes),
                    str(count),
                    f"{touches:.12g}",
                    f"{skew:.12g}",
                    "0",
                    "0",
                    role_flags(group.role),
                    "0",
                    "0",
                ]
            )
        )
    return rows


def append_runtime_state(rows: list[str], kv_cache_gib: float, recurrent_state_gib: float) -> None:
    for name, role, gib in (
        ("kv_cache_runtime", "kv_cache", kv_cache_gib),
        ("recurrent_state_runtime", "recurrent_state", recurrent_state_gib),
    ):
        if gib <= 0:
            continue
        nbytes = int(gib * 1024**3)
        rows.append(
            "\t".join(
                [name, role, str(nbytes), "1", "1", "0", "0", "0", "ram_required,gpu", "0", "0"]
            )
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, help="output TSV; default stdout")
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--expert-skew", type=float, default=0.8)
    parser.add_argument("--kv-cache-gib", type=float, default=0.0)
    parser.add_argument("--recurrent-state-gib", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        print(f"error: not a directory: {model_dir}", file=sys.stderr)
        return 2
    try:
        config = read_config(model_dir)
        inferred_experts = nested_find(config, ("num_experts", "num_local_experts", "n_routed_experts"))
        inferred_top_k = nested_find(
            config,
            ("num_experts_per_token", "num_selected_experts", "moe_top_k", "top_k"),
        )
        n_experts = args.num_experts if args.num_experts is not None else int(inferred_experts or 0)
        top_k = args.top_k if args.top_k is not None else int(inferred_top_k or 0)
        if n_experts < 0 or top_k < 0 or top_k > n_experts:
            raise ValueError(f"invalid expert geometry: num_experts={n_experts}, top_k={top_k}")
        if not 0.0 <= args.expert_skew <= 4.0:
            raise ValueError("--expert-skew must be between 0 and 4")
        if args.kv_cache_gib < 0 or args.recurrent_state_gib < 0:
            raise ValueError("runtime state sizes must be non-negative")

        tensors: list[Tensor] = []
        seen: set[str] = set()
        files = sorted(model_dir.glob("*.safetensors"))
        if not files:
            raise ValueError(f"no *.safetensors files found in {model_dir}")
        for path in files:
            for tensor in read_safetensors_header(path):
                if tensor.name in seen:
                    raise ValueError(f"duplicate tensor name across shards: {tensor.name}")
                seen.add(tensor.name)
                tensors.append(tensor)
        groups = build_groups(tensors)
        rows = manifest_rows(groups, n_experts, top_k, args.expert_skew)
        append_runtime_state(rows, args.kv_cache_gib, args.recurrent_state_gib)
        text = (
            "# generated by lattice/tools/scan_safetensors.py\n"
            f"# model_dir={model_dir} tensors={len(tensors)} shards={len(files)} "
            f"num_experts={n_experts} top_k={top_k}\n"
            "# name\trole\tbytes_each\tcount\ttouches_per_token\tskew\tcpu_ms\tgpu_ms\tflags\tmin_ram\tmin_vram\n"
            + "\n".join(rows)
            + "\n"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0
    except (OSError, ValueError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
