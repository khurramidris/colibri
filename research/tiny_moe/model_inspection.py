"""Inspect a Tiny-MoE safetensors header without loading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ALIGNMENT = 4096
MAX_HEADER_BYTES = 64 * 1024 * 1024
LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    absolute_offset: int
    byte_count: int
    category: str
    layer_id: int | None


def _is_url(source: str | os.PathLike[str]) -> bool:
    return str(source).startswith(("http://", "https://"))


def _read_local(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(length)
    if len(data) != length:
        raise ValueError(f"short local read at {offset}: {len(data)} != {length}")
    return data


def _read_remote(url: str, offset: int, length: int) -> bytes:
    end = offset + length - 1
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={offset}-{end}", "User-Agent": "colibri-gate3"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise ValueError(
                "server did not honor a byte range; refusing full download"
            )
        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {offset}-{end}/"):
            raise ValueError(f"unexpected Content-Range: {content_range!r}")
        data = response.read(length)
    if len(data) != length:
        raise ValueError(f"short remote read at {offset}: {len(data)} != {length}")
    return data


def read_range(source: str | os.PathLike[str], offset: int, length: int) -> bytes:
    if offset < 0 or length < 0:
        raise ValueError("negative range")
    if _is_url(source):
        return _read_remote(str(source), offset, length)
    return _read_local(Path(source), offset, length)


def _source_size(source: str | os.PathLike[str]) -> int | None:
    if not _is_url(source):
        return Path(source).stat().st_size
    request = urllib.request.Request(str(source), method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        value = response.headers.get("Content-Length")
    return int(value) if value is not None else None


def read_header(source: str | os.PathLike[str]) -> tuple[dict[str, Any], int]:
    first = read_range(source, 0, 8)
    header_bytes = struct.unpack("<Q", first)[0]
    if header_bytes <= 0 or header_bytes > MAX_HEADER_BYTES:
        raise ValueError(f"invalid safetensors header length: {header_bytes}")
    raw = read_range(source, 8, header_bytes)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise TypeError("safetensors header is not an object")
    return header, header_bytes + 8


def _category(name: str) -> tuple[str, int | None]:
    match = LAYER_RE.match(name)
    if match:
        layer_id = int(match.group(1))
        suffix = match.group(2)
        if suffix in {"moe.w13", "moe.w2"}:
            return "routed_expert", layer_id
        return "trunk", layer_id
    if name in {"embed_tokens.weight", "lm_head.weight", "norm.weight"}:
        return "persistent", None
    return "unclassified", None


def inspect_checkpoint(source: str | os.PathLike[str]) -> dict[str, Any]:
    header, data_start = read_header(source)
    source_size = _source_size(source)
    tensors: list[TensorRecord] = []
    for name, value in sorted(header.items()):
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise TypeError(f"tensor {name!r} has malformed metadata")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if (
            dtype not in DTYPE_BYTES
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
        ):
            raise ValueError(f"tensor {name!r} has unsupported metadata")
        if len(offsets) != 2 or any(not isinstance(v, int) for v in offsets):
            raise ValueError(f"tensor {name!r} has malformed offsets")
        start, end = offsets
        if start < 0 or end < start:
            raise ValueError(f"tensor {name!r} has invalid offsets")
        count = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                raise ValueError(f"tensor {name!r} has invalid shape")
            count *= dimension
        byte_count = end - start
        expected = count * DTYPE_BYTES[dtype]
        if byte_count != expected:
            raise ValueError(
                f"tensor {name!r}: {byte_count} bytes != {expected} expected"
            )
        category, layer_id = _category(name)
        tensors.append(
            TensorRecord(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                data_offsets=(start, end),
                absolute_offset=data_start + start,
                byte_count=byte_count,
                category=category,
                layer_id=layer_id,
            )
        )

    layers: dict[int, list[TensorRecord]] = {}
    for record in tensors:
        if record.layer_id is not None:
            layers.setdefault(record.layer_id, []).append(record)
    layer_summary = []
    for layer_id in sorted(layers):
        layer_records = sorted(layers[layer_id], key=lambda item: item.name)
        trunk = [item for item in layer_records if item.category == "trunk"]
        experts = [item for item in layer_records if item.category == "routed_expert"]
        layer_summary.append(
            {
                "layer_id": layer_id,
                "tensor_names": [item.name for item in layer_records],
                "trunk_tensor_names": [item.name for item in trunk],
                "routed_expert_tensor_names": [item.name for item in experts],
                "trunk_payload_bytes": sum(item.byte_count for item in trunk),
                "routed_expert_payload_bytes": sum(item.byte_count for item in experts),
                "source_span": {
                    "start": min(item.absolute_offset for item in layer_records),
                    "end": max(
                        item.absolute_offset + item.byte_count for item in layer_records
                    ),
                },
            }
        )

    category_bytes = {
        category: sum(item.byte_count for item in tensors if item.category == category)
        for category in ("persistent", "trunk", "routed_expert", "unclassified")
    }
    result = {
        "format": "safetensors-header-manifest-v1",
        "source": str(source),
        "source_size_bytes": source_size,
        "safetensors_header_bytes": data_start,
        "alignment_bytes": ALIGNMENT,
        "tensor_count": len(tensors),
        "metadata": header.get("__metadata__"),
        "categories": category_bytes,
        "architecture": {
            "vocab_size": 32000,
            "hidden_size": 512,
            "num_layers": len(layer_summary),
            "attention_heads": 8,
            "kv_lora_rank": 96,
            "qk_nope_dim": 48,
            "qk_rope_dim": 16,
            "routed_experts": 8,
            "shared_experts": 1,
            "top_k": 2,
            "moe_intermediate_size": 1024,
            "checkpoint_dtype": sorted({item.dtype for item in tensors}),
            "weight_tying": True,
        },
        "layers": layer_summary,
        "tensors": [asdict(item) for item in tensors],
    }
    if source_size is not None and tensors:
        last = max(item.absolute_offset + item.byte_count for item in tensors)
        if last > source_size:
            raise ValueError(
                f"tensor data ends at {last}, beyond source size {source_size}"
            )
    return result


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", default="base")
    parser.add_argument(
        "--hub-commit", default="bb4d6474d9123ce6e1523b340bb6f8dd0877e56a"
    )
    parser.add_argument(
        "--source-repo-commit", default="e380414b734c5a0834c9e7783d6f44a84728f5f2"
    )
    args = parser.parse_args()
    manifest = inspect_checkpoint(args.checkpoint)
    manifest["model"] = {
        "hub_id": "AbdelrhmanEbied/Tiny-MoE",
        "variant": args.variant,
        "hub_commit": args.hub_commit,
        "source_repo": "https://github.com/AbdelrhmanEbied/Tiny-MoE",
        "source_repo_commit": args.source_repo_commit,
    }
    if not _is_url(args.checkpoint):
        manifest["source_sha256"] = sha256_file(args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "tensor_count": manifest["tensor_count"],
                "layers": manifest["architecture"]["num_layers"],
                "categories": manifest["categories"],
                "source_sha256": manifest.get("source_sha256"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
