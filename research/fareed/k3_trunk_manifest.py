#!/usr/bin/env python3
"""Build and validate a layer-addressable Kimi K3 trunk manifest.

This tool reads Colibri-repacked safetensors headers only. It excludes routed
experts, proves that each layer's always-active tensors occupy one contiguous
physical span, and optionally copies those exact spans into the synthetic trunk
container used by Gate 1.

No tensor is dequantized or re-encoded.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

from synthetic_trunk import pack_trunk

_LAYER_RE = re.compile(r"^(?:language_model\.)?model\.layers\.(\d+)\.")
_EXPERT_MARKER = ".block_sparse_moe.experts."


class ManifestError(RuntimeError):
    """Raised when a repacked snapshot cannot be represented layer-by-layer."""


@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype: str
    shape: list[int]
    relative_start: int
    relative_end: int


@dataclass(frozen=True)
class LayerRecord:
    layer_id: int
    shard: str
    source_data_start: int
    source_span_start: int
    source_span_end: int
    payload_bytes: int
    tensors: list[TensorRecord]


@dataclass(frozen=True)
class GlobalRecord:
    shard: str
    name: str
    dtype: str
    shape: list[int]
    nbytes: int


def read_safetensors_header(path: str | os.PathLike[str]) -> tuple[dict, int]:
    target = Path(path)
    with target.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ManifestError(f"{target}: truncated safetensors length")
        header_bytes = struct.unpack("<Q", raw)[0]
        if header_bytes <= 0 or header_bytes > target.stat().st_size - 8:
            raise ManifestError(f"{target}: invalid safetensors header length")
        raw_header = handle.read(header_bytes)
        if len(raw_header) != header_bytes:
            raise ManifestError(f"{target}: truncated safetensors header")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{target}: invalid safetensors JSON") from exc
    if not isinstance(header, dict):
        raise ManifestError(f"{target}: header must be an object")
    header.pop("__metadata__", None)
    return header, 8 + header_bytes


def _layer_id(name: str) -> int | None:
    match = _LAYER_RE.match(name)
    return int(match.group(1)) if match else None


def _is_expert(name: str) -> bool:
    return _EXPERT_MARKER in name


def _validate_tensor_entry(path: Path, name: str, value: object) -> tuple[int, int, str, list[int]]:
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: {name}: tensor metadata is not an object")
    try:
        start, end = value["data_offsets"]
        dtype = value["dtype"]
        shape = value["shape"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"{path}: {name}: malformed tensor metadata") from exc
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        raise ManifestError(f"{path}: {name}: invalid data offsets")
    if not isinstance(dtype, str) or not isinstance(shape, list) or not all(
        isinstance(dim, int) and dim >= 0 for dim in shape
    ):
        raise ManifestError(f"{path}: {name}: invalid dtype or shape")
    return start, end, dtype, shape


def build_manifest(source_dir: str | os.PathLike[str]) -> dict[str, object]:
    source = Path(source_dir)
    shards = sorted(source.glob("model-*.safetensors"))
    if not shards:
        raise ManifestError(f"{source}: no model-*.safetensors shards found")

    layers: dict[int, LayerRecord] = {}
    globals_: list[GlobalRecord] = []
    total_expert_bytes = 0
    total_trunk_bytes = 0

    for shard in shards:
        header, data_start = read_safetensors_header(shard)
        entries: list[tuple[str, int, int, str, list[int]]] = []
        for name, value in header.items():
            start, end, dtype, shape = _validate_tensor_entry(shard, name, value)
            entries.append((name, start, end, dtype, shape))
        entries.sort(key=lambda item: item[1])

        previous_end = 0
        for name, start, end, _, _ in entries:
            if start != previous_end:
                raise ManifestError(
                    f"{shard}: safetensors data is not contiguous before {name}: "
                    f"expected {previous_end}, got {start}"
                )
            previous_end = end
        if data_start + previous_end != shard.stat().st_size:
            raise ManifestError(f"{shard}: header offsets do not cover the complete file")

        layer_ids = {
            layer_id
            for name, _, _, _, _ in entries
            if not _is_expert(name) and (layer_id := _layer_id(name)) is not None
        }
        if len(layer_ids) > 1:
            raise ManifestError(f"{shard}: contains trunk tensors from multiple layers: {layer_ids}")

        for name, start, end, dtype, shape in entries:
            if _is_expert(name):
                total_expert_bytes += end - start
            elif _layer_id(name) is None:
                globals_.append(
                    GlobalRecord(
                        shard=shard.name,
                        name=name,
                        dtype=dtype,
                        shape=shape,
                        nbytes=end - start,
                    )
                )

        if not layer_ids:
            continue
        layer_id = next(iter(layer_ids))
        if layer_id in layers:
            raise ManifestError(f"layer {layer_id} appears in more than one shard")

        trunk_entries = [item for item in entries if not _is_expert(item[0])]
        if any(_layer_id(name) != layer_id for name, *_ in trunk_entries):
            foreign = [name for name, *_ in trunk_entries if _layer_id(name) != layer_id]
            raise ManifestError(
                f"{shard}: layer shard mixes globals or foreign tensors with layer {layer_id}: "
                f"{foreign[:3]}"
            )

        span_start = min(start for _, start, _, _, _ in trunk_entries)
        span_end = max(end for _, _, end, _, _ in trunk_entries)
        cursor = span_start
        tensor_records: list[TensorRecord] = []
        for name, start, end, dtype, shape in trunk_entries:
            if start != cursor:
                raise ManifestError(
                    f"{shard}: non-expert span is interleaved before {name}; "
                    "one-read layer reconstruction is impossible"
                )
            tensor_records.append(
                TensorRecord(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    relative_start=start - span_start,
                    relative_end=end - span_start,
                )
            )
            cursor = end
        if cursor != span_end:
            raise ManifestError(f"{shard}: non-expert span has a gap")
        if any(start < span_end for name, start, _, _, _ in entries if _is_expert(name)):
            raise ManifestError(
                f"{shard}: routed expert bytes are interleaved with the trunk span"
            )

        payload_bytes = span_end - span_start
        total_trunk_bytes += payload_bytes
        layers[layer_id] = LayerRecord(
            layer_id=layer_id,
            shard=shard.name,
            source_data_start=data_start,
            source_span_start=span_start,
            source_span_end=span_end,
            payload_bytes=payload_bytes,
            tensors=tensor_records,
        )

    ordered = [layers[layer_id] for layer_id in sorted(layers)]
    if ordered:
        expected = list(range(ordered[0].layer_id, ordered[-1].layer_id + 1))
        actual = [record.layer_id for record in ordered]
        if actual != expected:
            raise ManifestError(f"layer ids are not contiguous: {actual[:4]}...{actual[-4:]}")

    return {
        "format": "cerno-k3-trunk-manifest-v1",
        "source_dir": str(source.resolve()),
        "layers": [
            {
                **{key: value for key, value in asdict(record).items() if key != "tensors"},
                "tensors": [asdict(tensor) for tensor in record.tensors],
            }
            for record in ordered
        ],
        "globals": [asdict(record) for record in globals_],
        "summary": {
            "shards_scanned": len(shards),
            "layers": len(ordered),
            "trunk_payload_bytes": total_trunk_bytes,
            "expert_payload_bytes": total_expert_bytes,
            "global_payload_bytes": sum(record.nbytes for record in globals_),
            "max_layer_payload_bytes": max(
                (record.payload_bytes for record in ordered), default=0
            ),
        },
    }


def _copy_span(
    source_path: Path,
    *,
    absolute_start: int,
    length: int,
    block_bytes: int = 8 << 20,
) -> bytes:
    chunks: list[bytes] = []
    with source_path.open("rb") as handle:
        handle.seek(absolute_start)
        remaining = length
        while remaining:
            chunk = handle.read(min(block_bytes, remaining))
            if not chunk:
                raise ManifestError(f"{source_path}: short read while copying trunk span")
            chunks.append(chunk)
            remaining -= len(chunk)
    return b"".join(chunks)


def pack_manifest(
    manifest: dict[str, object],
    *,
    source_dir: str | os.PathLike[str],
    output_bin: str | os.PathLike[str],
    output_json: str | os.PathLike[str],
) -> dict[str, object]:
    source = Path(source_dir)
    layer_payloads: list[bytes] = []
    packed_layers: list[dict[str, object]] = []

    raw_layers = manifest.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ManifestError("manifest contains no layers")
    for raw in raw_layers:
        if not isinstance(raw, dict):
            raise ManifestError("invalid layer record")
        shard = source / str(raw["shard"])
        data_start = int(raw["source_data_start"])
        span_start = int(raw["source_span_start"])
        payload_bytes = int(raw["payload_bytes"])
        payload = _copy_span(
            shard,
            absolute_start=data_start + span_start,
            length=payload_bytes,
        )
        layer_payloads.append(payload)
        packed_layers.append(
            {
                "layer_id": int(raw["layer_id"]),
                "payload_bytes": payload_bytes,
                "source_shard": shard.name,
                "tensors": raw["tensors"],
            }
        )

    entries = pack_trunk(output_bin, layer_payloads)
    for packed, entry in zip(packed_layers, entries):
        packed["packed_offset"] = entry.offset
        packed["packed_length"] = entry.length
        packed["crc32"] = entry.crc32

    output = {
        "format": "cerno-k3-packed-trunk-v1",
        "source_manifest_format": manifest.get("format"),
        "trunk_file": str(Path(output_bin).name),
        "layers": packed_layers,
        "summary": {
            "layers": len(packed_layers),
            "payload_bytes": sum(int(layer["payload_bytes"]) for layer in packed_layers),
            "file_bytes": Path(output_bin).stat().st_size,
        },
    }
    target = Path(output_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trunk-bin", type=Path)
    parser.add_argument("--packed-manifest", type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.source_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))

    if bool(args.trunk_bin) != bool(args.packed_manifest):
        raise SystemExit("--trunk-bin and --packed-manifest must be provided together")
    if args.trunk_bin and args.packed_manifest:
        packed = pack_manifest(
            manifest,
            source_dir=args.source_dir,
            output_bin=args.trunk_bin,
            output_json=args.packed_manifest,
        )
        print(json.dumps(packed["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
